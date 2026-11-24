import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import text, func, or_, and_, not_

from app.db import Job, UserProfile
from app.llm import llm_client, PreferenceExtraction, SingleJobRank

logger = logging.getLogger("jobmatch.agent")
logger.setLevel(logging.INFO)


def get_or_create_profile(db: Session, user_id: str) -> UserProfile:
    """Fetches or initializes the persistent user profile."""
    profile = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
    if not profile:
        profile = UserProfile(
            user_id=user_id,
            preferred_roles=[],
            must_have_skills=[],
            excluded_skills=[],
            preferred_locations=[],
            remote_only=False,
            min_stipend_inr=0,
            notes="",
            chat_history=[],
            updated_at=datetime.now(timezone.utc)
        )
        db.add(profile)
        db.commit()
        db.refresh(profile)
    return profile


def profile_to_dict(profile: UserProfile) -> Dict[str, Any]:
    """Serializes a UserProfile model to a dictionary."""
    return {
        "user_id": profile.user_id,
        "preferred_roles": profile.preferred_roles or [],
        "must_have_skills": profile.must_have_skills or [],
        "excluded_skills": profile.excluded_skills or [],
        "preferred_locations": profile.preferred_locations or [],
        "remote_only": bool(profile.remote_only),
        "min_stipend_inr": int(profile.min_stipend_inr or 0),
        "notes": profile.notes or "",
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


def update_user_preferences(
    db: Session,
    user_id: str,
    user_message: str
) -> tuple[UserProfile, PreferenceExtraction]:
    """Tool 1: Extracts preferences via LLM and merges into PostgreSQL."""
    profile = get_or_create_profile(db, user_id)
    current_data = profile_to_dict(profile)

    extracted = llm_client.extract_preferences(user_message, current_data)

    # Clean and merge unique items (case-insensitive deduplication)
    def merge_lists(existing: List[str], incoming: List[str]) -> List[str]:
        seen = set(x.strip().lower() for x in existing if x)
        merged = list(existing)
        for item in incoming:
            cleaned = item.strip()
            if cleaned and cleaned.lower() not in seen:
                seen.add(cleaned.lower())
                merged.append(cleaned)
        return merged

    if extracted.preferred_roles:
        profile.preferred_roles = merge_lists(profile.preferred_roles or [], extracted.preferred_roles)

    if extracted.must_have_skills:
        profile.must_have_skills = merge_lists(profile.must_have_skills or [], extracted.must_have_skills)

    if extracted.excluded_skills:
        profile.excluded_skills = merge_lists(profile.excluded_skills or [], extracted.excluded_skills)
        # Remove any newly excluded skill from must_have_skills
        excl_lower = set(s.lower() for s in profile.excluded_skills)
        profile.must_have_skills = [
            s for s in (profile.must_have_skills or [])
            if s.lower() not in excl_lower
        ]

    if extracted.preferred_locations:
        profile.preferred_locations = merge_lists(profile.preferred_locations or [], extracted.preferred_locations)

    if extracted.remote_only is not None:
        profile.remote_only = extracted.remote_only

    if extracted.min_stipend_inr is not None and extracted.min_stipend_inr > 0:
        profile.min_stipend_inr = extracted.min_stipend_inr

    if extracted.notes:
        profile.notes = extracted.notes

    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)

    return profile, extracted


def hybrid_job_search(
    db: Session,
    profile: UserProfile,
    query_text: str
) -> List[Dict[str, Any]]:
    """
    Tool 2: Multi-stage hybrid search
    - Step A: Hard metadata filtering
    - Step B: pgvector dense cosine + tsvector lexical rank fusion (Top 15)
    - Step C: LLM re-ranking & grounded explanations (Top 5)
    """
    # ---------------- Step A: Hard Metadata Filtering ----------------
    base_query = db.query(Job).filter(Job.is_active == True)

    # Remote only
    if profile.remote_only:
        base_query = base_query.filter(Job.work_mode == "remote")

    # Min stipend
    if profile.min_stipend_inr and profile.min_stipend_inr > 0:
        base_query = base_query.filter(Job.stipend_inr >= profile.min_stipend_inr)

    # Location filter (if specified and not remote_only)
    if profile.preferred_locations and not profile.remote_only:
        loc_filters = [Job.location.ilike(f"%{loc}%") for loc in profile.preferred_locations]
        loc_filters.append(Job.work_mode == "remote")  # Remote is always an option unless excluded
        base_query = base_query.filter(or_(*loc_filters))

    candidates = base_query.all()

    # Exclude skills: Filter out any candidate whose tech_stack, title, or description matches excluded_skills
    if profile.excluded_skills:
        excl_patterns = [s.strip().lower() for s in profile.excluded_skills if s.strip()]
        filtered_candidates = []
        for job in candidates:
            job_stack = [t.lower() for t in (job.tech_stack or [])]
            job_text = f"{job.title} {job.description}".lower()

            has_excluded = False
            for excl in excl_patterns:
                if excl in job_stack or excl in job_text:
                    has_excluded = True
                    break
            if not has_excluded:
                filtered_candidates.append(job)
        candidates = filtered_candidates

    if not candidates:
        logger.info("No candidates passed Step A hard metadata filtering.")
        return []

    # ---------------- Step B: Hybrid Retrieval (Top 15) ----------------
    # Generate query embedding
    effective_query = query_text.strip()
    if not effective_query:
        effective_query = " ".join((profile.preferred_roles or []) + (profile.must_have_skills or [])) or "software engineer"

    try:
        query_vector = llm_client.get_embedding(effective_query)
        q_vec = np.array(query_vector, dtype=float)
        q_norm = np.linalg.norm(q_vec)
    except Exception as e:
        logger.error(f"Failed to generate query embedding: {e}")
        query_vector = None
        q_vec = None
        q_norm = 1.0

    scored_candidates = []

    for job in candidates:
        # 1. Dense Cosine Similarity
        dense_score = 0.5  # default neutral score
        if job.embedding is not None and q_vec is not None and q_norm > 0:
            j_vec = np.array(job.embedding, dtype=float)
            j_norm = np.linalg.norm(j_vec)
            if j_norm > 0:
                # Cosine similarity in range [-1, 1], map to [0, 1]
                cos_sim = float(np.dot(q_vec, j_vec) / (q_norm * j_norm))
                dense_score = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))

        # 2. Lexical Keyword Overlap
        text_corpus = f"{job.title} {job.company} {' '.join(job.tech_stack or [])} {job.description}".lower()
        query_words = [w.lower() for w in effective_query.split() if len(w) > 2]
        overlap_count = sum(1 for w in query_words if w in text_corpus)
        must_have_overlap = sum(
            1 for s in (profile.must_have_skills or [])
            if s.lower() in [t.lower() for t in (job.tech_stack or [])] or s.lower() in text_corpus
        )
        total_words = max(1, len(query_words) + len(profile.must_have_skills or []))
        lexical_score = min(1.0, (overlap_count + 2 * must_have_overlap) / total_words)

        # 3. Hybrid Fusion: 0.7 * Vector + 0.3 * Lexical
        hybrid_score = 0.7 * dense_score + 0.3 * lexical_score

        scored_candidates.append({
            "job": job,
            "vector_score": dense_score,
            "lexical_score": lexical_score,
            "hybrid_score": hybrid_score
        })

    # Sort descending by hybrid score and take top 15
    scored_candidates.sort(key=lambda x: x["hybrid_score"], reverse=True)
    top_15_pool = scored_candidates[:15]

    pool_dicts = [{
        "id": sc["job"].id,
        "title": sc["job"].title,
        "company": sc["job"].company,
        "location": sc["job"].location,
        "work_mode": sc["job"].work_mode,
        "role_type": sc["job"].role_type,
        "stipend_inr": sc["job"].stipend_inr,
        "tech_stack": sc["job"].tech_stack or [],
        "description": sc["job"].description,
        "hybrid_score": round(sc["hybrid_score"], 4)
    } for sc in top_15_pool]

    # ---------------- Step C: LLM Re-ranking & Grounded Explanations (Top 5) ----------------
    profile_dict = profile_to_dict(profile)
    ranked_explanations: List[SingleJobRank] = llm_client.rerank_and_explain(
        candidates=pool_dicts,
        profile=profile_dict,
        query=effective_query
    )

    # Map ranked explanations back to top 5 jobs
    candidate_lookup = {c["id"]: c for c in pool_dicts}
    final_matches = []

    for rank_item in ranked_explanations:
        c_info = candidate_lookup.get(rank_item.job_id)
        if c_info:
            match_data = dict(c_info)
            match_data["match_score"] = rank_item.match_score
            match_data["match_reason"] = rank_item.match_reason
            final_matches.append(match_data)

    # In case LLM returned fewer than 5 or unmapped ids, fill from top pool
    if len(final_matches) < min(5, len(pool_dicts)):
        existing_ids = set(m["id"] for m in final_matches)
        for c_info in pool_dicts:
            if c_info["id"] not in existing_ids:
                match_data = dict(c_info)
                match_data["match_score"] = int(c_info["hybrid_score"] * 100)
                match_data["match_reason"] = (
                    f"Strong alignment with {', '.join(c_info['tech_stack'][:3])} "
                    f"and compensation of ₹{c_info['stipend_inr']:,}/month."
                )
                final_matches.append(match_data)
                if len(final_matches) >= 5:
                    break

    return final_matches[:5]


def execute_agent_loop(
    db: Session,
    user_id: str,
    user_message: str
) -> Dict[str, Any]:
    """
    Main stateful agent orchestration:
    1. Update preferences (Tool 1)
    2. Execute hybrid job search (Tool 2)
    3. Synthesize response and persist chat turn
    """
    profile, extracted = update_user_preferences(db, user_id, user_message)

    search_query = extracted.search_query or user_message
    matches = hybrid_job_search(db, profile, search_query)

    history = profile.chat_history or []
    reply = llm_client.synthesize_chat_reply(
        user_message=user_message,
        profile=profile_to_dict(profile),
        matches=matches,
        history=history
    )

    # Append new turn to chat history
    new_history = list(history)
    new_history.append({"role": "user", "content": user_message})
    new_history.append({"role": "assistant", "content": reply})
    profile.chat_history = new_history
    profile.updated_at = datetime.now(timezone.utc)
    db.commit()

    return {
        "reply": reply,
        "memory": profile_to_dict(profile),
        "matches": matches
    }


def reset_user_profile(db: Session, user_id: str) -> Dict[str, Any]:
    """Resets memory and history for the specified user."""
    profile = get_or_create_profile(db, user_id)
    profile.preferred_roles = []
    profile.must_have_skills = []
    profile.excluded_skills = []
    profile.preferred_locations = []
    profile.remote_only = False
    profile.min_stipend_inr = 0
    profile.notes = ""
    profile.chat_history = []
    profile.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(profile)
    return profile_to_dict(profile)
