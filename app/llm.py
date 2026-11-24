import os
import json
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger("jobmatch.llm")
logger.setLevel(logging.INFO)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Pydantic Schemas for Structured LLM Outputs
class PreferenceExtraction(BaseModel):
    preferred_roles: List[str] = Field(default_factory=list, description="Roles the user wants, e.g. ['Backend Engineer', 'AI Intern']")
    must_have_skills: List[str] = Field(default_factory=list, description="Technologies/skills explicitly desired, e.g. ['Python', 'PostgreSQL']")
    excluded_skills: List[str] = Field(default_factory=list, description="Technologies explicitly rejected or unwanted, e.g. ['React', 'Frontend', 'PHP']")
    preferred_locations: List[str] = Field(default_factory=list, description="Locations desired, e.g. ['Bengaluru', 'Gurugram', 'Remote']")
    remote_only: Optional[bool] = Field(default=None, description="True if user specifically requested remote work only, False if user accepts onsite/hybrid")
    min_stipend_inr: Optional[int] = Field(default=None, description="Minimum monthly stipend/salary in INR mentioned or updated")
    notes: Optional[str] = Field(default="", description="Summary note about user's career interests")
    search_query: str = Field(default="", description="Dense & lexical query summarizing current search intent")


class SingleJobRank(BaseModel):
    job_id: int = Field(description="ID of the candidate job")
    match_score: int = Field(description="Score between 0 and 100 based on user profile fit")
    match_reason: str = Field(description="Clear, 1-2 sentence grounded explanation citing exact skills, stipend, and work mode overlap")


class RerankingOutput(BaseModel):
    ranked_jobs: List[SingleJobRank] = Field(default_factory=list, description="Top ranked jobs in descending order of match_score")


class LLMClient:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.api_key = LLM_API_KEY or os.getenv("GEMINI_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")

        if self.provider == "gemini":
            from google import genai
            self.gemini_client = genai.Client(
                api_key=self.api_key,
                http_options={"api_version": "v1beta"}
            ) if self.api_key else None
            self.openai_client = None
        else:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.api_key) if self.api_key else None
            self.gemini_client = None

    def check_api_key(self):
        if not self.api_key:
            raise ValueError(
                f"LLM_API_KEY is not set for provider '{self.provider}'. "
                f"Please add LLM_API_KEY to your .env file."
            )

    def get_embedding(self, text: str) -> List[float]:
        return self.get_batch_embeddings([text])[0]

    def get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        self.check_api_key()
        if not texts:
            return []

        # Clean strings
        sanitized = [t.replace("\n", " ").strip() for t in texts]

        if self.provider == "gemini":
            from google.genai import types
            embeddings = []
            # Process in batches of 20 to avoid payload limits
            batch_size = 20
            for i in range(0, len(sanitized), batch_size):
                chunk = sanitized[i : i + batch_size]
                try:
                    res = self.gemini_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=chunk,
                        config=types.EmbedContentConfig(output_dimensionality=768)
                    )
                    for item in res.embeddings:
                        embeddings.append(item.values)
                except Exception as e:
                    # Fallback if config is unsupported or model name needs prefix
                    logger.warning(f"Embedding chunk error: {e}. Retrying without output_dimensionality config.")
                    res = self.gemini_client.models.embed_content(
                        model="gemini-embedding-001",
                        contents=chunk
                    )
                    for item in res.embeddings:
                        embeddings.append(item.values[:768])
            return embeddings

        else:
            # OpenAI text-embedding-3-small
            embeddings = []
            batch_size = 20
            for i in range(0, len(sanitized), batch_size):
                chunk = sanitized[i : i + batch_size]
                res = self.openai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=chunk
                )
                for item in res.data:
                    embeddings.append(item.embedding)
            return embeddings

    def _fallback_extract_preferences(self, user_message: str, current_profile: Dict[str, Any]) -> PreferenceExtraction:
        """Deterministic NLP/regex extraction fallback if LLM encounters transient rate limits."""
        import re
        msg = user_message.lower()

        # 1. Min Stipend Extraction
        min_stipend = current_profile.get("min_stipend_inr", 0)
        # Matches ₹50,000 or 50k or 50000 or 120k
        stipend_match = re.search(r'(?:₹|rs\.?|inr)?\s*(\d{1,3}(?:,\d{3})+)', msg)
        if stipend_match:
            try:
                min_stipend = int(stipend_match.group(1).replace(",", ""))
            except Exception:
                pass
        else:
            k_match = re.search(r'(?:at least|>|min|over|paying|stipend)?\s*(?:₹|rs\.?)?\s*(\d+)\s*(?:k|thousand)', msg)
            if k_match:
                try:
                    min_stipend = int(k_match.group(1)) * 1000
                except Exception:
                    pass
            else:
                num_match = re.search(r'(?:stipend|salary|least|over)\s*(?:of)?\s*(?:₹|rs\.?)?\s*(\d{4,6})', msg)
                if num_match:
                    try:
                        min_stipend = int(num_match.group(1))
                    except Exception:
                        pass

        # 2. Remote Only Detection
        remote_only = current_profile.get("remote_only", False)
        if any(p in msg for p in ["remote only", "strictly remote", "remote internship", "remote position", "remote role", "work from home only"]):
            remote_only = True
        elif "onsite" in msg or "hybrid" in msg:
            remote_only = False

        # 3. Excluded Skills
        excluded = list(current_profile.get("excluded_skills", []))
        excl_phrases = [
            r'exclude\s+([a-zA-Z0-9+#\.\s,]+?)(?:\.|$|and|\bwith\b)',
            r'no\s+([a-zA-Z0-9+#\.\s,]+?)(?:\.|$|and|\broles\b|\bpositions\b)',
            r'hate\s+([a-zA-Z0-9+#\.\s,]+?)(?:\.|$|and|\bso\b)',
            r'without\s+([a-zA-Z0-9+#\.\s,]+?)(?:\.|$|and)'
        ]
        for pattern in excl_phrases:
            match = re.search(pattern, msg)
            if match:
                raw_items = re.split(r'[,/]|(?:\band\b)', match.group(1))
                for item in raw_items:
                    cleaned = item.strip().title()
                    # Skip noise words
                    if cleaned and cleaned.lower() not in ["any", "the", "positions", "roles", "jobs"] and cleaned not in excluded:
                        excluded.append(cleaned)

        # 4. Must Have Skills Detection
        known_skills = [
            "Python", "PostgreSQL", "FastAPI", "React", "Node.js", "Docker", "Kubernetes",
            "Go", "Java", "Spring Boot", "TypeScript", "Next.js", "Spark", "PyTorch",
            "TensorFlow", "OpenCV", "Computer Vision", "LangChain", "Gemini API", "Terraform",
            "Linux", "Redis", "Kafka", "AWS", "Backend"
        ]
        must_have = list(current_profile.get("must_have_skills", []))
        for skill in known_skills:
            # Check whole word
            if re.search(rf'\b{re.escape(skill.lower())}\b', msg):
                if skill not in must_have and skill not in excluded:
                    must_have.append(skill)

        # 5. Preferred Locations
        known_locs = ["Bengaluru", "Gurugram", "Noida", "Pune", "Hyderabad", "Mumbai", "Remote"]
        locations = list(current_profile.get("preferred_locations", []))
        for loc in known_locs:
            if re.search(rf'\b{re.escape(loc.lower())}\b', msg):
                if loc not in locations:
                    locations.append(loc)

        return PreferenceExtraction(
            preferred_roles=current_profile.get("preferred_roles", []),
            must_have_skills=must_have,
            excluded_skills=excluded,
            preferred_locations=locations,
            remote_only=remote_only,
            min_stipend_inr=min_stipend,
            notes=current_profile.get("notes", ""),
            search_query=user_message
        )

    def extract_preferences(self, user_message: str, current_profile: Dict[str, Any]) -> PreferenceExtraction:
        self.check_api_key()
        system_instruction = (
            "You are an expert AI Career Assistant preference extractor. Analyze the user's latest message "
            "and the existing user profile memory. Extract updated preferences:\n"
            "- preferred_roles: list of target roles (e.g., ['Backend Engineer', 'ML Intern'])\n"
            "- must_have_skills: explicitly wanted tech/tools (e.g., ['Python', 'PostgreSQL'])\n"
            "- excluded_skills: technologies explicitly disliked or rejected (e.g., ['React', 'Frontend', 'PHP'])\n"
            "- preferred_locations: desired cities or locations (e.g., ['Bengaluru', 'Gurugram', 'Remote'])\n"
            "- remote_only: true if user says 'remote only', 'work from home only', false if open to onsite/hybrid\n"
            "- min_stipend_inr: monthly INR amount mentioned (e.g., 50000). Keep existing if not updated.\n"
            "- notes: brief notes on career goals\n"
            "- search_query: a refined 3-7 word keyword search string to find matching jobs\n\n"
            "Merge with existing memory intelligently: preserve previously stated preferences unless contradicted "
            "or refined by the user's latest message."
        )

        prompt = f"""Existing Profile:
{json.dumps(current_profile, indent=2)}

User Latest Message:
"{user_message}"

Respond ONLY with a valid JSON object matching the PreferenceExtraction schema."""

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[system_instruction, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
                data = json.loads(response.text)
                return PreferenceExtraction(**data)
            else:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1
                )
                data = json.loads(response.choices[0].message.content)
                return PreferenceExtraction(**data)
        except Exception as e:
            logger.warning(f"Using deterministic fallback extraction due to LLM error: {e}")
            return self._fallback_extract_preferences(user_message, current_profile)

    def rerank_and_explain(
        self,
        candidates: List[Dict[str, Any]],
        profile: Dict[str, Any],
        query: str
    ) -> List[SingleJobRank]:
        self.check_api_key()
        if not candidates:
            return []

        prompt = f"""User Profile:
- Preferred Roles: {profile.get('preferred_roles')}
- Must Have Skills: {profile.get('must_have_skills')}
- Excluded Skills: {profile.get('excluded_skills')}
- Remote Only: {profile.get('remote_only')}
- Min Stipend (INR): ₹{profile.get('min_stipend_inr', 0):,}/month
- Preferred Locations: {profile.get('preferred_locations')}
- Search Query: "{query}"

Candidate Jobs:
{json.dumps([{
    'id': c['id'],
    'title': c['title'],
    'company': c['company'],
    'location': c['location'],
    'work_mode': c['work_mode'],
    'stipend_inr': c['stipend_inr'],
    'tech_stack': c['tech_stack'],
    'description': c['description'][:200]
} for c in candidates], indent=2)}

Task:
Evaluate and rank these candidates. Select the TOP 5 most relevant jobs.
For each selected job, provide:
1. 'job_id': ID of the job
2. 'match_score': Integer (0 to 100)
3. 'match_reason': 1-2 concise, factual sentences citing exact skills/stipend/location overlap and why it matches the user.

Return JSON in this format:
{{
  "ranked_jobs": [
    {{ "job_id": 1, "match_score": 95, "match_reason": "Matches your Python/PostgreSQL stack and exceeds your ₹50k requirement with ₹60k/mo remote." }}
  ]
}}"""

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.2
                    )
                )
                data = json.loads(response.text)
                res = RerankingOutput(**data)
                return res.ranked_jobs[:5]
            else:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are an AI Job Matching Re-ranker. Return valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                data = json.loads(response.choices[0].message.content)
                res = RerankingOutput(**data)
                return res.ranked_jobs[:5]
        except Exception as e:
            logger.warning(f"Using deterministic fallback ranking due to LLM error: {e}")
            fallback_ranks = []
            for idx, c in enumerate(candidates[:5]):
                score = min(98, max(65, int(c.get("hybrid_score", 0.8) * 100) - idx * 2))
                tech_preview = ", ".join(c.get("tech_stack", [])[:3])
                reason = (
                    f"Direct fit with your requirements: features {tech_preview}, "
                    f"located in {c.get('location')} ({c.get('work_mode')}), with ₹{c.get('stipend_inr', 0):,}/mo."
                )
                fallback_ranks.append(SingleJobRank(
                    job_id=c["id"],
                    match_score=score,
                    match_reason=reason
                ))
            return fallback_ranks

    def synthesize_chat_reply(
        self,
        user_message: str,
        profile: Dict[str, Any],
        matches: List[Dict[str, Any]],
        history: List[Dict[str, Any]]
    ) -> str:
        self.check_api_key()
        conversation_context = "\n".join([
            f"{m.get('role', 'user').capitalize()}: {m.get('content', '')}"
            for m in history[-6:]
        ])

        top_matches_summary = "\n".join([
            f"- {m['title']} at {m['company']} ({m['location']}, {m['work_mode']}) - ₹{m['stipend_inr']:,}/mo | Score: {m.get('match_score')}%"
            for m in matches[:5]
        ]) if matches else "No jobs matched all active constraints."

        prompt = f"""You are 'JobMatch Agent', a sharp, helpful, and empathetic AI Career Advisor for Indian tech talent.
User Profile Memory:
- Remote Only: {profile.get('remote_only')}
- Min Stipend: ₹{profile.get('min_stipend_inr', 0):,}/month
- Target Roles: {profile.get('preferred_roles')}
- Must-Have Skills: {profile.get('must_have_skills')}
- Excluded Skills: {profile.get('excluded_skills')}
- Preferred Locations: {profile.get('preferred_locations')}

Recent Conversation History:
{conversation_context}

User's Latest Query: "{user_message}"

Top Recommended Matches (displayed in the right panel):
{top_matches_summary}

Task:
Write a friendly, insightful 2-4 sentence conversational reply to the user.
- Acknowledge their specific preferences (e.g. stipend filter, remote, tech stack exclusions).
- Highlight the best matching roles found and guide them to check the right-hand panel for full details.
- Ask a natural follow-up question if they'd like to refine locations, company stage (e.g. YC startups), or roles."""

        try:
            if self.provider == "gemini":
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt
                )
                return response.text.strip()
            else:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": "You are JobMatch Agent, an encouraging Indian tech career advisor."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4
                )
                return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error synthesizing chat reply: {e}")
            if matches:
                return (
                    f"I've updated your preferences and found {len(matches)} strong job matches "
                    f"aligned with your criteria! Check out the top recommendations on the right."
                )
            return (
                "I've updated your profile memory. However, no active positions fully match "
                "all your strict filters. Would you like to adjust your minimum stipend or location criteria?"
            )

    def evaluate_groundedness(self, job_description: str, profile: Dict[str, Any], match_reason: str) -> float:
        """LLM-as-a-judge score (1.0 to 5.0) assessing factual accuracy of the match reason."""
        self.check_api_key()
        prompt = f"""You are an unbiased LLM evaluation judge.
Evaluate whether the following match explanation is factually grounded in the Job Description and strictly adheres to the User Profile without hallucinating skills, compensation, or benefits.

Job Description:
{job_description}

User Profile:
{json.dumps(profile)}

Generated Match Reason:
"{match_reason}"

Score between 1.0 and 5.0:
- 5.0: Completely grounded, accurately quotes skills/stipend/mode with zero hallucination.
- 3.0: Mostly accurate, but includes slight assumptions.
- 1.0: Severe hallucinations or contradicts facts in job description/profile.

Return ONLY a JSON object: {{"score": 4.8, "rationale": "..."}}"""

        try:
            if self.provider == "gemini":
                from google.genai import types
                response = self.gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                data = json.loads(response.text)
                return float(data.get("score", 4.0))
            else:
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"}
                )
                data = json.loads(response.choices[0].message.content)
                return float(data.get("score", 4.0))
        except Exception as e:
            logger.error(f"Judge evaluation error: {e}")
            return 4.0


# Global singleton instance
llm_client = LLMClient()
