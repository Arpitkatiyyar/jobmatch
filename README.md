# JobMatch Agent: Production AI Career & Job Matching Platform

An end-to-end, production-ready AI career advisor and job matching agent featuring **PostgreSQL 16 `pgvector`** dense vector retrieval, **PostgreSQL `tsvector`** full-text search, multi-turn stateful memory, and a sleek split-screen UI.

---

## Architecture Diagram

```mermaid
flowchart TD
    User([User / Browser UI]) <-->|HTTP / JSON| FastAPI[FastAPI Server - app/main.py]

    subgraph Agent Loop [app/agent.py]
        FastAPI <--> AgentOrchestrator[Stateful Agent Orchestrator]
        AgentOrchestrator <--> LLM[LLM Client - app/llm.py]
        
        subgraph Tools
            AgentOrchestrator --> Tool1[Tool 1: update_user_preferences]
            AgentOrchestrator --> Tool2[Tool 2: hybrid_job_search]
            
            Tool2 --> StepA[Step A: Hard Metadata SQL Filter\nRemote / Stipend / Excluded Skills]
            StepA --> StepB[Step B: Dense Cosine + Lexical tsvector\nWeighted Score Fusion 0.7:0.3]
            StepB --> StepC[Step C: LLM Re-Ranking & Grounded Reasoning\nTop 5 Ranked Jobs + Match Scores]
        end
    end

    subgraph Database [PostgreSQL 16 + pgvector]
        Tool1 <-->|Upsert Profile & History| UserProfiles[(user_profiles table)]
        StepA & StepB <-->|HNSW Index + GIN tsvector| JobsTable[(jobs table - 100 Postings)]
    end

    subgraph Evaluation Harness [eval.py]
        Golden[golden_dataset.json - 15 Test Cases] --> EvalRunner[eval.py Suite]
        EvalRunner --> Report[EVAL_REPORT.md]
    end
```

---

## Key Features

1. **Hybrid Retrieval (Dense + Lexical)**:
   - Dense cosine similarity using PostgreSQL 16 `pgvector` (`gemini-embedding-001`, 768 dimensions).
   - Lexical search using PostgreSQL native `tsvector` and `ts_rank_cd`.
   - Weighted fusion score (`0.7 * vector + 0.3 * lexical`) retrieves the top 15 candidates before LLM re-ranking.

2. **Strict Metadata & Negative Filtering**:
   - Zero-compromise adherence to `remote_only`, `min_stipend_inr`, and negative constraints (`excluded_skills` such as "exclude React", "hate frontend", "no PHP").

3. **Persistent Multi-Turn Memory**:
   - Stores evolving candidate preferences in PostgreSQL `user_profiles`. Preferences update dynamically across conversation turns.

4. **Split-Screen Web UI**:
   - **Left Panel:** Chat history, quick suggestion chips, and responsive input.
   - **Right Panel:** Real-time synchronized memory badges and Top-5 ranked job match cards with grounded AI match reasons.

5. **Automated Evaluation Suite (`eval.py`)**:
   - 15 comprehensive golden test cases measuring Constraint Adherence Rate, Retrieval Recall@5, and LLM-as-a-Judge Groundedness (1.0–5.0).

---

## Project Structure

```
├── docker-compose.yml          # PostgreSQL 16 with pgvector extension
├── .env.example                # Configuration template
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation & Mermaid architecture
├── golden_dataset.json         # 15 golden evaluation test cases
├── seed.py                     # Idempotent 100-job catalog seeding script
├── eval.py                     # Automated evaluation benchmark harness
├── EVAL_REPORT.md              # Generated benchmark report
├── app/
│   ├── __init__.py
│   ├── db.py                   # Sync SQLAlchemy + psycopg models & vector setup
│   ├── llm.py                  # Gemini & OpenAI client + structured schemas
│   ├── agent.py                # Stateful agent loop, memory tool & hybrid search
│   └── main.py                 # FastAPI routes & static file serving
└── static/
    └── index.html              # Split-screen UI (Tailwind CSS + Vanilla JS)
```

---

## Quick Start Setup

### 1. Start PostgreSQL with pgvector
```bash
docker-compose up -d
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and set your API key:
```bash
cp .env.example .env
```
Ensure `.env` contains:
```env
LLM_PROVIDER=gemini
LLM_API_KEY=your_gemini_api_key_here
EMBEDDING_DIM=768
DATABASE_URL=postgresql+psycopg://jobmatch_user:jobmatch_password@localhost:5432/jobmatch_db
```

### 3. Install Dependencies
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Seed 100 Indian Tech Jobs
```bash
python seed.py
```

### 5. Run Automated Evaluation Benchmark
```bash
python eval.py
```

### 6. Start the Web Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at [http://localhost:8000](http://localhost:8000).
