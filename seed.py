import os
import sys
import logging
from typing import List, Dict, Any
from sqlalchemy import text, func
from sqlalchemy.orm import Session

from app.db import init_db, SessionLocal, Job, EMBEDDING_DIM
from app.llm import llm_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("seed")

RAW_SEED_JOBS: List[Dict[str, Any]] = [
    # --- AI / LLM / Agent Roles ---
    {
        "title": "AI Agent & Backend Intern",
        "company": "Sarvam AI",
        "location": "Bengaluru",
        "work_mode": "remote",
        "role_type": "internship",
        "stipend_inr": 60000,
        "tech_stack": ["Python", "FastAPI", "PostgreSQL", "LangChain", "Gemini API"],
        "description": "Design and build autonomous AI agents using Gemini and open-source LLMs. Build high-throughput async FastAPI backends with pgvector retrieval and tool calling pipelines."
    },
    {
        "title": "Generative AI Research Intern",
        "company": "Krutrim SI Designs",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 50000,
        "tech_stack": ["Python", "PyTorch", "Hugging Face", "Transformers", "CUDA"],
        "description": "Train and evaluate indicative Indian language foundation models. Fine-tune LoRA adapters and optimize inference latency with vLLM and TensorRT-LLM."
    },
    {
        "title": "Full-Stack AI Engineer",
        "company": "InVideo AI",
        "location": "Remote",
        "work_mode": "remote",
        "role_type": "full-time",
        "stipend_inr": 130000,
        "tech_stack": ["Python", "Next.js", "TypeScript", "PostgreSQL", "OpenAI", "FFmpeg"],
        "description": "Develop multimodal video generation workflows. Connect Next.js frontend with distributed Python worker pipelines using Celery, Redis, and OpenAI APIs."
    },
    {
        "title": "Junior Machine Learning Engineer",
        "company": "Wadhwani AI",
        "location": "Pune",
        "work_mode": "onsite",
        "role_type": "full-time",
        "stipend_inr": 75000,
        "tech_stack": ["Python", "TensorFlow", "Computer Vision", "Docker", "FastAPI"],
        "description": "Deploy edge computer vision models for agriculture and public health screening programs across rural India. Optimize ONNX models for mobile runtime."
    },
    {
        "title": "LLM Infrastructure Engineer",
        "company": "Yellow.ai",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "full-time",
        "stipend_inr": 140000,
        "tech_stack": ["Python", "Go", "Kubernetes", "vLLM", "Triton", "PostgreSQL"],
        "description": "Scale conversational enterprise agents handling 50M+ sessions. Maintain high-availability model inference clusters and pgvector embeddings indexing."
    },
    {
        "title": "AI Application Developer Intern",
        "company": "Cohesity Labs",
        "location": "Pune",
        "work_mode": "remote",
        "role_type": "internship",
        "stipend_inr": 45000,
        "tech_stack": ["Python", "LangChain", "VectorDB", "FastAPI", "Docker"],
        "description": "Build RAG enterprise search connectors for unstructured cloud data. Benchmark dense vector vs BM25 hybrid search pipelines."
    },
    {
        "title": "Applied AI Scientist (Search & Recommendations)",
        "company": "Zepto",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "full-time",
        "stipend_inr": 160000,
        "tech_stack": ["Python", "PyTorch", "Spark", "Redis", "Kafka", "PostgreSQL"],
        "description": "Build 10-minute grocery search rankers, personalized store feed recommendations, and real-time embedding matching serving millions of daily orders."
    },

    # --- Python / Backend / Distributed Systems ---
    {
        "title": "Backend Engineering Intern (Python)",
        "company": "Hasura",
        "location": "Bengaluru",
        "work_mode": "remote",
        "role_type": "internship",
        "stipend_inr": 55000,
        "tech_stack": ["Python", "PostgreSQL", "GraphQL", "Docker", "SQLAlchemy"],
        "description": "Build developer tooling and metadata connectors for PostgreSQL, Citus, and CockroachDB. Write performant SQL and async Python services."
    },
    {
        "title": "Backend SDE-1 (Python/FastAPI)",
        "company": "Postman",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "full-time",
        "stipend_inr": 115000,
        "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Redis", "AWS"],
        "description": "Build scalable backend APIs powering Postman's API monitoring and collaborative mock servers. High concurrency and strict API schema contracts."
    },
    {
        "title": "Python Backend Intern",
        "company": "Khatabook",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 40000,
        "tech_stack": ["Python", "Django", "PostgreSQL", "Celery", "Redis"],
        "description": "Help scale merchant ledger backend microservices handling millions of daily bookkeeping entries across tier 2/3 Indian businesses."
    },
    {
        "title": "Core Platform Engineer",
        "company": "Razorpay",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "full-time",
        "stipend_inr": 150000,
        "tech_stack": ["Go", "Python", "PostgreSQL", "Kafka", "Kubernetes"],
        "description": "Design ultra-reliable payment settlement routing and real-time transaction processing engines with five-nines uptime."
    },
    {
        "title": "Backend Systems Intern",
        "company": "Groww",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "internship",
        "stipend_inr": 50000,
        "tech_stack": ["Java", "Spring Boot", "PostgreSQL", "Kafka", "Redis"],
        "description": "Work on wealth management order execution routing, portfolio ledger calculations, and low-latency stock market data ingest."
    },
    {
        "title": "Distributed Systems SDE",
        "company": "Zerodha",
        "location": "Bengaluru",
        "work_mode": "remote",
        "role_type": "full-time",
        "stipend_inr": 145000,
        "tech_stack": ["Go", "Python", "PostgreSQL", "Redis", "Linux"],
        "description": "Work on the Kite trading platform backend. High throughput, zero-bloat architecture, and direct database optimizations without heavy ORMs."
    },
    {
        "title": "Database Platform Intern",
        "company": "BrowserStack",
        "location": "Remote",
        "work_mode": "remote",
        "role_type": "internship",
        "stipend_inr": 55000,
        "tech_stack": ["Python", "PostgreSQL", "Linux", "Docker", "Bash"],
        "description": "Optimize multi-tenant PostgreSQL clusters, connection poolers (PgBouncer), and automated database replication failovers."
    },
    {
        "title": "Backend Software Engineer (Golang/Python)",
        "company": "Cred",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "full-time",
        "stipend_inr": 165000,
        "tech_stack": ["Go", "Python", "PostgreSQL", "DynamoDB", "Kafka"],
        "description": "Power real-time rewards attribution, credit card bill payments, and merchant checkout APIs under massive flash sales."
    },
    {
        "title": "Python Data Backend Intern",
        "company": "Urban Company",
        "location": "Gurugram",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 45000,
        "tech_stack": ["Python", "FastAPI", "PostgreSQL", "Airflow", "ClickHouse"],
        "description": "Build high-throughput event logging pipelines and operational dashboards tracking service partner logistics and dispatch matching."
    },

    # --- Full Stack / Web Engineering ---
    {
        "title": "Full-Stack Developer Intern",
        "company": "Slice",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "internship",
        "stipend_inr": 45000,
        "tech_stack": ["React", "Node.js", "TypeScript", "PostgreSQL", "AWS"],
        "description": "Build frictionless customer onboarding flows and credit card management mobile-web portals with clean UI animations."
    },
    {
        "title": "Full-Stack SDE-1",
        "company": "Swiggy",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "full-time",
        "stipend_inr": 120000,
        "tech_stack": ["React", "TypeScript", "Java", "PostgreSQL", "Kafka"],
        "description": "Develop Swiggy Instamart customer and partner portal features. Optimize web bundle sizes, micro-frontends, and order tracking APIs."
    },
    {
        "title": "Full Stack Engineer (Remote)",
        "company": "Atlan",
        "location": "Remote",
        "work_mode": "remote",
        "role_type": "full-time",
        "stipend_inr": 125000,
        "tech_stack": ["Next.js", "Python", "FastAPI", "PostgreSQL", "Docker"],
        "description": "Help craft the modern data workspace interface. Build reactive metadata catalog interfaces and schema lineage graphs."
    },
    {
        "title": "Full-Stack Web Intern",
        "company": "PhysicsWallah",
        "location": "Noida",
        "work_mode": "onsite",
        "role_type": "internship",
        "stipend_inr": 30000,
        "tech_stack": ["React", "Node.js", "MongoDB", "Express", "Tailwind CSS"],
        "description": "Develop student test-taking interfaces, video lecture streaming player controls, and community doubt discussion forums."
    },
    {
        "title": "Frontend Engineer (React/TypeScript)",
        "company": "Licious",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "full-time",
        "stipend_inr": 95000,
        "tech_stack": ["React", "TypeScript", "Next.js", "Redux", "Tailwind CSS"],
        "description": "Develop responsive e-commerce web storefronts with server-side rendering, instant checkout, and cart sync."
    },
    {
        "title": "Frontend Engineering Intern",
        "company": "Meesho",
        "location": "Bengaluru",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 40000,
        "tech_stack": ["React", "JavaScript", "HTML5", "CSS3", "Webpack"],
        "description": "Work on the seller hub web application. Build performant product listing catalogs and analytics charts."
    },

    # --- Cloud / DevOps / SRE ---
    {
        "title": "DevOps & Cloud Engineer Intern",
        "company": "CleverTap",
        "location": "Mumbai",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 35000,
        "tech_stack": ["Docker", "Kubernetes", "Terraform", "AWS", "Python"],
        "description": "Automate multi-region AWS cloud deployments, write Infrastructure-as-Code with Terraform, and monitor multi-tenant Prometheus metrics."
    },
    {
        "title": "Site Reliability Engineer (SRE)",
        "company": "Flipkart",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "full-time",
        "stipend_inr": 135000,
        "tech_stack": ["Linux", "Kubernetes", "Python", "Go", "Prometheus", "Kafka"],
        "description": "Ensure ultra-reliability of the Big Billion Days shopping infrastructure. Build chaos engineering tests and auto-scaling rules."
    },
    {
        "title": "Cloud Platform Intern",
        "company": "Ather Energy",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "internship",
        "stipend_inr": 35000,
        "tech_stack": ["Python", "Docker", "AWS IoT", "PostgreSQL", "Grafana"],
        "description": "Assist in building IoT telemetry ingestion pipelines from 100,000+ connected electric smart scooters across India."
    },
    {
        "title": "SRE / Infrastructure Intern",
        "company": "Jupiter Money",
        "location": "Bengaluru",
        "work_mode": "remote",
        "role_type": "internship",
        "stipend_inr": 45000,
        "tech_stack": ["Linux", "Kubernetes", "Docker", "Python", "Bash"],
        "description": "Assist in provisioning secure banking microservice clusters, CI/CD GitHub Actions pipelines, and secret management."
    },

    # --- Data & ML Engineering ---
    {
        "title": "Data Engineering Intern",
        "company": "Ola Electric",
        "location": "Bengaluru",
        "work_mode": "onsite",
        "role_type": "internship",
        "stipend_inr": 38000,
        "tech_stack": ["Python", "SQL", "Apache Spark", "Airflow", "PostgreSQL"],
        "description": "Build high-volume ETL pipelines aggregating battery telemetry and manufacturing quality metrics from the Gigafactory."
    },
    {
        "title": "Data Platform Engineer",
        "company": "Delhivery",
        "location": "Gurugram",
        "work_mode": "hybrid",
        "role_type": "full-time",
        "stipend_inr": 110000,
        "tech_stack": ["Python", "Spark", "Kafka", "PostgreSQL", "Snowflake"],
        "description": "Process millions of package tracking milestones per minute. Build route optimization feature stores and data lakes."
    },
    {
        "title": "MLOps Intern",
        "company": "Fractal Analytics",
        "location": "Gurugram",
        "work_mode": "hybrid",
        "role_type": "internship",
        "stipend_inr": 35000,
        "tech_stack": ["Python", "MLflow", "Docker", "Kubeflow", "AWS"],
        "description": "Build automated ML model validation, drift detection, and canary deployment pipelines for Fortune 500 retail analytics."
    },
    {
        "title": "Computer Vision Engineer",
        "company": "Locus.sh",
        "location": "Bengaluru",
        "work_mode": "remote",
        "role_type": "full-time",
        "stipend_inr": 115000,
        "tech_stack": ["Python", "OpenCV", "PyTorch", "Docker", "C++"],
        "description": "Develop 3D volume estimation and parcel dimension scanning algorithms deployed at logistics sorting hubs."
    }
]

# Expand programmatically to reach 100 realistic, distinct postings across top Indian tech ecosystems
COMPANIES = [
    ("Sarvam AI", "Bengaluru", "AI Startup"),
    ("Krutrim SI", "Bengaluru", "AI Startup"),
    ("InVideo", "Remote", "Generative Video"),
    ("Hasura", "Bengaluru", "Developer Tooling"),
    ("Postman", "Bengaluru", "Developer API Platform"),
    ("Zepto", "Bengaluru", "Quick Commerce"),
    ("Swiggy", "Bengaluru", "Food Delivery & Logistics"),
    ("Cred", "Bengaluru", "Fintech"),
    ("Groww", "Bengaluru", "Wealthtech"),
    ("Zerodha", "Bengaluru", "Fintech / Brokerage"),
    ("Razorpay", "Bengaluru", "Payments Infrastructure"),
    ("BrowserStack", "Remote", "Cloud Infrastructure"),
    ("Atlan", "Remote", "Data Collaboration"),
    ("Khatabook", "Bengaluru", "SME SaaS"),
    ("Urban Company", "Gurugram", "Services Marketplace"),
    ("Delhivery", "Gurugram", "Logistics Tech"),
    ("PhysicsWallah", "Noida", "Edtech"),
    ("Ather Energy", "Bengaluru", "EV Tech"),
    ("Ola Electric", "Bengaluru", "EV Manufacturing"),
    ("Jupiter Money", "Bengaluru", "Neobank"),
    ("CleverTap", "Mumbai", "Retention Cloud"),
    ("Licious", "Bengaluru", "D2C Food"),
    ("Meesho", "Bengaluru", "E-commerce Marketplace"),
    ("Apna", "Bengaluru", "Jobs Platform"),
    ("Darwinbox", "Hyderabad", "HR SaaS"),
    ("Zenoti", "Hyderabad", "Enterprise SaaS"),
    ("HighRadius", "Hyderabad", "Fintech AI"),
    ("Druva", "Pune", "Cloud Data Protection"),
    ("Icertis", "Pune", "Contract Intelligence"),
    ("Mindtickle", "Pune", "Sales Readiness")
]

TECH_PROFILES = [
    {
        "role": "Backend Engineer",
        "role_type": "full-time",
        "stipend_range": (85000, 150000),
        "stacks": [
            ["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
            ["Python", "Django", "PostgreSQL", "Celery", "AWS"],
            ["Go", "PostgreSQL", "gRPC", "Docker", "Kubernetes"],
            ["Java", "Spring Boot", "PostgreSQL", "Kafka", "Redis"]
        ],
        "desc": "Architect and maintain fault-tolerant microservices, database schemas, and external API integrations with high reliability."
    },
    {
        "role": "AI / ML Engineer",
        "role_type": "full-time",
        "stipend_range": (100000, 175000),
        "stacks": [
            ["Python", "PyTorch", "Transformers", "FastAPI", "PostgreSQL"],
            ["Python", "LangChain", "VectorDB", "OpenAI", "PostgreSQL"],
            ["Python", "Gemini API", "pgvector", "FastAPI", "Docker"],
            ["Python", "TensorFlow", "CUDA", "vLLM", "Docker"]
        ],
        "desc": "Build and optimize production machine learning pipelines, LLM retrieval systems, and embeddings-based search engines."
    },
    {
        "role": "Software Engineering Intern (Backend)",
        "role_type": "internship",
        "stipend_range": (35000, 65000),
        "stacks": [
            ["Python", "FastAPI", "PostgreSQL", "Git", "Linux"],
            ["Python", "Django", "PostgreSQL", "REST APIs"],
            ["Go", "PostgreSQL", "Docker", "Linux"],
            ["Java", "Spring Boot", "MySQL", "PostgreSQL"]
        ],
        "desc": "Collaborate with senior engineers to implement REST endpoints, write comprehensive unit tests, and optimize database queries."
    },
    {
        "role": "AI Research & Development Intern",
        "role_type": "internship",
        "stipend_range": (40000, 70000),
        "stacks": [
            ["Python", "Hugging Face", "PyTorch", "pgvector", "FastAPI"],
            ["Python", "LangChain", "Gemini API", "VectorDB"],
            ["Python", "Computer Vision", "OpenCV", "TensorFlow"]
        ],
        "desc": "Conduct experimental benchmarks on state-of-the-art vision and language models, document findings, and prototype agent workflows."
    },
    {
        "role": "Full-Stack Developer",
        "role_type": "full-time",
        "stipend_range": (90000, 140000),
        "stacks": [
            ["Next.js", "React", "TypeScript", "Python", "PostgreSQL"],
            ["React", "Node.js", "TypeScript", "PostgreSQL", "Docker"],
            ["Vue.js", "Python", "FastAPI", "PostgreSQL", "Tailwind CSS"]
        ],
        "desc": "Deliver end-to-end features from slick responsive frontend components to robust transactional backend endpoints."
    },
    {
        "role": "DevOps / Cloud Engineer",
        "role_type": "full-time",
        "stipend_range": (95000, 155000),
        "stacks": [
            ["Kubernetes", "Docker", "Terraform", "AWS", "Python"],
            ["Linux", "GitHub Actions", "Docker", "Prometheus", "Bash"],
            ["GCP", "Kubernetes", "Terraform", "Go", "Helm"]
        ],
        "desc": "Automate containerized deployments, manage Kubernetes clusters, configure automated alerting, and reduce infrastructure costs."
    }
]

MODES = ["remote", "hybrid", "onsite"]


def generate_full_seed_list() -> List[Dict[str, Any]]:
    """Builds a deterministic 100-job catalog combining curated roles and structured expansions."""
    job_list = list(RAW_SEED_JOBS)
    idx = len(job_list)

    comp_len = len(COMPANIES)
    prof_len = len(TECH_PROFILES)

    while len(job_list) < 100:
        company_name, base_loc, domain = COMPANIES[idx % comp_len]
        prof = TECH_PROFILES[idx % prof_len]
        mode = MODES[(idx + (idx // 3)) % len(MODES)]
        stack = prof["stacks"][idx % len(prof["stacks"])]

        loc = "Remote" if mode == "remote" else base_loc
        stipend_min, stipend_max = prof["stipend_range"]
        # Step in increments of 5000
        steps = (stipend_max - stipend_min) // 5000
        stipend = stipend_min + ((idx * 3) % (steps + 1)) * 5000

        title = f"{prof['role']} - {domain.split()[0]}"
        desc = (
            f"Join {company_name} in building next-generation {domain.lower()} solutions. "
            f"{prof['desc']} You will work with {', '.join(stack[:3])}."
        )

        job_list.append({
            "title": title,
            "company": company_name,
            "location": loc,
            "work_mode": mode,
            "role_type": prof["role_type"],
            "stipend_inr": stipend,
            "tech_stack": stack,
            "description": desc
        })
        idx += 1

    return job_list[:100]


def seed_database():
    """Initializes schema and idempotently seeds 100 jobs with embeddings and tsvector."""
    logger.info("Initializing database extension and schema...")
    init_db()

    db: Session = SessionLocal()
    try:
        all_jobs = generate_full_seed_list()
        logger.info(f"Generated {len(all_jobs)} structured job postings.")

        # Check existing jobs to achieve idempotency
        existing = db.query(Job.title, Job.company).all()
        existing_set = set((j[0], j[1]) for j in existing)
        logger.info(f"Found {len(existing_set)} already existing jobs in database.")

        jobs_to_insert = [
            j for j in all_jobs
            if (j["title"], j["company"]) not in existing_set
        ]

        if not jobs_to_insert:
            logger.info("Database is already seeded with all 100 jobs. Nothing to do.")
            return

        logger.info(f"Generating embeddings for {len(jobs_to_insert)} new job postings...")
        # Prepare text for embedding
        embedding_texts = [
            f"{j['title']} at {j['company']}. Location: {j['location']} ({j['work_mode']}). "
            f"Tech Stack: {', '.join(j['tech_stack'])}. Stipend: ₹{j['stipend_inr']}/month. {j['description']}"
            for j in jobs_to_insert
        ]

        try:
            embeddings = llm_client.get_batch_embeddings(embedding_texts)
        except Exception as e:
            logger.warning(f"Could not fetch real LLM embeddings ({e}). Using normalized pseudo-embeddings.")
            # Deterministic pseudo-random embedding fallback so tests can run without blocking
            import numpy as np
            embeddings = []
            for i, text_val in enumerate(embedding_texts):
                np.random.seed(abs(hash(text_val)) % (2**31))
                vec = np.random.randn(EMBEDDING_DIM).astype(float)
                vec /= np.linalg.norm(vec)
                embeddings.append(vec.tolist())

        # Insert new jobs
        inserted_count = 0
        for job_data, emb in zip(jobs_to_insert, embeddings):
            new_job = Job(
                title=job_data["title"],
                company=job_data["company"],
                location=job_data["location"],
                work_mode=job_data["work_mode"],
                role_type=job_data["role_type"],
                stipend_inr=job_data["stipend_inr"],
                tech_stack=job_data["tech_stack"],
                description=job_data["description"],
                is_active=True,
                embedding=emb
            )
            db.add(new_job)
            inserted_count += 1

        db.commit()
        logger.info(f"Successfully inserted {inserted_count} jobs.")

        # Update tsvector column across all jobs
        logger.info("Updating PostgreSQL search_tsv tsvector indexes...")
        db.execute(text("""
            UPDATE jobs
            SET search_tsv = to_tsvector('english',
                coalesce(title, '') || ' ' ||
                coalesce(company, '') || ' ' ||
                coalesce(array_to_string(tech_stack, ' '), '') || ' ' ||
                coalesce(description, '')
            );
        """))
        db.commit()
        logger.info("Search tsvector generated and indexed successfully.")

        total = db.query(Job).count()
        logger.info(f"Database now contains {total} active jobs.")

    except Exception as e:
        db.rollback()
        logger.error(f"Seeding failed: {e}")
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    seed_database()
