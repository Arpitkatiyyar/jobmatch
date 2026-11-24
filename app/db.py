import os
from datetime import datetime, timezone
from typing import Generator
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Text,
    DateTime,
    Index,
    text,
    func
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TSVECTOR
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from pgvector.sqlalchemy import Vector

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://jobmatch_user:jobmatch_password@localhost:5432/jobmatch_db"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    company = Column(String(255), nullable=False, index=True)
    location = Column(String(100), nullable=False, index=True)
    work_mode = Column(String(50), nullable=False, index=True)  # 'remote', 'onsite', 'hybrid'
    role_type = Column(String(50), nullable=False, index=True)  # 'internship', 'full-time'
    stipend_inr = Column(Integer, nullable=False, index=True)
    tech_stack = Column(ARRAY(String), nullable=False, default=list)
    description = Column(Text, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    embedding = Column(Vector(EMBEDDING_DIM), nullable=True)
    search_tsv = Column(TSVECTOR, nullable=True)

    __table_args__ = (
        Index("ix_jobs_search_tsv", "search_tsv", postgresql_using="gin"),
        Index(
            "ix_jobs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )


class UserProfile(Base):
    __tablename__ = "user_profiles"

    user_id = Column(String(100), primary_key=True, index=True)
    preferred_roles = Column(ARRAY(String), nullable=False, default=list)
    must_have_skills = Column(ARRAY(String), nullable=False, default=list)
    excluded_skills = Column(ARRAY(String), nullable=False, default=list)
    preferred_locations = Column(ARRAY(String), nullable=False, default=list)
    remote_only = Column(Boolean, default=False, nullable=False)
    min_stipend_inr = Column(Integer, default=0, nullable=False)
    notes = Column(Text, default="", nullable=False)
    chat_history = Column(JSONB, default=list, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )


def init_db():
    """Initializes extensions and creates tables."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """Dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
