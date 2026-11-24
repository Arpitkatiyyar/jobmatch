import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db import init_db, get_db
from app.agent import (
    execute_agent_loop,
    get_or_create_profile,
    profile_to_dict,
    reset_user_profile
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB extension and tables on startup
    try:
        init_db()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Warning: Database initialization deferred or failed: {e}")
    yield


app = FastAPI(
    title="JobMatch Agent API",
    description="Stateful AI Career & Job Matching Agent with pgvector and hybrid retrieval",
    version="1.0.0",
    lifespan=lifespan
)

# Request & Response Models
class ChatRequest(BaseModel):
    user_id: str = Field(default="demo_user", description="Unique user identifier")
    message: str = Field(..., min_length=1, description="User prompt or preference update")


class ResetResponse(BaseModel):
    status: str
    message: str
    memory: dict


# Endpoints
@app.post("/api/chat")
def chat_endpoint(req: ChatRequest, db: Session = Depends(get_db)):
    """Executes stateful tool-calling agent loop, updates memory, and returns top ranked matches."""
    try:
        result = execute_agent_loop(
            db=db,
            user_id=req.user_id,
            user_message=req.message
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/profile/{user_id}")
def get_profile_endpoint(user_id: str, db: Session = Depends(get_db)):
    """Retrieves current extracted profile memory and preferences."""
    profile = get_or_create_profile(db, user_id)
    return profile_to_dict(profile)


@app.post("/api/profile/{user_id}/reset", response_model=ResetResponse)
def reset_profile_endpoint(user_id: str, db: Session = Depends(get_db)):
    """Clears persistent memory and chat history for the user."""
    memory = reset_user_profile(db, user_id)
    return {
        "status": "success",
        "message": f"Profile memory reset for user '{user_id}'.",
        "memory": memory
    }


# Static files mount
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def serve_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "JobMatch Agent API is running. UI not found in static/."}
