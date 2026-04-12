from __future__ import annotations

from fastapi import Body, FastAPI
from pydantic import BaseModel

from ..models import CodeReviewAction, CodeReviewStepResult
from .metrics import build_leaderboard_snapshot
from .review_environment import CodeReviewEnvironment


class ResetRequest(BaseModel):
    task_name: str | None = None


env = CodeReviewEnvironment()
app = FastAPI(title="Code Review Quality Environment", version="0.1.0")

@app.get("/")
def root() -> dict:
    return {"name": "code-review-quality-env", "status": "ok", "tasks": env.tasks}


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/tasks")
def tasks() -> dict:
    return {"tasks": env.tasks}


@app.get("/metrics/leaderboard")
def metrics_leaderboard() -> dict:
    return build_leaderboard_snapshot()


@app.post("/reset", response_model=CodeReviewStepResult)
def reset(req: ResetRequest | None = Body(default=None)) -> CodeReviewStepResult:
    return env.reset(task_name=req.task_name if req else None)


@app.post("/step", response_model=CodeReviewStepResult)
def step(action: CodeReviewAction) -> CodeReviewStepResult:
    return env.step(action)


@app.get("/state")
def state() -> dict:
    return env.state().model_dump()
