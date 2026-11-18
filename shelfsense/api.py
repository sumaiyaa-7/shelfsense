"""REST API (FastAPI).

    uvicorn shelfsense.api:app --reload
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "What are the top 3 products by revenue in Edison?"}'
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .agents.orchestrator import Copilot

app = FastAPI(title="ShelfSense", version=__version__, description="Multi-agent AI copilot for retail operations")
copilot = Copilot()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class AskResponse(BaseModel):
    question: str
    route: str
    answer: str
    citations: list[str]
    sql: str | None
    columns: list[str]
    rows: list[list]
    retrieved_ids: list[str]
    trace_id: str
    latency_ms: float
    total_tokens: int


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm": getattr(copilot.llm, "name", "mock"), "version": __version__}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    return AskResponse(**copilot.ask(req.question).to_dict())


@app.get("/traces/{trace_id}")
def get_trace(trace_id: str) -> dict:
    trace = copilot.tracer.get(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail="trace not found")
    return trace
