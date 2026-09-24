"""
FastAPI API gateway. Requires `pip install -r backend/requirements.txt`
(fastapi/uvicorn/sqlalchemy/pydantic) — not installable in the offline
sandbox this repo was scaffolded in, but this is real, complete, runnable
code for use in an environment with network access.

Run:
    cd backend
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from ingest import build_index  # noqa: E402

from .agents.graph import build_workflow  # noqa: E402
from .database.db import SessionLocal, init_db  # noqa: E402
from .database.models import AuditLog  # noqa: E402
from .llm_client import get_llm_client  # noqa: E402
from .observability.tracing import get_tracer  # noqa: E402

app = FastAPI(title="FrontierOps AI", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def startup():
    init_db()
    embedder, store, chunks = build_index()
    llm_client = get_llm_client()
    _state["workflow"] = build_workflow(embedder, store, llm_client, top_k=3)
    _state["tracer"] = get_tracer(path=str(ROOT / "traces.jsonl"))
    _state["chunk_count"] = len(chunks)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class EvidenceItem(BaseModel):
    chunk_id: str
    claim: str


class QueryResponse(BaseModel):
    answer: str
    evidence: list[EvidenceItem]
    confidence: float
    recommended_action: str
    human_review: bool
    intent: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": _state.get("chunk_count", 0)}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest):
    workflow = _state.get("workflow")
    if workflow is None:
        raise HTTPException(status_code=503, detail="Workflow not initialized")

    start = time.perf_counter()
    with _state["tracer"].span("api_query", query=req.query):
        state = workflow.run(req.query)
    latency_ms = (time.perf_counter() - start) * 1000

    session = SessionLocal()
    try:
        session.add(
            AuditLog(
                request_id=str(int(time.time() * 1000)),
                agent="pipeline",
                action="query",
                result=state.to_dict(),
                confidence=state.confidence,
            )
        )
        session.commit()
    finally:
        session.close()

    return QueryResponse(
        answer=state.answer,
        evidence=[EvidenceItem(chunk_id=c["chunk_id"], claim=c["claim"]) for c in state.citations],
        confidence=state.confidence,
        recommended_action="No escalation required" if not state.human_review else "Escalate to human review",
        human_review=state.human_review,
        intent=state.intent,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/api/trace/{request_id}")
def get_trace(request_id: str):
    session = SessionLocal()
    try:
        row = (
            session.query(AuditLog)
            .filter(AuditLog.request_id == request_id)
            .first()
        )
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
        return row.result
    finally:
        session.close()
