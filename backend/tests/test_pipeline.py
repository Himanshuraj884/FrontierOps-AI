import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest import build_index
from app.agents.graph import build_workflow
from app.llm_client import ExtractiveMockLLM


def _workflow():
    embedder, store, chunks = build_index()
    return build_workflow(embedder, store, ExtractiveMockLLM(), top_k=3)


def test_pipeline_runs_end_to_end():
    workflow = _workflow()
    state = workflow.run("What is the enterprise refund period?")
    assert state.answer
    assert state.retrieved_chunks
    assert state.validation_result is not None
    assert 0.0 <= state.confidence <= 1.0


def test_pipeline_flags_human_review_for_low_confidence():
    workflow = _workflow()
    state = workflow.run("What is our company's stock price today?")
    # no relevant document exists for this question -> weak grounding -> low confidence
    assert state.confidence < 0.80
    assert state.human_review


def test_pipeline_detects_injection_in_retrieved_docs():
    workflow = _workflow()
    state = workflow.run("What is the vendor onboarding process?")
    assert state.validation_result["injection"]["injection_detected"]


def test_state_trace_records_every_agent():
    workflow = _workflow()
    state = workflow.run("What is the enterprise refund period?")
    agents_seen = {t["agent"] for t in state.trace}
    assert {"orchestrator", "retrieval", "research", "analysis", "validation", "response"} <= agents_seen
