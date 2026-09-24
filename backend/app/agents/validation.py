"""
Validation Agent — runs the guardrail pipeline (grounding, citation
verification, unsupported-claim detection, prompt-injection screening, PII
screening, schema validation) against the draft answer produced by the
Analysis Agent, and computes a confidence score used for the human-in-the-
loop decision.
"""
from __future__ import annotations

from ..guardrails import citations as citations_guard
from ..guardrails import grounding as grounding_guard
from ..guardrails import injection as injection_guard
from ..guardrails import pii as pii_guard
from ..guardrails import schema as schema_guard
from .state import AgentState

CONFIDENCE_THRESHOLD = 0.80


def _build_citations(state: AgentState):
    return [
        {"chunk_id": c["chunk_id"], "claim": state.analysis.get("conclusion", "")}
        for c in state.retrieved_chunks
    ]


# The offline HashingEmbedder fallback (see rag/embeddings.py) produces cosine
# similarities in a compressed range (empirically ~0.05-0.55 on this corpus,
# vs. ~0.3-0.9 for a real sentence-transformer model) because sparse hashed
# TF-IDF vectors for short queries against longer chunks rarely achieve high
# raw cosine overlap even when the match is correct. RETRIEVAL_SCORE_CALIBRATION
# rescales the raw score against the empirically observed "good match" ceiling
# for this embedder so the confidence gate is meaningful; swap to a real
# embedding backend (network access -> sentence-transformers) for a
# less-compressed native range and this calibration becomes unnecessary
# (set it back to 1.0).
RETRIEVAL_SCORE_CALIBRATION = 0.55


def _compute_confidence(retrieval_score: float, grounding: dict, injection_flagged: bool) -> float:
    calibrated_retrieval = min(1.0, retrieval_score / RETRIEVAL_SCORE_CALIBRATION)
    base = 0.45 * calibrated_retrieval + 0.55 * grounding["faithfulness"]
    if injection_flagged:
        base *= 0.5
    return round(max(0.0, min(1.0, base)), 4)


def run(state: AgentState) -> AgentState:
    answer_text = state.analysis.get("conclusion", "")
    citations = _build_citations(state)

    injection_result = injection_guard.scan_retrieved_chunks(state.retrieved_chunks)
    grounding_result = grounding_guard.check_grounding(answer_text, state.retrieved_chunks)
    citation_result = citations_guard.verify_citations(citations, state.retrieved_chunks)
    pii_result = pii_guard.scan_text(answer_text)

    top_score = state.retrieved_chunks[0]["score"] if state.retrieved_chunks else 0.0
    confidence = _compute_confidence(top_score, grounding_result, injection_result["injection_detected"])

    draft_response = {
        "answer": answer_text,
        "evidence": citations,
        "confidence": confidence,
        "recommended_action": "No escalation required" if confidence >= CONFIDENCE_THRESHOLD else "Escalate to human review",
        "human_review": confidence < CONFIDENCE_THRESHOLD,
    }
    schema_result = schema_guard.validate_response_schema(draft_response)

    validation_result = {
        "grounding": grounding_result,
        "citations": citation_result,
        "injection": injection_result,
        "pii": pii_result,
        "schema": schema_result,
        "passed": all(
            [
                grounding_result["passed"],
                citation_result["passed"],
                not injection_result["injection_detected"],
                pii_result["passed"],
                schema_result["passed"],
            ]
        ),
    }

    state.confidence = confidence
    state.citations = citations
    state.validation_result = validation_result
    state.human_review = confidence < CONFIDENCE_THRESHOLD or not validation_result["passed"]
    state.log(
        "validation",
        "guardrails_evaluated",
        confidence=confidence,
        passed=validation_result["passed"],
        human_review=state.human_review,
    )
    return state
