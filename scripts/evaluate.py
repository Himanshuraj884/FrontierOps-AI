"""
Runs the full six-agent pipeline over the fixed benchmark question set and
computes real evaluation metrics (nothing here is hardcoded — every number
comes from executing the pipeline against data/evaluation/questions.json).

Run: python scripts/evaluate.py
Writes: data/evaluation/results.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts"))

from ingest import build_index  # noqa: E402
from app.agents.graph import build_workflow  # noqa: E402
from app.guardrails.grounding import check_grounding  # noqa: E402
from app.llm_client import get_llm_client  # noqa: E402

QUESTIONS_PATH = ROOT / "data" / "evaluation" / "questions.json"
RESULTS_PATH = ROOT / "data" / "evaluation" / "results.json"


def context_precision_recall(retrieved_doc_ids, expected_doc_ids):
    if not expected_doc_ids:
        return None, None  # out-of-scope question: precision/recall not defined
    retrieved_set = set(retrieved_doc_ids)
    expected_set = set(expected_doc_ids)
    precision = len(retrieved_set & expected_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(retrieved_set & expected_set) / len(expected_set) if expected_set else 0.0
    return precision, recall


def run_evaluation():
    embedder, store, chunks = build_index()
    llm_client = get_llm_client()
    workflow = build_workflow(embedder, store, llm_client, top_k=3)

    questions = json.loads(QUESTIONS_PATH.read_text())

    per_question = []
    precisions, recalls, faithfulnesses, latencies = [], [], [], []
    citation_valid_total, citation_total = 0, 0
    unsupported_before, unsupported_after = 0, 0
    injection_caught = 0
    failures = 0

    for q in questions:
        start = time.perf_counter()
        try:
            state = workflow.run(q["question"])
        except Exception as e:  # pragma: no cover - defensive
            failures += 1
            per_question.append({"id": q["id"], "question": q["question"], "error": str(e)})
            continue
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        retrieved_doc_ids = [c["document_id"] for c in state.retrieved_chunks]
        precision, recall = context_precision_recall(retrieved_doc_ids, q["expected_documents"])
        if precision is not None:
            precisions.append(precision)
            recalls.append(recall)

        # "Before validation" baseline: take the raw analysis conclusion and
        # grade it for grounding without any guardrail gating.
        baseline_grounding = check_grounding(state.analysis.get("conclusion", ""), state.retrieved_chunks)
        if baseline_grounding["unsupported_claims"]:
            unsupported_before += 1

        faithfulnesses.append(state.validation_result["grounding"]["faithfulness"])

        # "After validation": only count an answer as unsupported-and-exposed
        # if it was NOT routed to human review (i.e. it was shown to the user
        # as-is) AND still had unsupported claims.
        if not state.human_review and state.validation_result["grounding"]["unsupported_claims"]:
            unsupported_after += 1

        citation_total += len(state.citations)
        citation_valid_total += len(state.validation_result["citations"]["valid_citations"])

        if state.validation_result["injection"]["injection_detected"]:
            injection_caught += 1

        per_question.append(
            {
                "id": q["id"],
                "question": q["question"],
                "intent": state.intent,
                "retrieved_documents": retrieved_doc_ids,
                "context_precision": precision,
                "context_recall": recall,
                "faithfulness": state.validation_result["grounding"]["faithfulness"],
                "confidence": state.confidence,
                "human_review": state.human_review,
                "injection_detected": state.validation_result["injection"]["injection_detected"],
                "latency_ms": round(latency_ms, 2),
            }
        )

    n = len(questions)
    results = {
        "num_questions": n,
        "num_failures": failures,
        "context_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
        "context_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
        "faithfulness": round(sum(faithfulnesses) / len(faithfulnesses), 4) if faithfulnesses else None,
        "citation_accuracy": round(citation_valid_total / citation_total, 4) if citation_total else None,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "success_rate": round((n - failures) / n, 4) if n else None,
        "unsupported_answers_before_validation": unsupported_before,
        "unsupported_answers_after_validation": unsupported_after,
        "unsupported_reduction_pct": (
            round((unsupported_before - unsupported_after) / unsupported_before * 100, 1)
            if unsupported_before
            else 0.0
        ),
        "injection_attempts_caught": injection_caught,
        "human_review_rate": round(sum(1 for p in per_question if p.get("human_review")) / n, 4) if n else None,
        "per_question": per_question,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    results = run_evaluation()
    print(json.dumps({k: v for k, v in results.items() if k != "per_question"}, indent=2))
    print(f"\nFull results written to {RESULTS_PATH}")
