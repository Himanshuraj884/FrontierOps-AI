"""
Analysis Agent — combines the question, retrieved evidence, and research
findings into a structured conclusion. Delegates the actual language
generation to the pluggable LLM client but constrains it to the evidence
text only (see llm_client.py) so it cannot invent unsupported claims here.
"""
from __future__ import annotations

from .state import AgentState


def _build_prompt(state: AgentState) -> str:
    evidence_text = "\n".join(
        f"[{c['document_id']}] {c['text']}" for c in state.retrieved_chunks
    )
    return (
        f"QUESTION: {state.query}\n\n"
        f"EVIDENCE: {evidence_text}\n\n"
        "Using ONLY the evidence above, state the conclusion. Do not add facts "
        "not present in the evidence."
    )


SYSTEM_PROMPT = (
    "You are the Analysis Agent in an enterprise RAG platform. You must ground "
    "every statement in the provided evidence and explicitly flag uncertainty "
    "when evidence is incomplete or conflicting."
)


def run(state: AgentState, llm_client) -> AgentState:
    if not state.retrieved_chunks:
        state.analysis = {
            "conclusion": "No relevant evidence was retrieved for this question.",
            "supporting_evidence": [],
            "uncertainty": "high",
        }
        state.log("analysis", "no_evidence")
        return state

    prompt = _build_prompt(state)
    conclusion = llm_client.generate(SYSTEM_PROMPT, prompt)

    uncertainty = "low"
    if state.research_findings.get("potential_exceptions_found"):
        uncertainty = "medium"
    if state.research_findings.get("evidence_document_count", 0) < 1:
        uncertainty = "high"

    state.analysis = {
        "conclusion": conclusion,
        "supporting_evidence": [c["chunk_id"] for c in state.retrieved_chunks],
        "uncertainty": uncertainty,
    }
    state.log("analysis", "conclusion_drafted", uncertainty=uncertainty)
    return state
