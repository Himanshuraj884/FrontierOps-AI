"""
Research Agent — inspects retrieved evidence for conflicts, exceptions, or
gaps before analysis. Looks for documents that reference exceptions to a
general rule (e.g. "Schedule C", "unless", "override") co-occurring with the
query's topic, and flags them for the Analysis Agent.
"""
from __future__ import annotations

import re

from .state import AgentState

_EXCEPTION_MARKERS = re.compile(
    r"\b(exception|unless|override|custom terms|special terms|schedule c)\b", re.IGNORECASE
)


def run(state: AgentState) -> AgentState:
    conflicting_docs = []
    for chunk in state.retrieved_chunks:
        if _EXCEPTION_MARKERS.search(chunk["text"]):
            conflicting_docs.append(chunk["document_id"])

    findings = {
        "potential_exceptions_found": bool(conflicting_docs),
        "documents_with_exceptions": sorted(set(conflicting_docs)),
        "evidence_document_count": len({c["document_id"] for c in state.retrieved_chunks}),
    }
    state.research_findings = findings
    state.log("research", "findings_computed", **findings)
    return state
