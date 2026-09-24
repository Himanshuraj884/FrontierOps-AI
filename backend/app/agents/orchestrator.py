"""Orchestrator Agent — classifies intent and sets up the workflow plan."""
from __future__ import annotations

import re

from .state import AgentState

_POLICY_KEYWORDS = {"policy", "refund", "terms", "eligible", "eligibility", "period", "days"}
_SECURITY_KEYWORDS = {"security", "encrypt", "encryption", "access", "incident", "breach"}
_SUPPORT_KEYWORDS = {"support", "ticket", "response", "escalate", "escalation"}
_ACCOUNT_KEYWORDS = {"admin", "role", "export", "viewer", "member", "permission"}


def _classify(query: str) -> str:
    tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    scores = {
        "policy_question": len(tokens & _POLICY_KEYWORDS),
        "security_question": len(tokens & _SECURITY_KEYWORDS),
        "support_question": len(tokens & _SUPPORT_KEYWORDS),
        "account_question": len(tokens & _ACCOUNT_KEYWORDS),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_question"


def run(state: AgentState) -> AgentState:
    state.intent = _classify(state.query)
    state.log("orchestrator", "intent_classified", intent=state.intent)
    return state
