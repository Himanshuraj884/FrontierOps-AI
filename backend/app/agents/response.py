"""Response Agent — assembles the final, citation-backed, structured answer
after validation has passed (or flags human review if it hasn't)."""
from __future__ import annotations

from .state import AgentState


def run(state: AgentState) -> AgentState:
    if state.human_review:
        state.answer = (
            f"{state.analysis.get('conclusion', '')}\n\n"
            f"[This answer is below the confidence threshold "
            f"({state.confidence:.2f} < 0.80) or failed a guardrail check and has "
            f"been routed to human review before being shown to the end user.]"
        )
    else:
        state.answer = state.analysis.get("conclusion", "")
    state.log("response", "final_answer_assembled", human_review=state.human_review)
    return state
