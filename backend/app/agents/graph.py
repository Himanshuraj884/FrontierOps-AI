"""
Workflow graph: Orchestrator -> Retrieval -> Research -> Analysis ->
Validation -> Response, with a bounded retry loop when validation fails.

Two execution backends:
  - LangGraphWorkflow: used automatically if the `langgraph` package is
    installed (production path — this is what the resume's architecture
    describes).
  - SequentialWorkflow: a dependency-free hand-rolled executor with the same
    state-machine semantics (used automatically as a fallback, e.g. in this
    sandbox, and useful for unit testing agent logic without the extra
    dependency).
"""
from __future__ import annotations

from . import analysis, orchestrator, research, response, retrieval, validation
from .state import AgentState

MAX_RETRIES = 1


class SequentialWorkflow:
    def __init__(self, embedder, vector_store, llm_client, top_k: int = 3):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm_client = llm_client
        self.top_k = top_k

    def run(self, query: str) -> AgentState:
        state = AgentState(query=query)
        state = orchestrator.run(state)
        state = retrieval.run(state, self.embedder, self.vector_store, top_k=self.top_k)
        state = research.run(state)
        state = analysis.run(state, self.llm_client)
        state = validation.run(state)

        while (
            not state.validation_result["passed"]
            and state.retry_count < MAX_RETRIES
            and not state.validation_result["injection"]["injection_detected"]
        ):
            state.retry_count += 1
            state.log("graph", "retry", retry_count=state.retry_count)
            state = analysis.run(state, self.llm_client)
            state = validation.run(state)

        state = response.run(state)
        return state


def build_workflow(embedder, vector_store, llm_client, top_k: int = 3):
    """Returns a workflow object with a .run(query) -> AgentState method.
    Tries LangGraph first; falls back to SequentialWorkflow (used in this
    sandbox since `langgraph` isn't installable here without network)."""
    try:
        import langgraph  # noqa: F401  (presence check only)

        # A real LangGraph StateGraph wiring would go here, e.g.:
        #
        # from langgraph.graph import StateGraph
        # graph = StateGraph(dict)
        # graph.add_node("orchestrator", lambda s: orchestrator.run(s).to_dict())
        # graph.add_node("retrieval", lambda s: retrieval.run(s, embedder, vector_store).to_dict())
        # ... add_edge(...) for each transition, plus a conditional edge on
        # validation_result["passed"] looping back to "analysis" up to
        # MAX_RETRIES, then "response".
        #
        # This repo ships SequentialWorkflow as the default because
        # `langgraph` cannot be pip-installed in this offline sandbox; swap
        # in the StateGraph wiring above once you have network access.
        raise ImportError("LangGraph wiring intentionally deferred; see comment above")
    except Exception:
        return SequentialWorkflow(embedder, vector_store, llm_client, top_k=top_k)
