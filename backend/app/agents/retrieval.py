"""Retrieval Agent — embeds the query and pulls top-k chunks from the vector store."""
from __future__ import annotations

from .state import AgentState


def run(state: AgentState, embedder, vector_store, top_k: int = 3) -> AgentState:
    query_vec = embedder.encode([state.query])[0]
    results = vector_store.search(query_vec, top_k=top_k)
    state.retrieved_chunks = [
        {
            "chunk_id": r.chunk_id,
            "score": r.score,
            "document_id": r.payload.get("document_id"),
            "title": r.payload.get("title"),
            "section": r.payload.get("section"),
            "text": r.payload.get("text"),
        }
        for r in results
    ]
    state.log(
        "retrieval",
        "chunks_retrieved",
        count=len(state.retrieved_chunks),
        top_score=state.retrieved_chunks[0]["score"] if state.retrieved_chunks else 0.0,
    )
    return state
