"""Citation verification: does every citation point to a real, retrieved
chunk, and does that chunk's text actually overlap with the claim it's
attached to?"""
from __future__ import annotations

from typing import Dict, List

from .grounding import _tokens


def verify_citations(citations: List[Dict], retrieved_chunks: List[Dict], threshold: float = 0.2) -> Dict:
    chunk_index = {c["chunk_id"]: c for c in retrieved_chunks}
    valid, invalid = [], []

    for cit in citations:
        chunk = chunk_index.get(cit.get("chunk_id"))
        if chunk is None:
            invalid.append({**cit, "reason": "citation references a chunk that was not retrieved"})
            continue
        claim_tokens = _tokens(cit.get("claim", ""))
        ev_tokens = _tokens(chunk["text"])
        overlap = len(claim_tokens & ev_tokens) / (len(claim_tokens) or 1)
        if overlap >= threshold:
            valid.append(cit)
        else:
            invalid.append({**cit, "reason": "cited text does not support the claim"})

    return {
        "valid_citations": valid,
        "invalid_citations": invalid,
        "passed": len(invalid) == 0,
    }
