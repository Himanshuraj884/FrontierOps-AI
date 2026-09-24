"""
Grounding guardrail.

Splits the generated answer into sentence-level claims and checks each one
against the retrieved evidence using token-overlap (Jaccard-style) scoring.
A claim is "supported" if its overlap with at least one evidence chunk
clears a threshold. This is a real, deterministic, explainable check —
appropriate for a fast guardrail layer that runs before/alongside a more
expensive LLM-based grounding judge in a production system.
"""
from __future__ import annotations

import re
from typing import Dict, List

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "and", "or", "for", "on", "in",
    "within", "can", "be", "this", "that", "it", "as", "by", "with", "was",
    "were", "will", "may", "must", "not",
}


def _tokens(text: str) -> set:
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in toks if t not in _STOPWORDS}


def split_claims(answer: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if s.strip()]


def check_grounding(answer: str, evidence_chunks: List[Dict], threshold: float = 0.25) -> Dict:
    claims = split_claims(answer)
    evidence_token_sets = [_tokens(c["text"]) for c in evidence_chunks]

    supported, unsupported = [], []
    for claim in claims:
        claim_tokens = _tokens(claim)
        if not claim_tokens:
            continue
        best_overlap = 0.0
        for ev_tokens in evidence_token_sets:
            if not ev_tokens:
                continue
            inter = len(claim_tokens & ev_tokens)
            union = len(claim_tokens | ev_tokens)
            jaccard = inter / union if union else 0.0
            coverage = inter / len(claim_tokens)  # how much of the claim is covered
            score = max(jaccard, coverage * 0.6)
            best_overlap = max(best_overlap, score)
        if best_overlap >= threshold:
            supported.append(claim)
        else:
            unsupported.append(claim)

    total = len(claims) or 1
    faithfulness = len(supported) / total
    return {
        "claims_total": len(claims),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "faithfulness": faithfulness,
        "passed": len(unsupported) == 0,
    }
