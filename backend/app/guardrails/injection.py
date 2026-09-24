"""
Prompt-injection scanner.

Retrieved document content is untrusted input. This scanner flags chunks
that contain classic instruction-override patterns (e.g. "ignore previous
instructions", "reveal the system prompt") so the pipeline can mark that
content as data-only and prevent it from being treated as a directive to
the agents or the underlying LLM call.
"""
from __future__ import annotations

import re
from typing import Dict, List

_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|above) instructions", re.IGNORECASE),
    re.compile(r"reveal (the )?system prompt", re.IGNORECASE),
    re.compile(r"disregard (your|the) (rules|guidelines|instructions)", re.IGNORECASE),
    re.compile(r"you are now (in )?(dan|developer mode|jailbroken)", re.IGNORECASE),
    re.compile(r"reveal .*(api key|credentials|secret)", re.IGNORECASE),
    re.compile(r"act as (if you (are|were)|an unrestricted)", re.IGNORECASE),
]


def scan_chunk(text: str) -> List[str]:
    hits = []
    for pattern in _PATTERNS:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return hits


def scan_retrieved_chunks(chunks: List[Dict]) -> Dict:
    flagged = []
    for c in chunks:
        hits = scan_chunk(c["text"])
        if hits:
            flagged.append({"chunk_id": c["chunk_id"], "document_id": c["document_id"], "patterns": hits})
    return {
        "flagged_chunks": flagged,
        "injection_detected": len(flagged) > 0,
        "passed": len(flagged) == 0,
    }
