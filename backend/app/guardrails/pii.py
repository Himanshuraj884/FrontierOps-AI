"""PII screening: regex-based detection (and optional redaction) of emails,
phone numbers, and SSN-like patterns in generated answers or evidence."""
from __future__ import annotations

import re
from typing import Dict

_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "phone": re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}"),
    "ssn_like": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def scan_text(text: str) -> Dict:
    findings = {}
    for label, pattern in _PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            findings[label] = matches
    return {
        "findings": findings,
        "pii_detected": len(findings) > 0,
        "passed": len(findings) == 0,
    }


def redact_text(text: str) -> str:
    redacted = text
    for label, pattern in _PATTERNS.items():
        redacted = pattern.sub(f"[REDACTED_{label.upper()}]", redacted)
    return redacted
