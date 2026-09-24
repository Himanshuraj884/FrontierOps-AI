"""
Structured-output schema validation.

Enforces that the final response contains exactly the fields the spec
requires: answer, evidence, confidence, recommended_action, human_review.
Written with plain Python so it has no dependency on pydantic being
installed (the production FastAPI app additionally validates this with a
pydantic model at the API boundary — see app/main.py).
"""
from __future__ import annotations

from typing import Any, Dict

REQUIRED_FIELDS = {
    "answer": str,
    "evidence": list,
    "confidence": (int, float),
    "recommended_action": str,
    "human_review": bool,
}


def validate_response_schema(response: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in response:
            errors.append(f"missing required field: {field}")
            continue
        if not isinstance(response[field], expected_type):
            errors.append(
                f"field '{field}' has wrong type: expected {expected_type}, got {type(response[field])}"
            )
    if "confidence" in response and isinstance(response["confidence"], (int, float)):
        if not (0.0 <= response["confidence"] <= 1.0):
            errors.append("confidence must be between 0.0 and 1.0")

    return {"errors": errors, "passed": len(errors) == 0}
