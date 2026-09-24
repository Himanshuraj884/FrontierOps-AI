"""
Pluggable LLM client.

- If ANTHROPIC_API_KEY is set (and network access is available), calls the
  real Anthropic Messages API via urllib (stdlib only, no extra dependency).
- Otherwise falls back to ExtractiveMockLLM: a deterministic, dependency-free
  "reasoner" that answers strictly from the evidence it's given, using
  keyword-overlap sentence selection. This is a real algorithm (not a canned
  string), which is what lets the demo/evaluation scripts run end-to-end in
  environments with no LLM API access configured.

Swap the backend by setting ANTHROPIC_API_KEY, or by passing a different
client into the agents.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Optional


class LLMClient:
    def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-6"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        self.model = model

    def generate(self, system: str, prompt: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "max_tokens": 1024,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "content-type": "application/json",
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        parts = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
        return "\n".join(parts)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set:
    return set(_TOKEN_RE.findall(text.lower()))


class ExtractiveMockLLM(LLMClient):
    """Deterministic, offline, dependency-free stand-in for a real LLM call.

    It never invents claims: it only selects and lightly assembles sentences
    that already exist in the evidence passed to it. This is intentionally
    conservative so grounding/faithfulness checks downstream have something
    real to validate against.
    """

    def generate(self, system: str, prompt: str) -> str:
        # The agents pass structured prompts; parse out QUESTION and EVIDENCE
        q_match = re.search(r"QUESTION:\s*(.+)", prompt)
        ev_match = re.search(r"EVIDENCE:\s*(.*)", prompt, re.DOTALL)
        question = q_match.group(1).strip() if q_match else ""
        evidence_block = ev_match.group(1).strip() if ev_match else ""

        q_tokens = _tokens(question)
        sentences = re.split(r"(?<=[.!?])\s+", evidence_block)
        scored = []
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            overlap = len(q_tokens & _tokens(s))
            if overlap > 0:
                scored.append((overlap, s))
        scored.sort(key=lambda x: -x[0])
        best = [s for _, s in scored[:2]]
        if not best:
            return "The available evidence does not clearly answer this question."
        return " ".join(best)


def get_llm_client() -> LLMClient:
    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return AnthropicLLMClient()
        except Exception:
            pass
    return ExtractiveMockLLM()
