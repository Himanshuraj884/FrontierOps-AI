"""
Shared workflow state passed between agents. Modeled as a plain dict-backed
dataclass so it works whether the graph is executed by hand-rolled
sequencing (this repo's default, dependency-free path) or by LangGraph
(see graph.py — LangGraph is used automatically if installed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    query: str
    intent: str = ""
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    research_findings: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)
    answer: str = ""
    citations: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    validation_result: Optional[Dict[str, Any]] = None
    human_review: bool = False
    retry_count: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def log(self, agent: str, event: str, **kwargs) -> None:
        self.trace.append({"agent": agent, "event": event, **kwargs})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent,
            "retrieved_chunks": self.retrieved_chunks,
            "research_findings": self.research_findings,
            "analysis": self.analysis,
            "answer": self.answer,
            "citations": self.citations,
            "confidence": self.confidence,
            "validation_result": self.validation_result,
            "human_review": self.human_review,
            "retry_count": self.retry_count,
            "trace": self.trace,
        }
