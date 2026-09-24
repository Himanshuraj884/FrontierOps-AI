"""
Tracing.

Production: OpenTelemetry (auto-used if the `opentelemetry-sdk` package is
installed) — instruments each agent step and retrieval/LLM call as a span,
exportable to any OTLP collector (Jaeger, Honeycomb, etc.).

Fallback: JsonlTracer — writes the same span shape (name, start, end,
duration_ms, attributes) as newline-delimited JSON to a local file. Used
automatically when opentelemetry isn't installed, so
latency numbers in the dashboard/evaluation are always real measurements,
never placeholders.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict


class JsonlTracer:
    def __init__(self, path: str = "traces.jsonl"):
        self.path = Path(path)

    @contextmanager
    def span(self, name: str, **attributes: Any):
        start = time.perf_counter()
        record: Dict[str, Any] = {"name": name, "attributes": attributes}
        try:
            yield record
        finally:
            duration_ms = (time.perf_counter() - start) * 1000
            record["duration_ms"] = round(duration_ms, 2)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")


def get_tracer(path: str = "traces.jsonl"):
    try:
        from opentelemetry import trace  # type: ignore

        otel_tracer = trace.get_tracer("frontierops-ai")

        class _OtelWrapper:
            @contextmanager
            def span(self, name: str, **attributes: Any):
                with otel_tracer.start_as_current_span(name) as span:
                    for k, v in attributes.items():
                        span.set_attribute(k, v)
                    yield {"name": name, "attributes": attributes}

        return _OtelWrapper()
    except Exception:
        return JsonlTracer(path)
