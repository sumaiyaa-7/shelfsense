"""Lightweight tracing for AI pipelines.

Every request produces a tree of spans (router -> agent -> retrieval / LLM /
SQL) with latency, token usage, and arbitrary attributes. Finished traces are
appended to a JSONL file so they can be inspected, replayed, or loaded into a
dataframe for analysis. The same idea as OpenTelemetry / LangSmith, kept
dependency-free so it is easy to read.
"""
from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

_current: contextvars.ContextVar[Optional["Span"]] = contextvars.ContextVar("span", default=None)


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    parent_id: Optional[str] = None
    start: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    attrs: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)
    status: str = "ok"

    def add_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.attrs["input_tokens"] = self.attrs.get("input_tokens", 0) + input_tokens
        self.attrs["output_tokens"] = self.attrs.get("output_tokens", 0) + output_tokens

    def total_tokens(self) -> int:
        own = self.attrs.get("input_tokens", 0) + self.attrs.get("output_tokens", 0)
        return own + sum(c.total_tokens() for c in self.children)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


def current_span() -> Optional[Span]:
    return _current.get()


class Tracer:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.recent: dict[str, dict] = {}  # in-memory cache of recent traces for the API

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = _current.get()
        trace_id = parent.trace_id if parent else uuid.uuid4().hex[:16]
        span = Span(name=name, trace_id=trace_id, parent_id=parent.span_id if parent else None, attrs=dict(attrs))
        if parent:
            parent.children.append(span)
        token = _current.set(span)
        try:
            yield span
        except Exception as e:
            span.status = "error"
            span.attrs["error"] = f"{type(e).__name__}: {e}"
            raise
        finally:
            span.duration_ms = (time.perf_counter() - span.start) * 1000
            _current.reset(token)
            if parent is None:
                self._export(span)

    def _export(self, root: Span) -> None:
        record = {"trace_id": root.trace_id, "ts": time.time(), "total_tokens": root.total_tokens(), **root.to_dict()}
        self.recent[root.trace_id] = record
        if len(self.recent) > 500:
            self.recent.pop(next(iter(self.recent)))
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")

    def get(self, trace_id: str) -> Optional[dict]:
        return self.recent.get(trace_id)
