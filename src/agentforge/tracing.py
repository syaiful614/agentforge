"""Structured tracing: every LLM call, tool call and decision becomes a JSONL event."""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class _Sink:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: dict[str, Any]) -> None:
        with self.lock:
            self.events.append(event)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")


class Tracer:
    def __init__(
        self,
        path: Path | None = None,
        run_id: str | None = None,
        _sink: _Sink | None = None,
        _ctx: dict[str, Any] | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self._sink = _sink or _Sink(path)
        self._ctx = _ctx or {}

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._sink.events

    def bind(self, **ctx: Any) -> Tracer:
        """Child tracer sharing the same sink, tagging events with extra context."""
        return Tracer(run_id=self.run_id, _sink=self._sink, _ctx={**self._ctx, **ctx})

    def event(self, type: str, **data: Any) -> None:
        self._sink.write(
            {"ts": time.time(), "run_id": self.run_id, "type": type, **self._ctx, **data}
        )

    @contextmanager
    def span(self, type: str, **data: Any) -> Iterator[dict[str, Any]]:
        start = time.perf_counter()
        extra: dict[str, Any] = {}
        try:
            yield extra
        finally:
            ms = round((time.perf_counter() - start) * 1000, 1)
            self.event(type, duration_ms=ms, **data, **extra)

    def summary(self) -> dict[str, Any]:
        counts = Counter(e["type"] for e in self.events)
        tokens = sum(
            e.get("input_tokens", 0) + e.get("output_tokens", 0)
            for e in self.events
            if e["type"] == "llm_call"
        )
        return {"run_id": self.run_id, "events": dict(counts), "total_tokens": tokens}
