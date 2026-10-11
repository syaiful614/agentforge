"""Deterministic fake LLMs so the whole framework is testable offline, for free."""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable
from typing import Any

from agentforge.llm import LLMResponse, Usage

_ids = itertools.count(1)


def text(t: str, tokens: tuple[int, int] = (10, 5), stop: str = "end_turn") -> LLMResponse:
    return LLMResponse([{"type": "text", "text": t}], stop, Usage(*tokens))


def tool_use(*calls: tuple[str, dict[str, Any]], tokens: tuple[int, int] = (10, 5)) -> LLMResponse:
    content = [
        {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": n, "input": i} for n, i in calls
    ]
    return LLMResponse(content, "tool_use", Usage(*tokens))


class ScriptedLLM:
    """Returns queued responses in order; records every call for assertions."""

    model = "fake"

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> LLMResponse:
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of responses")
        item = self.script.pop(0)
        on_text = kwargs.get("on_text")
        if on_text is not None and item.text:  # simulate token-by-token streaming
            for i in range(0, len(item.text), 3):
                on_text(item.text[i : i + 3])
        return item


class RouterLLM:
    """Decides the response from the request itself; safe under parallel workers."""

    model = "fake-router"

    def __init__(self, handler: Callable[[dict[str, Any]], LLMResponse]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def generate(self, **kwargs: Any) -> LLMResponse:
        with self._lock:
            self.calls.append(kwargs)
        return self.handler(kwargs)
