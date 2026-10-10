"""Safety rails: resource budgets, human approval, and loop detection."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .llm import Usage

if TYPE_CHECKING:
    from .tools import Tool


class BudgetExceeded(Exception):
    """Raised when a token or wall-clock budget is exhausted."""


class MaxStepsExceeded(BudgetExceeded):
    """Raised when an agent exceeds its allowed number of LLM steps."""


class Budget:
    """Thread-safe budget that can be shared by many agents (orchestrator + workers).

    `max_steps` is enforced per agent run; tokens and time are global to the budget.
    """

    def __init__(
        self, max_steps: int = 15, max_tokens: int = 200_000, timeout_s: float = 300.0
    ) -> None:
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self._used = 0
        self._start = time.monotonic()
        self._lock = threading.Lock()

    @property
    def used_tokens(self) -> int:
        return self._used

    def record(self, usage: Usage) -> None:
        with self._lock:
            self._used += usage.total

    def check(self) -> None:
        if self._used >= self.max_tokens:
            raise BudgetExceeded(f"token budget exhausted ({self._used}/{self.max_tokens})")
        if time.monotonic() - self._start > self.timeout_s:
            raise BudgetExceeded(f"time budget exhausted ({self.timeout_s:.0f}s)")


ApprovalPolicy = Callable[["Tool", dict[str, Any]], bool]


def allow_all(tool: Tool, tool_input: dict[str, Any]) -> bool:
    return True


def deny_all(tool: Tool, tool_input: dict[str, Any]) -> bool:
    return False


def read_only_only(tool: Tool, tool_input: dict[str, Any]) -> bool:
    """Safe default: anything that needs approval is refused."""
    return not tool.requires_approval


def cli_prompt(tool: Tool, tool_input: dict[str, Any]) -> bool:
    """Human-in-the-loop approval on the terminal. Denies when not interactive."""
    if not sys.stdin.isatty():
        return False
    preview = json.dumps(tool_input, ensure_ascii=False)[:300]
    print(f"\n[approval] {tool.name}({preview})", file=sys.stderr)
    return input("Allow? [y/N] ").strip().lower() in {"y", "yes"}


class LoopDetector:
    """Counts identical (tool, input) calls to catch agents stuck in a rut."""

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._seen: Counter[str] = Counter()

    def record(self, name: str, tool_input: dict[str, Any]) -> bool:
        """Returns True when this exact call has now been made `threshold`+ times."""
        key = f"{name}:{json.dumps(tool_input, sort_keys=True, default=str)}"
        self._seen[key] += 1
        return self._seen[key] >= self.threshold
