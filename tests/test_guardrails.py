import pytest

from agentforge.guardrails import (
    Budget,
    BudgetExceeded,
    LoopDetector,
    allow_all,
    deny_all,
    read_only_only,
)
from agentforge.llm import Usage
from agentforge.tools import tool


def test_token_budget():
    b = Budget(max_tokens=100)
    b.check()
    b.record(Usage(60, 50))
    with pytest.raises(BudgetExceeded):
        b.check()


def test_time_budget():
    b = Budget(timeout_s=-1)
    with pytest.raises(BudgetExceeded):
        b.check()


def test_loop_detector_counts_identical_calls_only():
    d = LoopDetector(threshold=3)
    assert not d.record("t", {"a": 1})
    assert not d.record("t", {"a": 2})
    assert not d.record("t", {"a": 1})
    assert d.record("t", {"a": 1})


def test_policies():
    @tool(read_only=False, requires_approval=True)
    def danger() -> str:
        """Dangerous."""
        return "x"

    @tool
    def safe() -> str:
        """Safe."""
        return "x"

    assert allow_all(danger, {}) and not deny_all(safe, {})
    assert read_only_only(safe, {}) and not read_only_only(danger, {})
