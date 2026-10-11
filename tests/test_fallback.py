"""Planner fallback: opt-in degradation that is always reported, never silent."""

import importlib.util
from pathlib import Path

import pytest

from agentforge import Orchestrator, Tracer, cli, default_registry
from agentforge.llm import LLMError
from agentforge.orchestrator import PlanError
from fakes import RouterLLM, ScriptedLLM, text, tool_use


def _bad_plan():
    plan = {"subtasks": [{"id": "t1", "goal": "g", "role": "nonsense"}]}
    return tool_use(("submit_plan", plan))


def _broken_planner_then(answer):
    # planner retries = 3 attempts, all invalid; then the fallback agent answers
    return ScriptedLLM([_bad_plan(), _bad_plan(), _bad_plan(), text(answer)])


# ------------------------------------------------------------------ orchestrator
def test_fallback_answers_with_a_single_agent_and_is_marked_degraded(tmp_path):
    llm = _broken_planner_then("fallback answer")
    tracer = Tracer()
    orch = Orchestrator(llm, default_registry(tmp_path), fallback=True, tracer=tracer)
    res = orch.run("do the thing")

    assert res.output == "fallback answer" and res.ok
    assert res.degraded is True and "unknown role 'nonsense'" in res.degraded_reason
    assert list(res.subtask_results) == ["fallback"] and res.review_rounds == 0
    assert any(e["type"] == "planner_fallback" for e in tracer.events)


def test_fallback_agent_gets_the_workers_tools_and_nothing_more(tmp_path):
    llm = _broken_planner_then("ok")
    Orchestrator(llm, default_registry(tmp_path), fallback=True).run("x")
    offered = {t["name"] for t in llm.calls[-1]["tools"]}
    assert {"calculator", "read_file", "list_dir"} <= offered
    assert "write_file" not in offered  # no worker may write, so the fallback may not either


def test_fallback_is_off_by_default(tmp_path):
    llm = ScriptedLLM([_bad_plan()] * 3)
    with pytest.raises(LLMError, match="unknown role"):
        Orchestrator(llm, default_registry(tmp_path)).run("x")


def test_provider_errors_are_not_masked_by_the_fallback(tmp_path):
    class Down:
        model = "down"

        def generate(self, **kwargs):
            raise LLMError("server down")

    with pytest.raises(LLMError, match="server down"):
        Orchestrator(Down(), default_registry(tmp_path), fallback=True).run("x")


def test_a_working_planner_is_never_degraded(tmp_path):
    def handle(kw):
        choice = kw.get("tool_choice")
        if choice and choice["name"] == "submit_plan":
            plan = {"subtasks": [{"id": "t1", "goal": "go", "role": "researcher"}]}
            return tool_use(("submit_plan", plan))
        if choice and choice["name"] == "submit_review":
            return tool_use(("submit_review", {"approved": True}))
        return text("final" if "SYNTHESIZER" in kw["system"] else "worker done")

    orch = Orchestrator(RouterLLM(handle), default_registry(tmp_path), fallback=True)
    res = orch.run("x")
    assert res.degraded is False and res.degraded_reason is None and "t1" in res.subtask_results


# ------------------------------------------------------------------ CLI
@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTFORGE_TRACE_DIR", str(tmp_path / "traces"))

    def install(llm):
        monkeypatch.setattr(cli, "make_llm", lambda *a, **k: llm)
        return ["--workspace", str(tmp_path / "ws")]

    return install


def test_cli_fallback_warns_loudly_but_returns_the_answer(cli_env, capsys):
    ws = cli_env(_broken_planner_then("fallback answer"))
    assert cli.main(["run", "x", "--orchestrate", "--fallback", *ws]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "fallback answer"
    assert "warning: planner failed" in out.err and "single-agent mode (degraded)" in out.err


def test_cli_without_fallback_fails_with_a_clean_error(cli_env, capsys):
    ws = cli_env(ScriptedLLM([_bad_plan()] * 3))
    assert cli.main(["run", "x", "--orchestrate", *ws]) == 1
    assert "error: submit_plan: unknown role" in capsys.readouterr().err


# ------------------------------------------------------------------ eval harness
def _evals():
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("run_evals4", root / "evals" / "run_evals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_eval_marks_fallback_runs_and_can_require_real_orchestration():
    mod = _evals()
    base = {"name": "o", "mode": "orchestrate", "task": "t", "checks": {"contains_any": ["answer"]}}

    lenient = mod.run_case(base, _broken_planner_then("an answer"), fallback=True)
    assert lenient["degraded"] is True and lenient["passed"] is True

    strict_case = {**base, "checks": {**base["checks"], "no_fallback": True}}
    strict = mod.run_case(strict_case, _broken_planner_then("an answer"), fallback=True)
    assert strict["degraded"] is True and strict["passed"] is False
    assert any("fallback" in f for f in strict["failures"])


def test_eval_without_the_flag_records_a_failed_run_not_a_degraded_one():
    mod = _evals()
    case = {"name": "o", "mode": "orchestrate", "task": "t", "checks": {}}
    row = mod.safe_run(case, ScriptedLLM([_bad_plan()] * 3))
    assert row["passed"] is False and row["degraded"] is False
    assert "unknown role" in row["failures"][0]


def test_the_orchestrated_eval_case_requires_real_orchestration():
    import json

    cases = json.loads((Path(__file__).parent.parent / "evals" / "cases.json").read_text())
    orchestrated = [c for c in cases if c.get("mode") == "orchestrate"]
    assert orchestrated and all(c["checks"].get("no_fallback") for c in orchestrated)


def test_plan_error_still_propagates_from_direct_validation():
    from agentforge.orchestrator import Plan, Subtask, validate_plan

    plan = Plan(subtasks=[Subtask(id="a", goal="g", role="nope")])
    with pytest.raises(PlanError):
        validate_plan(plan, {"writer"})
