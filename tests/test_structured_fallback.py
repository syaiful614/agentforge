import importlib.util
import json
from pathlib import Path

import pytest

from agentforge import ToolRegistry
from agentforge.llm import LLMError
from agentforge.orchestrator import Plan
from agentforge.structured import StructuredOutputError, structured_output
from fakes import ScriptedLLM, text, tool_use

GOOD = '{"subtasks": [{"id": "a", "goal": "g", "role": "writer"}]}'


def _run(llm):
    return structured_output(
        llm, system="s", prompt="p", tool_name="submit_plan", tool_description="d", model_cls=Plan
    )


def test_accepts_json_in_plain_text_when_model_ignores_forced_tool():
    plan = _run(ScriptedLLM([text(f"Sure! Here is the plan: {GOOD} Hope it helps.")]))
    assert plan.subtasks[0].id == "a"


def test_accepts_fenced_json_and_argument_wrappers():
    fenced = f"```json\n{GOOD}\n```"
    assert _run(ScriptedLLM([text(fenced)])).subtasks[0].role == "writer"
    wrapped = '{"name": "submit_plan", "arguments": ' + GOOD + "}"
    assert _run(ScriptedLLM([text(wrapped)])).subtasks[0].goal == "g"


def test_retry_message_includes_schema_and_then_succeeds():
    llm = ScriptedLLM([text("I cannot do that"), text(GOOD)])
    assert _run(llm).subtasks[0].id == "a"
    retry = llm.calls[1]["messages"][-1]["content"]
    assert isinstance(retry, str) and "JSON schema" in retry and "subtasks" in retry


def test_structured_failure_is_an_llm_error_so_cli_shows_a_clean_message():
    with pytest.raises(StructuredOutputError) as info:
        _run(ScriptedLLM([text("nope")] * 3))
    assert isinstance(info.value, LLMError)


def test_eval_session_survives_a_provider_failure():
    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("run_evals2", root / "evals" / "run_evals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def boom(case, llm, fallback=False):
        raise LLMError("server down")

    mod.run_case = boom
    row = mod.safe_run({"name": "x", "task": "t", "checks": {}}, object())
    assert row["passed"] is False and "server down" in row["failures"][0]


# ------------------------------------------------------------------ stringified nested JSON
def test_nested_lists_sent_as_json_strings_are_coerced():
    payload = {"subtasks": json.dumps([{"id": "a", "goal": "g", "role": "writer"}])}
    assert _run(ScriptedLLM([tool_use(("submit_plan", payload))])).subtasks[0].id == "a"


def test_review_with_issues_as_a_string_is_accepted():
    from agentforge.orchestrator import Review

    llm = ScriptedLLM([tool_use(("submit_review", {"approved": True, "issues": "[]"}))])
    review = structured_output(
        llm,
        system="s",
        prompt="p",
        tool_name="submit_review",
        tool_description="d",
        model_cls=Review,
    )
    assert review.approved is True and review.issues == []


def test_coercion_also_works_inside_argument_wrappers():
    inner = {"subtasks": json.dumps([{"id": "a", "goal": "g", "role": "writer"}])}
    wrapped = json.dumps({"name": "submit_plan", "arguments": inner})
    assert _run(ScriptedLLM([text(wrapped)])).subtasks[0].goal == "g"


def test_real_string_fields_that_look_like_json_are_left_alone():
    from pydantic import BaseModel

    class Note(BaseModel):
        body: str

    note = structured_output(
        ScriptedLLM([tool_use(("submit_note", {"body": "[1, 2, 3]"}))]),
        system="s", prompt="p", tool_name="submit_note", tool_description="d", model_cls=Note,
    )  # fmt: skip
    assert note.body == "[1, 2, 3]"


def test_invalid_json_in_a_list_field_still_fails_cleanly():
    bad = {"subtasks": "[{not json"}
    with pytest.raises(StructuredOutputError):
        _run(ScriptedLLM([tool_use(("submit_plan", bad))] * 3))


# ------------------------------------------------------------------ semantic validation + repair
def _plan_call(role):
    return tool_use(("submit_plan", {"subtasks": [{"id": "t1", "goal": "g", "role": role}]}))


def test_validator_failure_is_fed_back_and_the_model_can_repair_it():
    from agentforge.orchestrator import validate_plan

    llm = ScriptedLLM([_plan_call("analytic-tool"), _plan_call("analyst")])
    plan = structured_output(
        llm, system="s", prompt="p", tool_name="submit_plan", tool_description="d",
        model_cls=Plan, validate=lambda p: validate_plan(p, {"analyst", "writer"}),
    )  # fmt: skip
    assert plan.subtasks[0].role == "analyst"
    feedback = llm.calls[1]["messages"][-1]["content"][0]
    assert feedback["is_error"] and "unknown role 'analytic-tool'" in feedback["content"]
    assert "allowed" in feedback["content"]


def test_planner_that_invents_a_role_gets_a_second_chance():
    from agentforge import Orchestrator

    llm = ScriptedLLM([_plan_call("analytic-tool"), _plan_call("analyst")])
    plan = Orchestrator(llm, ToolRegistry())._plan("task")
    assert plan.subtasks[0].role == "analyst"
    assert "ONLY these role names" in llm.calls[0]["system"]


def test_a_planner_that_never_fixes_the_plan_fails_cleanly_not_with_a_raw_plan_error():
    from agentforge import Orchestrator

    llm = ScriptedLLM([_plan_call("nonsense")] * 3)
    with pytest.raises(LLMError, match="unknown role 'nonsense'"):
        Orchestrator(llm, ToolRegistry()).run("task")


def test_eval_session_records_a_bad_plan_as_a_failed_run():
    from agentforge.orchestrator import PlanError

    root = Path(__file__).parent.parent
    spec = importlib.util.spec_from_file_location("run_evals3", root / "evals" / "run_evals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def bad_plan(case, llm, fallback=False):
        raise PlanError("dependency cycle detected")

    mod.run_case = bad_plan
    row = mod.safe_run({"name": "x", "task": "t", "checks": {}}, object())
    assert row["passed"] is False and "cycle" in row["failures"][0]
