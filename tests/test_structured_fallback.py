import importlib.util
from pathlib import Path

import pytest

from agentforge.llm import LLMError
from agentforge.orchestrator import Plan
from agentforge.structured import StructuredOutputError, structured_output
from fakes import ScriptedLLM, text

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

    def boom(case, llm):
        raise LLMError("server down")

    mod.run_case = boom
    row = mod.safe_run({"name": "x", "task": "t", "checks": {}}, object())
    assert row["passed"] is False and "server down" in row["failures"][0]
