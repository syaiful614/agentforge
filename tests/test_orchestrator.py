import pytest

from agentforge import Budget, Orchestrator, ToolRegistry
from agentforge.builtin_tools import calculator
from agentforge.orchestrator import Plan, PlanError, Subtask, default_workers, validate_plan
from agentforge.structured import StructuredOutputError, structured_output
from fakes import RouterLLM, ScriptedLLM, text, tool_use

ROLES = {"researcher", "analyst", "writer"}


def st(id_, deps=(), role="analyst"):
    return Subtask(id=id_, goal=f"goal {id_}", role=role, depends_on=list(deps))


def test_validate_plan_layers():
    plan = Plan(subtasks=[st("a"), st("b"), st("c", ["a", "b"]), st("d", ["c"])])
    layers = validate_plan(plan, ROLES)
    assert [sorted(s.id for s in layer) for layer in layers] == [["a", "b"], ["c"], ["d"]]


@pytest.mark.parametrize(
    "subtasks,msg",
    [
        ([st("a", ["b"]), st("b", ["a"])], "cycle"),
        ([st("a", ["zzz"])], "unknown task"),
        ([st("a"), st("a")], "duplicate"),
        ([st("a", role="hacker")], "unknown role"),
        ([st("a", ["a"])], "itself"),
    ],
)
def test_validate_plan_rejects_bad_dags(subtasks, msg):
    with pytest.raises(PlanError, match=msg):
        validate_plan(Plan(subtasks=subtasks), ROLES)


def test_structured_output_repairs_invalid_first_attempt():
    llm = ScriptedLLM(
        [
            tool_use(("submit_plan", {"subtasks": []})),  # violates min_length=1
            tool_use(("submit_plan", {"subtasks": [{"id": "a", "goal": "g", "role": "writer"}]})),
        ]
    )
    plan = structured_output(
        llm, system="s", prompt="p", tool_name="submit_plan", tool_description="d", model_cls=Plan
    )
    assert plan.subtasks[0].id == "a"
    # the retry carried a tool_result error back to the model (valid API shape)
    retry_msgs = llm.calls[1]["messages"]
    assert retry_msgs[-1]["content"][0]["type"] == "tool_result"
    assert llm.calls[0]["tool_choice"] == {"type": "tool", "name": "submit_plan"}


def test_structured_output_gives_up():
    llm = ScriptedLLM([text("no tool")] * 3)
    with pytest.raises(StructuredOutputError):
        structured_output(
            llm, system="s", prompt="p", tool_name="t", tool_description="d", model_cls=Plan
        )


def _handler(review_verdicts):
    verdicts = list(review_verdicts)

    def handle(kw):
        choice = kw.get("tool_choice")
        if choice and choice["name"] == "submit_plan":
            return tool_use(
                (
                    "submit_plan",
                    {
                        "subtasks": [
                            {"id": "t1", "goal": "compute A", "role": "analyst"},
                            {"id": "t2", "goal": "compute B", "role": "analyst"},
                            {
                                "id": "t3",
                                "goal": "write up",
                                "role": "writer",
                                "depends_on": ["t1", "t2"],
                            },
                        ]
                    },
                )
            )
        if choice and choice["name"] == "submit_review":
            ok = verdicts.pop(0)
            return tool_use(
                ("submit_review", {"approved": ok, "issues": [] if ok else ["add units"]})
            )
        system = kw["system"]
        if "SYNTHESIZER" in system:
            return text(
                "FINAL: " + ("revised" if "add units" in kw["messages"][0]["content"] else "v1")
            )
        goal = kw["messages"][0]["content"]
        last = kw["messages"][-1]
        if "compute A" in goal and last["role"] == "user" and isinstance(last["content"], str):
            return tool_use(("calculator", {"expression": "2+3"}))
        if "compute B" in goal and last["role"] == "user" and isinstance(last["content"], str):
            return tool_use(("calculator", {"expression": "4*5"}))
        if "compute" in goal:
            return text("result computed")
        assert "result computed" in goal, "writer must receive dependency context"
        return text("written")

    return handle


def test_orchestrator_end_to_end_with_review_loop():
    llm = RouterLLM(_handler([False, True]))
    orch = Orchestrator(llm, ToolRegistry([calculator]), max_review_rounds=2)
    res = orch.run("do the thing")

    assert res.output == "FINAL: revised"  # rejected once, re-synthesized with feedback
    assert res.ok and res.review_rounds == 1
    assert set(res.subtask_results) == {"t1", "t2", "t3"}
    assert res.subtask_results["t1"].tool_calls == ["calculator"]
    assert res.tokens > 0


def test_workers_get_least_privilege_tools():
    llm = RouterLLM(_handler([True]))
    Orchestrator(llm, ToolRegistry([calculator]), max_review_rounds=1).run("x")
    writer_calls = [
        c for c in llm.calls if "technical writer" in c["system"] and not c.get("tool_choice")
    ]
    assert writer_calls and all(c["tools"] is None for c in writer_calls)
    analyst_calls = [
        c for c in llm.calls if "quantitative analyst" in c["system"] and not c.get("tool_choice")
    ]
    assert all([t["name"] for t in c["tools"]] == ["calculator"] for c in analyst_calls)


def test_budget_exhaustion_degrades_gracefully():
    llm = RouterLLM(_handler([True]))
    orch = Orchestrator(llm, ToolRegistry([calculator]), budget=Budget(max_steps=5, max_tokens=1))
    res = orch.run("x")
    assert not res.ok
    assert "budget" in res.output.lower()


def test_default_workers_reference_real_builtin_tools():
    import tempfile
    from pathlib import Path

    from agentforge.builtin_tools import default_registry
    from agentforge.memory import LongTermMemory

    reg = default_registry(Path(tempfile.mkdtemp()), LongTermMemory())
    for spec in default_workers().values():
        assert all(n in reg for n in spec.tools)
