import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
spec = importlib.util.spec_from_file_location("run_evals", ROOT / "evals" / "run_evals.py")
run_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_evals)


def test_grade_passes_and_fails_on_each_check_type(tmp_path):
    checks = {
        "contains_any": ["403"],
        "contains_all": ["east"],
        "tool_called": ["calculator"],
        "tool_not_called": ["write_file"],
        "file_absent": ["pwned.txt"],
        "max_steps": 3,
    }
    assert run_evals.grade(checks, "East is 403", ["calculator"], 2, tmp_path) == []

    (tmp_path / "pwned.txt").write_text("x")
    failures = run_evals.grade(checks, "nothing", ["write_file"], 9, tmp_path)
    assert len(failures) == 6


def test_cases_file_is_well_formed_and_fixtures_exist():
    cases = json.loads((ROOT / "evals" / "cases.json").read_text())
    assert len({c["name"] for c in cases}) == len(cases)
    assert all("task" in c and "checks" in c for c in cases)
    for f in ("sales.csv", "injected.txt"):
        assert (ROOT / "evals" / "fixtures" / f).exists()


def test_live_evals_require_api_key(monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("AGENTFORGE_PROVIDER", raising=False)
    monkeypatch.setattr("sys.argv", ["run_evals"])
    assert run_evals.main() == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_run_case_records_the_answer_and_flags_safety_violations():
    from fakes import ScriptedLLM, text, tool_use

    case = {"name": "x", "task": "t", "checks": {"tool_not_called": ["write_file"]}}
    benign = run_evals.run_case(case, ScriptedLLM([text("all good")]))
    assert benign["safety_ok"] is True and benign["output"] == "all good" and benign["passed"]

    risky_llm = ScriptedLLM(
        [tool_use(("write_file", {"path": "pwned.txt", "content": "x"})), text("done")]
    )
    risky = run_evals.run_case(case, risky_llm)
    assert risky["safety_ok"] is False and not risky["passed"]
    assert any(f.startswith("forbidden tool") for f in risky["failures"])
