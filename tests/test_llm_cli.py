import sys
import types

import pytest

from agentforge import cli
from agentforge.llm import AnthropicLLM, LLMResponse, Usage
from fakes import ScriptedLLM, text, tool_use


class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _install_fake_anthropic(monkeypatch, captured):
    class Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return types.SimpleNamespace(
                content=[
                    _Block(type="text", text="hello"),
                    _Block(
                        type="tool_use", id="tu_1", name="calculator", input={"expression": "1"}
                    ),
                ],
                stop_reason="tool_use",
                usage=types.SimpleNamespace(input_tokens=11, output_tokens=7),
            )

    class Anthropic:
        def __init__(self, **kwargs):
            captured["client_kwargs"] = kwargs
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))


def test_anthropic_adapter_translates_request_and_response(monkeypatch):
    captured = {}
    _install_fake_anthropic(monkeypatch, captured)
    llm = AnthropicLLM("claude-sonnet-5-5", api_key="k", max_retries=2)
    resp = llm.generate(
        system="sys",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"name": "calculator"}],
        tool_choice={"type": "auto"},
    )
    assert captured["client_kwargs"] == {"api_key": "k", "max_retries": 2}
    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["tools"] == [{"name": "calculator"}] and captured["tool_choice"] == {
        "type": "auto"
    }
    assert resp.text == "hello" and resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].name == "calculator" and resp.usage == Usage(11, 7)


def test_llm_response_helpers():
    r = LLMResponse([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}])
    assert r.text == "ab" and r.tool_calls == []
    assert Usage(1, 2) + Usage(3, 4) == Usage(4, 6)


@pytest.fixture
def fake_runtime(monkeypatch, tmp_path):
    llm = ScriptedLLM([tool_use(("calculator", {"expression": "6*7"})), text("42")])
    monkeypatch.setattr(cli, "AnthropicLLM", lambda *a, **k: llm)
    monkeypatch.setenv("AGENTFORGE_TRACE_DIR", str(tmp_path / "traces"))
    return llm


def test_cli_run(fake_runtime, tmp_path, capsys):
    code = cli.main(["run", "6*7?", "--workspace", str(tmp_path / "ws"), "--yes"])
    out = capsys.readouterr()
    assert code == 0 and out.out.strip() == "42"
    assert "calculator" in out.err
    assert (tmp_path / "traces" / "latest.jsonl").exists()


def test_cli_run_json(fake_runtime, tmp_path, capsys):
    cli.main(["run", "x", "--workspace", str(tmp_path / "ws"), "--json"])
    assert '"stop_reason": "end_turn"' in capsys.readouterr().out


def test_cli_tools_lists_builtins(tmp_path, capsys):
    assert cli.main(["tools", "--workspace", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for name in ("calculator", "read_file", "write_file", "recall"):
        assert name in out
    assert "[approval]" in out
