import io
import json
import urllib.error

import pytest

from agentforge.llm import LLMError, OpenAICompatLLM, _to_openai_messages, make_llm

TOOL = {"name": "calculator", "description": "math", "input_schema": {"type": "object"}}


def _reply(message, finish="stop", usage=(12, 4)):
    return {
        "choices": [{"message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
    }


def test_message_translation_round_trip_shapes():
    msgs = [
        {"role": "user", "content": "what is 2+2?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "c1",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ],
        },
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "4"}]},
    ]
    out = _to_openai_messages("SYS", msgs)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "what is 2+2?"}
    call = out[2]["tool_calls"][0]
    assert call["id"] == "c1" and json.loads(call["function"]["arguments"]) == {"expression": "2+2"}
    assert out[3] == {"role": "tool", "tool_call_id": "c1", "content": "4"}


def test_generate_parses_tool_calls_and_usage(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="http://x/v1")
    seen = {}

    def fake_post(payload):
        seen.update(payload)
        return _reply(
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": "t1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": '{"expression": "1+1"}'},
                    }
                ],
            },
            finish="tool_calls",
        )

    monkeypatch.setattr(llm, "_post", fake_post)
    resp = llm.generate(
        system="s",
        messages=[{"role": "user", "content": "x"}],
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "calculator"},
    )
    assert seen["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert seen["tool_choice"] == {"type": "function", "function": {"name": "calculator"}}
    assert resp.stop_reason == "tool_use"
    assert resp.tool_calls[0].input == {"expression": "1+1"}
    assert resp.usage.total == 16


def test_generate_text_length_and_bad_json_args(monkeypatch):
    llm = OpenAICompatLLM("m")
    monkeypatch.setattr(llm, "_post", lambda p: _reply({"content": "partial"}, finish="length"))
    r = llm.generate(system="s", messages=[{"role": "user", "content": "x"}])
    assert r.text == "partial" and r.stop_reason == "max_tokens"

    bad = {"tool_calls": [{"id": "1", "function": {"name": "calculator", "arguments": "{oops"}}]}
    monkeypatch.setattr(llm, "_post", lambda p: _reply(bad, finish="tool_calls"))
    r = llm.generate(system="s", messages=[{"role": "user", "content": "x"}], tools=[TOOL])
    assert r.tool_calls[0].input == {}


def test_http_errors_become_friendly_llm_errors(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="http://x/v1", max_retries=0)

    def boom(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", boom)
    with pytest.raises(LLMError, match="invalid API key"):
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}])

    def down(req, timeout):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", down)
    with pytest.raises(LLMError, match="Is the server running"):
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}])


def test_retries_on_429_then_succeeds(monkeypatch):
    llm = OpenAICompatLLM("m", max_retries=2)
    monkeypatch.setattr("time.sleep", lambda s: None)
    calls = {"n": 0}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(_reply({"content": "ok"})).encode()

    def flaky(req, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 429, "slow", {}, io.BytesIO(b""))
        return Resp()

    monkeypatch.setattr("urllib.request.urlopen", flaky)
    assert llm.generate(system="s", messages=[{"role": "user", "content": "x"}]).text == "ok"
    assert calls["n"] == 2


def test_make_llm_factory():
    assert make_llm("openai", "llama3.1", base_url="http://h/v1").model == "llama3.1"
    with pytest.raises(LLMError, match="Set --model"):
        make_llm("openai", "claude-sonnet-5-5")
    with pytest.raises(LLMError, match="unknown provider"):
        make_llm("nope", "m")


def test_cli_prints_friendly_error_instead_of_traceback(monkeypatch, tmp_path, capsys):
    from agentforge import cli

    def explode(*a, **k):
        raise LLMError("Anthropic account has no credit.")

    monkeypatch.setattr(cli, "make_llm", explode)
    code = cli.main(["run", "x", "--workspace", str(tmp_path)])
    assert code == 1 and "error: Anthropic account has no credit." in capsys.readouterr().err


def test_timeouts_become_friendly_errors_and_are_not_retried(monkeypatch):
    llm = OpenAICompatLLM("m", base_url="http://x/v1", timeout=7, max_retries=3)
    calls = {"n": 0}

    def slow(req, timeout):
        calls["n"] += 1
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", slow)
    with pytest.raises(LLMError, match="within 7s.*AGENTFORGE_TIMEOUT"):
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}])
    assert calls["n"] == 1  # a timeout is expensive; never retry it blindly

    def connect_timeout(req, timeout):
        raise urllib.error.URLError(TimeoutError("connect timed out"))

    monkeypatch.setattr("urllib.request.urlopen", connect_timeout)
    with pytest.raises(LLMError, match="no response"):
        llm.generate(system="s", messages=[{"role": "user", "content": "x"}])


def test_timeout_setting_flows_into_factory(monkeypatch):
    from agentforge.config import Settings

    monkeypatch.setenv("AGENTFORGE_TIMEOUT", "600")
    assert Settings.from_env().timeout_s == 600.0
    llm = make_llm("openai", "m", timeout=600)
    assert llm.timeout == 600  # type: ignore[attr-defined]
