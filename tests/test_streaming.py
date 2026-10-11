"""Streaming: SSE parsing for OpenAI-compatible servers, the Anthropic SDK path, agent + CLI."""

import io
import json
import sys
import types
import urllib.error

import pytest

from agentforge import Agent, ToolRegistry, cli
from agentforge.builtin_tools import calculator
from agentforge.llm import AnthropicLLM, LLMError, LLMResponse, OpenAICompatLLM, Usage
from fakes import ScriptedLLM, text, tool_use

MSGS = [{"role": "user", "content": "hi"}]


# ------------------------------------------------------------------ SSE fixtures
class FakeStream:
    """Stands in for the urlopen() response: a context manager iterating raw SSE lines."""

    def __init__(self, items):
        self.items = items

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for item in self.items:
            if isinstance(item, Exception):
                raise item
            yield (item + "\n").encode()


def chunk(content=None, tool_calls=None, finish=None, usage=None, choices=True):
    delta = {}
    if content is not None:
        delta["content"] = content
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    out = {"choices": [{"delta": delta, "finish_reason": finish}] if choices else []}
    if usage:
        out["usage"] = {"prompt_tokens": usage[0], "completion_tokens": usage[1]}
    return out


def sse(*chunks, done=True):
    lines = []
    for c in chunks:
        lines += ["data: " + json.dumps(c), ""]
    if done:
        lines.append("data: [DONE]")
    return lines


@pytest.fixture
def server(monkeypatch):
    """Patch urlopen to serve `server.items`; captures the request body."""
    state = types.SimpleNamespace(items=[], bodies=[])

    def fake_urlopen(req, timeout):
        state.bodies.append(json.loads(req.data))
        return FakeStream(state.items)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return state


def _llm():
    return OpenAICompatLLM("m", base_url="http://x/v1", max_retries=0)


# ------------------------------------------------------------------ OpenAI-compatible
def test_text_deltas_are_forwarded_live_and_reply_is_assembled(server):
    server.items = sse(
        chunk("Hel"), chunk("lo"), chunk(finish="stop"), chunk(choices=False, usage=(11, 4))
    )
    seen = []
    resp = _llm().generate(system="s", messages=MSGS, on_text=seen.append)
    assert seen == ["Hel", "lo"]
    assert resp.text == "Hello" and resp.stop_reason == "end_turn" and resp.usage == Usage(11, 4)
    body = server.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}


def test_tool_call_arguments_split_across_chunks_are_reassembled(server):
    fn = lambda name=None, args=None: {  # noqa: E731
        k: v for k, v in (("name", name), ("arguments", args)) if v is not None
    }
    server.items = sse(
        chunk(tool_calls=[{"index": 0, "id": "c1", "function": fn("calculator", "")}]),
        chunk(tool_calls=[{"index": 0, "function": fn(args='{"expres')}]),
        chunk(tool_calls=[{"index": 0, "function": fn(args='sion": "2+2"}')}]),
        chunk(finish="tool_calls"),
    )
    resp = _llm().generate(system="s", messages=MSGS, on_text=lambda d: None)
    (call,) = resp.tool_calls
    assert (call.id, call.name, call.input) == ("c1", "calculator", {"expression": "2+2"})
    assert resp.stop_reason == "tool_use"


def test_servers_that_send_finished_tool_calls_without_ids(server):
    full = {"index": 0, "function": {"name": "calculator", "arguments": {"expression": "1+1"}}}
    server.items = sse(chunk(tool_calls=[full], finish="stop"))
    (call,) = _llm().generate(system="s", messages=MSGS, on_text=lambda d: None).tool_calls
    assert call.id == "call_0" and call.input == {"expression": "1+1"}


def test_keepalives_blank_lines_and_garbage_are_ignored(server):
    server.items = [": keep-alive", "", "data: not json", "event: ping", *sse(chunk("ok"))]
    seen = []
    assert _llm().generate(system="s", messages=MSGS, on_text=seen.append).text == "ok"
    assert seen == ["ok"]


def test_usage_is_estimated_when_the_server_does_not_report_it(server):
    server.items = sse(chunk("x" * 40))
    resp = _llm().generate(system="s", messages=MSGS, on_text=lambda d: None)
    assert resp.usage.output_tokens == 10 and resp.usage.input_tokens > 0


def test_connection_dropped_mid_stream_raises_a_clear_error_after_partial_text(server):
    server.items = [*sse(chunk("partial"), done=False), ConnectionResetError("reset by peer")]
    seen = []
    with pytest.raises(LLMError, match="was interrupted"):
        _llm().generate(system="s", messages=MSGS, on_text=seen.append)
    assert seen == ["partial"]  # what was already shown stays shown; nothing is retried


def test_stall_mid_stream_is_reported_as_a_timeout(server):
    server.items = [*sse(chunk("a"), done=False), TimeoutError("timed out")]
    with pytest.raises(LLMError, match="AGENTFORGE_TIMEOUT"):
        _llm().generate(system="s", messages=MSGS, on_text=lambda d: None)


def test_error_event_inside_the_stream(server):
    server.items = ["data: " + json.dumps({"error": {"message": "overloaded"}})]
    with pytest.raises(LLMError, match="stream error"):
        _llm().generate(system="s", messages=MSGS, on_text=lambda d: None)


def test_http_error_before_streaming_starts_is_clean_and_emits_nothing(monkeypatch):
    def deny(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 401, "no", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", deny)
    seen = []
    with pytest.raises(LLMError, match="invalid API key"):
        _llm().generate(system="s", messages=MSGS, on_text=seen.append)
    assert seen == []


def test_without_a_callback_the_request_is_not_streamed(server, monkeypatch):
    llm = _llm()
    monkeypatch.setattr(llm, "_post", lambda p: server.bodies.append(p) or {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}], "usage": {},
    })  # fmt: skip
    assert llm.generate(system="s", messages=MSGS).text == "hi"
    assert "stream" not in server.bodies[0]


# ------------------------------------------------------------------ Anthropic SDK path
class _Block:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_anthropic_streams_via_the_sdk_and_returns_the_final_message(monkeypatch):
    captured = {}

    class Stream:
        text_stream = ["Hel", "lo"]

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return types.SimpleNamespace(
                content=[
                    _Block(type="text", text="Hello"),
                    _Block(type="tool_use", id="tu", name="calculator", input={"expression": "1"}),
                ],
                stop_reason="tool_use",
                usage=types.SimpleNamespace(input_tokens=5, output_tokens=2),
            )

    class Messages:
        def create(self, **kw):
            raise AssertionError("create() must not be used when streaming")

        def stream(self, **kw):
            captured.update(kw)
            return Stream()

    class Anthropic:
        def __init__(self, **kw):
            self.messages = Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=Anthropic))
    seen = []
    resp = AnthropicLLM("m", api_key="k").generate(
        system="s", messages=MSGS, tools=[{"name": "calculator"}], on_text=seen.append
    )
    assert seen == ["Hel", "lo"] and resp.text == "Hello"
    assert resp.tool_calls[0].name == "calculator" and resp.usage == Usage(5, 2)
    assert captured["tools"] == [{"name": "calculator"}]


# ------------------------------------------------------------------ agent loop
def test_agent_streams_text_and_passes_nothing_when_disabled():
    deltas = []
    llm = ScriptedLLM([text("hello world")])
    res = Agent(llm, ToolRegistry(), on_text=deltas.append).run("x")
    assert "".join(deltas) == "hello world" == res.output

    quiet = ScriptedLLM([text("hello")])
    Agent(quiet, ToolRegistry()).run("x")
    assert "on_text" not in quiet.calls[0]  # providers see no change unless streaming is on


def _text_then_tool():
    return LLMResponse(
        [
            {"type": "text", "text": "Checking."},
            {"type": "tool_use", "id": "t1", "name": "calculator", "input": {"expression": "1+1"}},
        ],
        "tool_use",
        Usage(1, 1),
    )


def test_steps_are_separated_by_a_blank_line_only_when_text_was_streamed():
    deltas = []
    llm = ScriptedLLM([_text_then_tool(), text("Done")])
    Agent(llm, ToolRegistry([calculator]), on_text=deltas.append).run("x")
    assert "".join(deltas) == "Checking.\n\nDone"

    deltas.clear()
    silent = ScriptedLLM([tool_use(("calculator", {"expression": "1+1"})), text("Done")])
    Agent(silent, ToolRegistry([calculator]), on_text=deltas.append).run("x")
    assert "".join(deltas) == "Done"  # a tool-only step adds no stray separator


def test_on_tool_fires_before_the_tool_runs():
    events = []
    llm = ScriptedLLM([tool_use(("calculator", {"expression": "2+2"})), text("4")])
    Agent(llm, ToolRegistry([calculator]), on_tool=lambda n, i: events.append((n, i))).run("x")
    assert events == [("calculator", {"expression": "2+2"})]


# ------------------------------------------------------------------ CLI
@pytest.fixture
def cli_llm(monkeypatch, tmp_path):
    def install(llm):
        monkeypatch.setattr(cli, "make_llm", lambda *a, **k: llm)
        monkeypatch.setenv("AGENTFORGE_TRACE_DIR", str(tmp_path / "traces"))
        return ["--workspace", str(tmp_path / "ws")]

    return install


def test_cli_stream_prints_the_answer_once(cli_llm, capsys):
    ws = cli_llm(ScriptedLLM([text("42")]))
    assert cli.main(["run", "x", "--stream", *ws]) == 0
    assert capsys.readouterr().out == "42\n"  # streamed, not printed a second time


def test_cli_stream_announces_tools_on_stderr(cli_llm, capsys):
    ws = cli_llm(
        ScriptedLLM([tool_use(("calculator", {"expression": "6*7"})), text("The answer is 42")])
    )
    assert cli.main(["run", "x", "--stream", *ws]) == 0
    out = capsys.readouterr()
    assert out.out == "The answer is 42\n"
    assert "· calculator(" in out.err and "[end_turn" in out.err


def test_cli_stream_falls_back_to_the_output_when_nothing_was_streamed(cli_llm, capsys):
    ws = cli_llm(ScriptedLLM([tool_use(("calculator", {"expression": "1+1"}))]))
    assert cli.main(["run", "x", "--stream", "--max-steps", "1", *ws]) == 2
    assert "[agent stopped early: max_steps]" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["--json", "--orchestrate"])
def test_cli_rejects_incompatible_stream_combinations(cli_llm, capsys, flag):
    ws = cli_llm(ScriptedLLM([]))
    assert cli.main(["run", "x", "--stream", flag, *ws]) == 1
    assert "error: --stream" in capsys.readouterr().err


def test_cli_chat_stream(cli_llm, capsys, monkeypatch):
    ws = cli_llm(ScriptedLLM([text("hello")]))
    lines = iter(["hi"])

    def fake_input(prompt=""):
        try:
            return next(lines)
        except StopIteration:
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli.main(["chat", "--stream", *ws]) == 0
    assert "agent> hello" in capsys.readouterr().out


# ------------------------------------------------------------------ real socket, real HTTP
def test_text_really_arrives_before_the_server_has_finished(monkeypatch):
    """A live HTTP server (chunked SSE) refuses to send chunk 2 until the client has shown chunk 1.

    If the client buffered the response, chunk 1 would never reach on_text early and the server
    would flag it. This is the proof that streaming is live, not just parsed after the fact.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    first_shown = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def _send(self, line):
            data = (line + "\n").encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._send("data: " + json.dumps(chunk("Hel")))
            self._send("")
            self.server.buffered = not first_shown.wait(5)  # did the client show "Hel" yet?
            self._send("data: " + json.dumps(chunk("lo", finish="stop")))
            self._send("")
            self._send("data: [DONE]")
            self._send("")
            self.wfile.write(b"0\r\n\r\n")

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.buffered = None
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        llm = OpenAICompatLLM("m", base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1")
        seen = []

        def on_text(delta):
            seen.append(delta)
            first_shown.set()

        resp = llm.generate(system="s", messages=MSGS, on_text=on_text)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert httpd.buffered is False, "client held the text back until the stream ended"
    assert seen == ["Hel", "lo"] and resp.text == "Hello"
