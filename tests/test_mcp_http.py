"""MCP over Streamable HTTP, tested against a real local HTTP server (examples/)."""

import importlib.util
import json
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentforge import (
    Agent,
    MCPHTTPClient,
    MCPToolbox,
    ToolRegistry,
    cli,
    load_mcp_config,
    mcp_tools,
)
from agentforge.mcp_client import MCPError
from fakes import ScriptedLLM, text, tool_use

EXAMPLES = Path(__file__).parent.parent / "examples"
STDIO_SERVER = str(EXAMPLES / "mcp_demo_server.py")
_server_module = None


@pytest.fixture(scope="module", autouse=True)
def _env():
    mp = pytest.MonkeyPatch()
    mp.setenv("MCP_DEMO_TEST_TOOLS", "1")  # exposes `sleep` (read before the demo core loads)
    mp.setenv("NO_PROXY", "127.0.0.1,localhost")
    mp.setenv("no_proxy", "127.0.0.1,localhost")
    yield
    mp.undo()


def _load_server_module():
    global _server_module
    if _server_module is None:
        spec = importlib.util.spec_from_file_location(
            "mcp_demo_http_server", EXAMPLES / "mcp_demo_http_server.py"
        )
        _server_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_server_module)
    return _server_module


@pytest.fixture
def start_server():
    started = []

    def start(**options):
        httpd = _load_server_module().DemoHTTPServer(("127.0.0.1", 0), **options)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        httpd.url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
        started.append(httpd)
        return httpd

    yield start
    for httpd in started:
        httpd.shutdown()
        httpd.server_close()


def _client(srv, **kw):
    return MCPHTTPClient("remote", srv.url, timeout=10, **kw)


# ------------------------------------------------------------------ protocol
def test_handshake_tools_calls_and_errors(start_server):
    srv = start_server()
    with _client(srv) as c:
        assert c.server_info["name"] == "agentforge-demo"
        assert c.session_id and c.protocol_version
        names = {t["name"] for t in c.list_tools()}
        assert {"word_stats", "slugify", "add_note", "sleep"} <= names
        assert "crash" not in names  # a process-killing tool is never exposed over HTTP
        assert c.call_tool("word_stats", {"text": "the cat and the dog"}) == (
            "words=5 chars=19 unique=4",
            False,
        )
        out, is_error = c.call_tool("fail", {"message": "boom"})
        assert is_error and "boom" in out
        with pytest.raises(MCPError, match="unknown tool"):
            c.call_tool("nope", {})


def test_session_and_protocol_version_headers(start_server):
    srv = start_server()
    with _client(srv) as c:
        c.list_tools()
        sid, version = c.session_id, c.protocol_version
    posts = [r for r in srv.requests if r["method"] == "POST"]
    first, later = posts[0]["headers"], posts[1:]
    assert "mcp-session-id" not in first and "mcp-protocol-version" not in first
    assert "application/json" in first["accept"] and "text/event-stream" in first["accept"]
    assert len(later) == 2  # notifications/initialized + tools/list
    assert all(r["headers"]["mcp-session-id"] == sid for r in later)
    assert all(r["headers"]["mcp-protocol-version"] == version for r in later)


def test_event_stream_responses_with_notifications_and_server_requests(start_server):
    srv = start_server(sse=True)
    with _client(srv) as c:
        assert c.call_tool("slugify", {"text": "Hello World"}) == ("hello-world", False)
    # the server pinged us mid-stream; the client must have answered it
    assert {"jsonrpc": "2.0", "id": "srv-1", "result": {}} in srv.replies


def test_concurrent_calls_share_one_session(start_server):
    srv = start_server()
    with _client(srv) as c:
        with ThreadPoolExecutor(8) as pool:
            out = list(
                pool.map(lambda i: c.call_tool("slugify", {"text": f"Hello {i}"})[0], range(20))
            )
    assert out == [f"hello-{i}" for i in range(20)]


def test_close_ends_the_session_once(start_server):
    srv = start_server()
    c = _client(srv).start()
    sid = c.session_id
    c.close()
    c.close()
    assert srv.deleted == [sid] and c.session_id is None


def test_expired_session_is_reported_clearly(start_server):
    srv = start_server()
    with _client(srv) as c:
        srv.sessions.clear()
        with pytest.raises(MCPError, match="expired"):
            c.list_tools()


def test_timeout_does_not_poison_the_connection(start_server):
    srv = start_server()
    with _client(srv) as c:
        start = time.monotonic()
        with pytest.raises(MCPError, match="did not answer 'tools/call' within 0.3s"):
            c.call_tool("sleep", {"seconds": 2.0}, timeout=0.3)
        assert time.monotonic() - start < 1.5
        assert c.call_tool("slugify", {"text": "Still Works"})[0] == "still-works"


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_unreachable_server_is_reported_clearly():
    # Linux refuses a closed loopback port at once; Windows takes ~2 s and some firewalls just
    # drop the packets. Both are an unreachable server and both must give a clear MCPError.
    url = f"http://127.0.0.1:{_closed_port()}/mcp"
    with pytest.raises(
        MCPError, match="cannot reach MCP server 'remote'|did not answer 'initialize'"
    ):
        MCPHTTPClient("remote", url, timeout=5).start()


def test_demo_server_ignores_client_disconnects_but_reports_real_errors(start_server, capsys):
    srv = start_server()
    try:
        raise ConnectionAbortedError("client went away")
    except ConnectionAbortedError:
        srv.handle_error(None, ("127.0.0.1", 1))
    assert capsys.readouterr().err == ""
    try:
        raise ValueError("a real bug")
    except ValueError:
        srv.handle_error(None, ("127.0.0.1", 1))
    assert "ValueError" in capsys.readouterr().err


# ------------------------------------------------------------------ credentials and safety
def test_bearer_token_from_the_environment(start_server, monkeypatch):
    srv = start_server(token="s3cret-token")
    monkeypatch.setenv("TEST_MCP_TOKEN", "s3cret-token")
    with _client(srv, headers={"Authorization": "Bearer ${TEST_MCP_TOKEN}"}) as c:
        assert c.list_tools()

    with pytest.raises(MCPError, match="rejected the credentials") as wrong:
        _client(srv, headers={"Authorization": "Bearer guess"}).start()
    assert "guess" not in str(wrong.value) and "s3cret" not in str(wrong.value)
    with pytest.raises(MCPError, match="rejected the credentials"):
        _client(srv).start()


def test_unset_environment_variable_in_a_header_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("NOT_SET_TOKEN", raising=False)
    with pytest.raises(MCPError, match="unset environment variable NOT_SET_TOKEN"):
        MCPHTTPClient("r", "http://127.0.0.1:1/mcp", headers={"Authorization": "${NOT_SET_TOKEN}"})


def test_redirects_are_refused_so_credentials_never_travel(start_server):
    srv = start_server()
    url = srv.url.replace("/mcp", "/redirect")
    with pytest.raises(MCPError, match="redirect"):
        MCPHTTPClient("remote", url, headers={"Authorization": "Bearer x"}).start()
    # urllib would have replayed the request (with Authorization) at /mcp; nothing else was hit
    assert [r["path"] for r in srv.requests] == ["/redirect"]


@pytest.mark.parametrize(
    "url,msg",
    [
        ("http://example.com/mcp", "refusing plain http"),
        ("ftp://example.com/mcp", "must be http"),
        ("not a url", "must be http"),
    ],
)
def test_url_validation(url, msg):
    with pytest.raises(MCPError, match=msg):
        MCPHTTPClient("r", url)


def test_plain_http_is_allowed_only_for_loopback():
    for url in ("http://localhost:9/mcp", "http://127.0.0.1:9/mcp", "http://[::1]:9/mcp"):
        MCPHTTPClient("r", url)
    MCPHTTPClient("r", "https://example.com/mcp")


def test_header_values_cannot_inject_line_breaks():
    with pytest.raises(MCPError, match="line break"):
        MCPHTTPClient("r", "https://example.com/mcp", headers={"X": "a\r\nInjected: 1"})


# ------------------------------------------------------------------ with agent, config, CLI
def test_agent_uses_http_tools_and_write_tools_need_approval(start_server):
    srv = start_server()
    with _client(srv) as c:
        registry = ToolRegistry(mcp_tools(c))
        llm = ScriptedLLM(
            [tool_use(("remote__word_stats", {"text": "the cat the dog"})), text("ok")]
        )
        assert Agent(llm, registry).run("x").ok
        assert "words=4" in llm.calls[1]["messages"][-1]["content"][0]["content"]

        llm = ScriptedLLM([tool_use(("remote__add_note", {"text": "hi"})), text("ok")])
        Agent(llm, registry).run("x")
        assert "denied" in llm.calls[1]["messages"][-1]["content"][0]["content"].lower()
        assert c.call_tool("list_notes", {})[0] == "(no notes)"


def _config(tmp_path, servers):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}))
    return path


def test_toolbox_mixes_stdio_and_http_servers(start_server, tmp_path, monkeypatch):
    srv = start_server(token="tok")
    monkeypatch.setenv("TEST_MCP_TOKEN", "tok")
    cfg = _config(
        tmp_path,
        {
            "local": {"command": sys.executable, "args": [STDIO_SERVER]},
            "remote": {"url": srv.url, "headers": {"Authorization": "Bearer ${TEST_MCP_TOKEN}"}},
        },
    )
    box = MCPToolbox(load_mcp_config(cfg)).start()
    sid = box.clients[1].session_id
    registry = ToolRegistry()
    box.register_into(registry)
    assert "local__slugify" in registry and "remote__slugify" in registry
    assert (
        registry.get("remote__word_stats").run({"text": "a b"}).content
        == "words=2 chars=3 unique=2"
    )
    box.close()
    assert srv.deleted == [sid]


@pytest.mark.parametrize(
    "server,msg",
    [
        ({"command": "x", "url": "https://x/mcp"}, "exactly one of"),
        ({"args": []}, "needs a string 'command'"),
        ({"url": 5}, "needs a string 'url'"),
        ({"url": "https://x/mcp", "type": "sse"}, "not supported"),
        ({"url": "https://x/mcp", "headers": {"A": 1}}, "'headers' must map"),
    ],
)
def test_http_config_validation(tmp_path, server, msg):
    with pytest.raises(MCPError, match=msg):
        load_mcp_config(_config(tmp_path, {"a": server}))


def test_http_config_accepts_explicit_http_type(tmp_path):
    (spec,) = load_mcp_config(
        _config(tmp_path, {"a": {"type": "http", "url": "https://x/mcp", "headers": {"K": "v"}}})
    )
    assert spec.url == "https://x/mcp" and spec.headers == {"K": "v"} and spec.command == ""


def test_cli_lists_tools_of_an_http_server(start_server, tmp_path, capsys):
    srv = start_server()
    cfg = _config(tmp_path, {"remote": {"url": srv.url}})
    assert cli.main(["mcp-tools", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "remote__word_stats" in out and "[read-only]" in out and "[approval]" in out


def test_cli_reports_http_auth_failures_cleanly(start_server, tmp_path, capsys):
    srv = start_server(token="needed")
    cfg = _config(tmp_path, {"remote": {"url": srv.url}})
    assert cli.main(["mcp-tools", str(cfg)]) == 1
    assert "error: MCP server 'remote' rejected the credentials" in capsys.readouterr().err
