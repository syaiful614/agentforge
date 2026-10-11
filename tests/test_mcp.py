"""MCP client tests. Most run against a *real* subprocess (examples/mcp_demo_server.py)."""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from agentforge import (
    Agent,
    MCPClient,
    MCPError,
    MCPToolbox,
    ToolRegistry,
    allow_all,
    cli,
    load_mcp_config,
    mcp_tools,
)
from fakes import ScriptedLLM, text, tool_use

SERVER = str(Path(__file__).parent.parent / "examples" / "mcp_demo_server.py")


@pytest.fixture
def client():
    with MCPClient("demo", sys.executable, [SERVER], timeout=10) as c:
        yield c


@pytest.fixture
def test_client():
    env = {"MCP_DEMO_TEST_TOOLS": "1", "VISIBLE": "yes"}
    with MCPClient("demo", sys.executable, [SERVER], env=env, timeout=10) as c:
        yield c


def _tool_output(llm, call_index=1):
    return llm.calls[call_index]["messages"][-1]["content"][0]["content"]


# ------------------------------------------------------------------ protocol
def test_handshake_and_tool_discovery(client):
    assert client.server_info["name"] == "agentforge-demo"
    assert client.protocol_version
    names = {t["name"] for t in client.list_tools()}
    assert {"word_stats", "slugify", "add_note", "list_notes", "fail"} <= names
    assert "crash" not in names  # test-only tools stay hidden


def test_call_success_and_tool_level_error(client):
    assert client.call_tool("word_stats", {"text": "the cat and the dog"}) == (
        "words=5 chars=19 unique=4",
        False,
    )
    out, is_error = client.call_tool("fail", {"message": "boom"})
    assert is_error and "boom" in out


def test_protocol_error_for_unknown_tool(client):
    with pytest.raises(MCPError, match="unknown tool"):
        client.call_tool("nope", {})


def test_concurrent_requests_are_matched_to_the_right_response(client):
    def work(i):
        return client.call_tool("slugify", {"text": f"Hello World {i}"})[0]

    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(work, range(20)))
    assert results == [f"hello-world-{i}" for i in range(20)]


def test_timeout_then_connection_keeps_working(test_client):
    with pytest.raises(MCPError, match="did not answer 'tools/call' within 0.2s"):
        test_client.call_tool("sleep", {"seconds": 1.0}, timeout=0.2)
    # the late reply to the abandoned request must be ignored, not mismatched
    assert test_client.call_tool("slugify", {"text": "Still Works"}, timeout=10)[0] == "still-works"


def test_server_crash_fails_pending_and_later_calls(test_client):
    with pytest.raises(MCPError, match="exited unexpectedly"):
        test_client.call_tool("crash", {})
    with pytest.raises(MCPError, match="exited unexpectedly"):
        test_client.call_tool("slugify", {"text": "x"})


def test_close_is_idempotent_and_stops_the_process():
    c = MCPClient("demo", sys.executable, [SERVER], timeout=10).start()
    proc = c._proc
    c.close()
    c.close()
    assert proc.poll() is not None


def test_bad_command_gives_clear_error():
    with pytest.raises(MCPError, match="cannot start MCP server 'x'"):
        MCPClient("x", "definitely-not-a-real-binary-xyz").start()


# ------------------------------------------------------------------ security
def test_server_does_not_inherit_secrets_but_gets_explicit_env(monkeypatch):
    monkeypatch.setenv("SECRET_TEST_KEY", "topsecret")
    with MCPClient(
        "demo", sys.executable, [SERVER], env={"MCP_DEMO_TEST_TOOLS": "1", "VISIBLE": "yes"}
    ) as c:
        assert c.call_tool("getenv", {"name": "SECRET_TEST_KEY"})[0] == "<unset>"
        assert c.call_tool("getenv", {"name": "VISIBLE"})[0] == "yes"


# ------------------------------------------------------------------ adapter
def test_tools_are_namespaced_and_default_to_requiring_approval(client):
    by_name = {t.name: t for t in mcp_tools(client)}
    assert by_name["demo__word_stats"].parallel_safe
    assert by_name["demo__add_note"].requires_approval and not by_name["demo__add_note"].read_only
    assert by_name["demo__word_stats"].description.startswith("[MCP server: demo]")


def test_hints_are_ignored_when_not_trusted(client):
    assert all(t.requires_approval for t in mcp_tools(client, trust_read_only_hint=False))


def test_include_and_exclude_filters_use_original_names(client):
    assert [t.name for t in mcp_tools(client, include=["slugify"])] == ["demo__slugify"]
    names = {t.name for t in mcp_tools(client, exclude=["add_note", "fail"])}
    assert "demo__add_note" not in names and "demo__fail" not in names and "demo__slugify" in names


def test_tool_run_maps_results_errors_and_missing_args(client):
    tools = {t.name: t for t in mcp_tools(client)}
    ok = tools["demo__word_stats"].run({"text": "a b c"})
    assert not ok.is_error and ok.content == "words=3 chars=5 unique=3"
    failed = tools["demo__fail"].run({"message": "kaput"})
    assert failed.is_error and failed.content == "RuntimeError: kaput"
    missing = tools["demo__word_stats"].run({})
    assert missing.is_error and "missing required field(s): text" in missing.content


class _StubClient:
    name = "my server"

    def list_tools(self):
        return [
            {
                "name": "do.thing",
                "description": "x" * 5000,
                "inputSchema": {"$schema": "x", "type": "object"},
            },
            {"name": "bad", "inputSchema": "not-a-schema"},
            {"description": "no name"},
        ]

    def call_tool(self, name, args, timeout=None):
        return "ok", False


def test_names_are_sanitised_descriptions_capped_and_schemas_repaired():
    tools = mcp_tools(_StubClient())
    assert [t.name for t in tools] == ["my_server__do_thing", "my_server__bad"]
    assert len(tools[0].description) < 1100
    assert "$schema" not in tools[0].input_schema
    assert tools[1].input_schema == {"type": "object", "properties": {}}


# ------------------------------------------------------------------ with the agent loop
def test_agent_uses_mcp_tool_end_to_end(client):
    llm = ScriptedLLM([tool_use(("demo__word_stats", {"text": "the cat the dog"})), text("done")])
    res = Agent(llm, ToolRegistry(mcp_tools(client))).run("count words")
    assert res.ok and res.tool_calls == ["demo__word_stats"]
    assert "words=4" in _tool_output(llm)
    assert {t["name"] for t in llm.calls[0]["tools"]} >= {"demo__word_stats", "demo__add_note"}


def test_write_tool_denied_by_default_and_allowed_with_approval(client):
    registry = ToolRegistry(mcp_tools(client))
    llm = ScriptedLLM([tool_use(("demo__add_note", {"text": "hi"})), text("ok")])
    Agent(llm, registry).run("save note")  # default policy: read_only_only
    assert "denied" in _tool_output(llm).lower()
    assert client.call_tool("list_notes", {})[0] == "(no notes)"

    llm = ScriptedLLM([tool_use(("demo__add_note", {"text": "hi"})), text("ok")])
    Agent(llm, registry, approval=allow_all).run("save note")
    assert "saved note #1" in _tool_output(llm)


# ------------------------------------------------------------------ config + toolbox + CLI
def _config(tmp_path, servers, bom=False):
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": servers}), encoding="utf-8-sig" if bom else "utf-8")
    return path


def test_config_roundtrip_and_windows_bom(tmp_path):
    servers = {"demo": {"command": sys.executable, "args": [SERVER], "include": ["slugify"]}}
    (spec,) = load_mcp_config(_config(tmp_path, servers, bom=True))
    assert spec.name == "demo" and spec.include == ["slugify"] and spec.trust_read_only_hint


@pytest.mark.parametrize(
    "servers,msg",
    [
        ({"bad name!": {"command": "x"}}, "invalid MCP server name"),
        ({"a": {"args": []}}, "needs a string 'command'"),
        ({"a": {"command": "x", "args": "no"}}, "'args' must be a list"),
        ({"a": {"command": "x", "env": {"K": 1}}}, "'env' must map"),
        ({}, "non-empty 'mcpServers'"),
    ],
)
def test_config_validation(tmp_path, servers, msg):
    with pytest.raises(MCPError, match=msg):
        load_mcp_config(_config(tmp_path, servers))


def test_config_missing_or_malformed_file(tmp_path):
    with pytest.raises(MCPError, match="cannot read"):
        load_mcp_config(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    with pytest.raises(MCPError, match="not valid JSON"):
        load_mcp_config(bad)


def test_toolbox_registers_tools_and_cleans_up(tmp_path):
    servers = {"demo": {"command": sys.executable, "args": [SERVER]}}
    box = MCPToolbox(load_mcp_config(_config(tmp_path, servers))).start()
    procs = [c._proc for c in box.clients]
    reg = ToolRegistry()
    box.register_into(reg)
    assert "demo__slugify" in reg
    box.close()
    assert all(p.poll() is not None for p in procs)


def test_toolbox_start_failure_leaves_no_processes(tmp_path):
    servers = {
        "good": {"command": sys.executable, "args": [SERVER]},
        "bad": {"command": "definitely-not-a-real-binary-xyz"},
    }
    box = MCPToolbox(load_mcp_config(_config(tmp_path, servers)))
    with pytest.raises(MCPError):
        box.start()
    assert box.clients == []


def test_cli_mcp_tools_lists_discovered_tools(tmp_path, capsys):
    cfg = _config(tmp_path, {"demo": {"command": sys.executable, "args": [SERVER]}})
    assert cli.main(["mcp-tools", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "demo__word_stats" in out and "[read-only]" in out and "[approval]" in out


def test_cli_reports_bad_mcp_config_cleanly(tmp_path, capsys):
    assert cli.main(["mcp-tools", str(tmp_path / "missing.json")]) == 1
    assert "error: cannot read MCP config" in capsys.readouterr().err


def test_timeout_returns_promptly_even_if_the_server_is_busy(test_client):
    start = time.monotonic()
    with pytest.raises(MCPError):
        test_client.call_tool("sleep", {"seconds": 2.0}, timeout=0.2)
    assert time.monotonic() - start < 1.0


def test_cli_run_exposes_mcp_tools_to_the_agent(tmp_path, monkeypatch, capsys):
    cfg = _config(tmp_path, {"demo": {"command": sys.executable, "args": [SERVER]}})
    llm = ScriptedLLM([tool_use(("demo__slugify", {"text": "Hello World"})), text("slug ok")])
    monkeypatch.setattr(cli, "make_llm", lambda *a, **k: llm)
    monkeypatch.setenv("AGENTFORGE_TRACE_DIR", str(tmp_path / "traces"))
    code = cli.main(["run", "x", "--workspace", str(tmp_path / "ws"), "--mcp-config", str(cfg)])
    assert code == 0 and capsys.readouterr().out.strip() == "slug ok"
    assert "hello-world" in _tool_output(llm)


# ------------------------------------------------------------------ null-argument pruning
def test_null_optional_args_are_dropped_so_strict_servers_accept_them(client):
    # Reproduces what a small model does: fill every optional field, with null for "unused".
    # The strict demo server rejects null for a number, exactly like a schema validator would.
    out, is_error = client.call_tool("truncate_text", {"text": "hello world", "max_chars": None})
    assert is_error and "received NoneType" in out

    tools = {t.name: t for t in mcp_tools(client)}
    fixed = tools["demo__truncate_text"].run({"text": "hello world", "max_chars": None})
    assert not fixed.is_error and fixed.content == "hello world"
    kept = tools["demo__truncate_text"].run({"text": "hello world", "max_chars": 5})
    assert kept.content == "hello"


def test_prune_keeps_nulls_the_schema_allows():
    from agentforge.mcp_client import prune_null_args

    schema = {
        "properties": {
            "a": {"type": ["string", "null"]},
            "b": {"anyOf": [{"type": "number"}, {"type": "null"}]},
            "c": {"type": "number"},
        }
    }
    out = prune_null_args({"a": None, "b": None, "c": None, "unknown": None, "ok": 1}, schema)
    assert out == {"a": None, "b": None, "ok": 1}


# ------------------------------------------------------------------ trace shows tool errors
def _traced_run(client, trace_outputs, calls):
    from agentforge import Tracer

    tracer = Tracer()
    llm = ScriptedLLM([tool_use(*calls), text("done")])
    agent = Agent(llm, ToolRegistry(mcp_tools(client)), tracer=tracer, trace_outputs=trace_outputs)
    agent.run("x")
    return [e for e in tracer.events if e["type"] == "tool_call"]


def test_trace_records_error_text_by_default_but_not_successful_output(client):
    events = _traced_run(
        client,
        "errors",
        [("demo__fail", {"message": "kaput"}), ("demo__slugify", {"text": "Secret Title"})],
    )
    by_tool = {e["tool"]: e for e in events}
    assert "kaput" in by_tool["demo__fail"]["output_preview"]
    assert "output_preview" not in by_tool["demo__slugify"]


def test_trace_output_modes(client):
    call = [("demo__slugify", {"text": "Secret Title"})]
    assert "secret-title" in _traced_run(client, "all", call)[0]["output_preview"]
    assert (
        "output_preview" not in _traced_run(client, "none", [("demo__fail", {"message": "m"})])[0]
    )
    with pytest.raises(ValueError, match="trace_outputs"):
        Agent(ScriptedLLM([]), ToolRegistry(), trace_outputs="everything")
