"""MCP (Model Context Protocol) client over stdio, implemented straight from the spec.

MCP lets an agent use tools hosted by *external* servers (filesystem, GitHub, databases...)
without writing each tool by hand. The wire format is JSON-RPC 2.0, one JSON message per line,
over the server process's stdin/stdout. This module has no dependency beyond the stdlib.

Security posture (servers are third-party code, treat them as untrusted):
  * The server process gets a *minimal* environment, never your API keys. Pass what it needs
    explicitly via the config's "env".
  * MCP tools require human approval by default. A tool is auto-allowed only when the server
    marks it `readOnlyHint: true` and you left `trust_read_only_hint` on (hints are advisory
    and come from the server, so only connect servers you trust).
  * Tool descriptions are length-capped (a "tool poisoning" surface) and tool results are
    wrapped as untrusted data by the agent loop like any other tool output.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from ._version import __version__
from .tools import Tool, ToolError, ToolRegistry

PROTOCOL_VERSION = "2025-06-18"
MAX_DESCRIPTION_CHARS = 1000

_SAFE_ENV_KEYS = (
    "PATH", "PATHEXT", "SYSTEMROOT", "COMSPEC", "WINDIR", "HOME", "USERPROFILE",
    "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR", "LANG", "LC_ALL",
)  # fmt: skip


class MCPError(RuntimeError):
    """Protocol, transport or configuration failure when talking to an MCP server."""


def _base_env() -> dict[str, str]:
    return {k: v for k in _SAFE_ENV_KEYS if (v := os.environ.get(k)) is not None}


class _Pending:
    def __init__(self) -> None:
        self.event = threading.Event()
        self.response: dict[str, Any] | None = None
        self.error: str | None = None


class MCPClient:
    """One connection to one MCP server (a child process speaking JSON-RPC on stdio).

    Thread-safe: requests are matched to responses by id, so the agent can run several
    read-only tool calls concurrently over the same connection.
    """

    def __init__(
        self,
        name: str,
        command: str,
        args: Iterable[str] = (),
        *,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args)
        self.env = dict(env or {})
        self.cwd = cwd
        self.timeout = timeout
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str | None = None

        self._proc: subprocess.Popen[bytes] | None = None
        self._stdin: IO[bytes] | None = None
        self._threads: list[threading.Thread] = []
        self._write_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, _Pending] = {}
        self._closed = False
        self._stderr_tail: deque[str] = deque(maxlen=40)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> MCPClient:
        exe = shutil.which(self.command) or self.command  # resolves npx -> npx.cmd on Windows
        try:
            proc = subprocess.Popen(
                [exe, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**_base_env(), **self.env},
                cwd=self.cwd,
            )
        except OSError as exc:
            raise MCPError(
                f"cannot start MCP server '{self.name}' ({self.command}): {exc}"
            ) from exc
        assert proc.stdin and proc.stdout and proc.stderr
        self._proc, self._stdin = proc, proc.stdin
        for target, stream in ((self._read_loop, proc.stdout), (self._drain_stderr, proc.stderr)):
            t = threading.Thread(target=target, args=(stream,), daemon=True)
            t.start()
            self._threads.append(t)

        try:
            result = self._request(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "agentforge", "version": __version__},
                },
            )
            self.server_info = result.get("serverInfo") or {}
            self.protocol_version = result.get("protocolVersion")
            self._notify("notifications/initialized")
        except BaseException:
            self.close()
            raise
        return self

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for t in self._threads:
            t.join(timeout=2)
        for stream in (proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass

    def __enter__(self) -> MCPClient:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ MCP operations
    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None
        for _ in range(50):  # pagination guard
            res = self._request("tools/list", {"cursor": cursor} if cursor else {})
            tools.extend(t for t in res.get("tools", []) if isinstance(t, dict))
            cursor = res.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(
        self, name: str, arguments: dict[str, Any], timeout: float | None = None
    ) -> tuple[str, bool]:
        """Returns (text, is_error). Protocol-level failures raise MCPError."""
        res = self._request("tools/call", {"name": name, "arguments": arguments}, timeout)
        return _content_to_text(res), bool(res.get("isError"))

    # ------------------------------------------------------------------ transport
    def _send(self, obj: dict[str, Any]) -> None:
        stdin = self._stdin
        if stdin is None or self._proc is None:
            raise MCPError(f"MCP server '{self.name}' is not running")
        data = (json.dumps(obj) + "\n").encode("utf-8")
        try:
            with self._write_lock:
                stdin.write(data)
                stdin.flush()
        except (OSError, ValueError) as exc:
            raise MCPError(self._exit_message()) from exc

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        pending = _Pending()
        with self._state_lock:
            if self._closed:
                raise MCPError(self._exit_message())
            self._next_id += 1
            rid = self._next_id
            self._pending[rid] = pending
        try:
            self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        except MCPError:
            with self._state_lock:
                self._pending.pop(rid, None)
            raise

        wait = self.timeout if timeout is None else timeout
        if not pending.event.wait(wait):
            with self._state_lock:
                self._pending.pop(rid, None)
            try:  # tell the server we gave up; a late reply is ignored anyway
                self._notify("notifications/cancelled", {"requestId": rid, "reason": "timeout"})
            except MCPError:
                pass
            raise MCPError(f"MCP server '{self.name}' did not answer '{method}' within {wait:g}s")
        if pending.error:
            raise MCPError(pending.error)
        msg = pending.response or {}
        if "error" in msg:
            err = msg["error"] if isinstance(msg["error"], dict) else {}
            raise MCPError(
                f"MCP '{method}' failed: {err.get('message', 'unknown error')} "
                f"(code {err.get('code', '?')})"
            )
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    def _read_loop(self, stream: IO[bytes]) -> None:
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self._stderr_tail.append(f"[non-JSON on stdout] {line[:200]}")
                continue
            if not isinstance(msg, dict):
                continue
            if "method" in msg:
                self._handle_server_message(msg)
            elif "id" in msg:
                with self._state_lock:
                    pending = self._pending.pop(msg["id"], None)
                if pending is not None:  # unknown id = late reply after a timeout: ignore
                    pending.response = msg
                    pending.event.set()
        with self._state_lock:  # EOF: server is gone, fail everything still waiting
            self._closed = True
            waiting, self._pending = list(self._pending.values()), {}
        for pending in waiting:
            pending.error = self._exit_message()
            pending.event.set()

    def _handle_server_message(self, msg: dict[str, Any]) -> None:
        """Server->client traffic. We advertise no capabilities, so only `ping` is answered."""
        if "id" not in msg:
            return  # notification (logging, progress, list_changed...): ignored
        if msg.get("method") == "ping":
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": -32601, "message": "Method not found"},
            }
        try:
            self._send(reply)
        except MCPError:
            pass

    def _drain_stderr(self, stream: IO[bytes]) -> None:
        for raw in iter(stream.readline, b""):
            self._stderr_tail.append(raw.decode("utf-8", errors="replace").strip()[:300])

    def _exit_message(self) -> str:
        tail = " | ".join(list(self._stderr_tail)[-5:])
        base = f"MCP server '{self.name}' exited unexpectedly"
        return f"{base} (stderr: {tail})" if tail else base


def _content_to_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content") or []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            res = block.get("resource") or {}
            parts.append(str(res.get("text") or f"[resource: {res.get('uri', '?')}]"))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} content omitted]")
        else:
            parts.append(json.dumps(block, default=str))
    if not parts and result.get("structuredContent") is not None:
        parts.append(json.dumps(result["structuredContent"], default=str))
    return "\n".join(parts) or "(empty result)"


# --------------------------------------------------------------------------- tool adapter
_NAME_BAD = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_name(server: str, tool: str) -> str:
    return _NAME_BAD.sub("_", f"{server}__{tool}")[:64]  # provider tool-name limits


def _make_caller(client: MCPClient, tool_name: str) -> Any:
    def call(args: dict[str, Any]) -> str:
        text, is_error = client.call_tool(tool_name, args)
        if is_error:
            raise ToolError(text)
        return text

    return call


def mcp_tools(
    client: MCPClient,
    *,
    include: Iterable[str] | None = None,
    exclude: Iterable[str] = (),
    trust_read_only_hint: bool = True,
) -> list[Tool]:
    """Expose a server's tools as AgentForge `Tool`s, named `<server>__<tool>`.

    `include`/`exclude` match the server's original tool names (allow-list your attack surface).
    """
    allow = None if include is None else set(include)
    deny = set(exclude)
    tools: list[Tool] = []
    seen: set[str] = set()
    for spec in client.list_tools():
        orig = spec.get("name")
        if not isinstance(orig, str) or not orig:
            continue
        if (allow is not None and orig not in allow) or orig in deny:
            continue
        annotations = spec.get("annotations") or {}
        read_only = bool(trust_read_only_hint and annotations.get("readOnlyHint") is True)
        description = str(spec.get("description") or orig).strip()[:MAX_DESCRIPTION_CHARS]
        schema = spec.get("inputSchema")
        if not isinstance(schema, dict) or schema.get("type") != "object":
            schema = {"type": "object", "properties": {}}
        schema = {k: v for k, v in schema.items() if k != "$schema"}

        name = _safe_name(client.name, orig)
        if name in seen:
            raise MCPError(f"tool name collision after sanitising: {name}")
        seen.add(name)
        tools.append(
            Tool(
                name=name,
                description=f"[MCP server: {client.name}] {description}",
                input_schema=schema,
                fn=_make_caller(client, orig),
                model=None,
                read_only=read_only,
                requires_approval=not read_only,
            )
        )
    return tools


# --------------------------------------------------------------------------- config + toolbox
@dataclass
class MCPServerSpec:
    name: str
    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    include: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    trust_read_only_hint: bool = True


def load_mcp_config(path: str | Path) -> list[MCPServerSpec]:
    """Load the de-facto standard `{"mcpServers": {name: {command, args, env}}}` file."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))  # tolerate Windows BOM
    except OSError as exc:
        raise MCPError(f"cannot read MCP config {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MCPError(f"MCP config {path} is not valid JSON: {exc}") from exc

    servers = data.get("mcpServers") if isinstance(data, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise MCPError(f"MCP config {path} needs a non-empty 'mcpServers' object")

    specs: list[MCPServerSpec] = []
    for name, cfg in servers.items():
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
            raise MCPError(f"invalid MCP server name '{name}' (use letters, digits, _ and -)")
        if not isinstance(cfg, dict) or not isinstance(cfg.get("command"), str):
            raise MCPError(f"MCP server '{name}' needs a string 'command'")
        args = cfg.get("args", [])
        env = cfg.get("env", {})
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise MCPError(f"MCP server '{name}': 'args' must be a list of strings")
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise MCPError(f"MCP server '{name}': 'env' must map names to strings")
        specs.append(
            MCPServerSpec(
                name=name,
                command=cfg["command"],
                args=args,
                env=env,
                cwd=cfg.get("cwd"),
                include=cfg.get("include"),
                exclude=cfg.get("exclude", []),
                trust_read_only_hint=bool(cfg.get("trustReadOnlyHints", True)),
            )
        )
    return specs


class MCPToolbox:
    """Starts every configured server, collects their tools, and shuts them down together."""

    def __init__(self, specs: list[MCPServerSpec], timeout: float = 30.0) -> None:
        self.specs = specs
        self.timeout = timeout
        self.clients: list[MCPClient] = []
        self.tools: list[Tool] = []

    def start(self) -> MCPToolbox:
        try:
            for spec in self.specs:
                client = MCPClient(
                    spec.name,
                    spec.command,
                    spec.args,
                    env=spec.env,
                    cwd=spec.cwd,
                    timeout=self.timeout,
                ).start()
                self.clients.append(client)
                self.tools += mcp_tools(
                    client,
                    include=spec.include,
                    exclude=spec.exclude,
                    trust_read_only_hint=spec.trust_read_only_hint,
                )
        except BaseException:
            self.close()
            raise
        return self

    def register_into(self, registry: ToolRegistry) -> None:
        for t in self.tools:
            registry.register(t)

    def close(self) -> None:
        for c in self.clients:
            c.close()
        self.clients.clear()

    def __enter__(self) -> MCPToolbox:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()
