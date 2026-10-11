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

import http.client
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
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


class MCPConnection:
    """What an agent needs from an MCP server, whatever the transport (stdio or HTTP)."""

    name: str
    timeout: float
    server_info: dict[str, Any]
    protocol_version: str | None

    def _request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

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


class MCPClient(MCPConnection):
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


# --------------------------------------------------------------------------- HTTP transport
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Never follow redirects: urllib would replay our headers (incl. Authorization) elsewhere."""

    def redirect_request(
        self, req: Any, fp: Any, code: Any, msg: Any, headers: Any, newurl: Any
    ) -> None:
        return None


def _expand_env(value: str, where: str) -> str:
    """Replace ${NAME} with the client's environment variable, so secrets stay out of config."""

    def sub(match: re.Match[str]) -> str:
        found = os.environ.get(match.group(1))
        if found is None:
            raise MCPError(f"{where} references unset environment variable {match.group(1)}")
        return found

    return _ENV_REF.sub(sub, value)


class MCPHTTPClient(MCPConnection):
    """MCP over Streamable HTTP: each JSON-RPC message is a POST to one endpoint.

    The server answers a request with plain JSON or with a server-sent-event stream (which may
    carry notifications and server requests before the actual response). A session id returned
    by `initialize` is echoed on every later request, and the negotiated protocol version is sent
    in `MCP-Protocol-Version`. Every call is its own HTTP request, so calls are naturally
    concurrent.

    Security: plain http is accepted only for localhost, redirects are refused (credentials must
    not travel to another host), and header values may reference environment variables as
    `${NAME}` so tokens never need to be written into a config file.
    """

    def __init__(
        self,
        name: str,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        parts = urllib.parse.urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise MCPError(f"MCP server '{name}': url must be http(s)://host/path, got {url!r}")
        if parts.scheme == "http" and parts.hostname not in _LOOPBACK_HOSTS:
            raise MCPError(
                f"MCP server '{name}': refusing plain http to {parts.hostname}; "
                "use https (plain http is allowed only for localhost)"
            )
        self.name = name
        self.url = url
        self.timeout = timeout
        self.server_info = {}
        self.protocol_version = None
        self.session_id: str | None = None
        self._headers: dict[str, str] = {}
        for key, value in (headers or {}).items():
            expanded = _expand_env(value, f"MCP server '{name}' header '{key}'")
            if "\r" in key + expanded or "\n" in key + expanded:
                raise MCPError(f"MCP server '{name}': header '{key}' contains a line break")
            self._headers[key] = expanded
        self._next_id = 0
        self._id_lock = threading.Lock()
        self._opener = urllib.request.build_opener(_NoRedirect)

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> MCPHTTPClient:
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
        sid, self.session_id = self.session_id, None
        if not sid:
            return
        headers = {**self._headers, "Mcp-Session-Id": sid}
        if self.protocol_version:
            headers["MCP-Protocol-Version"] = self.protocol_version
        req = urllib.request.Request(self.url, headers=headers, method="DELETE")
        try:  # ending the session is best-effort; servers may answer 405 or be gone already
            self._opener.open(req, timeout=5).close()
        except (OSError, http.client.HTTPException):
            pass

    def __enter__(self) -> MCPHTTPClient:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport
    def _timeout_error(self, wait: float, method: str) -> MCPError:
        return MCPError(f"MCP server '{self.name}' did not answer '{method}' within {wait:g}s")

    def _http_error(self, exc: urllib.error.HTTPError) -> str:
        if exc.code in (401, 403):
            return (
                f"MCP server '{self.name}' rejected the credentials (HTTP {exc.code}); "
                "check the 'headers' in your MCP config"
            )
        if exc.code == 404 and self.session_id:
            return f"MCP session with '{self.name}' expired or was closed (HTTP 404); reconnect"
        if 300 <= exc.code < 400:
            return (
                f"MCP server '{self.name}' answered with a redirect (HTTP {exc.code}) to "
                f"{exc.headers.get('Location')}; refusing to follow it so credentials are not "
                "sent to another address"
            )
        return f"MCP server '{self.name}' returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}"  # noqa: E501

    def _post(self, message: dict[str, Any], wait: float) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            **self._headers,
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol_version and message.get("method") != "initialize":
            headers["MCP-Protocol-Version"] = self.protocol_version
        req = urllib.request.Request(
            self.url, data=json.dumps(message).encode(), headers=headers, method="POST"
        )
        method = str(message.get("method"))
        try:
            return self._opener.open(req, timeout=wait)
        except urllib.error.HTTPError as exc:
            raise MCPError(self._http_error(exc)) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise self._timeout_error(wait, method) from exc
            raise MCPError(
                f"cannot reach MCP server '{self.name}' at {self.url}: {exc.reason}"
            ) from exc
        except TimeoutError as exc:
            raise self._timeout_error(wait, method) from exc

    def _notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        message = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._post(message, self.timeout).close()  # servers answer 202 Accepted with no body

    def _request(
        self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None
    ) -> dict[str, Any]:
        wait = self.timeout if timeout is None else timeout
        with self._id_lock:
            self._next_id += 1
            rid = self._next_id
        message = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        resp = self._post(message, wait)
        try:
            with resp:
                if method == "initialize" and resp.headers.get("Mcp-Session-Id"):
                    self.session_id = resp.headers["Mcp-Session-Id"]
                if resp.headers.get_content_type() == "text/event-stream":
                    msg = self._read_sse(resp, rid, wait)
                else:
                    msg = self._read_json(resp, rid)
        except TimeoutError as exc:
            raise self._timeout_error(wait, method) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise MCPError(
                f"connection to MCP server '{self.name}' was interrupted: {exc}"
            ) from exc

        if "error" in msg:
            err = msg["error"] if isinstance(msg["error"], dict) else {}
            raise MCPError(
                f"MCP '{method}' failed: {err.get('message', 'unknown error')} "
                f"(code {err.get('code', '?')})"
            )
        result = msg.get("result")
        return result if isinstance(result, dict) else {}

    def _read_json(self, resp: Any, rid: int) -> dict[str, Any]:
        body = resp.read()
        if not body:
            raise MCPError(f"MCP server '{self.name}' sent no response to a request")
        try:
            data = json.loads(body)
        except json.JSONDecodeError as exc:
            raise MCPError(f"MCP server '{self.name}' sent invalid JSON: {exc}") from exc
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("id") == rid:
                return item
        raise MCPError(f"MCP server '{self.name}' answered a different request id")

    def _read_sse(self, resp: Any, rid: int, wait: float) -> dict[str, Any]:
        """Read events until the response to `rid` arrives; answer server requests on the way."""
        data_lines: list[str] = []
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                found = self._handle_event(data_lines, rid, wait)
                data_lines = []
                if found is not None:
                    return found
            elif line.startswith("data:"):
                chunk = line[5:]
                data_lines.append(chunk[1:] if chunk.startswith(" ") else chunk)
        found = self._handle_event(data_lines, rid, wait)  # a final event without a blank line
        if found is not None:
            return found
        raise MCPError(f"MCP server '{self.name}' ended the stream without a response")

    def _handle_event(self, data_lines: list[str], rid: int, wait: float) -> dict[str, Any] | None:
        if not data_lines:
            return None
        try:
            msg = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            return None
        if not isinstance(msg, dict):
            return None
        if "method" in msg:
            if "id" in msg:  # a request from the server; we advertise no capabilities
                self._reply_to_server(msg, wait)
            return None  # notifications (logging, progress, ...) are ignored
        if msg.get("id") == rid and ("result" in msg or "error" in msg):
            return msg
        return None

    def _reply_to_server(self, msg: dict[str, Any], wait: float) -> None:
        if msg.get("method") == "ping":
            reply: dict[str, Any] = {"jsonrpc": "2.0", "id": msg["id"], "result": {}}
        else:
            reply = {
                "jsonrpc": "2.0",
                "id": msg["id"],
                "error": {"code": -32601, "message": "Method not found"},
            }
        try:
            self._post(reply, wait).close()
        except (MCPError, OSError):
            pass


# --------------------------------------------------------------------------- tool adapter
_NAME_BAD = re.compile(r"[^a-zA-Z0-9_-]")


def _safe_name(server: str, tool: str) -> str:
    return _NAME_BAD.sub("_", f"{server}__{tool}")[:64]  # provider tool-name limits


def _allows_null(prop: Any) -> bool:
    """Does this JSON-schema property explicitly accept null?"""
    if not isinstance(prop, dict):
        return False
    kind = prop.get("type")
    if kind == "null" or (isinstance(kind, list) and "null" in kind):
        return True
    return any(_allows_null(b) for key in ("anyOf", "oneOf") for b in prop.get(key, []))


def prune_null_args(args: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Drop `null` values for properties that don't accept null.

    Small models often fill every optional parameter, sending null for the ones they don't
    need; strict servers reject that (e.g. "expected number, received null"). Omitting the key
    is what the model meant.
    """
    props = schema.get("properties") or {}
    return {k: v for k, v in args.items() if v is not None or _allows_null(props.get(k))}


def _make_caller(client: MCPConnection, tool_name: str, schema: dict[str, Any]) -> Any:
    def call(args: dict[str, Any]) -> str:
        text, is_error = client.call_tool(tool_name, prune_null_args(args, schema))
        if is_error:
            raise ToolError(text)
        return text

    return call


def mcp_tools(
    client: MCPConnection,
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
                fn=_make_caller(client, orig, schema),
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
    command: str = ""
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    include: list[str] | None = None
    exclude: list[str] = field(default_factory=list)
    trust_read_only_hint: bool = True
    url: str | None = None  # set for Streamable HTTP servers instead of `command`
    headers: dict[str, str] = field(default_factory=dict)


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
        if not isinstance(cfg, dict):
            raise MCPError(
                f"MCP server '{name}' needs a string 'command' (stdio) or a 'url' (http)"
            )
        command, url = cfg.get("command"), cfg.get("url")
        if command is None and url is None:
            raise MCPError(
                f"MCP server '{name}' needs a string 'command' (stdio) or a 'url' (http)"
            )
        if command is not None and url is not None:
            raise MCPError(f"MCP server '{name}' needs exactly one of 'command' or 'url'")
        if command is not None and not isinstance(command, str):
            raise MCPError(f"MCP server '{name}' needs a string 'command'")
        if url is not None:
            if not isinstance(url, str):
                raise MCPError(f"MCP server '{name}' needs a string 'url'")
            if cfg.get("type") not in (None, "http", "streamable-http"):
                raise MCPError(
                    f"MCP server '{name}': transport '{cfg.get('type')}' is not supported "
                    "(use stdio, or Streamable HTTP via 'url')"
                )
        headers = cfg.get("headers", {})
        if not isinstance(headers, dict) or not all(isinstance(v, str) for v in headers.values()):
            raise MCPError(f"MCP server '{name}': 'headers' must map names to strings")
        args = cfg.get("args", [])
        env = cfg.get("env", {})
        if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
            raise MCPError(f"MCP server '{name}': 'args' must be a list of strings")
        if not isinstance(env, dict) or not all(isinstance(v, str) for v in env.values()):
            raise MCPError(f"MCP server '{name}': 'env' must map names to strings")
        specs.append(
            MCPServerSpec(
                name=name,
                command=command or "",
                url=url,
                headers=headers,
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
        self.clients: list[MCPConnection] = []
        self.tools: list[Tool] = []

    def start(self) -> MCPToolbox:
        try:
            for spec in self.specs:
                client: MCPConnection
                if spec.url:
                    client = MCPHTTPClient(
                        spec.name, spec.url, headers=spec.headers, timeout=self.timeout
                    ).start()
                else:
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
