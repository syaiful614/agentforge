#!/usr/bin/env python3
"""A tiny MCP server over Streamable HTTP (stdlib only), for demos and tests.

    python examples/mcp_demo_http_server.py --port 8808 [--token SECRET] [--sse]

Then point a config at it:
    {"mcpServers": {"remote": {"url": "http://127.0.0.1:8808/mcp",
                               "headers": {"Authorization": "Bearer ${MCP_TOKEN}"}}}}

It issues a session id on `initialize`, can require a bearer token, and with --sse answers every
request with a server-sent-event stream that also carries a notification and a server `ping`
request (which a client must answer), so the whole transport can be exercised locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import mcp_demo_server as core  # noqa: E402  (same folder; shares the demo tools)

HIDDEN_TOOLS = {"crash"}  # never expose a tool that kills the process inside an HTTP server


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def _record(self, method: str) -> None:
        headers = {k.lower(): v for k, v in self.headers.items()}
        self.server.requests.append({"method": method, "path": self.path, "headers": headers})  # type: ignore[attr-defined]

    def _plain(self, status: int, headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        return not token or self.headers.get("Authorization") == f"Bearer {token}"

    def _send_json(self, obj: dict[str, Any], headers: dict[str, str]) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_sse(self, obj: dict[str, Any], headers: dict[str, str]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()  # HTTP/1.0: the stream ends when the connection closes
        events = [
            {"jsonrpc": "2.0", "method": "notifications/message", "params": {"data": "hi"}},
            {"jsonrpc": "2.0", "id": "srv-1", "method": "ping"},
            obj,
        ]
        self.wfile.write(b": keep-alive\n\n")
        for event in events:
            self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
        self.wfile.flush()

    def _dispatch(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            return {"result": {
                "protocolVersion": params.get("protocolVersion", core.PROTOCOL),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agentforge-demo", "version": "0.1.0"},
            }}  # fmt: skip
        if method == "tools/list":
            return {"result": {"tools": [t for t in core.TOOLS if t["name"] not in HIDDEN_TOOLS]}}
        if method == "tools/call":
            if params.get("name") in HIDDEN_TOOLS:
                return {"error": {"code": -32602, "message": f"unknown tool {params['name']}"}}
            return core.execute(params)
        if method == "ping":
            return {"result": {}}
        return {"error": {"code": -32601, "message": "Method not found"}}

    def do_POST(self) -> None:
        srv: Any = self.server
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self._record("POST")
        if self.path == "/redirect":
            self._plain(302, {"Location": f"http://127.0.0.1:{srv.server_address[1]}/mcp"})
            return
        if self.path != "/mcp":
            self._plain(404)
            return
        if not self._authorized():
            self._plain(401, {"WWW-Authenticate": "Bearer"})
            return
        try:
            msg = json.loads(raw)
        except ValueError:
            self._plain(400)
            return
        if "method" not in msg:  # a client reply to one of our server requests
            srv.replies.append(msg)
            self._plain(202)
            return

        method, id_, params = msg["method"], msg.get("id"), msg.get("params") or {}
        headers: dict[str, str] = {}
        if method == "initialize":
            sid = uuid.uuid4().hex
            srv.sessions.add(sid)
            headers["Mcp-Session-Id"] = sid
        else:
            sid = self.headers.get("Mcp-Session-Id") or ""
            if not sid:
                self._plain(400)
                return
            if sid not in srv.sessions:
                self._plain(404)
                return
        if id_ is None:  # notification
            self._plain(202)
            return
        response = {"jsonrpc": "2.0", "id": id_, **self._dispatch(method, params)}
        (self._send_sse if srv.sse else self._send_json)(response, headers)

    def do_GET(self) -> None:  # we do not offer a server-initiated stream
        self._record("GET")
        self._plain(405)

    def do_DELETE(self) -> None:
        srv: Any = self.server
        self._record("DELETE")
        if not self._authorized():
            self._plain(401)
            return
        sid = self.headers.get("Mcp-Session-Id") or ""
        if sid in srv.sessions:
            srv.sessions.discard(sid)
            srv.deleted.append(sid)
            self._plain(200)
        else:
            self._plain(404)


class DemoHTTPServer(ThreadingHTTPServer):
    def __init__(
        self, address: tuple[str, int], *, token: str | None = None, sse: bool = False
    ) -> None:
        super().__init__(address, Handler)
        self.token = token
        self.sse = sse
        self.sessions: set[str] = set()
        self.requests: list[dict[str, Any]] = []
        self.replies: list[dict[str, Any]] = []
        self.deleted: list[str] = []

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up (timeout, Ctrl+C) is normal; anything else is a real bug.
        if isinstance(sys.exc_info()[1], ConnectionError):
            return
        super().handle_error(request, client_address)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8808)
    ap.add_argument("--token", help="require 'Authorization: Bearer <token>'")
    ap.add_argument("--sse", action="store_true", help="answer with event streams")
    args = ap.parse_args()
    server = DemoHTTPServer((args.host, args.port), token=args.token, sse=args.sse)
    print(f"listening on http://{args.host}:{server.server_address[1]}/mcp", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
