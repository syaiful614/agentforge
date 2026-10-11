#!/usr/bin/env python3
"""A tiny, dependency-free MCP server (stdio transport) for demos and tests.

It speaks just enough of the protocol (initialize, tools/list, tools/call, ping) to show how
an MCP server works, and doubles as a fixture for AgentForge's MCP client tests.

    python examples/mcp_demo_server.py        # then type JSON-RPC lines on stdin

Extra tools used only by the test-suite appear when MCP_DEMO_TEST_TOOLS=1.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from typing import Any

PROTOCOL = "2025-06-18"
NOTES: list[str] = []


def _tool(
    name: str,
    description: str,
    props: dict[str, Any],
    read_only: bool,
    required: list[str] | None = None,
) -> dict[str, Any]:
    required = list(props) if required is None else required
    return {
        "name": name,
        "description": description,
        "inputSchema": {"type": "object", "properties": props, "required": required},
        "annotations": {"readOnlyHint": read_only},
    }


TEXT = {"text": {"type": "string"}}
TOOLS = [
    _tool("word_stats", "Count words, characters and unique words in a text.", TEXT, True),
    _tool("slugify", "Turn a title into a URL slug.", TEXT, True),
    _tool("add_note", "Save a note in server memory (a write operation).", TEXT, False),
    _tool(
        "truncate_text",
        "Cut a text to at most max_chars characters (max_chars is optional).",
        {"text": {"type": "string"}, "max_chars": {"type": "number"}},
        True,
        required=["text"],
    ),
    _tool("list_notes", "List saved notes.", {}, True),
    _tool("fail", "Always fails; shows error handling.", {"message": {"type": "string"}}, True),
]
if os.environ.get("MCP_DEMO_TEST_TOOLS"):
    TOOLS += [
        _tool("sleep", "Sleep for N seconds.", {"seconds": {"type": "number"}}, True),
        _tool("crash", "Kill the server process.", {}, True),
        _tool("getenv", "Read an environment variable.", {"name": {"type": "string"}}, True),
    ]


def call(name: str, args: dict[str, Any]) -> str:
    if name == "word_stats":
        text = str(args["text"])
        words = re.findall(r"\w+", text)
        return f"words={len(words)} chars={len(text)} unique={len({w.lower() for w in words})}"
    if name == "slugify":
        return re.sub(r"[^a-z0-9]+", "-", str(args["text"]).lower()).strip("-")
    if name == "truncate_text":
        limit = args.get("max_chars", len(str(args["text"])))
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            # strict like a zod/JSON-schema validator: null is not a number
            raise ValueError(f"max_chars: expected number, received {type(limit).__name__}")
        return str(args["text"])[: int(limit)]
    if name == "add_note":
        NOTES.append(str(args["text"]))
        return f"saved note #{len(NOTES)}"
    if name == "list_notes":
        return "\n".join(f"{i}. {n}" for i, n in enumerate(NOTES, 1)) or "(no notes)"
    if name == "fail":
        raise RuntimeError(str(args["message"]))
    if os.environ.get("MCP_DEMO_TEST_TOOLS"):
        if name == "sleep":
            time.sleep(float(args["seconds"]))
            return "slept"
        if name == "crash":
            os._exit(3)
        if name == "getenv":
            return os.environ.get(str(args["name"]), "<unset>")
    raise KeyError(name)


def send(obj: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")  # one message per line; never embed raw newlines
    sys.stdout.flush()


def reply(id_: Any, result: dict[str, Any]) -> None:
    send({"jsonrpc": "2.0", "id": id_, "result": result})


def reply_error(id_: Any, code: int, message: str) -> None:
    send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}})


def handle_call(id_: Any, params: dict[str, Any]) -> None:
    try:
        text, is_error = call(params.get("name", ""), params.get("arguments") or {}), False
    except KeyError as exc:
        reply_error(id_, -32602, f"unknown tool {exc}")
        return
    except Exception as exc:  # tool failures are results with isError, not protocol errors
        text, is_error = f"{type(exc).__name__}: {exc}", True
    reply(id_, {"content": [{"type": "text", "text": text}], "isError": is_error})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method, id_, params = msg.get("method"), msg.get("id"), msg.get("params") or {}
        if method == "initialize":
            reply(id_, {
                "protocolVersion": params.get("protocolVersion", PROTOCOL),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "agentforge-demo", "version": "0.1.0"},
            })  # fmt: skip
        elif method == "tools/list":
            reply(id_, {"tools": TOOLS})
        elif method == "tools/call":
            handle_call(id_, params)
        elif method == "ping":
            reply(id_, {})
        elif id_ is not None:  # unknown request; notifications (no id) are ignored
            reply_error(id_, -32601, "Method not found")


if __name__ == "__main__":
    main()
