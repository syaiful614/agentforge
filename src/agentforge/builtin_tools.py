"""Batteries-included tools: safe math, time, sandboxed files, long-term memory."""

from __future__ import annotations

import ast
import math
import operator
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any
from zoneinfo import ZoneInfo

from pydantic import Field

from .memory import LongTermMemory
from .tools import Tool, ToolRegistry, tool

# --------------------------------------------------------------------- calculator
_BINOPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type, Callable[..., Any]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _safe_eval(node: ast.AST) -> Any:
    """Whitelist-based AST evaluator: never uses eval(), so no code execution."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BINOPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError(f"unsupported expression element: {type(node).__name__}")


@tool
def calculator(
    expression: Annotated[str, Field(description="Math expression, e.g. '(17*23)+sqrt(144)'")],
) -> str:
    """Evaluate an arithmetic expression exactly. Use this instead of doing math in your head.
    Supports + - * / // % **, parentheses, pi, e, and the functions sqrt, log, log10, sin, cos,
    tan, abs, round, min, max.
    """
    if len(expression) > 200:
        raise ValueError("expression too long")
    result = _safe_eval(ast.parse(expression.strip(), mode="eval"))
    return f"{result:.12g}" if isinstance(result, float) else str(result)


@tool
def current_time(
    timezone: Annotated[str, Field(description="IANA timezone, e.g. 'Asia/Jakarta'")] = "UTC",
) -> str:
    """Get the current date and time in a given timezone."""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z (%A)")


# --------------------------------------------------------------------- files
def make_file_tools(root: Path, allow_write: bool = True) -> list[Tool]:
    """File tools confined to `root`. Path traversal and symlink escapes are rejected."""
    root = Path(root).resolve()

    def resolve(rel: str) -> Path:
        p = (root / rel).resolve()
        if not p.is_relative_to(root):
            raise PermissionError(f"path escapes workspace: {rel}")
        return p

    @tool
    def list_dir(
        path: Annotated[str, Field(description="Directory relative to the workspace")] = ".",
    ) -> str:
        """List files and folders in a workspace directory."""
        d = resolve(path)
        if not d.is_dir():
            raise NotADirectoryError(path)
        entries = sorted(d.iterdir(), key=lambda p: (p.is_file(), p.name))
        return "\n".join(f"{p.name}{'/' if p.is_dir() else ''}" for p in entries) or "(empty)"

    @tool
    def read_file(path: Annotated[str, Field(description="File relative to the workspace")]) -> str:
        """Read a UTF-8 text file from the workspace."""
        return resolve(path).read_text(encoding="utf-8", errors="replace")

    @tool
    def grep(
        pattern: Annotated[str, Field(description="Regular expression")],
        path: Annotated[str, Field(description="Directory or file to search")] = ".",
    ) -> str:
        """Search workspace text files for a regex. Returns up to 50 'file:line: text' matches."""
        rx = re.compile(pattern)
        base = resolve(path)
        files = [base] if base.is_file() else sorted(p for p in base.rglob("*") if p.is_file())
        hits: list[str] = []
        for f in files[:200]:
            if any(part.startswith(".") for part in f.relative_to(root).parts):
                continue
            if f.stat().st_size > 1_000_000:
                continue
            try:
                for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{f.relative_to(root)}:{n}: {line.strip()[:200]}")
                        if len(hits) >= 50:
                            return "\n".join(hits)
            except UnicodeDecodeError:
                continue
        return "\n".join(hits) or "no matches"

    tools_: list[Tool] = [list_dir, read_file, grep]

    if allow_write:

        @tool(read_only=False, requires_approval=True)
        def write_file(
            path: Annotated[str, Field(description="File relative to the workspace")],
            content: Annotated[str, Field(description="Full file content")],
        ) -> str:
            """Create or overwrite a text file in the workspace. Requires human approval."""
            p = resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"wrote {len(content)} chars to {path}"

        tools_.append(write_file)
    return tools_


# --------------------------------------------------------------------- memory
def make_memory_tools(ltm: LongTermMemory) -> list[Tool]:
    @tool(read_only=False)
    def remember(
        text: Annotated[str, Field(description="A durable fact worth remembering")],
        tags: Annotated[list[str], Field(description="Short topical tags")] = [],  # noqa: B006
    ) -> str:
        """Save a durable fact to long-term memory for future conversations."""
        return f"saved memory #{ltm.add(text, tags)}"

    @tool
    def recall(
        query: Annotated[str, Field(description="What to look up")],
        k: Annotated[int, Field(ge=1, le=10)] = 5,
    ) -> str:
        """Search long-term memory for facts saved earlier."""
        rows = ltm.search(query, k)
        return "\n".join(f"- {r.text} [{', '.join(r.tags)}]" for r in rows) or "no memories found"

    return [remember, recall]


def default_registry(workspace: Path, ltm: LongTermMemory | None = None) -> ToolRegistry:
    reg = ToolRegistry([calculator, current_time, *make_file_tools(workspace)])
    if ltm is not None:
        for t in make_memory_tools(ltm):
            reg.register(t)
    return reg
