"""Tool definition, schema generation, validation and safe execution.

A tool is a plain typed Python function. The JSON schema sent to the model is
generated from the type hints (via pydantic), and model-produced arguments are
validated before the function ever runs. Failures are returned to the model as
error results so it can self-correct instead of crashing the loop.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

MAX_OUTPUT_CHARS = 8_000


def clean_schema(schema: Any) -> Any:
    """Strip pydantic's auto-generated `title` noise from a JSON schema."""
    if isinstance(schema, dict):
        return {
            k: clean_schema(v)
            for k, v in schema.items()
            if not (k == "title" and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [clean_schema(v) for v in schema]
    return schema


def truncate(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated {len(text) - limit} chars]"


class ToolError(Exception):
    """Raise from a tool to report a failure to the model without a type-name prefix."""


@dataclass
class ToolResult:
    content: str
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    model: type[BaseModel] | None = None  # None => raw JSON-schema tool (e.g. from MCP)
    read_only: bool = True
    requires_approval: bool = False

    def to_api(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    @property
    def parallel_safe(self) -> bool:
        return self.read_only and not self.requires_approval

    def run(self, raw: dict[str, Any] | None) -> ToolResult:
        call: Callable[[], Any]
        if self.model is None:
            # Raw JSON-schema tool: only check that required fields are present.
            payload = raw if isinstance(raw, dict) else {}
            missing = [k for k in self.input_schema.get("required", []) if k not in payload]
            if missing:
                fields = ", ".join(missing)
                return ToolResult(
                    f"Invalid arguments for '{self.name}': missing required field(s): {fields}",
                    True,
                )
            call = lambda: self.fn(payload)  # noqa: E731
        else:
            try:
                args = self.model.model_validate(raw or {})
            except ValidationError as exc:
                detail = "; ".join(
                    f"{'.'.join(str(p) for p in e['loc']) or 'input'}: {e['msg']}"
                    for e in exc.errors()
                )
                return ToolResult(f"Invalid arguments for '{self.name}': {detail}", True)
            kwargs = {k: getattr(args, k) for k in type(args).model_fields}
            call = lambda: self.fn(**kwargs)  # noqa: E731

        try:
            out = call()
        except ToolError as exc:
            return ToolResult(truncate(str(exc)), True)
        except Exception as exc:  # tool bugs must never kill the agent loop
            return ToolResult(f"{type(exc).__name__}: {exc}", True)
        if not isinstance(out, str):
            out = json.dumps(out, default=str, ensure_ascii=False)
        return ToolResult(truncate(out))


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    read_only: bool = True,
    requires_approval: bool = False,
) -> Any:
    """Decorator turning a typed function into a `Tool`.

    Use `Annotated[int, Field(description=...)]` to document parameters.
    """

    def build(f: Callable[..., Any]) -> Tool:
        desc = description or inspect.getdoc(f)
        if not desc:
            raise ValueError(f"Tool '{f.__name__}' needs a docstring/description for the model")
        hints = get_type_hints(f, include_extras=True)
        fields: dict[str, Any] = {}
        for pname, param in inspect.signature(f).parameters.items():
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[pname] = (hints.get(pname, str), default)
        model = create_model(f"{f.__name__}_args", **fields)
        schema = clean_schema(model.model_json_schema())
        return Tool(
            name=name or f.__name__,
            description=desc.strip(),
            input_schema=schema,
            fn=f,
            model=model,
            read_only=read_only,
            requires_approval=requires_approval,
        )

    return build(fn) if fn is not None else build


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_api() for t in self._tools.values()]

    def subset(self, names: Iterable[str]) -> ToolRegistry:
        """Least-privilege view: only the named tools (unknown names are ignored)."""
        return ToolRegistry(self._tools[n] for n in names if n in self._tools)
