"""Reliable structured output: forced tool call, with a JSON-in-text fallback for weak models.

Strong models honour a forced tool call. Smaller/local models (or servers that ignore
`tool_choice`) often reply with JSON in plain text instead, so we accept that too, validate it
against the pydantic model, and feed validation errors back for self-repair.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM, LLMError, Usage
from .tools import clean_schema

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(LLMError):
    pass


def _extract_json(text: str) -> Any | None:
    """Pull the first parseable JSON object out of free text or a ```json fence."""
    candidates = [m.group(1) for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        start, end = cand.find("{"), cand.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(cand[start : end + 1])
            except json.JSONDecodeError:
                continue
    return None


def _attempts(candidate: Any) -> list[Any]:
    """The payload itself, plus common wrappers weak models add: {"arguments": {...}}."""
    out = [candidate]
    if isinstance(candidate, dict):
        out += [
            candidate[k] for k in ("arguments", "parameters") if isinstance(candidate.get(k), dict)
        ]
    return out


def structured_output(
    llm: LLM,
    *,
    system: str,
    prompt: str,
    tool_name: str,
    tool_description: str,
    model_cls: type[T],
    max_tokens: int = 2048,
    retries: int = 2,
    on_usage: Callable[[Usage], None] | None = None,
) -> T:
    """Get a validated `model_cls` from the model, repairing bad output up to `retries` times."""
    schema = clean_schema(model_cls.model_json_schema())
    tool = {"name": tool_name, "description": tool_description, "input_schema": schema}
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    last_error = "unknown"

    for _ in range(retries + 1):
        resp = llm.generate(
            system=system,
            messages=messages,
            tools=[tool],
            max_tokens=max_tokens,
            tool_choice={"type": "tool", "name": tool_name},
        )
        if on_usage:
            on_usage(resp.usage)

        call = next((c for c in resp.tool_calls if c.name == tool_name), None)
        candidate = call.input if call is not None else _extract_json(resp.text)

        if candidate is not None:
            for attempt in _attempts(candidate):
                try:
                    return model_cls.model_validate(attempt)
                except ValidationError as exc:
                    last_error = str(exc)
        else:
            last_error = "model did not call the required tool"

        feedback: Any
        if call is not None:
            feedback = [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "is_error": True,
                    "content": f"Invalid output: {last_error}. Call {tool_name} again, corrected.",
                }
            ]
        else:
            feedback = (
                f"Invalid or missing output ({last_error[:300]}). Call {tool_name}, or reply with "
                f"ONLY a JSON object matching this JSON schema:\n{json.dumps(schema)}"
            )
        messages = [
            *messages,
            {"role": "assistant", "content": resp.content or [{"type": "text", "text": "(none)"}]},
            {"role": "user", "content": feedback},
        ]
    raise StructuredOutputError(f"{tool_name}: {last_error}")
