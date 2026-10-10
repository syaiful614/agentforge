"""Reliable structured output via forced tool calls + validation + self-repair retries."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .llm import LLM, Usage
from .tools import clean_schema

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    pass


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
    """Force the model to call a single tool whose schema is `model_cls`, then validate.

    On invalid output the validation error is fed back so the model can correct itself.
    """
    tool = {
        "name": tool_name,
        "description": tool_description,
        "input_schema": clean_schema(model_cls.model_json_schema()),
    }
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
        if call is not None:
            try:
                return model_cls.model_validate(call.input)
            except ValidationError as exc:
                last_error = str(exc)
            feedback: Any = [
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "is_error": True,
                    "content": f"Invalid output: {last_error}. Call {tool_name} again, corrected.",
                }
            ]
        else:
            last_error = "model did not call the required tool"
            feedback = f"You must call {tool_name}. Try again."

        messages = [
            *messages,
            {"role": "assistant", "content": resp.content or [{"type": "text", "text": "(none)"}]},
            {"role": "user", "content": feedback},
        ]
    raise StructuredOutputError(f"{tool_name}: {last_error}")
