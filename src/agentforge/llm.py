"""Provider abstraction.

The whole framework talks to a tiny `LLM` protocol. Messages use the Anthropic
content-block format as the canonical internal representation, which keeps the
agent loop free of provider-specific glue and makes it trivial to fake in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens
        )


@dataclass
class LLMResponse:
    content: list[dict[str, Any]]
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(b["text"] for b in self.content if b.get("type") == "text")

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(b["id"], b["name"], dict(b.get("input") or {}))
            for b in self.content
            if b.get("type") == "tool_use"
        ]


class LLM(Protocol):
    model: str

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        tool_choice: dict[str, Any] | None = None,
    ) -> LLMResponse: ...


class AnthropicLLM:
    """Anthropic Messages API adapter.

    - Retries/backoff on 429/5xx are delegated to the SDK (`max_retries`).
    - The system prompt is marked cacheable so long, stable prompts are cheap
      across the many calls an agent loop makes.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        max_retries: int = 4,
        cache_system: bool = True,
    ) -> None:
        import anthropic  # imported lazily so tests/offline use need no key

        self.model = model
        self._cache_system = cache_system
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        tool_choice: dict[str, Any] | None = None,
    ) -> LLMResponse:
        system_param: Any = system
        if self._cache_system:
            system_param = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "system": system_param,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        resp = self._client.messages.create(**kwargs)

        content: list[dict[str, Any]] = []
        for block in resp.content:
            if block.type == "text":
                content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                content.append(
                    {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                )
        usage = Usage(resp.usage.input_tokens, resp.usage.output_tokens)
        return LLMResponse(content, resp.stop_reason or "end_turn", usage)
