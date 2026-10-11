"""Provider abstraction.

The whole framework talks to a tiny `LLM` protocol. Messages use the Anthropic
content-block format as the canonical internal representation, which keeps the
agent loop free of provider-specific glue and makes it trivial to fake in tests.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(RuntimeError):
    """A provider failure with a human-readable message (auth, billing, network...)."""


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
        self._api_error = getattr(anthropic, "APIError", None)
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

        try:
            resp = self._client.messages.create(**kwargs)
        except Exception as exc:
            if self._api_error is not None and isinstance(exc, self._api_error):
                raise LLMError(_friendly_anthropic_error(exc)) from exc
            raise

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


def _friendly_anthropic_error(exc: Exception) -> str:
    msg = str(exc)
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "Anthropic API key is invalid. Check ANTHROPIC_API_KEY."
    if "credit balance" in msg.lower():
        return "Anthropic account has no credit. Add credit under Plans & Billing in the Console."
    return f"Anthropic API error ({status}): {msg}"


# --------------------------------------------------------------------------- OpenAI-compatible
def _to_openai_messages(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}]
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            out.append({"role": m["role"], "content": content})
            continue
        if m["role"] == "assistant":
            text = "".join(b["text"] for b in content if b.get("type") == "text")
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {"name": b["name"], "arguments": json.dumps(b["input"])},
                }
                for b in content
                if b.get("type") == "tool_use"
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        else:
            for b in content:
                if b.get("type") == "tool_result":
                    body = (
                        b["content"] if isinstance(b["content"], str) else json.dumps(b["content"])
                    )
                    out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
                elif b.get("type") == "text":
                    out.append({"role": "user", "content": b["text"]})
    return out


class OpenAICompatLLM:
    """Adapter for any OpenAI-compatible /chat/completions endpoint.

    Works with Ollama (http://localhost:11434/v1), Groq, Gemini's OpenAI endpoint,
    OpenRouter, vLLM, LM Studio... Uses only the standard library (no extra dependency).
    The model must support tool calling for agent features to work.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434/v1",
        api_key: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url, data=json.dumps(payload).encode(), headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")[:500]
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                hint = " (invalid API key?)" if exc.code in (401, 403) else ""
                raise LLMError(f"{self.base_url} returned HTTP {exc.code}{hint}: {body}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise LLMError(f"cannot reach {url}: {exc.reason}. Is the server running?") from exc
        raise LLMError("unreachable")  # pragma: no cover

    def generate(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 2048,
        tool_choice: dict[str, Any] | None = None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _to_openai_messages(system, messages),
            "max_tokens": max_tokens,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t["description"],
                        "parameters": t["input_schema"],
                    },
                }
                for t in tools
            ]
            if tool_choice and tool_choice.get("type") == "tool":
                payload["tool_choice"] = {
                    "type": "function",
                    "function": {"name": tool_choice["name"]},
                }

        data = self._post(payload)
        choice = data["choices"][0]
        message = choice["message"]

        content: list[dict[str, Any]] = []
        if message.get("content"):
            content.append({"type": "text", "text": message["content"]})
        for tc in message.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except json.JSONDecodeError:
                args = {}
            content.append(
                {"type": "tool_use", "id": tc["id"], "name": tc["function"]["name"], "input": args}
            )

        finish = choice.get("finish_reason")
        stop = {"tool_calls": "tool_use", "length": "max_tokens"}.get(finish or "", "end_turn")
        u = data.get("usage") or {}
        return LLMResponse(
            content, stop, Usage(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))
        )


def make_llm(
    provider: str,
    model: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
) -> LLM:
    """Factory: provider is 'anthropic' or 'openai' (any OpenAI-compatible endpoint)."""
    if provider == "anthropic":
        return AnthropicLLM(model, api_key=api_key)
    if provider == "openai":
        if model.startswith("claude"):
            raise LLMError("Set --model (or AGENTFORGE_MODEL) to a model your provider serves.")
        return OpenAICompatLLM(model, base_url or "http://localhost:11434/v1", api_key)
    raise LLMError(f"unknown provider '{provider}' (use 'anthropic' or 'openai')")
