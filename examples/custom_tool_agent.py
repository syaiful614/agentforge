"""Build your own agent in ~25 lines: custom tool + guardrails + tracing.

export ANTHROPIC_API_KEY=...
python examples/custom_tool_agent.py
"""

from typing import Annotated

from pydantic import Field

from agentforge import Agent, AnthropicLLM, Budget, ToolRegistry, Tracer, tool
from agentforge.builtin_tools import calculator


@tool
def get_exchange_rate(
    base: Annotated[str, Field(description="ISO currency code, e.g. USD")],
    quote: Annotated[str, Field(description="ISO currency code, e.g. IDR")],
) -> dict:
    """Look up an exchange rate. (Stubbed here; call a real FX API in production.)"""
    rates = {("USD", "IDR"): 16250.0, ("EUR", "IDR"): 17600.0}
    if (base, quote) not in rates:
        raise ValueError(f"unsupported pair {base}/{quote}")
    return {"base": base, "quote": quote, "rate": rates[(base, quote)]}


if __name__ == "__main__":
    agent = Agent(
        AnthropicLLM(),
        ToolRegistry([get_exchange_rate, calculator]),
        system_prompt="You are a finance assistant. Always use tools for rates and arithmetic.",
        budget=Budget(max_steps=8, max_tokens=50_000),
        tracer=Tracer("traces/example.jsonl"),
    )
    result = agent.run("How many IDR is 2,500 USD plus 1,000 EUR?")
    print(result.output)
    print(f"steps={result.steps} tokens={result.usage.total} tools={result.tool_calls}")
