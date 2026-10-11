# AgentForge

[![CI](https://github.com/YOUR_USERNAME/agentforge/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/agentforge/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

**A production-minded agent framework built from first principles — no LangChain, ~1,000 lines.**
Tool use, memory, guardrails, tracing, multi-agent orchestration and an eval harness,
designed to show *how agents actually work* and *what it takes to run them safely*.

> Bahasa Indonesia: lihat [README.id.md](README.id.md)

## Why this exists

Most agent demos are a `while True` loop around an LLM. Production agents fail in the gaps:
runaway cost, infinite loops, prompt injection, corrupted context, untestable behaviour.
AgentForge treats those as first-class design problems.

| Capability | How |
|---|---|
| **Agent loop** | Native tool use, parallel read-only tool calls, step/token/time budgets |
| **Typed tools** | `@tool` derives JSON schema from type hints, validates args, returns errors to the model |
| **Multi-agent** | Planner → DAG of parallel workers → synthesizer → critic (reflection loop) |
| **Structured output** | Forced tool call + pydantic validation + self-repair retries |
| **Memory** | Context compaction (never splits tool pairs) + persistent SQLite long-term memory |
| **Safety** | Default-deny approvals, loop detection, injection-resistant tool output, sandboxed file tools, AST-safe calculator |
| **Observability** | JSONL traces for every LLM/tool call with timings and tokens |
| **Evals** | Trajectory-aware live evals incl. a prompt-injection case; offline test-suite with fake LLMs |

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design decisions and threat model.

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/agentforge && cd agentforge
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env && export ANTHROPIC_API_KEY=sk-ant-...

agentforge tools                                  # list built-in tools
agentforge run "What is (17*23)+sqrt(144)?"       # single agent
agentforge run "Analyse sales.csv: best/worst region and ratio" \
  --workspace ./workspace --orchestrate           # planner + parallel workers + critic
agentforge chat                                   # interactive, with memory
```

Risky tools (`write_file`) prompt for approval on the terminal; pass `-y` to auto-approve.

## Providers (including free ones)

AgentForge talks to models through a tiny `LLM` protocol with two adapters:
`anthropic` (default) and `openai` (any OpenAI-compatible `/chat/completions` endpoint).
The model must support **tool calling**.

```bash
# Free + local: Ollama (https://ollama.com). Pick a tool-calling model from its library.
ollama pull <tool-calling-model>
agentforge run "What is 17*23?" --provider openai --model <tool-calling-model>
# defaults to http://localhost:11434/v1

# Hosted free tiers (e.g. Groq, Gemini, OpenRouter): use their OpenAI-compatible URL + key
export AGENTFORGE_PROVIDER=openai
export AGENTFORGE_BASE_URL=<provider openai-compatible url>
export AGENTFORGE_API_KEY=<your key>
export AGENTFORGE_MODEL=<model name>
python evals/run_evals.py
```

Free-tier limits and model names change often; check your provider's docs. Smaller models
may fail some eval cases (that is exactly what the eval harness is for).

## Use as a library

```python
from agentforge import Agent, AnthropicLLM, Budget, ToolRegistry, Tracer, tool
from agentforge.builtin_tools import calculator


@tool
def get_exchange_rate(base: str, quote: str) -> dict:
    """Look up an exchange rate."""
    return {"base": base, "quote": quote, "rate": 16250.0}


agent = Agent(
    AnthropicLLM(),
    ToolRegistry([get_exchange_rate, calculator]),
    budget=Budget(max_steps=8, max_tokens=50_000),
    tracer=Tracer("traces/run.jsonl"),
)
result = agent.run("How many IDR is 2,500 USD?")
print(result.output, result.steps, result.usage.total, result.tool_calls)
```

Full example: [examples/custom_tool_agent.py](examples/custom_tool_agent.py).

### Multi-agent

```python
from agentforge import Orchestrator, default_registry

orch = Orchestrator(AnthropicLLM(), default_registry(Path("workspace")), max_review_rounds=1)
res = orch.run("Compare Q1 and Q2 revenue in the workspace files and explain the change")
print(res.output)
print(res.plan.model_dump())
```

## Testing & evals

```bash
make all      # ruff + mypy + pytest (offline, deterministic, no API key needed)
make eval     # live evals against the real model (needs ANTHROPIC_API_KEY)
python evals/run_evals.py --repeat 3   # pass-rate under stochasticity
```

The unit suite (~55 tests) uses scripted/fake LLMs, so the loop, guardrails, memory
compaction, DAG planning and review loop are verified without network access.

## Project layout

```
src/agentforge/
  agent.py          agent loop (budget, approval, loop-detect, tracing)
  orchestrator.py   planner / DAG workers / synthesizer / critic
  tools.py          @tool, schema generation, validation, registry
  builtin_tools.py  calculator, time, sandboxed files, memory tools
  memory.py         context compaction + SQLite long-term memory
  guardrails.py     budgets, approval policies, loop detector
  structured.py     reliable structured output
  tracing.py        JSONL tracer
  llm.py            provider protocol + Anthropic adapter
evals/              live eval harness, cases, fixtures
tests/              offline test-suite with fake LLMs
```

## Roadmap
Streaming · MCP client · embedding-based memory · USD cost budgets · container sandbox for code tools.

## License
MIT
