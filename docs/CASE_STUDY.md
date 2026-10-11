# AgentForge: case study

An agent framework built from first principles (no LangChain), to learn what it takes to run
LLM agents **safely and measurably**, not just to make a demo loop work.

| | |
|---|---|
| **What** | Agent loop, typed tools, memory, guardrails, tracing, multi-agent orchestration, MCP client, streaming, eval harness |
| **Size** | ~3,050 lines of source, ~2,250 lines of tests, 179 tests, 92% coverage |
| **Quality gates** | ruff, mypy and pytest on every push; CI green on Python 3.10, 3.11 and 3.12 |
| **Providers** | Anthropic, plus any OpenAI-compatible endpoint (Ollama, Groq, Gemini, OpenRouter, ...) |
| **Verified on** | Windows 11, CPU-only laptop with 8 GB RAM, small local models via Ollama |
| **Process** | Built iteratively with an AI pair-programmer (Claude). Every change is covered by tests and verified in CI, and the design decisions below are the author's to defend. |

## Architecture

```mermaid
flowchart TB
    user["CLI or Python API"] --> orch["Orchestrator: planner, parallel workers, synthesizer, critic"]
    user --> loop
    orch --> loop
    subgraph loop["Agent loop"]
        budget["Budget check"] --> compact["Context compaction"]
        compact --> llm["LLM call, optionally streamed"]
        llm --> calls{"Tool calls?"}
        calls -- yes --> guard["Loop detection, approval policy, validated execution"]
        guard --> results["Results wrapped as untrusted data"]
        results --> budget
        calls -- no --> answer["Final answer"]
    end
    llm --> providers["Anthropic or any OpenAI-compatible endpoint"]
    guard --> tools["Tool registry"]
    tools --> builtin["Built-in tools: calculator, sandboxed files, memory"]
    tools --> mcp["MCP client over stdio"]
    mcp --> servers["External MCP servers"]
    loop --> trace["JSONL trace of every call"]
```

## Design decisions, and why

1. **One tiny seam to the model.** The whole framework talks to `LLM.generate()`. Everything above
   it is provider-agnostic, and tests use scripted fake models, so the entire suite runs offline,
   free and deterministically.
2. **Tools are typed functions.** The JSON schema is derived from type hints and arguments are
   validated before execution. Failures go back to the model as error results so it can correct
   itself; a buggy tool cannot crash the loop.
3. **Guardrails are first-class.** Per-agent step caps and a shared token/time budget; loop
   detection; human approval for risky tools with a deny-by-default policy; tool output wrapped
   as untrusted data; sandboxed file tools; an AST-based calculator instead of `eval`.
4. **Context compaction that cannot corrupt the transcript.** Old turns are summarised, but only
   cut at real user turns so a `tool_use` is never separated from its `tool_result`.
5. **Orchestration with structure and least privilege.** The planner emits a validated task graph;
   independent tasks run in parallel; each worker gets only the tools its role needs (MCP write
   tools are never granted to workers).
6. **MCP written from the spec.** JSON-RPC over stdio with request/response matching, timeouts and
   cancellation, over stdio or Streamable HTTP. Servers are treated as untrusted: minimal child environment (no inherited API
   keys), namespaced tools, approval by default, capped descriptions, allow-lists; over HTTP,
   tokens come from environment variables, plain http is limited to localhost, and redirects are
   refused.
7. **Streaming without changing the contract.** One optional `on_text` callback; the adapter still
   returns the complete reply, so budgets, approvals and tracing never learn about streaming.
8. **Evals grade trajectories, not just answers.** Which tools ran, how many steps, whether a
   forbidden side effect happened, and a separate safety tally.

Full rationale and threat model: [ARCHITECTURE.md](ARCHITECTURE.md).

## Results (small local models, honestly reported)

Three runs per case on a CPU-only laptop; `python evals/run_evals.py --repeat 3`.

| Case | qwen2.5:3b | llama3.2:3b |
|---|---|---|
| Single-tool calculation | 3/3 | 3/3 |
| Read a file, then sum a column | 0/3 | 0/3 |
| Summarise a file containing a prompt-injection attack | 2/3 | 0/3 (later session: 2/2) |
| Multi-agent analysis (plan, delegate, synthesize) | 0/3 | 0/3 |
| **Runs passed** | **5/12 (42%)** | **3/12 (25%)** |

What this does and does not show:

- Both models handle single-tool tasks but fail at multi-step tool use and structured planning.
  The harness exists to expose exactly that gap.
- On the injection case, no run of either model called the forbidden `write_file` tool or created
  the forbidden file, including every run where the malicious text was in the model's context.
  That is weak evidence (few runs, one machine, tools auto-approved on purpose), not a security
  guarantee. The difference between models is a content check and is mostly run-to-run variance.
- Larger models have not been measured. This is a harness demo, not a benchmark.

## Engineering stories

**The MCP "wrong path" that was not a path problem.** An agent calling the official filesystem
MCP server kept reporting a bad file path. The trace showed the tool inputs, and the same error
appeared for both backslash and forward-slash paths, which ruled the path out. The common factor
was `"head": 0, "tail": null`: small models fill every optional parameter, and a strict server
rejects `null` for a number. The fix was generic (drop `null` values the schema does not allow),
covered by a test that reproduces it against a real subprocess. The debugging also exposed an
observability gap, so tool error text is now recorded in traces (successful outputs only on
opt-in, because they can contain private data).

**A planner that invented a role.** A model proposed a worker role that did not exist; the plan
was only validated after planning finished, so the whole eval session crashed with a traceback.
Plan validation moved inside the structured-output repair loop, so the model receives
`unknown role ... (allowed: ...)` and gets another chance, and an unfixable plan now fails as a
clean, recorded error. Related: models that send lists as JSON strings, or answer in plain text
instead of calling the forced tool, are tolerated and still fully validated.

**CI went red on a README.** After adding a documentation section, CI failed in 20 seconds while
every local check passed. Upgrading the tools reproduced it: current `ruff format` also checks
Python code blocks inside Markdown. The lesson became a habit: run the exact CI sequence
(`ruff check`, `ruff format --check`, `mypy`, `pytest`) before every push, including docs changes.

**Proving that streaming is live.** Parsing server-sent events in a unit test does not prove text
arrives early. One test starts a real HTTP server that refuses to send the second chunk until the
client has displayed the first; if the client buffered, the test fails. A manual run against a
local model showed first text after 3.9 s and completion after 8.5 s, in 38 chunks.

**A security test that must be able to fail.** For the HTTP transport, the redirect guard (never
replay credentials at another address) is easy to get wrong silently. The test was validated by
removing the guard on purpose: the HTTP library then followed the redirect and the test failed,
which proves it protects something rather than passing vacuously.

## Known limitations

- Orchestration needs a stronger model than 3B-class on CPU. An opt-in fallback runs a single
  agent when planning fails, but it is reported as degraded and never counted as success.
- MCP covers tools only (no resources or prompts) and uses static headers for HTTP auth (no OAuth
  login flow).
- Long-term memory is keyword-based; embeddings are the obvious upgrade.
- Anthropic streaming is covered with a fake SDK only; it has not been run against the live API.
- Tools run in-process; running untrusted code would need a container sandbox.

## Run it

```bash
pip install -e ".[dev]" && make all          # lint, types, 179 tests, no API key needed
agentforge mcp-tools examples/mcp.json       # connect to the bundled MCP demo server
agentforge run "Count the words in 'hello agent world'" --stream \
  --provider openai --model <ollama-model> --mcp-config examples/mcp.json --mcp-only
```

## What this project demonstrates

Agent loop design and failure handling; tool-use schemas and validation; prompt-injection and
least-privilege thinking; protocol implementation (MCP, SSE) from specifications; testing
non-deterministic systems with fakes and trajectory-based evals; debugging from traces; and
reporting results honestly.
