# AgentForge (Bahasa Indonesia)

Framework agent AI **siap produksi yang dibangun dari nol** (tanpa LangChain, ±1.000 baris).
Mencakup tool use, memori, guardrails, tracing, orkestrasi multi-agent, dan eval harness.

## Fitur utama
- **Agent loop**: tool use native, eksekusi paralel untuk tool read-only, batas langkah/token/waktu.
- **Tool bertipe**: `@tool` membuat JSON schema dari type hint dan memvalidasi argumen.
- **Multi-agent**: Planner → DAG worker paralel → synthesizer → critic (reflection).
- **Memori**: kompaksi konteks yang aman (tidak memisahkan `tool_use` dan `tool_result`) + SQLite.
- **Keamanan**: approval default-deny, deteksi loop, tool output tahan prompt injection,
  tool file ter-sandbox, kalkulator berbasis AST (bukan `eval`).
- **Observability**: trace JSONL untuk setiap panggilan LLM dan tool.
- **Evals**: evaluasi live berbasis trajectory, termasuk uji ketahanan prompt injection.

## Mulai cepat
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
agentforge run "Berapa (17*23)+sqrt(144)?"
agentforge run "Analisis sales.csv" --workspace ./workspace --orchestrate
make all    # lint + type-check + test (offline, tanpa API key)
```

Detail desain dan threat model: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
