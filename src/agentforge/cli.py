"""Command line interface: `agentforge run | chat | tools`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent
from .builtin_tools import default_registry
from .config import Settings
from .guardrails import ApprovalPolicy, Budget, allow_all, cli_prompt
from .llm import LLM, LLMError, make_llm
from .memory import LongTermMemory
from .orchestrator import Orchestrator
from .tools import ToolRegistry
from .tracing import Tracer


def _build(
    args: argparse.Namespace,
) -> tuple[LLM, ToolRegistry, Tracer, ApprovalPolicy, Budget]:
    s = Settings.from_env()
    workspace = Path(args.workspace or s.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".agentforge").mkdir(exist_ok=True)
    ltm = LongTermMemory(str(workspace / ".agentforge" / "memory.db"))
    tools = default_registry(workspace, ltm)
    tracer = Tracer(Path(args.trace) if args.trace else s.trace_dir / "latest.jsonl")
    approval = allow_all if args.yes else cli_prompt
    budget = Budget(max_steps=args.max_steps, max_tokens=args.max_tokens)
    provider = args.provider or s.provider
    llm = make_llm(
        provider,
        args.model or s.model,
        api_key=s.key_for(provider),
        base_url=args.base_url or s.base_url,
    )
    return llm, tools, tracer, approval, budget


def _cmd_run(args: argparse.Namespace) -> int:
    llm, tools, tracer, approval, budget = _build(args)
    if args.orchestrate:
        orch = Orchestrator(
            llm,
            tools,
            budget=budget,
            approval=approval,
            tracer=tracer,
            on_event=lambda m: print(f"  · {m}", file=sys.stderr),
        )
        orch_res = orch.run(args.task)
        print(orch_res.output)
        print(f"\n[{orch_res.tokens} tokens | run {orch_res.run_id}]", file=sys.stderr)
        return 0 if orch_res.ok else 2

    agent = Agent(llm, tools, budget=budget, approval=approval, tracer=tracer)
    res = agent.run(args.task)
    if args.json:
        print(json.dumps({**res.__dict__, "usage": res.usage.total}, default=str, indent=2))
    else:
        print(res.output)
        print(
            f"\n[{res.stop_reason} | {res.steps} steps | {res.usage.total} tokens | "
            f"tools: {', '.join(res.tool_calls) or '-'}]",
            file=sys.stderr,
        )
    return 0 if res.ok else 2


def _cmd_chat(args: argparse.Namespace) -> int:
    llm, tools, tracer, approval, budget = _build(args)
    agent = Agent(llm, tools, budget=budget, approval=approval, tracer=tracer)
    print("AgentForge chat. Ctrl-D to exit, /reset to clear memory.", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except EOFError:
            return 0
        if not line:
            continue
        if line == "/reset":
            agent.reset()
            continue
        res = agent.run(line)
        print(f"agent> {res.output}\n")


def _cmd_tools(args: argparse.Namespace) -> int:
    tools = default_registry(Path(args.workspace or "."), LongTermMemory())
    for name in tools.names:
        t = tools.get(name)
        assert t is not None
        flags = "approval" if t.requires_approval else ("read-only" if t.read_only else "write")
        print(f"{name:<14} [{flags}] {t.description.splitlines()[0]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model")
    common.add_argument("--provider", choices=["anthropic", "openai"], help="LLM backend")
    common.add_argument(
        "--base-url", help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1"
    )
    common.add_argument("--workspace", help="Directory the file tools are confined to")
    common.add_argument("--trace", help="JSONL trace output path")
    common.add_argument("--max-steps", type=int, default=15)
    common.add_argument("--max-tokens", type=int, default=200_000, help="Token budget")
    common.add_argument("-y", "--yes", action="store_true", help="Auto-approve risky tools")

    p = argparse.ArgumentParser(prog="agentforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="Run a single task")
    r.add_argument("task")
    r.add_argument("--orchestrate", action="store_true", help="Use planner/worker/critic mode")
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_cmd_run)

    c = sub.add_parser("chat", parents=[common], help="Interactive session")
    c.set_defaults(fn=_cmd_chat)

    t = sub.add_parser("tools", help="List built-in tools")
    t.add_argument("--workspace")
    t.set_defaults(fn=_cmd_tools)

    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
