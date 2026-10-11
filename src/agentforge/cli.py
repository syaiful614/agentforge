"""Command line interface: `agentforge run | chat | tools`."""

from __future__ import annotations

import argparse
import atexit
import json
import sys
from pathlib import Path
from typing import Any, NamedTuple

from .agent import Agent
from .builtin_tools import default_registry
from .config import Settings
from .guardrails import ApprovalPolicy, Budget, allow_all, cli_prompt
from .llm import LLM, LLMError, make_llm
from .mcp_client import MCPError, MCPToolbox, load_mcp_config
from .memory import LongTermMemory
from .orchestrator import Orchestrator, default_workers, with_extra_tools
from .tools import ToolRegistry
from .tracing import Tracer


class Runtime(NamedTuple):
    llm: LLM
    tools: ToolRegistry
    tracer: Tracer
    approval: ApprovalPolicy
    budget: Budget
    mcp_read_only: list[str]  # MCP tools safe to hand to orchestrator workers


def _build(args: argparse.Namespace) -> Runtime:
    mcp_config = getattr(args, "mcp_config", None)
    mcp_only = getattr(args, "mcp_only", False)
    if mcp_only and not mcp_config:
        raise MCPError("--mcp-only requires --mcp-config")

    s = Settings.from_env()
    workspace = Path(args.workspace or s.workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / ".agentforge").mkdir(exist_ok=True)
    ltm = LongTermMemory(str(workspace / ".agentforge" / "memory.db"))
    tools = default_registry(workspace, ltm)
    mcp_read_only: list[str] = []
    if mcp_config:
        toolbox = MCPToolbox(load_mcp_config(mcp_config)).start()
        atexit.register(toolbox.close)  # never leave server processes behind
        mcp_read_only = [t.name for t in toolbox.tools if t.parallel_safe]
        if mcp_only:
            tools = ToolRegistry(toolbox.tools)  # hide built-ins: fewer choices for small models
        else:
            toolbox.register_into(tools)
    tracer = Tracer(Path(args.trace) if args.trace else s.trace_dir / "latest.jsonl")
    approval = allow_all if args.yes else cli_prompt
    budget = Budget(max_steps=args.max_steps, max_tokens=args.max_tokens)
    provider = args.provider or s.provider
    llm = make_llm(
        provider,
        args.model or s.model,
        api_key=s.key_for(provider),
        base_url=args.base_url or s.base_url,
        timeout=s.timeout_s,
    )
    return Runtime(llm, tools, tracer, approval, budget, mcp_read_only)


class CLIError(Exception):
    """Invalid combination of command-line options."""


class _StreamPrinter:
    """Writes streamed text to stdout immediately and remembers whether anything was shown."""

    def __init__(self) -> None:
        self.printed = False

    def __call__(self, delta: str) -> None:
        self.printed = True
        sys.stdout.write(delta)
        sys.stdout.flush()


def _announce_tool(name: str, tool_input: dict[str, Any]) -> None:
    preview = json.dumps(tool_input, ensure_ascii=False)[:80]
    print(f"\n  · {name}({preview})", file=sys.stderr, flush=True)


def _cmd_run(args: argparse.Namespace) -> int:
    stream = bool(getattr(args, "stream", False))
    if stream and args.orchestrate:
        raise CLIError("--stream is not supported with --orchestrate (workers run in parallel)")
    if stream and args.json:
        raise CLIError("--stream cannot be combined with --json")

    rt = _build(args)
    llm, tools, tracer, approval, budget = rt[:5]
    if args.orchestrate:
        orch = Orchestrator(
            llm,
            tools,
            workers=with_extra_tools(default_workers(), rt.mcp_read_only),
            budget=budget,
            approval=approval,
            tracer=tracer,
            fallback=args.fallback,
            on_event=lambda m: print(f"  · {m}", file=sys.stderr),
        )
        orch_res = orch.run(args.task)
        print(orch_res.output)
        if orch_res.degraded:
            print(
                f"\nwarning: planner failed ({orch_res.degraded_reason}); "
                "answered in single-agent mode (degraded)",
                file=sys.stderr,
            )
        print(f"\n[{orch_res.tokens} tokens | run {orch_res.run_id}]", file=sys.stderr)
        return 0 if orch_res.ok else 2

    printer = _StreamPrinter()
    agent = Agent(
        llm,
        tools,
        budget=budget,
        approval=approval,
        tracer=tracer,
        on_text=printer if stream else None,
        on_tool=_announce_tool if stream else None,
    )
    res = agent.run(args.task)
    if args.json:
        print(json.dumps({**res.__dict__, "usage": res.usage.total}, default=str, indent=2))
    else:
        if stream:
            print()  # finish the streamed line
            if not printer.printed:  # e.g. stopped by a budget before any text arrived
                print(res.output)
        else:
            print(res.output)
        print(
            f"\n[{res.stop_reason} | {res.steps} steps | {res.usage.total} tokens | "
            f"tools: {', '.join(res.tool_calls) or '-'}]",
            file=sys.stderr,
        )
    return 0 if res.ok else 2


def _cmd_chat(args: argparse.Namespace) -> int:
    llm, tools, tracer, approval, budget = _build(args)[:5]
    stream = bool(getattr(args, "stream", False))
    printer = _StreamPrinter()
    agent = Agent(
        llm,
        tools,
        budget=budget,
        approval=approval,
        tracer=tracer,
        on_text=printer if stream else None,
        on_tool=_announce_tool if stream else None,
    )
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
        if stream:
            printer.printed = False
            print("agent> ", end="", flush=True)
            res = agent.run(line)
            if not printer.printed:
                print(res.output, end="")
            print("\n")
        else:
            res = agent.run(line)
            print(f"agent> {res.output}\n")


def _cmd_mcp_tools(args: argparse.Namespace) -> int:
    with MCPToolbox(load_mcp_config(args.config)) as box:
        for client in box.clients:
            info = client.server_info
            print(f"# {client.name}: {info.get('name', '?')} (protocol {client.protocol_version})")
        for t in box.tools:
            flags = "read-only" if t.parallel_safe else "approval"
            print(f"{t.name:<28} [{flags}] {t.description.splitlines()[0][:70]}")
    return 0


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
    common.add_argument(
        "--mcp-config", help="JSON file listing MCP servers (see examples/mcp.json)"
    )
    common.add_argument(
        "--mcp-only",
        action="store_true",
        help="Expose only MCP tools to the agent (hide the built-in tools)",
    )
    common.add_argument("--workspace", help="Directory the file tools are confined to")
    common.add_argument("--trace", help="JSONL trace output path")
    common.add_argument("--max-steps", type=int, default=15)
    common.add_argument("--max-tokens", type=int, default=200_000, help="Token budget")
    common.add_argument("--stream", action="store_true", help="Print the answer as it is generated")
    common.add_argument("-y", "--yes", action="store_true", help="Auto-approve risky tools")

    p = argparse.ArgumentParser(prog="agentforge", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", parents=[common], help="Run a single task")
    r.add_argument("task")
    r.add_argument("--orchestrate", action="store_true", help="Use planner/worker/critic mode")
    r.add_argument(
        "--fallback",
        action="store_true",
        help="With --orchestrate: if planning fails, use one agent (reported as degraded)",
    )
    r.add_argument("--json", action="store_true")
    r.set_defaults(fn=_cmd_run)

    c = sub.add_parser("chat", parents=[common], help="Interactive session")
    c.set_defaults(fn=_cmd_chat)

    m = sub.add_parser("mcp-tools", help="Connect to MCP servers and list their tools")
    m.add_argument("config", help="MCP config JSON file")
    m.set_defaults(fn=_cmd_mcp_tools)

    t = sub.add_parser("tools", help="List built-in tools")
    t.add_argument("--workspace")
    t.set_defaults(fn=_cmd_tools)

    args = p.parse_args(argv)
    try:
        return int(args.fn(args))
    except (LLMError, MCPError, CLIError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
