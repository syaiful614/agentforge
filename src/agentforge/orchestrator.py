"""Multi-agent orchestration: plan -> parallel workers (DAG) -> synthesize -> critique.

1. Planner  : one forced-tool-call produces a validated task DAG (Plan).
2. Workers  : each subtask runs in a fresh, least-privilege Agent. Independent
              subtasks within a DAG layer run in parallel threads.
3. Synthesizer: merges worker outputs into one answer.
4. Critic   : optional reflection loop; if the answer is rejected it is re-synthesized
              with the critic's issues (workers are not re-run, keeping cost bounded).
All agents share one Budget, so token/time limits are global.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from .agent import Agent, AgentResult
from .guardrails import ApprovalPolicy, Budget, BudgetExceeded, read_only_only
from .llm import LLM, Usage
from .structured import structured_output
from .tools import ToolRegistry
from .tracing import Tracer


class Subtask(BaseModel):
    id: str = Field(description="Short unique id, e.g. 't1'")
    goal: str = Field(description="A self-contained instruction for the worker")
    role: str = Field(description="Which worker role should execute it")
    depends_on: list[str] = Field(default_factory=list, description="Ids that must finish first")


class Plan(BaseModel):
    subtasks: list[Subtask] = Field(min_length=1, max_length=8)


class Review(BaseModel):
    approved: bool = Field(description="True if the answer fully and correctly solves the task")
    issues: list[str] = Field(default_factory=list, description="Concrete problems to fix")


class PlanError(ValueError):
    pass


def validate_plan(plan: Plan, roles: set[str]) -> list[list[Subtask]]:
    """Validate the DAG and return execution layers (Kahn's algorithm)."""
    by_id = {s.id: s for s in plan.subtasks}
    if len(by_id) != len(plan.subtasks):
        raise PlanError("duplicate subtask ids")
    for s in plan.subtasks:
        if s.role not in roles:
            raise PlanError(f"unknown role '{s.role}' in {s.id} (allowed: {sorted(roles)})")
        for d in s.depends_on:
            if d not in by_id:
                raise PlanError(f"{s.id} depends on unknown task '{d}'")
            if d == s.id:
                raise PlanError(f"{s.id} depends on itself")

    remaining = dict(by_id)
    done: set[str] = set()
    layers: list[list[Subtask]] = []
    while remaining:
        ready = [s for s in remaining.values() if set(s.depends_on) <= done]
        if not ready:
            raise PlanError("dependency cycle detected")
        layers.append(ready)
        done |= {s.id for s in ready}
        for s in ready:
            del remaining[s.id]
    return layers


@dataclass
class WorkerSpec:
    system_prompt: str
    tools: list[str] = field(default_factory=list)


def default_workers() -> dict[str, WorkerSpec]:
    return {
        "researcher": WorkerSpec(
            "You are a meticulous researcher. Gather facts using the available tools and report "
            "them precisely, citing file names or sources. Do not speculate.",
            ["list_dir", "read_file", "grep", "recall"],
        ),
        "analyst": WorkerSpec(
            "You are a quantitative analyst. Use the calculator for every computation and show "
            "the key numbers behind your conclusions.",
            ["calculator", "read_file", "list_dir", "current_time"],
        ),
        "writer": WorkerSpec(
            "You are a clear technical writer. Turn the provided material into well-structured "
            "prose. Do not invent facts that are not in the provided material.",
            [],
        ),
    }


def with_extra_tools(
    workers: dict[str, WorkerSpec],
    names: list[str],
    roles: tuple[str, ...] = ("researcher", "analyst"),
) -> dict[str, WorkerSpec]:
    """Grant extra (e.g. read-only MCP) tools to selected worker roles, leaving others untouched.

    Worker tool lists are fixed names, so tools discovered at runtime would otherwise never
    reach them. The writer role stays tool-free by default (least privilege).
    """
    out: dict[str, WorkerSpec] = {}
    for role, spec in workers.items():
        extra = [n for n in names if n not in spec.tools] if role in roles else []
        out[role] = WorkerSpec(spec.system_prompt, [*spec.tools, *extra])
    return out


PLANNER_SYSTEM = (
    "You are a planning agent. Decompose the user's task into the FEWEST subtasks that "
    "solve it (often 1-4). Make subtasks independent where possible so they can run in "
    "parallel; use depends_on only when a task truly needs another's output. Each goal must be "
    "self-contained. Available worker roles:\n{roles}"
)
SYNTH_SYSTEM = (
    "You are the SYNTHESIZER. Combine the workers' results into one accurate, well-organised "
    "final answer to the original task. Resolve conflicts explicitly; never invent facts. "
    "If a worker FAILED, say what is missing."
)
CRITIC_SYSTEM = (
    "You are a strict reviewer. Judge whether the answer completely and correctly addresses "
    "the task, using only the worker evidence provided. Approve if it does."
)


@dataclass
class OrchestratorResult:
    output: str
    plan: Plan
    subtask_results: dict[str, AgentResult]
    review_rounds: int
    tokens: int
    run_id: str

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.subtask_results.values())


class Orchestrator:
    def __init__(
        self,
        llm: LLM,
        tools: ToolRegistry,
        *,
        workers: dict[str, WorkerSpec] | None = None,
        budget: Budget | None = None,
        approval: ApprovalPolicy = read_only_only,
        tracer: Tracer | None = None,
        max_review_rounds: int = 1,
        max_parallel_workers: int = 4,
        on_event: Callable[[str], None] | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.workers = workers or default_workers()
        self.budget = budget or Budget(max_steps=10, max_tokens=400_000, timeout_s=600)
        self.approval = approval
        self.tracer = (tracer or Tracer()).bind(agent="orchestrator")
        self._root_tracer = tracer or self.tracer
        self.max_review_rounds = max_review_rounds
        self.max_parallel_workers = max_parallel_workers
        self.on_event = on_event or (lambda _msg: None)

    # ------------------------------------------------------------------ pipeline
    def run(self, task: str) -> OrchestratorResult:
        plan = self._plan(task)
        layers = validate_plan(plan, set(self.workers))
        self.tracer.event("plan", plan=plan.model_dump())
        self.on_event(f"plan: {len(plan.subtasks)} subtasks in {len(layers)} layer(s)")

        results: dict[str, AgentResult] = {}
        for layer in layers:
            with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as pool:
                outs = list(pool.map(lambda s: self._run_subtask(s, plan, results), layer))
            for sub, res in zip(layer, outs, strict=True):
                results[sub.id] = res
                self.on_event(f"{sub.id} [{sub.role}] -> {res.stop_reason}")

        answer, rounds = self._synthesize_with_review(task, plan, results)
        return OrchestratorResult(
            answer, plan, results, rounds, self.budget.used_tokens, self.tracer.run_id
        )

    # ------------------------------------------------------------------ stages
    def _record(self, usage: Usage) -> None:
        self.budget.record(usage)

    def _plan(self, task: str) -> Plan:
        roles = "\n".join(
            f"- {n}: {s.system_prompt.split('.')[0]}." for n, s in self.workers.items()
        )
        plan = structured_output(
            self.llm,
            system=PLANNER_SYSTEM.format(roles=roles),
            prompt=task,
            tool_name="submit_plan",
            tool_description="Submit the task decomposition.",
            model_cls=Plan,
            on_usage=self._record,
        )
        return plan

    def _run_subtask(
        self, sub: Subtask, plan: Plan, results: dict[str, AgentResult]
    ) -> AgentResult:
        spec = self.workers[sub.role]
        context = ""
        for dep in sub.depends_on:
            dep_res = results[dep]
            status = "" if dep_res.ok else f" (INCOMPLETE: {dep_res.stop_reason})"
            context += f"\n### Result of {dep}{status}\n{dep_res.output}\n"
        prompt = sub.goal + (f"\n\n## Context from earlier tasks{context}" if context else "")

        agent = Agent(
            self.llm,
            self.tools.subset(spec.tools),
            name=f"{sub.role}:{sub.id}",
            system_prompt=spec.system_prompt,
            budget=self.budget,
            approval=self.approval,
            tracer=self._root_tracer,
        )
        try:
            return agent.run(prompt)
        except Exception as exc:  # one failing worker must not sink the whole run
            self.tracer.event("worker_crashed", task_id=sub.id, error=repr(exc))
            return AgentResult(f"FAILED: {exc!r}", "error", 0)

    def _evidence(self, plan: Plan, results: dict[str, AgentResult]) -> str:
        parts = []
        for s in plan.subtasks:
            r = results[s.id]
            tag = "OK" if r.ok else f"FAILED ({r.stop_reason})"
            parts.append(f"## {s.id} [{s.role}] {tag}\nGoal: {s.goal}\nResult:\n{r.output}")
        return "\n\n".join(parts)

    def _synthesize_with_review(
        self, task: str, plan: Plan, results: dict[str, AgentResult]
    ) -> tuple[str, int]:
        evidence = self._evidence(plan, results)
        prompt = f"# Original task\n{task}\n\n# Worker results\n{evidence}"
        feedback = ""
        answer, rounds = "", 0
        try:
            for rounds in range(self.max_review_rounds + 1):
                self.budget.check()
                resp = self.llm.generate(
                    system=SYNTH_SYSTEM,
                    messages=[{"role": "user", "content": prompt + feedback}],
                    max_tokens=2048,
                )
                self._record(resp.usage)
                answer = resp.text
                if rounds >= self.max_review_rounds:
                    break
                review = structured_output(
                    self.llm,
                    system=CRITIC_SYSTEM,
                    prompt=f"{prompt}\n\n# Answer to review\n{answer}",
                    tool_name="submit_review",
                    tool_description="Submit the review verdict.",
                    model_cls=Review,
                    on_usage=self._record,
                )
                self.tracer.event("review", round=rounds, **review.model_dump())
                if review.approved:
                    break
                feedback = "\n\n# Reviewer issues to fix\n" + "\n".join(
                    f"- {i}" for i in review.issues
                )
        except BudgetExceeded as exc:
            self.tracer.event("stopped", reason=str(exc))
            answer = answer or f"[budget exhausted before synthesis]\n\n{evidence}"
        return answer, rounds
