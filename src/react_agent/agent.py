"""The ReAct loop as a LangGraph StateGraph: think -> act -> observe -> reflect, bounded.

Graph topology (every cycle passes through `gate`, the single place limits are enforced)::

    START -> gate -> think -> act -> reflect -> gate ...
               |       |       '-> gate (tool error streak ok)
               |       |-> critique -> finish -> END     (answer accepted)
               |       |           '-> gate              (answer rejected: revise)
               |       '-> gate                          (unparseable reply: retry)
               '-> degrade -> END   (cap / deadline / loop / errors / outage)
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from react_agent.llm import LLMClient, LLMError
from react_agent.models import (
    AgentConfig,
    Critique,
    Reflection,
    RunResult,
    Step,
    StopReason,
    ToolCall,
)
from react_agent.parsing import ParseError, parse_model
from react_agent.tools import ToolError, ToolRegistry, default_registry

SYSTEM = """You are a careful planning agent that solves tasks with tools.
Each turn, reply with ONE JSON object and nothing else:
  {{"thought": "<reasoning>", "action": {{"tool": "<name>", "input": {{...}}}}}}
or, when you have enough evidence:
  {{"thought": "<reasoning>", "final_answer": "<answer>"}}
Do not repeat a tool call that already failed or already gave its result; change approach.
Tools:
{tools}"""

REFLECT_SYSTEM = (
    'Review the latest step. Reply with JSON only: {"on_track": bool, "assessment": str, '
    '"next_hint": str}. Set on_track=false if you are repeating yourself or making no progress.'
)
CRITIQUE_SYSTEM = (
    "You are a strict reviewer. Check the proposed answer against the task and the evidence in "
    'the scratchpad. Reply with JSON only: {"acceptable": bool, "issues": str, '
    '"confidence": number between 0 and 1}.'
)
DEGRADE_SYSTEM = (
    "The agent had to stop early. Using only the evidence in the scratchpad, give the most "
    "useful partial answer you can. State plainly what is established and what is missing. "
    "Plain text, no JSON."
)

# Graph steps one cycle can consume (gate, think, act, reflect/critique) plus slack. The
# recursion limit is a backstop beneath max_iterations, not the primary bound.
STEPS_PER_CYCLE = 4


class Decision(BaseModel):
    thought: str = ""
    action: ToolCall | None = None
    final_answer: str | None = None


class AgentState(TypedDict, total=False):
    task: str
    started: float
    iteration: int
    steps: list[Step]
    seen: dict[str, int]  # tool-call signature -> times issued
    tool_errors: int  # consecutive
    parse_failures: int  # consecutive
    revisions: int
    decision: Decision | None  # latest parsed model decision
    draft: str | None  # latest proposed final answer
    critique: Critique | None
    warnings: list[str]
    stop_reason: StopReason | None
    answer: str


class ReActAgent:
    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.llm = llm
        self.tools = tools or default_registry()
        self.config = config or AgentConfig()
        self.clock = clock
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # ------------------------------------------------------------------ public

    def run(self, task: str, thread_id: str | None = None) -> RunResult:
        cfg = self.config
        thread_id = thread_id or uuid.uuid4().hex
        run_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": cfg.recursion_limit
            or cfg.max_iterations * STEPS_PER_CYCLE + 10,
        }
        initial: AgentState = {
            "task": task, "started": self.clock(), "iteration": 0, "steps": [], "seen": {},
            "tool_errors": 0, "parse_failures": 0, "revisions": 0, "decision": None,
            "draft": None, "critique": None, "warnings": [], "stop_reason": None, "answer": "",
        }
        try:
            state = self.graph.invoke(initial, run_config)
        except GraphRecursionError:
            # Backstop tripped: recover the last checkpointed state and degrade from it.
            state = dict(self.graph.get_state(run_config).values)
            state["warnings"] = [*state.get("warnings", []), "graph recursion limit reached"]
            state.update(self._degrade({**state, "stop_reason": StopReason.MAX_ITERATIONS}))
        return RunResult(
            task=task, answer=state["answer"], stop_reason=state["stop_reason"],
            degraded=state["stop_reason"] is not StopReason.ANSWERED,
            iterations=state["iteration"] if state["stop_reason"] is StopReason.ANSWERED
            else len(state["steps"]),
            steps=state["steps"], critique=state.get("critique"), revisions=state["revisions"],
            elapsed_seconds=self.clock() - state["started"], warnings=state["warnings"],
            thread_id=thread_id,
        )

    def checkpoints(self, thread_id: str) -> int:
        """Number of state snapshots persisted for a run (one per graph super-step)."""
        cfg = {"configurable": {"thread_id": thread_id}}
        return sum(1 for _ in self.graph.get_state_history(cfg))

    # ------------------------------------------------------------------- graph

    def _build_graph(self) -> Any:
        g = StateGraph(AgentState)
        for name, fn in [("gate", self._gate), ("think", self._think), ("act", self._act),
                         ("reflect", self._reflect), ("critique", self._critique),
                         ("finish", self._finish), ("degrade", self._degrade)]:
            g.add_node(name, fn)
        g.add_edge(START, "gate")
        g.add_conditional_edges("gate", self._after_gate, ["think", "degrade"])
        g.add_conditional_edges("think", self._after_think,
                                ["gate", "act", "critique", "degrade"])
        g.add_conditional_edges("act", self._after_act, ["reflect", "gate", "degrade"])
        g.add_edge("reflect", "gate")
        g.add_conditional_edges("critique", self._after_critique, ["finish", "gate"])
        g.add_edge("finish", END)
        g.add_edge("degrade", END)
        return g.compile(checkpointer=self.checkpointer)

    # ------------------------------------------------------------------- nodes

    def _gate(self, state: AgentState) -> dict:
        """The single enforcement point for the iteration cap and the wall-clock deadline."""
        cfg = self.config
        if state["iteration"] >= cfg.max_iterations:
            return {"stop_reason": StopReason.MAX_ITERATIONS}
        if cfg.deadline_seconds is not None and (
                self.clock() - state["started"] >= cfg.deadline_seconds):
            return {"stop_reason": StopReason.TIMEOUT}
        return {}

    @staticmethod
    def _after_gate(state: AgentState) -> str:
        return "degrade" if state.get("stop_reason") else "think"

    def _think(self, state: AgentState) -> dict:
        cfg = self.config
        i = state["iteration"] + 1
        try:
            raw = self.llm.complete(
                purpose="think", system=SYSTEM.format(tools=self.tools.describe()),
                prompt=self._scratchpad(state["task"], state["steps"]),
                max_tokens=cfg.max_tokens)
        except LLMError as exc:
            return {"iteration": i, "stop_reason": StopReason.LLM_ERROR,
                    "warnings": [*state["warnings"], f"think failed: {exc}"]}
        try:
            decision = parse_model(raw, Decision)
            if decision.action is None and decision.final_answer is None:
                raise ParseError("need either 'action' or 'final_answer'")
        except ParseError as exc:
            failures = state["parse_failures"] + 1
            step = Step(index=i, observation=f"Unparseable reply ({exc}). "
                        "Reply with a single JSON object.", error=True)
            update: dict = {"iteration": i, "parse_failures": failures, "decision": None,
                            "steps": [*state["steps"], step]}
            if failures >= cfg.max_consecutive_parse_failures:
                update["stop_reason"] = StopReason.PARSE_FAILURES
            return update
        return {"iteration": i, "parse_failures": 0, "decision": decision}

    @staticmethod
    def _after_think(state: AgentState) -> str:
        if state.get("stop_reason"):
            return "degrade"
        decision = state.get("decision")
        if decision is None:
            return "gate"  # unparseable: retry, still bounded by the gate
        return "critique" if decision.final_answer is not None else "act"

    def _act(self, state: AgentState) -> dict:
        """Run the chosen tool. Errors are observations, not exceptions."""
        cfg = self.config
        decision = state["decision"]
        assert decision is not None and decision.action is not None
        call = decision.action
        sig = call.signature()
        seen = dict(state["seen"])
        seen[sig] = seen.get(sig, 0) + 1
        step = Step(index=state["iteration"], thought=decision.thought, action=call)
        update: dict = {"seen": seen, "steps": [*state["steps"], step]}

        if seen[sig] >= cfg.repeat_limit:
            step.observation = f"Blocked: {call.tool} called {seen[sig]} times with same input."
            step.error = True
            update["stop_reason"] = StopReason.LOOP_DETECTED
            return update
        try:
            step.observation = self.tools.run(call.tool, call.input)
            update["tool_errors"] = 0
        except ToolError as exc:
            step.observation, step.error = f"Error: {exc}", True
            update["tool_errors"] = state["tool_errors"] + 1
            if update["tool_errors"] >= cfg.max_consecutive_tool_errors:
                update["stop_reason"] = StopReason.TOOL_FAILURES
        if seen[sig] == cfg.repeat_limit - 1:
            step.observation += " [warning: you already made this exact call; try something else]"
        return update

    def _after_act(self, state: AgentState) -> str:
        if state.get("stop_reason"):
            return "degrade"
        return "reflect" if self.config.reflect else "gate"

    def _reflect(self, state: AgentState) -> dict:
        step = state["steps"][-1]
        try:
            raw = self.llm.complete(
                purpose="reflect", system=REFLECT_SYSTEM,
                prompt=self._scratchpad(state["task"], state["steps"]),
                max_tokens=self.config.max_tokens)
            reflection = parse_model(raw, Reflection)
        except (LLMError, ParseError) as exc:  # advisory: never fatal
            return {"warnings": [*state["warnings"], f"reflection skipped: {exc}"]}
        observation = step.observation
        if not reflection.on_track and reflection.next_hint:
            observation += f" [reflection hint: {reflection.next_hint}]"
        step = step.model_copy(update={"reflection": reflection, "observation": observation})
        return {"steps": [*state["steps"][:-1], step]}

    def _critique(self, state: AgentState) -> dict:
        """Self-critique gate on a proposed final answer."""
        cfg = self.config
        decision = state["decision"]
        assert decision is not None and decision.final_answer is not None
        draft = decision.final_answer
        i, steps, warnings = state["iteration"], state["steps"], list(state["warnings"])
        critique: Critique | None = None
        if cfg.critique:
            prompt = f"{self._scratchpad(state['task'], steps)}\n\nProposed answer: {draft}"
            try:
                raw = self.llm.complete(purpose="critique", system=CRITIQUE_SYSTEM,
                                        prompt=prompt, max_tokens=cfg.max_tokens)
                critique = parse_model(raw, Critique)
            except (LLMError, ParseError) as exc:  # fail open: a dead critic can't block
                warnings.append(f"critique skipped: {exc}")
        update: dict = {"draft": draft, "critique": critique, "warnings": warnings}
        if critique and not critique.acceptable:
            if state["revisions"] < cfg.max_revisions:
                update["revisions"] = state["revisions"] + 1
                update["decision"] = None
                update["steps"] = [*steps, Step(
                    index=i, thought=decision.thought, note="answer_rejected",
                    observation=f"Your proposed answer was rejected by review: {critique.issues} "
                                "Gather what is missing or fix it.")]
            else:
                warnings.append(f"answer accepted after {state['revisions']} revisions "
                                f"despite: {critique.issues}")
        return update

    @staticmethod
    def _after_critique(state: AgentState) -> str:
        return "finish" if state.get("decision") is not None else "gate"

    def _finish(self, state: AgentState) -> dict:
        decision = state["decision"]
        assert decision is not None
        step = Step(index=state["iteration"], thought=decision.thought, note="final_answer")
        return {"answer": state["draft"], "stop_reason": StopReason.ANSWERED,
                "steps": [*state["steps"], step]}

    def _degrade(self, state: AgentState) -> dict:
        """Graceful degradation: best-effort answer, LLM first, deterministic fallback."""
        reason = state["stop_reason"]
        steps, draft, warnings = state["steps"], state.get("draft"), list(state["warnings"])
        answer = ""
        if reason is not StopReason.LLM_ERROR:  # provider is down; don't hammer it
            prompt = self._scratchpad(state["task"], steps)
            if draft:
                prompt += f"\nUnreviewed draft answer: {draft}"
            try:
                answer = self.llm.complete(purpose="degrade", system=DEGRADE_SYSTEM,
                                           prompt=prompt,
                                           max_tokens=self.config.max_tokens).strip()
            except LLMError as exc:
                warnings.append(f"degrade call failed: {exc}")
        if not answer:
            answer = self._fallback_answer(steps, draft, reason)
        return {"answer": answer, "warnings": warnings, "stop_reason": reason}

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _scratchpad(task: str, steps: list[Step]) -> str:
        lines = [f"Task: {task}", ""]
        for s in steps:
            if s.thought:
                lines.append(f"Thought {s.index}: {s.thought}")
            if s.action:
                lines.append(f"Action {s.index}: {s.action.tool} {s.action.input}")
            if s.observation:
                lines.append(f"Observation {s.index}: {s.observation}")
            if s.reflection and s.reflection.assessment:
                lines.append(f"Reflection {s.index}: {s.reflection.assessment}")
        lines.append("\nNext step (JSON only):")
        return "\n".join(lines)

    @staticmethod
    def _fallback_answer(steps: list[Step], draft: str | None, reason: StopReason) -> str:
        """No LLM available: report what was actually observed, never invent."""
        facts = [s.observation for s in steps if s.action and not s.error and s.observation]
        parts = [f"[partial result - stopped: {reason.value}]"]
        if draft:
            parts.append(f"Unreviewed draft: {draft}")
        parts.append("Evidence gathered: " + (" | ".join(facts) if facts else "none"))
        return " ".join(parts)


__all__ = ["AgentState", "ReActAgent"]
