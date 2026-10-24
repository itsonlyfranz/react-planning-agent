"""The ReAct loop: think -> act -> observe -> reflect, bounded on every axis."""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

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


class Decision(BaseModel):
    thought: str = ""
    action: ToolCall | None = None
    final_answer: str | None = None


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

    # ------------------------------------------------------------------ public

    def run(self, task: str) -> RunResult:
        cfg = self.config
        started = self.clock()
        steps: list[Step] = []
        warnings: list[str] = []
        seen: Counter[str] = Counter()
        tool_errors = parse_failures = revisions = 0
        draft: str | None = None

        def degrade(reason: StopReason) -> RunResult:
            return self._degrade(task, steps, reason, draft, warnings, started, revisions)

        for i in range(1, cfg.max_iterations + 1):
            if cfg.deadline_seconds is not None and self.clock() - started >= cfg.deadline_seconds:
                return degrade(StopReason.TIMEOUT)

            # THINK
            try:
                raw = self.llm.complete(
                    purpose="think",
                    system=SYSTEM.format(tools=self.tools.describe()),
                    prompt=self._scratchpad(task, steps),
                    max_tokens=cfg.max_tokens,
                )
            except LLMError as exc:
                warnings.append(f"think failed: {exc}")
                return degrade(StopReason.LLM_ERROR)
            try:
                decision = parse_model(raw, Decision)
                if decision.action is None and decision.final_answer is None:
                    raise ParseError("need either 'action' or 'final_answer'")
            except ParseError as exc:
                parse_failures += 1
                steps.append(Step(index=i, observation=f"Unparseable reply ({exc}). "
                                  "Reply with a single JSON object.", error=True))
                if parse_failures >= cfg.max_consecutive_parse_failures:
                    return degrade(StopReason.PARSE_FAILURES)
                continue
            parse_failures = 0

            # FINAL ANSWER -> self-critique gate
            if decision.final_answer is not None:
                draft = decision.final_answer
                critique = self._critique(task, steps, draft, warnings) if cfg.critique else None
                if critique and not critique.acceptable and revisions < cfg.max_revisions:
                    revisions += 1
                    steps.append(Step(
                        index=i, thought=decision.thought, observation=(
                            f"Your proposed answer was rejected by review: {critique.issues} "
                            "Gather what is missing or fix it."),
                        note="answer_rejected"))
                    continue
                if critique and not critique.acceptable:
                    warnings.append(f"answer accepted after {revisions} revisions despite: "
                                    f"{critique.issues}")
                steps.append(Step(index=i, thought=decision.thought, note="final_answer"))
                return RunResult(
                    task=task, answer=draft, stop_reason=StopReason.ANSWERED, degraded=False,
                    iterations=i, steps=steps, critique=critique, revisions=revisions,
                    elapsed_seconds=self.clock() - started, warnings=warnings)

            # ACT (guarded against repeated identical calls)
            call = decision.action
            assert call is not None
            sig = call.signature()
            seen[sig] += 1
            step = Step(index=i, thought=decision.thought, action=call)
            steps.append(step)
            if seen[sig] >= cfg.repeat_limit:
                step.observation = f"Blocked: {call.tool} called {seen[sig]} times with same input."
                step.error = True
                return degrade(StopReason.LOOP_DETECTED)

            # OBSERVE
            try:
                step.observation = self.tools.run(call.tool, call.input)
                tool_errors = 0
            except ToolError as exc:
                step.observation, step.error = f"Error: {exc}", True
                tool_errors += 1
                if tool_errors >= cfg.max_consecutive_tool_errors:
                    return degrade(StopReason.TOOL_FAILURES)
            if seen[sig] == cfg.repeat_limit - 1:
                step.observation += " [warning: you already made this exact call; try something else]"

            # REFLECT
            if cfg.reflect:
                step.reflection = self._reflect(task, steps, warnings)
                if step.reflection and not step.reflection.on_track and step.reflection.next_hint:
                    step.observation += f" [reflection hint: {step.reflection.next_hint}]"

        return degrade(StopReason.MAX_ITERATIONS)

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

    def _reflect(self, task: str, steps: list[Step], warnings: list[str]) -> Reflection | None:
        try:
            raw = self.llm.complete(purpose="reflect", system=REFLECT_SYSTEM,
                                    prompt=self._scratchpad(task, steps),
                                    max_tokens=self.config.max_tokens)
            return parse_model(raw, Reflection)
        except (LLMError, ParseError) as exc:  # reflection is advisory; never fatal
            warnings.append(f"reflection skipped: {exc}")
            return None

    def _critique(self, task: str, steps: list[Step], answer: str,
                  warnings: list[str]) -> Critique | None:
        prompt = f"{self._scratchpad(task, steps)}\n\nProposed answer: {answer}"
        try:
            raw = self.llm.complete(purpose="critique", system=CRITIQUE_SYSTEM, prompt=prompt,
                                    max_tokens=self.config.max_tokens)
            return parse_model(raw, Critique)
        except (LLMError, ParseError) as exc:  # fail open: an unavailable critic must not block
            warnings.append(f"critique skipped: {exc}")
            return None

    def _degrade(self, task: str, steps: list[Step], reason: StopReason, draft: str | None,
                 warnings: list[str], started: float, revisions: int) -> RunResult:
        """Graceful degradation: best-effort answer, LLM first, deterministic fallback."""
        answer = ""
        if reason is not StopReason.LLM_ERROR:  # provider is down; don't hammer it
            prompt = self._scratchpad(task, steps)
            if draft:
                prompt += f"\nUnreviewed draft answer: {draft}"
            try:
                answer = self.llm.complete(purpose="degrade", system=DEGRADE_SYSTEM,
                                           prompt=prompt, max_tokens=self.config.max_tokens).strip()
            except LLMError as exc:
                warnings.append(f"degrade call failed: {exc}")
        if not answer:
            answer = self._fallback_answer(steps, draft, reason)
        return RunResult(
            task=task, answer=answer, stop_reason=reason, degraded=True, iterations=len(steps),
            steps=steps, revisions=revisions, elapsed_seconds=self.clock() - started,
            warnings=warnings)

    @staticmethod
    def _fallback_answer(steps: list[Step], draft: str | None, reason: StopReason) -> str:
        """No LLM available: report what was actually observed, never invent."""
        facts = [s.observation for s in steps if s.action and not s.error and s.observation]
        parts = [f"[partial result - stopped: {reason.value}]"]
        if draft:
            parts.append(f"Unreviewed draft: {draft}")
        parts.append("Evidence gathered: " + (" | ".join(facts) if facts else "none"))
        return " ".join(parts)
