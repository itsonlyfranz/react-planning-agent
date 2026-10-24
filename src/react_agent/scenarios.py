"""Scripted offline scenarios, one per headline behaviour."""

from __future__ import annotations

from dataclasses import dataclass

from react_agent.fake import ScriptedLLM
from react_agent.llm import LLMError
from react_agent.models import AgentConfig


def _act(tool: str, thought: str = "", **inp) -> dict:
    return {"thought": thought, "action": {"tool": tool, "input": inp}}


def _final(answer: str, thought: str = "I have enough evidence.") -> dict:
    return {"thought": thought, "final_answer": answer}


@dataclass
class Scenario:
    name: str
    summary: str
    task: str
    llm: ScriptedLLM
    config: AgentConfig


def build_scenarios() -> list[Scenario]:
    task = "How many times taller is the Eiffel Tower than a 1.8 m person?"
    return [
        Scenario(
            "success", "Happy path: lookup, calculate, critique accepts.", task,
            ScriptedLLM({"think": [
                _act("lookup", "Need the tower height.", query="eiffel tower height"),
                _act("calculator", "Divide by 1.8.", expression="330 / 1.8"),
                _final("About 183 times taller."),
            ]}), AgentConfig()),
        Scenario(
            "self-critique", "First answer is rejected by the critic, then revised.", task,
            ScriptedLLM({
                "think": [
                    _act("lookup", query="eiffel tower height"),
                    _final("Much taller."),
                    _act("calculator", "Quantify it.", expression="330 / 1.8"),
                    _final("About 183 times taller."),
                ],
                "critique": [
                    {"acceptable": False, "issues": "Answer is not quantified.", "confidence": 0.2},
                    {"acceptable": True, "confidence": 0.9},
                ]}), AgentConfig()),
        Scenario(
            "infinite-loop", "Model repeats the same call forever; the guard stops it.", task,
            ScriptedLLM({"think": [_act("lookup", "Try again.", query="eiffel tower height")],
                         "degrade": [("The tower is 330 m tall; I never got to divide by 1.8 m "
                                      "(roughly 183x), so treat that as unverified.")]},
                        repeat_last={"think"}),
            AgentConfig(max_iterations=20)),
        Scenario(
            "max-iterations", "Model keeps varying calls; the hard cap stops it.",
            "Add up the first ten primes, one at a time.",
            ScriptedLLM({"think": [_act("calculator", expression=f"{n} + 0") for n in range(30)]}),
            AgentConfig(max_iterations=4)),
        Scenario(
            "tool-failures", "Tool keeps failing; degrade with whatever was learned.",
            "What is the population of Atlantis?",
            ScriptedLLM({"think": [_act("lookup", query=f"atlantis population {n}")
                                   for n in range(5)],
                         "degrade": [("No source has a population for Atlantis; nothing was "
                                      "established.")]}),
            AgentConfig(max_consecutive_tool_errors=3)),
        Scenario(
            "llm-outage", "Provider dies mid-run; deterministic fallback, no crash.", task,
            ScriptedLLM({"think": [_act("lookup", query="eiffel tower height"),
                                   LLMError("simulated outage")]}),
            AgentConfig()),
    ]
