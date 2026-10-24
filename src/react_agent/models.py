"""Pydantic data models shared across the agent."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StopReason(str, Enum):
    """Why a run ended. Everything except ANSWERED is a degraded exit."""

    ANSWERED = "answered"
    MAX_ITERATIONS = "max_iterations"
    LOOP_DETECTED = "loop_detected"
    TOOL_FAILURES = "tool_failures"
    PARSE_FAILURES = "parse_failures"
    TIMEOUT = "timeout"
    LLM_ERROR = "llm_error"


class AgentConfig(BaseModel):
    max_iterations: int = Field(8, ge=1, description="Hard cap on think/act cycles.")
    repeat_limit: int = Field(3, ge=2, description="Identical tool calls tolerated before stopping.")
    max_consecutive_tool_errors: int = Field(3, ge=1)
    max_consecutive_parse_failures: int = Field(3, ge=1)
    max_revisions: int = Field(2, ge=0, description="Times a critiqued answer may be revised.")
    deadline_seconds: float | None = Field(60.0, gt=0, description="Wall-clock budget per run.")
    reflect: bool = True
    critique: bool = True
    max_tokens: int = Field(1024, ge=1)
    recursion_limit: int | None = Field(
        None, ge=1, description="LangGraph backstop; default derived from max_iterations.")


class ToolCall(BaseModel):
    tool: str
    input: dict[str, Any] = Field(default_factory=dict)

    def signature(self) -> str:
        import json

        return f"{self.tool}:{json.dumps(self.input, sort_keys=True, default=str)}"


class Critique(BaseModel):
    acceptable: bool
    issues: str = ""
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class Reflection(BaseModel):
    on_track: bool = True
    assessment: str = ""
    next_hint: str = ""


class Step(BaseModel):
    """One observe-think-act-reflect cycle, kept as a full audit trail."""

    index: int
    thought: str = ""
    action: ToolCall | None = None
    observation: str = ""
    error: bool = False
    reflection: Reflection | None = None
    note: str = ""


class RunResult(BaseModel):
    task: str
    answer: str
    stop_reason: StopReason
    degraded: bool
    iterations: int
    steps: list[Step]
    critique: Critique | None = None
    revisions: int = 0
    elapsed_seconds: float = 0.0
    warnings: list[str] = Field(default_factory=list)
    thread_id: str = ""
