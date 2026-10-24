"""Local, deterministic tools. No network access."""

from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    """A tool failed in an expected way; the message is shown to the model as the observation."""


class Tool(BaseModel):
    name: str
    description: str
    fn: Callable[..., str]

    model_config = {"arbitrary_types_allowed": True}


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def describe(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, name: str, args: dict[str, Any]) -> str:
        """Run a tool; raises ToolError for any failure (unknown tool, bad args, tool bug)."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"unknown tool {name!r}; available: {sorted(self._tools)}")
        try:
            return str(tool.fn(**args))
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"bad arguments for {name}: {exc}") from exc
        except Exception as exc:  # a misbehaving tool must never crash the loop
            raise ToolError(f"{name} crashed: {type(exc).__name__}: {exc}") from exc


_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}

_MAX_INT_BITS = 4096


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("exponent too large")
        if (isinstance(node.op, ast.Pow) and isinstance(left, int) and isinstance(right, int)
                and right > 0 and abs(left).bit_length() * right > _MAX_INT_BITS):
            raise ToolError("result too large")
        result = _OPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > _MAX_INT_BITS:
            raise ToolError("result too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ToolError("unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic (no eval())."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except ZeroDivisionError as exc:
        raise ToolError("division by zero") from exc
    except SyntaxError as exc:
        raise ToolError(f"invalid expression: {expression!r}") from exc


KNOWLEDGE = {
    "paris population": "Paris has about 2.1 million residents (city proper).",
    "berlin population": "Berlin has about 3.7 million residents.",
    "tokyo population": "Tokyo (23 wards) has about 9.7 million residents.",
    "eiffel tower height": "The Eiffel Tower is 330 metres tall.",
}


def lookup(query: str) -> str:
    """Search a tiny built-in fact base."""
    q = query.lower().strip()
    hits = [v for k, v in KNOWLEDGE.items() if k in q or q in k]
    if not hits:
        raise ToolError(f"no results for {query!r}")
    return " ".join(hits)


def default_registry() -> ToolRegistry:
    return ToolRegistry(
        [
            Tool(name="calculator", description='Evaluate arithmetic. input: {"expression": str}',
                 fn=calculator),
            Tool(name="lookup", description='Look up a fact. input: {"query": str}', fn=lookup),
        ]
    )
