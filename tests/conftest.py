import pytest

from react_agent import AgentConfig, ReActAgent
from react_agent.fake import ScriptedLLM


def act(tool: str, thought: str = "", **inp) -> dict:
    return {"thought": thought, "action": {"tool": tool, "input": inp}}


def final(answer: str) -> dict:
    return {"thought": "done", "final_answer": answer}


@pytest.fixture
def make_agent():
    def _make(script, *, repeat_last=None, clock=None, **cfg):
        llm = ScriptedLLM(script, repeat_last=repeat_last)
        kwargs = {"clock": clock} if clock else {}
        return ReActAgent(llm, config=AgentConfig(**cfg), **kwargs), llm

    return _make
