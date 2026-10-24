"""Graceful degradation: every failure path still returns a useful RunResult."""

import pytest

from conftest import act
from react_agent import StopReason
from react_agent.llm import LLMError


@pytest.mark.parametrize("cfg,script,reason", [
    ({"max_iterations": 2}, [act("calculator", expression=f"{n}+1") for n in range(5)],
     StopReason.MAX_ITERATIONS),
    ({}, [act("lookup", query="paris population")] * 5, StopReason.LOOP_DETECTED),
])
def test_degraded_answer_comes_from_llm_summary(make_agent, cfg, script, reason):
    agent, llm = make_agent({"think": script, "degrade": ["partial: here is what I know"]},
                            repeat_last={"think"}, **cfg)
    r = agent.run("t")
    assert r.degraded and r.stop_reason is reason
    assert r.answer == "partial: here is what I know"
    assert llm.count("degrade") == 1


def test_fallback_when_degrade_call_also_fails(make_agent):
    agent, _ = make_agent({
        "think": [act("lookup", query="paris population")] * 3,
        "degrade": [LLMError("down")]}, repeat_last={"think"})
    r = agent.run("t")
    assert r.answer.startswith("[partial result - stopped: loop_detected]")
    assert "2.1 million" in r.answer  # deterministic fallback reports real evidence only
    assert any("degrade call failed" in w for w in r.warnings)


def test_fallback_never_reports_failed_tool_output_as_evidence(make_agent):
    agent, _ = make_agent({"think": [act("lookup", query=f"zz{n}") for n in range(3)],
                           "degrade": [LLMError("down")]})
    r = agent.run("t")
    assert "Evidence gathered: none" in r.answer


def test_reflection_failure_does_not_break_the_run(make_agent):
    agent, _ = make_agent({
        "think": [act("calculator", expression="1+1"), {"thought": "", "final_answer": "2"}],
        "reflect": [LLMError("flaky")]})
    r = agent.run("t")
    assert r.stop_reason is StopReason.ANSWERED
    assert any("reflection skipped" in w for w in r.warnings)


def test_result_is_serialisable(make_agent):
    agent, _ = make_agent({"think": [act("calculator", expression="2*2"),
                                     {"thought": "", "final_answer": "4"}]})
    assert '"stop_reason":"answered"' in agent.run("t").model_dump_json()


def test_langgraph_recursion_limit_is_a_graceful_backstop(make_agent):
    # Limit far below what max_iterations would allow: LangGraph aborts, agent degrades.
    agent, _ = make_agent({"think": [act("calculator", expression=f"{n}+1") for n in range(50)],
                           "degrade": ["partial after backstop"]},
                          max_iterations=40, recursion_limit=6)
    r = agent.run("t")
    assert r.degraded and r.answer == "partial after backstop"
    assert any("recursion limit" in w for w in r.warnings)
    assert 0 < len(r.steps) < 40


def test_every_run_is_checkpointed_per_thread(make_agent):
    agent, _ = make_agent({"think": [act("calculator", expression="1+1"),
                                     {"thought": "", "final_answer": "2"}]})
    r = agent.run("t", thread_id="run-1")
    assert r.thread_id == "run-1"
    assert agent.checkpoints("run-1") >= 6  # gate/think/act/reflect/gate/think/critique/finish...
    assert agent.checkpoints("never-ran") == 0
