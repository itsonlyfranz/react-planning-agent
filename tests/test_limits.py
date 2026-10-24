"""The agent cannot loop forever: every axis has a bound."""

from conftest import act
from react_agent import StopReason
from react_agent.llm import LLMError


def test_max_iterations_is_a_hard_cap(make_agent):
    varied = [act("calculator", expression=f"{n}+1") for n in range(50)]
    agent, llm = make_agent({"think": varied}, max_iterations=5)
    r = agent.run("t")
    assert r.stop_reason is StopReason.MAX_ITERATIONS and r.degraded
    assert llm.count("think") == 5


def test_identical_repeated_call_is_detected_and_stopped(make_agent):
    agent, llm = make_agent({"think": [act("lookup", query="paris population")]},
                            repeat_last={"think"}, max_iterations=50, repeat_limit=3)
    r = agent.run("t")
    assert r.stop_reason is StopReason.LOOP_DETECTED and r.degraded
    assert llm.count("think") == 3  # stopped long before max_iterations
    assert "already made this exact call" in r.steps[1].observation  # soft warning first


def test_same_tool_different_input_is_not_a_loop(make_agent):
    agent, _ = make_agent({"think": [act("calculator", expression=f"{n}+1") for n in range(4)]},
                          max_iterations=4)
    assert agent.run("t").stop_reason is StopReason.MAX_ITERATIONS


def test_consecutive_tool_errors_stop_the_run(make_agent):
    agent, _ = make_agent({"think": [act("lookup", query=f"nothing {n}") for n in range(9)]},
                          max_consecutive_tool_errors=3)
    r = agent.run("t")
    assert r.stop_reason is StopReason.TOOL_FAILURES
    assert len(r.steps) == 3 and all(s.error for s in r.steps)


def test_tool_error_streak_resets_on_success(make_agent):
    agent, _ = make_agent({"think": [
        act("lookup", query="x1"), act("lookup", query="x2"),
        act("calculator", expression="1+1"),
        act("lookup", query="x3"), act("lookup", query="x4"),
        {"thought": "", "final_answer": "ok"},
    ]}, max_consecutive_tool_errors=3)
    assert agent.run("t").stop_reason is StopReason.ANSWERED


def test_deadline_stops_the_run(make_agent):
    ticks = iter(range(0, 1000, 10))  # each clock read advances 10s
    agent, _ = make_agent({"think": [act("calculator", expression=f"{n}+1") for n in range(20)]},
                          clock=lambda: next(ticks), deadline_seconds=25, max_iterations=20)
    assert agent.run("t").stop_reason is StopReason.TIMEOUT


def test_unparseable_replies_are_retried_then_abandoned(make_agent):
    agent, llm = make_agent({"think": ["I refuse to use JSON"] * 5}, max_consecutive_parse_failures=3)
    r = agent.run("t")
    assert r.stop_reason is StopReason.PARSE_FAILURES
    assert llm.count("think") == 3


def test_parse_failure_recovers_when_model_corrects_itself(make_agent):
    agent, _ = make_agent({"think": ["garbage", '{"thought": "x", "final_answer": "fine"}']})
    r = agent.run("t")
    assert r.stop_reason is StopReason.ANSWERED and r.answer == "fine"


def test_unknown_tool_is_an_observation_not_a_crash(make_agent):
    agent, _ = make_agent({"think": [act("rm_rf"), {"thought": "", "final_answer": "ok"}]})
    r = agent.run("t")
    assert r.steps[0].error and "unknown tool" in r.steps[0].observation
    assert r.stop_reason is StopReason.ANSWERED


def test_llm_error_stops_cleanly(make_agent):
    agent, llm = make_agent({"think": [LLMError("boom")]})
    r = agent.run("t")
    assert r.stop_reason is StopReason.LLM_ERROR
    assert llm.count("degrade") == 0  # does not hammer a dead provider
