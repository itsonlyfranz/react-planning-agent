"""Self-critique gate on final answers."""

from conftest import act, final
from react_agent import StopReason
from react_agent.llm import LLMError


def test_rejected_answer_is_revised(make_agent):
    agent, llm = make_agent({
        "think": [final("vague"), act("calculator", expression="6*7"), final("42")],
        "critique": [{"acceptable": False, "issues": "not computed"}, {"acceptable": True}],
    })
    r = agent.run("t")
    assert r.answer == "42" and r.revisions == 1 and not r.degraded
    assert "not computed" in [p for k, p in llm.calls if k == "think"][1]


def test_revisions_are_bounded(make_agent):
    agent, llm = make_agent(
        {"think": [final("a"), final("b"), final("c"), final("d")],
         "critique": [{"acceptable": False, "issues": "bad"}] * 5},
        max_revisions=2)
    r = agent.run("t")
    assert llm.count("critique") == 3 and r.revisions == 2
    assert r.answer == "c" and r.stop_reason is StopReason.ANSWERED
    assert any("despite: bad" in w for w in r.warnings)  # low-confidence acceptance is surfaced


def test_critique_failure_fails_open_with_warning(make_agent):
    agent, _ = make_agent({"think": [final("ok")], "critique": [LLMError("down")]})
    r = agent.run("t")
    assert r.answer == "ok" and any("critique skipped" in w for w in r.warnings)


def test_critique_can_be_disabled(make_agent):
    agent, llm = make_agent({"think": [final("ok")]}, critique=False)
    agent.run("t")
    assert llm.count("critique") == 0


def test_rejected_draft_survives_into_degraded_answer(make_agent):
    agent, _ = make_agent(
        {"think": [final("my draft")], "critique": [{"acceptable": False, "issues": "x"}],
         "degrade": [LLMError("down")]},
        max_iterations=1)
    r = agent.run("t")
    assert r.stop_reason is StopReason.MAX_ITERATIONS
    assert "my draft" in r.answer
