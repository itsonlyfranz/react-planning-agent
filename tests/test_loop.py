"""Observe-think-act-reflect loop behaviour."""

from conftest import act, final
from react_agent import StopReason


def test_happy_path_uses_tools_and_answers(make_agent):
    agent, _ = make_agent({"think": [
        act("lookup", query="eiffel tower height"),
        act("calculator", expression="330 / 1.8"),
        final("about 183x"),
    ]})
    r = agent.run("ratio?")
    assert r.stop_reason is StopReason.ANSWERED and not r.degraded
    assert r.answer == "about 183x"
    assert [s.action.tool for s in r.steps if s.action] == ["lookup", "calculator"]
    assert "330 metres" in r.steps[0].observation
    assert r.steps[1].observation.startswith("183.33")


def test_reflection_runs_after_each_action_and_is_recorded(make_agent):
    agent, llm = make_agent({"think": [act("calculator", expression="1+1"), final("2")]})
    r = agent.run("t")
    assert llm.count("reflect") == 1
    assert r.steps[0].reflection is not None and r.steps[0].reflection.on_track


def test_off_track_reflection_hint_reaches_next_prompt(make_agent):
    agent, llm = make_agent({
        "think": [act("calculator", expression="1+1"), final("2")],
        "reflect": [{"on_track": False, "assessment": "stalling", "next_hint": "just answer"}],
    })
    agent.run("t")
    last_think_prompt = [p for purpose, p in llm.calls if purpose == "think"][-1]
    assert "just answer" in last_think_prompt


def test_reflection_can_be_disabled(make_agent):
    agent, llm = make_agent({"think": [act("calculator", expression="1+1"), final("2")]},
                            reflect=False)
    agent.run("t")
    assert llm.count("reflect") == 0


def test_prompt_contains_prior_observations(make_agent):
    agent, llm = make_agent({"think": [act("lookup", query="paris population"), final("2.1M")]})
    agent.run("t")
    second_prompt = [p for purpose, p in llm.calls if purpose == "think"][1]
    assert "Observation 1: Paris has about 2.1 million" in second_prompt



def test_reflections_survive_recursion_limit_recovery(make_agent):
    agent, _ = make_agent({
        "think": [act("calculator", expression=f"{n}+1") for n in range(50)],
        "reflect": [{"on_track": False, "assessment": "stalling", "next_hint": "just answer"}],
        "degrade": ["partial"],
    }, max_iterations=40, recursion_limit=8)
    r = agent.run("t")
    assert any("recursion limit" in w for w in r.warnings)
    assert r.steps and all(s.reflection is not None for s in r.steps)
    assert len(r.steps) == 2 and "just answer" in r.steps[0].observation
