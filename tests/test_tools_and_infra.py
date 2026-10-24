import json

import pytest

from react_agent.cli import main
from react_agent.fake import ScriptedLLM
from react_agent.llm import DEFAULT_MODEL, AnthropicClient, LLMError, load_dotenv
from react_agent.models import AgentConfig
from react_agent.parsing import ParseError, extract_json
from react_agent.tools import ToolError, calculator, default_registry
from react_agent.trace import append_trace


def test_calculator_basic_and_safe():
    assert calculator("2 * (3 + 4)") == "14"
    for bad in ["__import__('os').system('x')", "2 **", "1/0", "9**999"]:
        with pytest.raises(ToolError):
            calculator(bad)


@pytest.mark.parametrize("expr", ["((((2**100)**100)**100)**100)**100", "(10**99)**100",
                                  "*".join(["(9**99)"] * 14)])
def test_calculator_rejects_oversized_results(expr):
    with pytest.raises(ToolError, match="too large"):
        calculator(expr)


def test_registry_wraps_tool_crashes_and_bad_args():
    reg = default_registry()
    with pytest.raises(ToolError):
        reg.run("calculator", {"wrong": 1})
    with pytest.raises(ToolError):
        reg.run("nope", {})


def test_extract_json_handles_fences_and_chatter():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    with pytest.raises(ParseError):
        extract_json("no json here")


def test_config_validation():
    with pytest.raises(ValueError):
        AgentConfig(max_iterations=0)


def test_scripted_llm_exhaustion_raises():
    with pytest.raises(LLMError):
        ScriptedLLM().complete(purpose="think", system="", prompt="", max_tokens=1)


def test_anthropic_client_requires_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        AnthropicClient()


def test_model_is_configurable_by_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert AnthropicClient().model == DEFAULT_MODEL == "claude-sonnet-5-5"
    monkeypatch.setenv("REACT_AGENT_MODEL", "other-model")
    assert AnthropicClient().model == "other-model"


def test_load_dotenv_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# c\nFOO_X=from_file\nBAR_X='quoted'\n")
    monkeypatch.setenv("FOO_X", "from_env")
    monkeypatch.delenv("BAR_X", raising=False)
    load_dotenv(env)
    import os

    assert os.environ["FOO_X"] == "from_env" and os.environ["BAR_X"] == "quoted"
    monkeypatch.delenv("BAR_X")


def test_trace_is_appended_as_jsonl(tmp_path, make_agent):
    from conftest import final

    agent, _ = make_agent({"think": [final("x")]})
    path = tmp_path / "t" / "runs.jsonl"
    append_trace(agent.run("t"), path)
    assert json.loads(path.read_text().splitlines()[0])["answer"] == "x"


def test_demo_cli_runs_every_scenario_offline(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    for name in ["success", "self-critique", "infinite-loop", "max-iterations",
                 "tool-failures", "llm-outage"]:
        assert f"== {name}" in out
    assert "stop=loop_detected" in out and "stop=max_iterations" in out


def test_demo_unknown_scenario_exits_2(capsys):
    assert main(["demo", "bogus"]) == 2


class _FakeChat:
    def __init__(self, content=None, exc=None):
        self.content, self.exc, self.kwargs = content, exc, None

    def invoke(self, messages, **kwargs):
        self.kwargs = kwargs
        if self.exc:
            raise self.exc
        return type("M", (), {"content": self.content})()


def test_anthropic_client_flattens_block_content_and_passes_max_tokens():
    chat = _FakeChat([{"type": "text", "text": "he"}, {"type": "text", "text": "llo"}])
    client = AnthropicClient(chat_model=chat)
    assert client.complete(purpose="think", system="s", prompt="p", max_tokens=7) == "hello"
    assert chat.kwargs == {"max_tokens": 7}


def test_anthropic_client_normalises_provider_errors():
    client = AnthropicClient(chat_model=_FakeChat(exc=TimeoutError("slow")))
    with pytest.raises(LLMError, match="TimeoutError"):
        client.complete(purpose="think", system="s", prompt="p", max_tokens=7)
