# ReAct Planning Agent

**A ReAct agent with an observe → think → act → reflect loop that is bounded on every axis: it can't infinite loop, it critiques its own answers, and it degrades gracefully instead of crashing.**

Technologies: Python 3.11+, LangGraph, langchain-anthropic (Claude), Pydantic v2, pytest · Category: AI agents · Shows: you can build agents that don't infinite loop.

## Highlights

- The agent is a LangGraph `StateGraph` with typed state, explicit nodes (`gate`, `think`, `act`, `reflect`, `critique`, `finish`, `degrade`) and conditional edges; every cycle passes through one `gate` node that enforces the limits.
- Five independent stop conditions (iteration cap, repeated-call detection, wall-clock deadline, tool-error streak, parse-failure streak), each covered by a test.
- In the `infinite-loop` demo the model repeats one call with a cap of 20 iterations; the loop guard warns on call 2 and stops on call 3 (3 of 20 turns used).
- Self-critique gate on final answers with bounded revisions; low-confidence acceptance is surfaced in `warnings`, never silent.
- Two-tier graceful degradation: LLM-written partial answer, then a deterministic fallback built only from successful observations. Every early exit carries a machine-readable `StopReason`.
- LLM behind a one-method interface; the full test suite and the six-scenario demo run offline with no API key.

## Architecture

| Component | Responsibility |
|---|---|
| `ReActAgent` (`agent.py`) | Builds and runs the LangGraph `StateGraph` (typed `AgentState`, one node per phase, conditional edges); the `gate` node enforces the iteration cap and deadline, `act` enforces loop and error-streak limits |
| LangGraph runtime | `MemorySaver` checkpoints every super-step per `thread_id` (inspectable via `agent.checkpoints`); `recursion_limit` is a second-line backstop that degrades from the last checkpoint |
| `LLMClient` / `AnthropicClient` (`llm.py`) | One-method provider interface; Claude via `langchain-anthropic` `ChatAnthropic`, errors normalised to `LLMError` |
| `ScriptedLLM` (`fake.py`) | Offline fake that replays queued responses per role and can simulate outages or a stuck model |
| `ToolRegistry` (`tools.py`) | Dispatches tools; unknown tools, bad args and crashes become observations. Ships a safe AST calculator and a local fact lookup |
| Parsing (`parsing.py`) | Tolerant JSON extraction and Pydantic validation of model replies |
| Models (`models.py`) | `AgentConfig`, `Step`, `Critique`, `Reflection`, `RunResult`, `StopReason` |
| Scenarios and CLI (`scenarios.py`, `cli.py`) | Six scripted demo scenarios and the `demo` / `run` commands |
| Trace (`trace.py`) | Appends each run to a local JSONL file |

Graph (every cycle re-enters `gate`, the single place the iteration cap and deadline are enforced):

```
START -> gate --(cap/deadline hit)--> degrade -> END
          |
          v
        think --(LLM down / 3 bad replies)--> degrade
          |  \--(unparseable)--> gate
          |-- action --> act --(same call x3 / error streak)--> degrade
          |               \--> reflect --> gate
          '-- final_answer --> critique --(accepted)--> finish -> END
                                  '--(rejected, revisions left)--> gate
degrade: LLM best-effort partial answer; deterministic fallback if that fails too.
```

## What it demonstrates

| Capability | Where | Proved by |
|---|---|---|
| Observe/think/act/reflect loop | `agent.py` `ReActAgent.run` | `tests/test_loop.py` |
| Max iteration limit (hard cap) + LangGraph recursion-limit backstop | `AgentConfig.max_iterations`, `recursion_limit` | `test_max_iterations_is_a_hard_cap`, `test_langgraph_recursion_limit_is_a_graceful_backstop` |
| Repeated-call loop detection (warn, then stop) | `repeat_limit` | `test_identical_repeated_call_is_detected_and_stopped`, `test_same_tool_different_input_is_not_a_loop` |
| Other bounds: wall-clock deadline, consecutive tool errors, consecutive parse failures | `AgentConfig` | `tests/test_limits.py` |
| Self-critique of the final answer, bounded revisions | `_critique`, `max_revisions` | `tests/test_critique.py` |
| Reflection after each action (advisory, fed back into the next prompt) | `_reflect` | `test_off_track_reflection_hint_reaches_next_prompt` |
| Graceful degradation (LLM summary, then deterministic fallback) | `_degrade` | `tests/test_degradation.py` |
| Per-run checkpoints (LangGraph `MemorySaver`) | `ReActAgent.checkpoints` | `test_every_run_is_checkpointed_per_thread` |
| Survives tool crashes, bad JSON, provider outages | `ToolRegistry.run`, `parsing.py` | `test_unknown_tool_is_an_observation_not_a_crash`, `test_llm_error_stops_cleanly` |

## Quickstart

Python 3.11+. With [uv](https://docs.astral.sh/uv/) (or plain `pip install -e '.[dev]'`):

```bash
uv sync --extra dev           # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pytest                 # 45 tests, fully offline - no network, no API key
uv run react-agent demo       # six scripted scenarios, offline
```

### Demo

`react-agent demo [scenario]` replays scripted model behaviour through the real agent loop:

| Scenario | Shows |
|---|---|
| `success` | lookup → calculate → critique accepts |
| `self-critique` | vague answer rejected, agent gathers more, revised answer accepted |
| `infinite-loop` | model repeats a call forever → soft warning, then `loop_detected` after 3 turns (cap was 20) |
| `max-iterations` | model keeps varying calls → `max_iterations`, degraded answer |
| `tool-failures` | tool keeps failing → `tool_failures`, honest "nothing established" |
| `llm-outage` | provider dies mid-run → deterministic fallback reports only real evidence |

### Live with Claude

```bash
cp .env.example .env          # set ANTHROPIC_API_KEY (gitignored); optional REACT_AGENT_MODEL
uv run react-agent run "How many times taller is the Eiffel Tower than a 1.8 m person?" --trace traces/runs.jsonl
```

The default model is `claude-sonnet-5-5`; override with `REACT_AGENT_MODEL`. Exit code is 1 when
the run degraded, so it composes in scripts.

## Design decisions and trade-offs

- **Multiple independent bounds, not one.** An iteration cap alone still wastes the whole budget
  on a model that repeats itself, so the loop guard (signature = tool + canonical JSON input)
  warns on the second identical call and stops on the third. Wall-clock, error-streak and
  parse-failure bounds cover stalls that iteration counts do not.
- **Errors are observations.** Unknown tools, bad arguments and tool crashes are returned to the
  model as text so it can adapt; only a *streak* of failures ends the run.
- **Advisory vs. blocking LLM roles.** Reflection and critique fail *open* (recorded in
  `warnings`) so an unreliable helper call can't take down a run. The think call fails *closed*
  into degradation.
- **Critique is bounded.** After `max_revisions` the last answer is accepted but flagged in
  `warnings` rather than looping on perfectionism.
- **Degradation never invents.** The fallback answer is built only from successful tool
  observations and any unreviewed draft, and says it is partial. A degraded result is marked
  `degraded=True` with a machine-readable `StopReason`.
- **LangGraph where it fits.** Typed state + conditional edges make the control flow explicit and inspectable, and checkpointing gives a per-run audit trail and a recovery point for the recursion-limit backstop. Interrupts and parallel branches were deliberately not used: the run is autonomous and sequential, and adding them would be decoration. The recursion limit is derived from `max_iterations` (4 graph steps per cycle + slack) so it only fires if a bug lets the `gate` be bypassed.
- **LLM behind a one-method protocol** (`complete(purpose, system, prompt, max_tokens)`); the
  `purpose` tag lets a scripted fake respond per role. Cost: the model is prompted with a text
  scratchpad rather than native tool-use blocks, which keeps the loop provider-agnostic and
  trivially testable at the price of JSON-in-text parsing (handled tolerantly).
- **Local-only infrastructure.** Tools are in-process; traces are a JSONL file. No services.
- **Limits.** Reflection and critique add LLM calls per step (disable via `reflect=False` /
  `critique=False`); the built-in tools are intentionally tiny - the point is the control loop.

## Testing

`uv run pytest` runs entirely against `ScriptedLLM`, which replays queued responses per purpose
(and can raise `LLMError` to simulate outages, or repeat its last response to simulate a stuck
model). The deadline test injects a fake clock. Lint: `uv run ruff check .`
