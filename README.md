# ReAct Planning Agent

A ReAct (reason + act) agent with an **observe → think → act → reflect** loop that is
**provably bounded**: it cannot loop forever, it critiques its own answers, and when it has to
give up it still returns something useful instead of crashing or hanging.

> **Shows:** you can build agents that don't infinite loop.

## What it demonstrates

| Capability | Where | Proved by |
|---|---|---|
| Observe/think/act/reflect loop | `agent.py` `ReActAgent.run` | `tests/test_loop.py` |
| Max iteration limit (hard cap) | `AgentConfig.max_iterations` | `test_max_iterations_is_a_hard_cap` |
| Repeated-call loop detection (warn, then stop) | `repeat_limit` | `test_identical_repeated_call_is_detected_and_stopped`, `test_same_tool_different_input_is_not_a_loop` |
| Other bounds: wall-clock deadline, consecutive tool errors, consecutive parse failures | `AgentConfig` | `tests/test_limits.py` |
| Self-critique of the final answer, bounded revisions | `_critique`, `max_revisions` | `tests/test_critique.py` |
| Reflection after each action (advisory, fed back into the next prompt) | `_reflect` | `test_off_track_reflection_hint_reaches_next_prompt` |
| Graceful degradation (LLM summary, then deterministic fallback) | `_degrade` | `tests/test_degradation.py` |
| Survives tool crashes, bad JSON, provider outages | `ToolRegistry.run`, `parsing.py` | `test_unknown_tool_is_an_observation_not_a_crash`, `test_llm_error_stops_cleanly` |

## Architecture

```
            ┌────────────────────────── ReActAgent.run(task) ──────────────────────────┐
            │  for i in 1..max_iterations          (deadline checked every turn)        │
            │                                                                           │
  task ───▶ │  THINK   LLM -> {"thought", "action" | "final_answer"}   (parse errors    │
            │            │                                              retried, bounded)│
            │            ├─ final_answer ─▶ CRITIQUE ─ rejected? ─▶ revise (≤ max_revisions)
            │            │                      └─ accepted ─────▶ RunResult(ANSWERED) │
            │            └─ action ─▶ loop guard (same call ≥ repeat_limit?) ─▶ DEGRADE │
            │  ACT/OBSERVE   ToolRegistry.run  (errors become observations)             │
            │                 └ N consecutive errors ─────────────────────────▶ DEGRADE │
            │  REFLECT       LLM -> {"on_track", "assessment", "next_hint"}             │
            │                                                                           │
            │  cap / timeout / LLM failure / parse failures ─────────────────▶ DEGRADE  │
            └───────────────────────────────────────────────────────────────────────────┘
   DEGRADE: ask the LLM for a best-effort partial answer from the scratchpad;
            if that fails too, build one deterministically from successful observations.
```

Modules (`src/react_agent/`): `models.py` (Pydantic v2: config, steps, results, `StopReason`),
`llm.py` (client protocol + Claude client), `fake.py` (offline scripted LLM), `tools.py`
(sandboxed calculator, fact lookup), `parsing.py`, `agent.py`, `scenarios.py`, `cli.py`,
`trace.py` (JSONL run traces).

## Quickstart

Python 3.11+. With [uv](https://docs.astral.sh/uv/) (or plain `pip install -e '.[dev]'`):

```bash
uv sync --extra dev           # or: python -m venv .venv && .venv/bin/pip install -e '.[dev]'
uv run pytest                 # 37 tests, fully offline - no network, no API key
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
