"""CLI: `react-agent demo` (offline) or `react-agent run "<task>" --live`."""

from __future__ import annotations

import argparse
import sys

from react_agent.agent import ReActAgent
from react_agent.llm import AnthropicClient, LLMError, load_dotenv
from react_agent.models import AgentConfig, RunResult
from react_agent.scenarios import build_scenarios
from react_agent.trace import append_trace


def print_result(result: RunResult) -> None:
    for s in result.steps:
        act = f"{s.action.tool}({s.action.input})" if s.action else s.note or "-"
        flag = " !" if s.error else ""
        print(f"  {s.index}. {act}{flag}")
        if s.observation:
            print(f"       -> {s.observation[:110]}")
    tag = "DEGRADED" if result.degraded else "OK"
    print(f"  [{tag}] stop={result.stop_reason.value} iterations={result.iterations} "
          f"revisions={result.revisions}")
    print(f"  answer: {result.answer}")
    for w in result.warnings:
        print(f"  warning: {w}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="react-agent", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    demo = sub.add_parser("demo", help="run offline scripted scenarios (no API key)")
    demo.add_argument("scenario", nargs="?", help="scenario name (default: all)")
    run = sub.add_parser("run", help="run a task against Claude")
    run.add_argument("task")
    run.add_argument("--max-iterations", type=int, default=8)
    run.add_argument("--trace", help="append the run to this JSONL file")
    args = parser.parse_args(argv)

    if args.cmd == "demo":
        scenarios = build_scenarios()
        if args.scenario:
            scenarios = [s for s in scenarios if s.name == args.scenario]
            if not scenarios:
                print(f"unknown scenario; choose from: "
                      f"{[s.name for s in build_scenarios()]}", file=sys.stderr)
                return 2
        for sc in scenarios:
            print(f"\n== {sc.name}: {sc.summary}\n   task: {sc.task}")
            print_result(ReActAgent(sc.llm, config=sc.config).run(sc.task))
        return 0

    load_dotenv()
    try:
        client = AnthropicClient()
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    agent = ReActAgent(client, config=AgentConfig(max_iterations=args.max_iterations))
    result = agent.run(args.task)
    print_result(result)
    if args.trace:
        append_trace(result, args.trace)
    return 1 if result.degraded else 0


if __name__ == "__main__":
    raise SystemExit(main())
