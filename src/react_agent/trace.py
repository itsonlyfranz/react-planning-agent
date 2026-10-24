"""Local-file run traces (JSONL), one line per run."""

from __future__ import annotations

from pathlib import Path

from react_agent.models import RunResult


def append_trace(result: RunResult, path: str | Path = "traces/runs.jsonl") -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(result.model_dump_json() + "\n")
    return p
