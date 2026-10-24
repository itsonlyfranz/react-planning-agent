"""Offline scripted LLM used by the tests and the demo. No network, no API key."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from react_agent.llm import LLMError

_DEFAULTS: dict[str, str] = {
    "reflect": json.dumps({"on_track": True, "assessment": "Progressing.", "next_hint": ""}),
    "critique": json.dumps({"acceptable": True, "issues": "", "confidence": 0.9}),
    "degrade": "Best-effort answer from the evidence gathered so far.",
}


class ScriptedLLM:
    """Replays queued responses per purpose.

    A response may be a string, a dict (JSON-encoded), or an Exception (raised, to simulate
    provider outages). When a purpose's queue is empty and `repeat_last` is set, the last
    response is repeated - which is how the demo models an agent stuck in a loop.
    """

    def __init__(
        self,
        script: dict[str, list[Any]] | None = None,
        *,
        repeat_last: set[str] | None = None,
    ) -> None:
        self._queues: dict[str, deque[Any]] = defaultdict(deque)
        for purpose, items in (script or {}).items():
            self._queues[purpose].extend(items)
        self._last: dict[str, Any] = {}
        self._repeat_last = repeat_last or set()
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, purpose: str, system: str, prompt: str, max_tokens: int) -> str:
        self.calls.append((purpose, prompt))
        queue = self._queues[purpose]
        if queue:
            item = queue.popleft()
            self._last[purpose] = item
        elif purpose in self._repeat_last and purpose in self._last:
            item = self._last[purpose]
        elif purpose in _DEFAULTS:
            item = _DEFAULTS[purpose]
        else:
            raise LLMError(f"ScriptedLLM exhausted for purpose {purpose!r}")
        if isinstance(item, Exception):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def count(self, purpose: str) -> int:
        return sum(1 for p, _ in self.calls if p == purpose)
