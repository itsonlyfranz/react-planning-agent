"""Small LLM interface so the agent never depends on a concrete provider."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

DEFAULT_MODEL = "claude-sonnet-5-5"


class LLMError(RuntimeError):
    """Raised by any client when a completion cannot be produced."""


class LLMClient(Protocol):
    def complete(self, *, purpose: str, system: str, prompt: str, max_tokens: int) -> str:
        """Return the model's text. `purpose` is one of think/reflect/critique/degrade."""
        ...


def load_dotenv(path: str | Path = ".env") -> None:
    """Tiny .env loader (no dependency); existing environment variables win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


class AnthropicClient:
    """Claude via the official SDK. Any SDK failure is normalised to LLMError."""

    def __init__(self, model: str | None = None, api_key: str | None = None) -> None:
        import anthropic

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (see .env.example)")
        self.model = model or os.environ.get("REACT_AGENT_MODEL") or DEFAULT_MODEL
        self._client = anthropic.Anthropic(api_key=key, max_retries=2, timeout=30.0)

    def complete(self, *, purpose: str, system: str, prompt: str, max_tokens: int) -> str:
        import anthropic

        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.AnthropicError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
