"""Tolerant parsing of model output into typed structures."""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ParseError(ValueError):
    pass


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of text (handles code fences and chatter)."""
    start = text.find("{")
    if start == -1:
        raise ParseError("no JSON object found")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(obj, dict):
        raise ParseError("JSON is not an object")
    return obj


def parse_model(text: str, model: type[T]) -> T:
    try:
        return model.model_validate(extract_json(text))
    except ValidationError as exc:
        raise ParseError(f"schema mismatch: {exc.errors()[0]['msg']}") from exc
