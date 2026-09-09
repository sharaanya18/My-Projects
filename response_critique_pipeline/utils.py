"""Shared parsing helpers for the dataset's JSON-encoded columns."""
from __future__ import annotations

import json
from typing import Any, Dict, List


def load_context(context_json: str) -> List[Dict[str, str]]:
    turns = json.loads(context_json)
    if not isinstance(turns, list):
        raise ValueError("context must decode to a JSON list of chat turns")
    return turns


def load_candidates(candidates_json: str) -> List[str]:
    candidates = json.loads(candidates_json)
    if not isinstance(candidates, list) or len(candidates) != 16:
        raise ValueError("candidates must decode to a JSON list of exactly 16 strings")
    return candidates


def load_relevance(relevance_json: str) -> List[int]:
    relevance = json.loads(relevance_json)
    if not isinstance(relevance, list) or len(relevance) != 16:
        raise ValueError("relevance must decode to a JSON list of exactly 16 ints")
    for y in relevance:
        if y not in (-1, 0, 1):
            raise ValueError(f"relevance values must be in {{-1,0,1}}, got {y}")
    return [int(y) for y in relevance]


def count_nonzero_relevance(relevance_json: str) -> int:
    return sum(1 for y in load_relevance(relevance_json) if y != 0)


def format_context(turns: List[Dict[str, str]]) -> str:
    lines = []
    for turn in turns:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        lines.append(f"[{role}] {content}")
    return "\n".join(lines)
