"""Batched Anthropic API caller for the grounding-and-routing task.

For each row: sends one request containing the context, both responses, and
all 16 candidate critique statements; expects back a JSON array of 16
objects, each `{"reasoning": str, "p_a": float, "p_b": float}` aligned
position-for-position with the candidates.

- Malformed JSON (wrong shape, non-numeric fields, wrong length) is retried
  up to 2 additional times (3 attempts total) with a corrective follow-up.
- After exhausting retries, the row falls back to an all-zero vector
  (p_a = p_b = 0 for all 16 candidates -> score 0 for all).
- The full reasoning text for every candidate is persisted to a per-row log
  file for audit -- not just the scores.
- Results are cached per row id, so re-running the batch skips rows that
  already completed (resumable, doesn't re-pay for finished rows).
"""
from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import format_context, load_candidates, load_context

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
MAX_EXTRA_RETRIES = 2  # up to 2 retries after the first attempt => 3 tries total
N_CANDIDATES = 16

SYSTEM_PROMPT = """You are grounding candidate critique statements against a pair of \
candidate responses to a prompt.

You will be given a conversation, two candidate responses labeled "Response A" and \
"Response B", and a pool of exactly 16 candidate critique statements. Between 2 and 7 \
of the statements are genuine critique points that describe a real shortfall of \
Response A or Response B; the rest are distractors drawn from unrelated response \
pairs and do not apply to either response here. Every statement has had its response \
identifier masked to the neutral phrase "the response" -- you must decide from \
content alone whether a statement is genuine, and if so, which response it concerns. \
A genuine critique point may describe something a response fails to do, so absence \
of a behavior in the response text is itself evidence.

Respond with ONLY a JSON array of exactly 16 objects, in the same order as the \
candidates were given. Each object must have exactly these three keys:
  "reasoning": a short string explaining your judgement for this candidate,
  "p_a": a number in [0, 1], your probability this candidate is a genuine critique \
point concerning Response A,
  "p_b": a number in [0, 1], your probability this candidate is a genuine critique \
point concerning Response B.
A distractor should have both p_a and p_b close to 0. A genuine critique point should \
have exactly one of p_a, p_b close to 1 and the other close to 0.
Output nothing before or after the JSON array -- no prose, no code fences."""


def build_user_message(context_json: str, response_a: str, response_b: str, candidates_json: str) -> str:
    turns = load_context(context_json)
    candidates = load_candidates(candidates_json)
    candidates_block = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    return (
        f"Conversation:\n{format_context(turns)}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        f"Candidate critique statements (16, order-shuffled):\n{candidates_block}\n\n"
        "Return the JSON array now."
    )


def extract_json_array(text: str) -> Optional[Any]:
    """Best-effort extraction of a JSON array from a model response that may
    have stray prose or code fences around it."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    return None
    return None


def validate_parsed(parsed: Any, expected_len: int = N_CANDIDATES) -> Optional[str]:
    """Returns None if valid, else a short description of what's wrong."""
    if not isinstance(parsed, list):
        return "not a JSON array"
    if len(parsed) != expected_len:
        return f"expected {expected_len} entries, got {len(parsed)}"
    for i, entry in enumerate(parsed):
        if not isinstance(entry, dict):
            return f"entry {i} is not an object"
        for key in ("reasoning", "p_a", "p_b"):
            if key not in entry:
                return f"entry {i} missing key '{key}'"
        if not isinstance(entry["reasoning"], str):
            return f"entry {i} 'reasoning' is not a string"
        for key in ("p_a", "p_b"):
            v = entry[key]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return f"entry {i} '{key}' is not numeric"
            fv = float(v)
            if fv != fv or fv in (float("inf"), float("-inf")):
                return f"entry {i} '{key}' is not finite"
    return None


def fallback_result(reason: str) -> List[Dict[str, Any]]:
    return [
        {"reasoning": f"FALLBACK (all-zero): {reason}", "p_a": 0.0, "p_b": 0.0}
        for _ in range(N_CANDIDATES)
    ]


@dataclass
class RowResult:
    row_id: str
    result: List[Dict[str, Any]]
    attempts: int
    fallback: bool
    raw_responses: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class AnthropicRowCaller:
    """Wraps the Anthropic client so it can be swapped for a fake in tests."""

    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL, max_tokens: int = 4096):
        import anthropic  # local import: keep this an optional dependency

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens

    def __call__(self, system: str, user: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in message.content if getattr(block, "type", None) == "text")


def call_row(
    caller,
    row_id: str,
    context_json: str,
    response_a: str,
    response_b: str,
    candidates_json: str,
    max_extra_retries: int = MAX_EXTRA_RETRIES,
) -> RowResult:
    user_message = build_user_message(context_json, response_a, response_b, candidates_json)
    raw_responses: List[str] = []
    errors: List[str] = []

    for attempt in range(max_extra_retries + 1):
        prompt = user_message
        if attempt > 0:
            prompt = (
                user_message
                + "\n\nYour previous response was invalid: "
                + errors[-1]
                + "\nReturn ONLY a valid JSON array of exactly 16 objects as instructed."
            )
        try:
            raw = caller(SYSTEM_PROMPT, prompt)
        except Exception as exc:  # network/API error -- also retryable
            raw = ""
            raw_responses.append("")
            errors.append(f"API error: {exc}")
            logger.warning("row %s attempt %d API error: %s", row_id, attempt, exc)
            continue

        raw_responses.append(raw)
        parsed = extract_json_array(raw)
        if parsed is None:
            errors.append("response did not contain a parseable JSON array")
            continue
        problem = validate_parsed(parsed)
        if problem is not None:
            errors.append(problem)
            continue

        return RowResult(
            row_id=row_id,
            result=parsed,
            attempts=attempt + 1,
            fallback=False,
            raw_responses=raw_responses,
            errors=errors,
        )

    reason = errors[-1] if errors else "unknown failure"
    return RowResult(
        row_id=row_id,
        result=fallback_result(reason),
        attempts=max_extra_retries + 1,
        fallback=True,
        raw_responses=raw_responses,
        errors=errors,
    )


def cache_path(cache_dir: Path, row_id: str) -> Path:
    return Path(cache_dir) / f"{row_id}.json"


def is_cached(cache_dir: Path, row_id: str) -> bool:
    return cache_path(cache_dir, row_id).exists()


def load_cached(cache_dir: Path, row_id: str) -> RowResult:
    data = json.loads(cache_path(cache_dir, row_id).read_text())
    return RowResult(**data)


def save_result(cache_dir: Path, result: RowResult) -> None:
    """Persists the full per-row result -- including every candidate's full
    reasoning text, not just p_a/p_b -- as the audit log for this row. This
    file doubles as the resumability cache."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache_path(cache_dir, result.row_id).write_text(json.dumps(asdict(result), indent=2))


def run_batch(
    df,
    cache_dir: str,
    caller=None,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_workers: int = 4,
    force: bool = False,
    id_col: str = "id",
    context_col: str = "context",
    response_a_col: str = "response_a",
    response_b_col: str = "response_b",
    candidates_col: str = "candidates",
) -> "list[RowResult]":
    """Runs (or resumes) the batch over every row in df. Rows already present
    in cache_dir are skipped unless force=True, so a re-run only pays for
    rows that previously failed to complete."""
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

    if caller is None:
        caller = AnthropicRowCaller(api_key=api_key, model=model)

    rows = df.to_dict("records")
    pending = [r for r in rows if force or not is_cached(cache_dir_path, str(r[id_col]))]
    already_done = len(rows) - len(pending)
    if already_done:
        logger.info("skipping %d already-cached rows (resumed)", already_done)

    results: List[RowResult] = []
    for r in rows:
        row_id = str(r[id_col])
        if not force and is_cached(cache_dir_path, row_id):
            results.append(load_cached(cache_dir_path, row_id))

    def _work(r):
        row_id = str(r[id_col])
        result = call_row(
            caller,
            row_id,
            r[context_col],
            r[response_a_col],
            r[response_b_col],
            r[candidates_col],
        )
        save_result(cache_dir_path, result)
        return result

    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_work, r): r[id_col] for r in pending}
            for fut in as_completed(futures):
                results.append(fut.result())

    return results


def results_to_scores(results: "list[RowResult]") -> "dict[str, dict]":
    """Aggregates cached row results into id -> {p_a, p_b, score, fallback},
    the compact form downstream stages (calibration, submission) consume --
    kept separate from the full-reasoning per-row log files."""
    from .scoring import to_scores

    out = {}
    for r in results:
        p_a = [float(e["p_a"]) for e in r.result]
        p_b = [float(e["p_b"]) for e in r.result]
        out[r.row_id] = {
            "p_a": p_a,
            "p_b": p_b,
            "score": to_scores(p_a, p_b),
            "fallback": r.fallback,
        }
    return out
