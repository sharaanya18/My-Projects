"""Local, exact reproduction of the Eris "Response Critique Point Grounding"
grading pseudocode.

The spec's pseudocode calls an unspecified helper,
``argsort_desc_with_expected_tie_credit``, described only in a comment: rank
by descending |score| and, within a block of tied |score| values, credit
hits at the *expected* average precision over the block rather than an
arbitrary tie-break order, "so a constant vector scores exactly the pool
prevalence (never worst-case)".

The concrete rule implemented below (``average_precision``) is: group
candidates into contiguous blocks of equal |score|; within a block, every
true critique point in it is credited with the precision measured at the
*end* of the block (cumulative hits / cumulative items through the whole
block), rather than a per-position rank value. For singleton blocks (the
common case, no ties) this reduces exactly to the textbook
``hits / rank`` term, so it is a strict generalization of the pseudocode's
inner loop. For a fully-tied vector (e.g. an all-zero baseline) it reduces
to exactly ``R / 16``, matching the stated "exactly the pool prevalence"
property -- which a naive random-permutation expectation does not satisfy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

Number = float


def sign(x: float) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def average_precision(abs_scores: Sequence[float], is_true: Sequence[int]) -> float:
    """Rank by descending |score|; relevant = true critique points.

    Ties (equal |score|) are credited at the expected average precision over
    the tied block -- see module docstring for the exact rule -- so a
    constant vector scores exactly the pool prevalence.
    """
    n = len(abs_scores)
    if n == 0:
        return 0.0
    R = sum(is_true)
    if R == 0:
        return 0.0  # never happens: every pool has >= 2 true points

    order = sorted(range(n), key=lambda i: -abs_scores[i])

    psum = 0.0
    pos = 0    # cumulative items consumed so far
    hits = 0   # cumulative true items consumed so far
    i = 0
    while i < n:
        j = i
        val = abs_scores[order[i]]
        block_true = 0
        block_size = 0
        while j < n and abs_scores[order[j]] == val:
            block_true += is_true[order[j]]
            block_size += 1
            j += 1
        pos_end = pos + block_size
        hits_end = hits + block_true
        if block_true > 0:
            p_block = hits_end / pos_end
            psum += p_block * block_true
        pos, hits, i = pos_end, hits_end, j

    return psum / R


def row_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """scores in [-1,1], labels in {-1,0,1}, len 16 -> combined score in [0,1]."""
    is_true = [1 if y != 0 else 0 for y in labels]
    ground = average_precision([abs(s) for s in scores], is_true)

    num = sum(abs(s) for s, y in zip(scores, labels) if y != 0 and sign(s) == sign(y))
    den = sum(abs(s) for s, y in zip(scores, labels) if y != 0)
    n_true = sum(1 for y in labels if y != 0)
    if den > 0:
        w = min(1.0, den / n_true)
        route = w * (num / den) + (1 - w) * 0.5
    else:
        route = 0.5

    calib = 1.0 - sum((s - y) ** 2 for s, y in zip(scores, labels)) / (4 * len(labels))

    raw = ground * (route ** 2) * (calib ** 2)
    return max(0.0, min(1.0, raw))


@dataclass
class Row:
    scores: List[float]
    labels: List[int]
    id: Optional[str] = None
    domain: Optional[str] = None


def grade(rows: Iterable[Row], floor: float = 0.02) -> float:
    """Mean per-row score. Per the evaluation spec's prose, the mean (not
    each row) is floored at 0.02 so a valid submission never scores exactly
    zero."""
    scores = [row_score(r.scores, r.labels) for r in rows]
    if not scores:
        return floor
    return max(floor, sum(scores) / len(scores))


def is_malformed(scores: Optional[Sequence[float]], expected_len: int = 16) -> bool:
    """Mirrors the submission-format rules for what makes a relevance cell
    unreadable: not a list, wrong length, non-numeric/non-finite, or out of
    [-1, 1]."""
    if scores is None:
        return True
    if len(scores) != expected_len:
        return True
    for s in scores:
        try:
            f = float(s)
        except (TypeError, ValueError):
            return True
        if f != f or f in (float("inf"), float("-inf")):  # NaN / inf
            return True
        if f < -1.0 or f > 1.0:
            return True
    return False


def grade_submission(rows: Iterable[Row], floor: float = 0.02) -> float:
    """Like grade(), but rows whose .scores are malformed score 0.0 for that
    row instead of raising -- matching "malformed rows score 0, they do not
    void the submission"."""
    scores = []
    for r in rows:
        if is_malformed(r.scores) or is_malformed(r.labels, expected_len=len(r.scores) if r.scores else 16):
            scores.append(0.0)
        else:
            scores.append(row_score(r.scores, r.labels))
    if not scores:
        return floor
    return max(floor, sum(scores) / len(scores))


def to_score(p_a: float, p_b: float) -> float:
    """score = p_b - p_a, always in [-1, 1] given p_a, p_b in [0, 1]."""
    return p_b - p_a


def to_scores(p_a_list: Sequence[float], p_b_list: Sequence[float]) -> List[float]:
    return [to_score(a, b) for a, b in zip(p_a_list, p_b_list)]
