"""Mean average precision for the tag-ranking challenge.

For one case: walk the ranking from the top, and each time a correct code
is reached, record the fraction of codes seen so far that are correct;
average those fractions over the (up to 3) correct codes. The challenge
score is the mean of that value over every case.
"""
from __future__ import annotations

from typing import Sequence


def average_precision(ranked: Sequence[str], correct: set[str]) -> float:
    if not correct:
        return 0.0
    hits = 0
    precisions = []
    for i, code in enumerate(ranked, start=1):
        if code in correct:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return 0.0
    # Codes in `correct` that never appear in `ranked` count as misses
    # (contribute 0), matching "submitting fewer than 80 forfeits them".
    missing = len(correct) - hits
    total = sum(precisions) + 0.0 * missing
    return total / len(correct)


def mean_average_precision(rankings: dict[str, list[str]], answers: dict[str, set[str]]) -> float:
    scores = [average_precision(rankings[cid], answers[cid]) for cid in answers]
    return sum(scores) / len(scores)
