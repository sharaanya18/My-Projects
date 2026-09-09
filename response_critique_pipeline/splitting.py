"""Stratified split of train.csv into dev / calibration / final_check sets.

Stratified jointly by `domain` and by the count of non-zero relevance values
per row (an int in [2, 7]). Joint strata can be tiny (a handful of rows for
a rare domain x n_true combination), so this uses a manual, seeded largest-
remainder allocation per stratum rather than sklearn's StratifiedShuffleSplit,
which errors out when a class has too few members.
"""
from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .utils import count_nonzero_relevance

SPLIT_NAMES = ("dev", "calibration", "final_check")


def _allocate_counts(k: int, ratios: Sequence[float]) -> List[int]:
    """Largest-remainder apportionment of k items across len(ratios) bins.

    For k >= len(ratios), guarantees at least 1 item per bin so every split
    gets representation from every stratum that can afford it, then
    distributes the remainder proportionally to `ratios`.
    """
    n_bins = len(ratios)
    if k <= 0:
        return [0] * n_bins

    if k >= n_bins:
        counts = [1] * n_bins
        remaining = k - n_bins
    else:
        counts = [0] * n_bins
        remaining = k

    ideal = [remaining * r for r in ratios]
    floors = [int(x) for x in ideal]
    remainders = [ideal[i] - floors[i] for i in range(n_bins)]
    for i in range(n_bins):
        counts[i] += floors[i]
    leftover = remaining - sum(floors)
    # hand out leftover units to the bins with the largest fractional remainder
    order = sorted(range(n_bins), key=lambda i: -remainders[i])
    for i in range(leftover):
        counts[order[i % n_bins]] += 1
    return counts


def add_split_column(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    domain_col: str = "domain",
    relevance_col: str = "relevance",
    split_col: str = "split",
) -> pd.DataFrame:
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")

    out = df.copy()
    out["_n_true"] = out[relevance_col].map(count_nonzero_relevance)
    out["_stratum"] = list(zip(out[domain_col], out["_n_true"]))

    assignments = pd.Series(index=out.index, dtype=object)
    rng = random.Random(seed)

    for stratum, group in out.groupby("_stratum", sort=True):
        idx = list(group.index)
        rng.shuffle(idx)
        counts = _allocate_counts(len(idx), ratios)
        cursor = 0
        for split_name, n in zip(SPLIT_NAMES, counts):
            for i in idx[cursor:cursor + n]:
                assignments[i] = split_name
            cursor += n

    out[split_col] = assignments
    out = out.drop(columns=["_n_true", "_stratum"])
    return out


def split_dataframe(
    df: pd.DataFrame,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    domain_col: str = "domain",
    relevance_col: str = "relevance",
) -> Dict[str, pd.DataFrame]:
    """Returns {'dev': df, 'calibration': df, 'final_check': df}."""
    with_split = add_split_column(
        df, ratios=ratios, seed=seed, domain_col=domain_col, relevance_col=relevance_col
    )
    result = {}
    for name in SPLIT_NAMES:
        result[name] = (
            with_split[with_split["split"] == name]
            .drop(columns=["split"])
            .reset_index(drop=True)
        )
    return result


def summarize_strata(
    df: pd.DataFrame,
    split_col: str = "split",
    domain_col: str = "domain",
    relevance_col: str = "relevance",
) -> pd.DataFrame:
    """Diagnostic table: rows per (domain, n_true, split) cell, useful for
    spotting strata so rare they collapsed entirely into one split."""
    tmp = df.copy()
    tmp["_n_true"] = tmp[relevance_col].map(count_nonzero_relevance)
    return (
        tmp.groupby([domain_col, "_n_true", split_col])
        .size()
        .rename("count")
        .reset_index()
        .sort_values([domain_col, "_n_true", split_col])
    )
