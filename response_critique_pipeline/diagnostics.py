"""Per-domain diagnostic report on the final-check split.

Breaks the three grading factors -- grounding (average precision), routing,
and calibration -- down per domain, plus the combined row_score, and flags
any domain that lags the best domain by more than a threshold on a chosen
metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import pandas as pd

from .scoring import average_precision, row_score, sign
from .utils import load_relevance

METRIC_COLUMNS = ["grounding", "routing", "calibration", "row_score"]


def _routing_component(scores: Sequence[float], labels: Sequence[int]) -> float:
    num = sum(abs(s) for s, y in zip(scores, labels) if y != 0 and sign(s) == sign(y))
    den = sum(abs(s) for s, y in zip(scores, labels) if y != 0)
    n_true = sum(1 for y in labels if y != 0)
    if den > 0:
        w = min(1.0, den / n_true)
        return w * (num / den) + (1 - w) * 0.5
    return 0.5


def _calibration_component(scores: Sequence[float], labels: Sequence[int]) -> float:
    return 1.0 - sum((s - y) ** 2 for s, y in zip(scores, labels)) / (4 * len(labels))


def per_row_components(scores: Sequence[float], labels: Sequence[int]) -> Dict[str, float]:
    is_true = [1 if y != 0 else 0 for y in labels]
    return {
        "grounding": average_precision([abs(s) for s in scores], is_true),
        "routing": _routing_component(scores, labels),
        "calibration": _calibration_component(scores, labels),
        "row_score": row_score(scores, labels),
    }


def per_domain_metrics(
    df,
    score_col: str = "score",
    domain_col: str = "domain",
    relevance_col: str = "relevance",
) -> pd.DataFrame:
    """df must have one row per example with a 16-length `score` list
    (calibrated or raw, caller's choice) and the JSON `relevance` column."""
    records = []
    for _, row in df.iterrows():
        scores = row[score_col]
        labels = load_relevance(row[relevance_col]) if isinstance(row[relevance_col], str) else row[relevance_col]
        comp = per_row_components(scores, labels)
        comp[domain_col] = row[domain_col]
        records.append(comp)

    per_row = pd.DataFrame(records)
    agg = per_row.groupby(domain_col)[METRIC_COLUMNS].agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg["n_rows"] = per_row.groupby(domain_col).size()
    return agg.reset_index()


@dataclass
class DomainFlag:
    domain: str
    metric: str
    domain_value: float
    best_value: float
    gap: float


def flag_lagging_domains(
    metrics_df: pd.DataFrame,
    threshold: float = 0.05,
    metric: str = "row_score",
    domain_col: str = "domain",
) -> List[DomainFlag]:
    """Flags any domain whose mean `metric` trails the best-performing
    domain by more than `threshold`."""
    col = f"{metric}_mean"
    if col not in metrics_df.columns:
        raise KeyError(f"{col!r} not in metrics_df; expected output of per_domain_metrics()")

    best_value = metrics_df[col].max()
    flags = []
    for _, row in metrics_df.iterrows():
        gap = best_value - row[col]
        if gap > threshold:
            flags.append(
                DomainFlag(
                    domain=row[domain_col],
                    metric=metric,
                    domain_value=float(row[col]),
                    best_value=float(best_value),
                    gap=float(gap),
                )
            )
    return sorted(flags, key=lambda f: -f.gap)


def build_report(
    df,
    score_col: str = "score",
    domain_col: str = "domain",
    relevance_col: str = "relevance",
    threshold: float = 0.05,
    flag_metrics: Sequence[str] = ("grounding", "routing", "calibration", "row_score"),
) -> Dict[str, object]:
    metrics_df = per_domain_metrics(df, score_col=score_col, domain_col=domain_col, relevance_col=relevance_col)
    flags = []
    for m in flag_metrics:
        flags.extend(flag_lagging_domains(metrics_df, threshold=threshold, metric=m, domain_col=domain_col))
    return {"metrics": metrics_df, "flags": flags, "threshold": threshold}


def format_report(report: Dict[str, object]) -> str:
    lines = ["Per-domain diagnostic report", "=" * 40]
    lines.append(report["metrics"].to_string(index=False))
    lines.append("")
    flags = report["flags"]
    if not flags:
        lines.append(f"No domain lags another by more than {report['threshold']} on any tracked metric.")
    else:
        lines.append(f"Domains lagging by more than {report['threshold']}:")
        for f in flags:
            lines.append(
                f"  - {f.domain}: {f.metric} = {f.domain_value:.4f} "
                f"(best = {f.best_value:.4f}, gap = {f.gap:.4f})"
            )
    return "\n".join(lines)
