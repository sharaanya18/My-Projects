"""Submission writer with pre-save validation.

Validates, before writing anything to disk, that:
  - every test id appears exactly once (no missing, no duplicate, no extra),
  - every prediction is a list of exactly 16 finite floats in [-1, 1].

These mirror the grading spec's "outright rejected" failure modes (missing
id, unknown id, duplicate id, blank id, wrong column shape) -- catching them
locally before submission is far cheaper than a rejected upload.
"""
from __future__ import annotations

import csv
import json
import math
from typing import Dict, List, Sequence

N_CANDIDATES = 16


class SubmissionValidationError(ValueError):
    def __init__(self, issues: List[str]):
        self.issues = issues
        super().__init__("submission failed validation:\n" + "\n".join(f"  - {i}" for i in issues))


def _validate_scores(row_id: str, scores) -> List[str]:
    issues = []
    if not isinstance(scores, (list, tuple)):
        issues.append(f"{row_id}: relevance is not a list")
        return issues
    if len(scores) != N_CANDIDATES:
        issues.append(f"{row_id}: expected {N_CANDIDATES} values, got {len(scores)}")
        return issues
    for i, s in enumerate(scores):
        if isinstance(s, bool) or not isinstance(s, (int, float)):
            issues.append(f"{row_id}[{i}]: value {s!r} is not numeric")
            continue
        f = float(s)
        if math.isnan(f) or math.isinf(f):
            issues.append(f"{row_id}[{i}]: value is not finite")
            continue
        if f < -1.0 or f > 1.0:
            issues.append(f"{row_id}[{i}]: value {f} outside [-1, 1]")
    return issues


def validate_predictions(test_ids: Sequence[str], predictions: Dict[str, Sequence[float]]) -> None:
    issues: List[str] = []

    test_id_set = list(dict.fromkeys(test_ids))  # preserve order, dedupe for the membership checks below
    test_id_counts: Dict[str, int] = {}
    for tid in test_ids:
        test_id_counts[tid] = test_id_counts.get(tid, 0) + 1
    dup_test_ids = [tid for tid, c in test_id_counts.items() if c > 1]
    if dup_test_ids:
        issues.append(f"test set itself has duplicate ids: {dup_test_ids[:10]}")

    test_id_lookup = set(test_id_set)
    pred_ids = list(predictions.keys())
    pred_id_counts: Dict[str, int] = {}
    for pid in pred_ids:
        pred_id_counts[pid] = pred_id_counts.get(pid, 0) + 1
    dup_pred_ids = [pid for pid, c in pred_id_counts.items() if c > 1]
    if dup_pred_ids:
        issues.append(f"duplicate ids in predictions: {dup_pred_ids[:10]}")

    missing = [tid for tid in test_id_set if tid not in predictions]
    if missing:
        issues.append(f"missing predictions for {len(missing)} test id(s), e.g. {missing[:10]}")

    unknown = [pid for pid in dict.fromkeys(pred_ids) if pid not in test_id_lookup]
    if unknown:
        issues.append(f"predictions contain {len(unknown)} id(s) not in test set, e.g. {unknown[:10]}")

    for tid in test_id_set:
        if not tid or not str(tid).strip():
            issues.append("blank id encountered")
        if tid in predictions:
            issues.extend(_validate_scores(tid, predictions[tid]))

    if issues:
        raise SubmissionValidationError(issues)


def write_submission(
    test_ids: Sequence[str],
    predictions: Dict[str, Sequence[float]],
    out_path: str,
) -> None:
    """Validates predictions against test_ids, then writes id,relevance CSV.
    Raises SubmissionValidationError (with every issue found, not just the
    first) instead of writing a partially-valid file."""
    validate_predictions(test_ids, predictions)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "relevance"])
        for tid in test_ids:
            values = [float(v) for v in predictions[tid]]
            writer.writerow([tid, json.dumps(values)])
