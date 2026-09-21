#!/usr/bin/env python3
"""Shipd/Eris — backport review-capacity ranking. Final solution.

Task      : rank 586 change requests; metric NDCG@20 over one global list.
Approach  : parse the fixed-schema profile_text into a tabular frame, add
            scale-free ratio features, drop the two features with the most
            extreme train->test drift, and fit a LightGBM regressor on the
            metric's own gain (2^rel - 1), averaged over several seeds.

Why this and not something heavier -- every claim below was measured, see
FINAL_APPROACH.md and working/EXPERIMENT_LOG.md:
  * profile_text is a key=value record, not language; sentence embeddings
    would encode counts as words. A parsed tabular view strictly dominates
    TF-IDF over the same string (0.92 vs 0.87 NDCG@20, cluster-CV).
  * LambdaRank looked better on cluster-CV (+0.019) but was a coin flip on
    the shift-aware holdout (-0.018, 7/18 wins), so it is not used.
  * Regression on 2^r-1 beat regression on the raw label on BOTH views.

Usage:  python final_solution.py [--raw raw] [--out working/submission.csv]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from data import load, CAT_COLS                      # noqa: E402
from features import build                           # noqa: E402

# Dropped: the largest train->test drift in the whole schema.
# body_mention_count  train mean 2.39 -> test 0.66 (KS 0.443)
# body_mention_band   zero-mentions   42% -> 87%   (TVD 0.443)
# Removing them improved all three validation views (+0.005 / +0.006 / +0.008).
DROP = ["body_mention_count", "body_mention_band"]

# num_leaves=7 / min_child_samples=40 was within noise of the default on
# cluster-CV and slightly ahead on the shift holdout (+0.011, 6/9). Chosen for
# robustness under a project-disjoint, heavily shifted holdout, not for CV.
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=7,
              min_child_samples=40, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1, verbose=-1)
N_SEEDS = 10


def fit_predict(X: pd.DataFrame, y: np.ndarray, XT: pd.DataFrame,
                n_seeds: int = N_SEEDS) -> np.ndarray:
    """Seed-averaged regression on the metric's gain, averaged on the RAW scale.

    Averaging RAW scores, not ranks. Measured on sibling-disjoint folds:
        rank-averaging  -0.014 / -0.008 / -0.116 / -0.068   (0 of 4 wins)
        raw-averaging   +0.007 / -0.018 / +0.007 / +0.004   (3 of 4 wins)
    Rank-averaging flattens each seed's score into a uniform [0,1] grade, which
    discards how confident that seed was. For a top-20 metric the sharpness of
    the head of the list is exactly what matters, so that confidence is not
    something to average away.
    """
    gain = 2.0 ** np.asarray(y, dtype=float) - 1.0
    preds = [lgb.LGBMRegressor(random_state=s, **PARAMS).fit(X, gain).predict(XT)
             for s in range(n_seeds)]
    return np.mean(preds, axis=0)


def validate_submission(sub: pd.DataFrame, test_ids: pd.Series) -> None:
    """Every check the grader performs, run locally before writing."""
    assert list(sub.columns) == ["id", "prediction"], f"columns: {list(sub.columns)}"
    assert len(sub) == len(test_ids) == 586, f"row count {len(sub)} != 586"
    assert sub.id.duplicated().sum() == 0, "duplicate ids"
    assert set(sub.id) == set(test_ids), "id set differs from test.csv"
    assert (sub.id.values == test_ids.values).all(), "test id order not preserved"
    assert pd.api.types.is_numeric_dtype(sub.prediction), "prediction not numeric"
    assert np.isfinite(sub.prediction.to_numpy(dtype=float)).all(), "non-finite prediction"
    assert sub.prediction.notna().all(), "NaN prediction"
    assert sub.prediction.nunique() > 1, "constant prediction -- no ranking information"
    print("  submission checks: columns/rows/ids/order/dtype/finite/NaN/variance OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "raw"))
    ap.add_argument("--out", default=str(HERE / "working" / "submission.csv"))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()

    t0 = time.perf_counter()
    TR, TE = load(Path(a.raw))
    y = TR.target.to_numpy()
    print(f"loaded train={TR.shape} test={TE.shape}")

    X, XT = build(TR, TE, drop=DROP)
    assert list(X.columns) == list(XT.columns)
    print(f"features: {X.shape[1]}  (dropped {DROP})")

    pred = fit_predict(X, y, XT, n_seeds=a.seeds)
    print(f"fitted {a.seeds} seeds in {time.perf_counter()-t0:.1f}s")

    sub = pd.DataFrame({"id": TE.id.values, "prediction": pred})
    validate_submission(sub, TE.id)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out}  rows={len(sub)}  "
          f"range=[{sub.prediction.min():.4f}, {sub.prediction.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
