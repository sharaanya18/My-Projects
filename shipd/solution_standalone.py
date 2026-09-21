#!/usr/bin/env python3
"""Shipd/Eris — backport review-capacity ranking. Self-contained solution.

Single file, no local imports. Reads train.csv / train_targets.csv / test.csv
and writes working/submission.csv.

    python solution_standalone.py --raw . --out working/submission.csv

Metric: NDCG@20 over one global ranked list, gain 2^rel - 1.

APPROACH
    profile_text is a fixed 51-key `key=value` record, not prose, so this is a
    tabular ranking problem. Parse it, add scale-free ratio features, drop the
    two fields with the most extreme train->test drift, and fit LightGBM on the
    metric's own gain (2^rel - 1), averaged over 10 seeds.

WHY NOT SOMETHING BIGGER (all measured on project-disjoint folds)
    TF-IDF over the raw string      0.867  vs 0.918 for the parsed table
    LambdaRank                      +0.019 cluster-CV but -0.018 shift-holdout
    blending with LambdaRank        OOF rho 0.955 -- not complementary
    rank-based seed averaging       0 of 4 wins (harmful)
    sentence embeddings             profile_text is counts, not language

The training data memorises projects (title_digit_count maps 2->1.35, 3->1.25,
4->0.22, 5->1.35 -- an identifier, not a mechanism), and the hidden holdout is
project-disjoint, so random K-fold reports ~0.97 and means nothing. Every
choice above was made under sibling-disjoint grouped validation.

Requires: numpy, pandas, scikit-learn(optional), lightgbm.  CPU only, ~2s.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------
CAT_COLS = ['title_length_band', 'title_word_band', 'title_question_band',
            'body_length_band', 'body_word_band', 'body_structure',
            'body_question_band', 'body_reference_band', 'body_mention_band',
            'backport_term', 'release_term', 'test_term', 'bug_term',
            'docs_term', 'security_term', 'base_channel', 'change_scope']

# Largest train->test drift in the whole schema:
#   body_mention_count  train mean 2.39 -> test 0.66   (KS 0.443)
#   body_mention_band   zero-mentions 42% -> 87%       (TVD 0.443)
# Dropping them improved every validation view (+0.005 / +0.006 / +0.008).
DROP = ['body_mention_count', 'body_mention_band']

# num_leaves=7 / min_child_samples=40: within noise of the default on
# cluster-CV, slightly ahead on the shift holdout (+0.011, 6/9), and a smaller
# in-sample/OOF Spearman gap (0.166 vs 0.224). Chosen for robustness.
PARAMS = dict(n_estimators=400, learning_rate=0.05, num_leaves=7,
              min_child_samples=40, colsample_bytree=0.8,
              subsample=0.8, subsample_freq=1, verbose=-1)
N_SEEDS = 10


# --------------------------------------------------------------------------
# parsing and features
# --------------------------------------------------------------------------
def parse_profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = [dict(t.split('=', 1) for t in s.split() if '=' in t) for s in df.profile_text]
    out = pd.DataFrame(rows)
    out.insert(0, 'id', df.id.values)
    for c in out.columns:
        if c != 'id' and c not in CAT_COLS:
            out[c] = pd.to_numeric(out[c], errors='raise')
    return out


def derived(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and densities. Scale-free, so they survive a size shift better
    than raw counts do -- which matters when the adversarial train/test AUC
    is 0.975."""
    e = 1e-6
    d = pd.DataFrame(index=df.index)
    d['test_file_ratio']    = df.test_file_count / (df.file_count + e)
    d['docs_file_ratio']    = df.docs_file_count / (df.file_count + e)
    d['code_file_ratio']    = df.code_file_count / (df.file_count + e)
    d['config_file_ratio']  = df.config_file_count / (df.file_count + e)
    d['other_file_ratio']   = df.other_file_count / (df.file_count + e)
    d['add_del_ratio']      = df.additions / (df.deletions + e)
    d['churn_per_file']     = df.changed_lines / (df.file_count + e)
    d['max_file_share']     = df.max_file_changes / (df.changed_lines + e)
    d['del_share']          = df.deletions / (df.changed_lines + e)
    d['body_per_line']      = df.body_length / (df.body_line_count + 1)
    d['body_word_density']  = df.body_unique_word_count / (df.body_word_count + e)
    d['title_word_density'] = df.title_unique_word_count / (df.title_word_count + e)
    d['body_digit_rate']    = df.body_digit_count / (df.body_length + e)
    d['title_digit_rate']   = df.title_digit_count / (df.title_length + e)
    d['body_ref_rate']      = df.body_reference_count / (df.body_word_count + e)
    d['body_link_rate']     = df.body_link_count / (df.body_word_count + e)
    d['has_body']           = (df.body_length > 0).astype(float)
    d['has_tests']          = (df.test_file_count > 0).astype(float)
    d['has_docs']           = (df.docs_file_count > 0).astype(float)
    d['log_changed']        = np.log1p(df.changed_lines)
    d['log_files']          = np.log1p(df.file_count)
    d['log_body']           = np.log1p(df.body_length)
    return d


def build(TR: pd.DataFrame, TE: pd.DataFrame, drop=DROP):
    """Design matrices with identical columns and shared category levels.

    Category levels are taken from train+test together so that a level seen
    only in test (title_length_band=band_4 is one) does not become NaN.
    """
    num = [c for c in TR.columns if c not in ('id', 'target') and c not in CAT_COLS]
    ALL = pd.concat([TR.drop(columns=[c for c in ['target'] if c in TR]), TE],
                    ignore_index=True)
    dtypes = {c: pd.CategoricalDtype(sorted(ALL[c].astype(str).unique())) for c in CAT_COLS}

    def one(df):
        X = pd.concat([df[num].astype(float), derived(df)], axis=1)
        for c in CAT_COLS:
            X[c] = df[c].astype(str).astype(dtypes[c])
        return X

    A, B = one(TR), one(TE)
    keep = [c for c in A.columns if c not in set(drop)]
    return A[keep], B[keep]


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
def fit_predict(X, y, XT, n_seeds: int = N_SEEDS) -> np.ndarray:
    """Seed-averaged regression on the metric's gain, averaged on the RAW scale.

    Averaging RAW scores, not ranks. Measured on sibling-disjoint folds:
        rank-averaging  -0.014 / -0.008 / -0.116 / -0.068   (0 of 4 wins)
        raw-averaging   +0.007 / -0.018 / +0.007 / +0.004   (3 of 4 wins)
    Rank-averaging flattens each seed's score into a uniform [0,1] grade and
    discards how confident that seed was. For a top-20 metric the sharpness of
    the head of the list is exactly what matters.
    """
    gain = 2.0 ** np.asarray(y, dtype=float) - 1.0
    preds = [lgb.LGBMRegressor(random_state=s, **PARAMS).fit(X, gain).predict(XT)
             for s in range(n_seeds)]
    return np.mean(preds, axis=0)


# --------------------------------------------------------------------------
def validate_submission(sub: pd.DataFrame, test_ids) -> None:
    test_ids = pd.Series(test_ids)
    assert list(sub.columns) == ['id', 'prediction'], f"columns {list(sub.columns)}"
    assert len(sub) == len(test_ids) == 586, f"row count {len(sub)}"
    assert sub.id.duplicated().sum() == 0, "duplicate ids"
    assert set(sub.id) == set(test_ids), "id set differs from test.csv"
    assert (sub.id.values == test_ids.values).all(), "test id order not preserved"
    assert pd.api.types.is_numeric_dtype(sub.prediction), "prediction not numeric"
    assert np.isfinite(sub.prediction.to_numpy(dtype=float)).all(), "non-finite prediction"
    assert sub.prediction.nunique() > 1, "constant prediction -- no ranking information"
    print("  checks: columns/rows/ids/order/dtype/finite/variance OK")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='.', help='dir holding train.csv/test.csv/train_targets.csv')
    ap.add_argument('--out', default='working/submission.csv')
    ap.add_argument('--seeds', type=int, default=N_SEEDS)
    a = ap.parse_args()
    raw = Path(a.raw)

    t0 = time.perf_counter()
    tr = pd.read_csv(raw / 'train.csv')
    te = pd.read_csv(raw / 'test.csv')
    ta = pd.read_csv(raw / 'train_targets.csv')

    TR = parse_profile(tr).merge(ta, on='id', how='left')
    assert TR.target.notna().all(), "unmatched training ids"
    TR['target'] = TR.target.astype(int)
    TE = parse_profile(te)
    print(f"loaded train={TR.shape} test={TE.shape}")

    X, XT = build(TR, TE)
    assert list(X.columns) == list(XT.columns)
    print(f"features: {X.shape[1]} (dropped {DROP})")

    pred = fit_predict(X, TR.target.to_numpy(), XT, n_seeds=a.seeds)
    sub = pd.DataFrame({'id': TE.id.values, 'prediction': pred})
    validate_submission(sub, TE.id)

    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out, index=False)
    print(f"wrote {out} rows={len(sub)} in {time.perf_counter()-t0:.1f}s")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
