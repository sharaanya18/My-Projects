#!/usr/bin/env python3
"""Independent submission verification -- re-implements the grader's checks.

Deliberately does NOT import final_solution: a verifier that shares code with
the thing it verifies cannot catch a shared bug.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
SUB = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "working" / "submission.csv"
TEST = HERE / "raw" / "test.csv"
SAMPLE = HERE / "raw" / "sample_submission.csv"

TOP_K = 20
def ndcg_at_k(truth, prediction, k=TOP_K):
    truth = np.asarray(truth, dtype=float); prediction = np.asarray(prediction, dtype=float)
    cutoff = min(k, truth.size)
    gains = 2.0 ** truth - 1.0
    order = np.argsort(-prediction, kind="mergesort")[:cutoff]
    disc = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    dcg = float(np.sum(gains[order] * disc))
    idcg = float(np.sum(np.sort(gains)[::-1][:cutoff] * disc))
    return 0.0 if idcg <= 0.0 else dcg / idcg

fails = []
def check(cond, msg):
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}")
    if not cond: fails.append(msg)

sub = pd.read_csv(SUB)
test = pd.read_csv(TEST)
samp = pd.read_csv(SAMPLE)

print(f"verifying {SUB}")
check(list(sub.columns) == ["id", "prediction"], f"columns are exactly ['id','prediction'] (got {list(sub.columns)})")
check(len(sub) == 586, f"row count is 586 (got {len(sub)})")
check(len(sub) == len(test), "row count matches test.csv")
check(sub.id.duplicated().sum() == 0, "no duplicate ids")
check(set(sub.id) == set(test.id), "id set identical to test.csv")
check(set(sub.id) == set(samp.id), "id set identical to sample_submission.csv")
check((sub.id.values == test.id.values).all(), "test id ORDER preserved (not re-sorted)")
check(sub.id.map(type).eq(str).all(), "ids are strings")
check(sub.id.str.len().eq(12).all(), "all ids are 12 characters")
check(pd.api.types.is_numeric_dtype(sub.prediction), f"prediction is numeric (dtype {sub.prediction.dtype})")
p = sub.prediction.to_numpy(dtype=float)
check(np.isfinite(p).all(), "all predictions finite (no NaN/inf)")
check(sub.prediction.notna().all(), "no missing predictions")
check(sub.prediction.nunique() > 1, f"predictions vary ({sub.prediction.nunique()} distinct values)")
check(not set(np.unique(p)).issubset({0.0, 1.0, 2.0}), "predictions are scores, not class labels")

# the grader sorts both frames by id before scoring -- confirm that is lossless here
a = sub.sort_values("id").reset_index(drop=True)
check((a.id.values == np.sort(test.id.values)).all(), "sorting by id reproduces the test id order")

# sanity: the metric runs end-to-end on this submission with a dummy truth vector
rng = np.random.default_rng(0)
fake = rng.choice([0, 1, 2], len(sub), p=[.507, .252, .241])
s = ndcg_at_k(fake, p)
check(0.0 <= s <= 1.0, f"metric evaluates to a value in [0,1] against a dummy truth ({s:.4f})")

print(f"\n  prediction summary: min={p.min():.4f} max={p.max():.4f} "
      f"mean={p.mean():.4f} std={p.std():.4f}")
print(f"  top-5 ids by score: {sub.sort_values('prediction', ascending=False).id.head(5).tolist()}")
print(f"\n{'ALL CHECKS PASSED' if not fails else f'{len(fails)} CHECK(S) FAILED: {fails}'}")
sys.exit(1 if fails else 0)
