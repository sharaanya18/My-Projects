"""Validation designed to DETECT the failure mode that produced 0.247.

The previous scheme reported 0.84-0.93 for a model that scored 0.247. It failed
because every held-out group was still drawn from the same distribution the model
trained on, so memorised project structure kept working.

These views deliberately break that:

  V1 unfitted      -- zero-parameter rules. No parameters => memorisation is
                      impossible => the number measures relationship strength only.
  V2 extrapolate   -- train on the small/simple half of TRAIN, score the
                      large/complex half. Mimics the DIRECTION of the real shift
                      using train only (no test statistics).
  V3 region-holdout-- KMeans regions of feature space held out whole, so an
                      unseen "project" is an unseen region.
  V4 gap           -- in-sample minus out-of-sample. The old model hid a huge
                      gap; any candidate with a large gap is suspect.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from ndcg import ndcg_at_k

SAMPLE_N = 586


def sample_idx(n, draws=500, size=SAMPLE_N, seed=0):
    rng = np.random.default_rng(seed)
    size = min(size, n)
    return [rng.choice(n, size, replace=False) for _ in range(draws)]


def eval_ndcg(y, score, idx=None, draws=500, seed=0):
    y = np.asarray(y); score = np.asarray(score, dtype=float)
    idx = idx if idx is not None else sample_idx(len(y), draws, seed=seed)
    v = np.array([ndcg_at_k(y[i], score[i]) for i in idx])
    return float(v.mean()), float(v.std())


def complexity_axis(df, cols):
    """A train-only 'bigger/more structured' axis. The real shift runs this way
    (test has more lines, files, checklist items, headings), but the axis itself
    is built from TRAIN ONLY -- no test statistics touch it."""
    Z = np.column_stack([
        (np.log1p(df[c].astype(float)) - np.log1p(df[c].astype(float)).mean())
        / (np.log1p(df[c].astype(float)).std() + 1e-9) for c in cols])
    return Z.mean(axis=1)


def extrapolation_split(df, frac=0.35, cols=('body_line_count', 'file_count',
                                             'body_length', 'changed_lines')):
    """Train on the SIMPLE end, validate on the COMPLEX end. Pure extrapolation."""
    a = complexity_axis(df, list(cols))
    order = np.argsort(-a)
    n_va = int(len(df) * frac)
    return order[n_va:], order[:n_va]        # (train_idx, valid_idx)


def region_folds(X_num, n_regions=40, n_folds=5, seed=0):
    """Hold out whole KMeans regions of feature space."""
    Z = StandardScaler().fit_transform(np.asarray(X_num, dtype=float))
    km = KMeans(n_regions, n_init=10, random_state=seed).fit_predict(Z)
    sizes = pd.Series(km).value_counts()
    order = list(sizes.index)
    assign, tot = {}, np.zeros(n_folds)
    for g in order:
        f = int(np.argmin(tot)); assign[g] = f; tot[f] += sizes[g]
    return np.array([assign[g] for g in km]), km
