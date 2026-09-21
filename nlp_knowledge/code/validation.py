"""Validation utilities: grouped folds, OOF, leakage checks, thresholds.

Build the validation scheme BEFORE the first model.
See ../validation/*.md for the reasoning behind each function.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


# --------------------------------------------------------------------------
# group construction
# --------------------------------------------------------------------------

def pair_group_ids(left, right) -> np.ndarray:
    """Connected-component ids over a pair graph.

    If text X appears in pairs (X,A) and (X,B), a random split puts one in
    train and one in valid and the model memorises X. Grouping by connected
    component prevents that. This is THE leakage fix for pair tasks.
    """
    parent: dict[str, str] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    keys_l = [_norm_key(t) for t in left]
    keys_r = [_norm_key(t) for t in right]
    for a, b in zip(keys_l, keys_r):
        union(a, b)

    roots = {}
    out = np.empty(len(keys_l), dtype=np.int64)
    for i, a in enumerate(keys_l):
        r = find(a)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]

    # Edge case worth catching early: if the pair graph is densely connected,
    # everything collapses into one or two components and GroupKFold becomes
    # impossible. That is a real finding about the dataset, not a bug here --
    # you need a different split (by time, by source, or by a held-out set of
    # entities) and should say so explicitly.
    n_groups = len(roots)
    if n_groups < 5 and len(keys_l) > 20:
        import warnings
        warnings.warn(
            f"pair_group_ids: only {n_groups} connected component(s) over "
            f"{len(keys_l)} pairs -- the pair graph is nearly fully connected, "
            "so grouped CV cannot separate train from validation. Choose a "
            "different split axis and document the leakage risk.",
            stacklevel=2,
        )
    return out


def _norm_key(text: str) -> str:
    return hashlib.md5(" ".join(str(text).lower().split()).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# duplicate / near-duplicate detection
# --------------------------------------------------------------------------

def exact_duplicate_report(texts, labels=None) -> dict:
    """Exact duplicates, and (if labels given) duplicates with CONFLICTING labels.

    Conflicting-label duplicates are a direct lower bound on the label noise
    rate -- and a cap on the achievable score.
    """
    buckets = defaultdict(list)
    for i, t in enumerate(texts):
        buckets[_norm_key(t)].append(i)
    dup_groups = [v for v in buckets.values() if len(v) > 1]
    n_dup_rows = sum(len(v) for v in dup_groups)

    report = {
        "n_rows": len(list(texts)) if not hasattr(texts, "__len__") else len(texts),
        "n_unique": len(buckets),
        "n_duplicate_groups": len(dup_groups),
        "n_duplicate_rows": n_dup_rows,
    }
    if labels is not None:
        labels = np.asarray(labels)
        conflicting = [g for g in dup_groups if len(set(labels[g].tolist())) > 1]
        report["n_conflicting_groups"] = len(conflicting)
        report["n_conflicting_rows"] = sum(len(g) for g in conflicting)
        denom = report["n_rows"] or 1
        report["min_label_noise_rate"] = report["n_conflicting_rows"] / denom
        report["_conflicting_groups"] = conflicting[:50]
    report["_duplicate_groups"] = dup_groups[:50]
    return report


def near_duplicate_pairs(texts, threshold: float = 0.95, max_features: int = 50000,
                         always_return_top: int = 0):
    """Char-3gram TF-IDF cosine near-duplicates above `threshold`.

    CALIBRATE `threshold` on your own corpus -- it is not universal. IDF makes
    the scale corpus-dependent: on a tiny corpus two near-identical strings can
    score ~0.83. Pass `always_return_top=N` to get the N highest-scoring pairs
    regardless of the threshold, and eyeball the distribution before choosing.

    O(n^2) in the similarity step -- fine up to a few thousand rows. For large
    corpora use MinHash/datasketch instead.
    """
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                          max_features=max_features)
    X = vec.fit_transform(texts)
    sims = (X @ X.T).toarray()
    np.fill_diagonal(sims, 0.0)
    if always_return_top:
        flat = np.triu(sims, k=1)
        idx = np.argsort(flat, axis=None)[::-1][:always_return_top]
        ii, jj = np.unravel_index(idx, flat.shape)
    else:
        ii, jj = np.where(sims >= threshold)
    seen, out = set(), []
    for i, j in zip(ii, jj):
        key = (min(int(i), int(j)), max(int(i), int(j)))
        if key not in seen and sims[i, j] > 0:
            seen.add(key)
            out.append((key[0], key[1], float(sims[i, j])))
    return sorted(out, key=lambda x: -x[2])


# --------------------------------------------------------------------------
# adversarial validation
# --------------------------------------------------------------------------

def adversarial_validation(train_texts, test_texts, n_splits: int = 5,
                           top_k: int = 25, random_state: int = 0) -> dict:
    """Can a model tell train from test?

    AUC ~0.5  -> same distribution; random CV is defensible.
    AUC ~1.0  -> trivially separable: leakage, preprocessing difference, or shift.
    In between -> partial shift; use the probabilities to build a test-like
                  validation set.

    Returns AUC plus the most discriminative tokens in each direction.
    """
    texts = list(train_texts) + list(test_texts)
    y = np.r_[np.zeros(len(list(train_texts))), np.ones(len(list(test_texts)))]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, max_features=100000)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    auc = cross_val_score(pipe, texts, y, cv=cv, scoring="roc_auc", n_jobs=-1)

    fitted = clone(pipe).fit(texts, y)
    names = np.asarray(fitted.named_steps["tfidf"].get_feature_names_out())
    coef = fitted.named_steps["clf"].coef_[0]
    order = np.argsort(coef)

    return {
        "auc_mean": float(auc.mean()),
        "auc_std": float(auc.std()),
        "verdict": _adv_verdict(auc.mean()),
        "test_like_tokens": names[order[::-1][:top_k]].tolist(),
        "train_like_tokens": names[order[:top_k]].tolist(),
    }


def _adv_verdict(auc: float) -> str:
    if auc < 0.6:
        return "OK: train and test look alike; random CV is defensible."
    if auc < 0.8:
        return "PARTIAL SHIFT: build a test-like validation set; inspect top tokens."
    return "SEVERE: trivially separable. Investigate leakage / preprocessing / shift."


# --------------------------------------------------------------------------
# OOF
# --------------------------------------------------------------------------

def oof_predict(estimator, X, y, cv, X_test=None, proba: bool = True):
    """Out-of-fold predictions + averaged test predictions.

    Returns (oof, test_mean, fold_ids). The k fold-models' averaged test
    predictions are a free ensemble -- you never need to refit on all data.
    """
    n = len(y)
    fold_ids = np.full(n, -1, dtype=np.int64)
    oof = None
    test_acc = []

    for k, (tr, va) in enumerate(cv.split(X, y, groups=getattr(cv, "_groups", None))):
        model = clone(estimator)
        Xtr = X.iloc[tr] if hasattr(X, "iloc") else [X[i] for i in tr] if isinstance(X, list) else X[tr]
        Xva = X.iloc[va] if hasattr(X, "iloc") else [X[i] for i in va] if isinstance(X, list) else X[va]
        model.fit(Xtr, np.asarray(y)[tr])

        pv = model.predict_proba(Xva) if proba else model.predict(Xva)
        pv = np.asarray(pv)
        if oof is None:
            oof = np.zeros((n,) + pv.shape[1:], dtype=float)
        oof[va] = pv
        fold_ids[va] = k

        if X_test is not None:
            pt = model.predict_proba(X_test) if proba else model.predict(X_test)
            test_acc.append(np.asarray(pt))

    test_mean = np.mean(test_acc, axis=0) if test_acc else None
    assert (fold_ids >= 0).all(), "some rows were never in a validation fold"
    return oof, test_mean, fold_ids


# --------------------------------------------------------------------------
# thresholds
# --------------------------------------------------------------------------

def tune_threshold(y_true, scores, metric, n_grid: int = 200,
                   prefer_plateau: bool = True) -> dict:
    """Sweep a decision threshold and report the optimum AND its stability.

    A narrow spike is fitted to noise. `prefer_plateau=True` returns the
    centre of the widest near-optimal plateau instead of the bare argmax --
    a robust threshold beats an optimal one.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    lo, hi = float(scores.min()), float(scores.max())
    grid = np.linspace(lo, hi, n_grid)
    vals = np.array([metric(y_true, (scores >= t).astype(int)) for t in grid])

    best_i = int(np.argmax(vals))
    best_v = float(vals[best_i])

    tol = 0.995 * best_v if best_v > 0 else best_v
    near = vals >= tol
    # widest contiguous run of near-optimal thresholds
    best_run, run_start, cur_start = (0, 0), None, None
    for i, ok in enumerate(near):
        if ok and cur_start is None:
            cur_start = i
        if (not ok or i == len(near) - 1) and cur_start is not None:
            end = i if not ok else i + 1
            if end - cur_start > best_run[0]:
                best_run, run_start = (end - cur_start, cur_start), cur_start
            cur_start = None

    plateau_t = float(grid[run_start + best_run[0] // 2]) if run_start is not None else float(grid[best_i])
    chosen = plateau_t if prefer_plateau else float(grid[best_i])

    return {
        "threshold": chosen,
        "threshold_argmax": float(grid[best_i]),
        "best_metric": best_v,
        "metric_at_chosen": float(metric(y_true, (scores >= chosen).astype(int))),
        "metric_at_0.5": float(metric(y_true, (scores >= 0.5).astype(int))),
        "plateau_width": int(best_run[0]),
        "grid": grid,
        "values": vals,
    }


def tune_thresholds_multilabel(Y_true, S, metric, **kw) -> list[dict]:
    """One threshold PER LABEL -- label frequencies differ by orders of magnitude."""
    Y_true, S = np.asarray(Y_true), np.asarray(S)
    return [tune_threshold(Y_true[:, j], S[:, j], metric, **kw)
            for j in range(S.shape[1])]


# --------------------------------------------------------------------------
# model agreement (ensembling precondition)
# --------------------------------------------------------------------------

def ensemble_worth_it(oof_a, oof_b, y_true, metric) -> dict:
    """Are two models complementary enough to be worth blending?

    Checks correlation, disagreement rate, and -- the decisive number -- who
    is right on the rows where they disagree.
    """
    a, b = np.asarray(oof_a, dtype=float), np.asarray(oof_b, dtype=float)
    if a.ndim > 1:
        a = a[:, -1]
    if b.ndim > 1:
        b = b[:, -1]
    y = np.asarray(y_true)

    corr = float(np.corrcoef(a, b)[0, 1])
    pa, pb = (a >= 0.5).astype(int), (b >= 0.5).astype(int)
    dis = pa != pb
    n_dis = int(dis.sum())

    out = {
        "pearson_corr": corr,
        "disagreement_rate": n_dis / len(y) if len(y) else 0.0,
        "metric_a": float(metric(y, pa)),
        "metric_b": float(metric(y, pb)),
        "metric_mean_blend": float(metric(y, ((a + b) / 2 >= 0.5).astype(int))),
    }
    if n_dis:
        out["a_right_on_disagreements"] = float((pa[dis] == y[dis]).mean())
        out["b_right_on_disagreements"] = float((pb[dis] == y[dis]).mean())
    # The decisive number is who is right where they disagree -- use it.
    if corr > 0.95:
        out["verdict"] = "SKIP: correlation > 0.95, unlikely to help"
    elif n_dis == 0:
        out["verdict"] = "SKIP: the models never disagree"
    else:
        ra = out["a_right_on_disagreements"]
        rb = out["b_right_on_disagreements"]
        if max(ra, rb) >= 0.8 and abs(ra - rb) >= 0.3:
            winner = "A" if ra > rb else "B"
            out["verdict"] = (
                f"SKIP: on disagreements {winner} is right {max(ra, rb):.0%} of the "
                f"time vs {min(ra, rb):.0%}. Just use {winner}."
            )
        else:
            out["verdict"] = ("TEST: plausibly complementary -- confirm the blend "
                              "beats the best single member on the same folds")
    return out
