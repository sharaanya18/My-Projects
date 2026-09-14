#!/usr/bin/env python3
"""
Unseen Epigram Census -- predict the six-count editorial genre census of the
withheld epigrams of each manuscript.

Task recap
----------
Every query is one manuscript.  We see `revealed_occurrences` (Greek text of
some of its catalogued epigrams) plus `total_count` / `hidden_count`.  We must
predict, for the `hidden_count` epigrams we never see, how many carry each of
six opaque editorial genre codes g0..g5.  An epigram may carry several codes,
so the six counts can sum to more than `hidden_count`.

Scoring is mean Bray-Curtis similarity  S = 1 - |p-y|_1 / (|p|_1 + |y|_1).

Modelling idea
--------------
Revealed and hidden epigrams are two subsets of the *same* manuscript, so the
genre *mixture* of the hidden part is well approximated by the genre mixture of
the revealed part.  We therefore estimate a per-manuscript genre *rate* vector
r in [0,1]^6 from the revealed Greek text and predict

        count_g = round( r_g * hidden_count ).

r is learnt with ridge regression on mean-pooled per-epigram TF-IDF: pooling
the L2-normalised TF-IDF of each individual epigram and then applying a linear
map is exactly a per-epigram linear genre scorer averaged over the manuscript,
which matches the mixture assumption while needing only the aggregate targets
we are given (the only supervision that exists -- no per-epigram label is ever
observed).

Two further pieces matter for the metric:

* Ensembling several TF-IDF views (char n-grams, word unigrams, and a joint
  view) plus a k-NN mixture estimator stabilises r.
* Ridge predictions are shrunk toward the mean, which is wrong for a rounded
  L1-style metric: Bray-Curtis rewards committing to the genres that are
  actually present.  A per-genre affine recalibration  r -> a_g*r + b_g  is
  fitted by coordinate search directly on out-of-fold Bray-Curtis, which
  restores the spread and is worth ~+0.03.

Everything is fitted from the released public files only; runs on CPU in a
couple of minutes.
"""

import argparse
import csv
import json
import os
import re
import sys
import unicodedata

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize

N_GENRES = 6
GENRE_COLS = ["g%d" % i for i in range(N_GENRES)]


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_labels(path):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["query_id"]] = np.array([float(row[c]) for c in GENRE_COLS])
    return out


_KEEP = re.compile(r"[^Ͱ-Ͽἀ-῿\s]")


def normalise(text):
    """Lower-case, strip diacritics and non-Greek clutter (editorial brackets,
    punctuation, abbreviation marks), collapse whitespace."""
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def docs_of(rows):
    """One list of normalised epigram strings per manuscript."""
    return [[normalise(o["text"]) for o in r["revealed_occurrences"]] for r in rows]


def counts_of(rows):
    hidden = np.array([r["hidden_count"] for r in rows], dtype=float)
    total = np.array([r["total_count"] for r in rows], dtype=float)
    return hidden, total


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
class PooledTfidf:
    """Mean-pooled per-epigram TF-IDF.

    Each epigram is vectorised and L2-normalised on its own, then averaged over
    the manuscript.  This makes the manuscript representation a genre *mixture*
    rather than a bag of all its words, so long epigrams cannot dominate.
    """

    def __init__(self, specs):
        self.specs = specs

    def fit(self, doc_lists):
        flat = [d for docs in doc_lists for d in docs]
        self.vecs = []
        for spec in self.specs:
            spec = dict(spec)
            weight = spec.pop("w", 1.0)
            vec = TfidfVectorizer(sublinear_tf=True, **spec).fit(flat)
            self.vecs.append((vec, weight))
        return self

    def transform(self, doc_lists):
        parts = []
        for vec, weight in self.vecs:
            pooled = [
                sp.csr_matrix(normalize(vec.transform(docs)).mean(axis=0))
                for docs in doc_lists
            ]
            parts.append(normalize(sp.vstack(pooled).tocsr()) * weight)
        return sp.hstack(parts).tocsr() if len(parts) > 1 else parts[0]


def meta_features(rows):
    """Manuscript-level side information (size correlates with the mixture)."""
    out = []
    for r in rows:
        texts = [o["text"] for o in r["revealed_occurrences"]]
        out.append(
            [
                np.log(r["hidden_count"]),
                np.log(r["total_count"]),
                r["hidden_count"] / max(r["total_count"], 1),
                np.log(np.mean([len(t) for t in texts]) + 1.0),
                np.mean([t.count("\n") + 1 for t in texts]) / 10.0,
                np.log(len(texts) + 1.0),
            ]
        )
    return np.asarray(out, dtype=float)


def attach_meta(X, M, weight):
    if weight <= 0:
        return X
    return sp.hstack([X, sp.csr_matrix(M * weight)]).tocsr()


# --------------------------------------------------------------------------
# estimators
# --------------------------------------------------------------------------
def knn_rate(X_train, R_train, X_query, k=25, power=3.0):
    """Similarity-weighted mixture of the rate vectors of nearby manuscripts."""
    S = np.asarray((X_query @ X_train.T).todense())
    k = min(k, S.shape[1])
    out = np.zeros((S.shape[0], R_train.shape[1]))
    for i in range(S.shape[0]):
        idx = np.argpartition(-S[i], k - 1)[:k]
        w = np.maximum(S[i][idx], 0.0) ** power
        if w.sum() <= 0:
            w = np.ones_like(w)
        out[i] = (w[:, None] * R_train[idx]).sum(0) / w.sum()
    return out


# Ensemble members: (name, tfidf specs, ridge alpha or None for k-NN, meta weight, blend weight)
MEMBERS = [
    ("char", [dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3)], 0.3, 0.0, 1.0),
    ("char_meta", [dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3)], 0.3, 0.15, 0.6),
    ("word", [dict(analyzer="word", ngram_range=(1, 1), min_df=1)], 0.2, 0.0, 0.8),
    (
        "charword",
        [
            dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3),
            dict(analyzer="word", ngram_range=(1, 1), min_df=1, w=0.7),
        ],
        0.5,
        0.0,
        0.8,
    ),
    ("knn", [dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3)], None, 0.0, 0.4),
]


def fit_predict_members(L_fit, M_fit, R_fit, targets):
    """Fit every ensemble member on (L_fit, R_fit); predict rates for each
    (doc_lists, meta) pair in `targets`.  Returns a list of blended rate
    matrices, one per target."""
    acc = [np.zeros((len(L), N_GENRES)) for L, _ in targets]
    wsum = 0.0
    cache = {}
    for name, specs, alpha, mw, bw in MEMBERS:
        key = json.dumps(specs, sort_keys=True)
        if key not in cache:
            feat = PooledTfidf(specs).fit(L_fit)
            cache[key] = (feat, feat.transform(L_fit))
        feat, X_fit = cache[key]
        X_tgts = [feat.transform(L) for L, _ in targets]
        if alpha is None:
            preds = [knn_rate(X_fit, R_fit, Xt) for Xt in X_tgts]
        else:
            model = Ridge(alpha=alpha).fit(attach_meta(X_fit, M_fit, mw), R_fit)
            preds = [
                model.predict(attach_meta(Xt, M, mw))
                for Xt, (_, M) in zip(X_tgts, targets)
            ]
        for a, p in zip(acc, preds):
            a += bw * p
        wsum += bw
    return [np.clip(a / wsum, 0.0, None) for a in acc]


# --------------------------------------------------------------------------
# metric, calibration and rounding
# --------------------------------------------------------------------------
def bray_curtis(p, y):
    denom = p.sum() + y.sum()
    return 1.0 - np.abs(p - y).sum() / denom if denom > 0 else 1.0


def mean_bc(P, Y):
    return float(np.mean([bray_curtis(P[i], Y[i]) for i in range(len(Y))]))


def to_counts(R, hidden, a=None, b=None):
    """Affine-recalibrate rates, scale by hidden_count, round into the legal
    integer range [0, hidden_count]."""
    if a is not None:
        R = R * a[None, :] + b[None, :]
    R = np.clip(R, 0.0, 1.6)
    return np.clip(np.rint(R * hidden[:, None]), 0.0, hidden[:, None])


A_GRID = [0.6, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 2.5]
B_GRID = [-0.15, -0.1, -0.07, -0.05, -0.02, 0.0, 0.02, 0.05, 0.1, 0.15, 0.25]


def fit_calibration(R, hidden, Y, max_passes=30):
    """Coordinate search of a per-genre affine map maximising rounded
    out-of-fold Bray-Curtis -- the metric itself, not a surrogate."""
    a = np.ones(N_GENRES)
    b = np.zeros(N_GENRES)

    def sc(a, b):
        return mean_bc(to_counts(R, hidden, a, b), Y)

    best = sc(a, b)
    for _ in range(max_passes):
        improved = False
        for g in range(N_GENRES):
            for av in A_GRID:
                for bv in B_GRID:
                    a2, b2 = a.copy(), b.copy()
                    a2[g], b2[g] = av, bv
                    s = sc(a2, b2)
                    if s > best + 1e-9:
                        best, a, b = s, a2, b2
                        improved = True
        if not improved:
            break
    return a, b, best


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def out_of_fold(L, M, R, n_splits=5, seed=0):
    oof = np.zeros((len(L), N_GENRES))
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=seed).split(
        np.arange(len(L))
    ):
        preds = fit_predict_members(
            [L[i] for i in tr_idx],
            M[tr_idx],
            R[tr_idx],
            [([L[i] for i in te_idx], M[te_idx])],
        )
        oof[te_idx] = preds[0]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", help="directory holding the public files")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument(
        "--use-validation",
        action="store_true",
        help="fold validation.jsonl into the fitting set for the final model",
    )
    args = ap.parse_args()

    D = args.data
    train = read_jsonl(os.path.join(D, "train.jsonl"))
    valid = read_jsonl(os.path.join(D, "validation.jsonl"))
    test = read_jsonl(os.path.join(D, "test.jsonl"))
    y_train = read_labels(os.path.join(D, "train_labels.csv"))
    y_valid = read_labels(os.path.join(D, "validation_labels.csv"))

    L_tr, L_va, L_te = docs_of(train), docs_of(valid), docs_of(test)
    h_tr, _ = counts_of(train)
    h_va, _ = counts_of(valid)
    h_te, _ = counts_of(test)
    Y_tr = np.array([y_train[r["query_id"]] for r in train])
    Y_va = np.array([y_valid[r["query_id"]] for r in valid])
    R_tr, R_va = Y_tr / h_tr[:, None], Y_va / h_va[:, None]

    M_tr, M_va, M_te = meta_features(train), meta_features(valid), meta_features(test)
    mu, sd = M_tr.mean(0), M_tr.std(0) + 1e-9
    M_tr, M_va, M_te = (M_tr - mu) / sd, (M_va - mu) / sd, (M_te - mu) / sd

    # ---- out-of-fold rates on train: used to fit the calibration and to report
    print("fitting out-of-fold rates on train ...", flush=True)
    oof = out_of_fold(L_tr, M_tr, R_tr, n_splits=args.folds)
    print("  train OOF, uncalibrated : %.4f" % mean_bc(to_counts(oof, h_tr), Y_tr))
    a, b, oof_cal = fit_calibration(oof, h_tr, Y_tr)
    print("  train OOF, calibrated   : %.4f" % oof_cal)
    print("  calibration a = %s" % np.round(a, 2))
    print("  calibration b = %s" % np.round(b, 2))

    # ---- honest check on the held-out validation manuscripts
    (P_va,) = fit_predict_members(L_tr, M_tr, R_tr, [(L_va, M_va)])
    print("  validation, uncalibrated: %.4f" % mean_bc(to_counts(P_va, h_va), Y_va))
    print("  validation, calibrated  : %.4f" % mean_bc(to_counts(P_va, h_va, a, b), Y_va))

    # ---- final model, then test predictions
    if args.use_validation:
        L_fit = L_tr + L_va
        M_fit = np.vstack([M_tr, M_va])
        R_fit = np.vstack([R_tr, R_va])
        # refit the calibration on OOF over the enlarged fitting set
        oof2 = out_of_fold(L_fit, M_fit, R_fit, n_splits=args.folds)
        h_fit = np.concatenate([h_tr, h_va])
        Y_fit = np.vstack([Y_tr, Y_va])
        a, b, s2 = fit_calibration(oof2, h_fit, Y_fit)
        print("  train+val OOF, calibrated: %.4f" % s2)
    else:
        L_fit, M_fit, R_fit = L_tr, M_tr, R_tr

    (P_te,) = fit_predict_members(L_fit, M_fit, R_fit, [(L_te, M_te)])
    counts = to_counts(P_te, h_te, a, b).astype(int)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["query_id"] + GENRE_COLS)
        for r, row in zip(test, counts):
            w.writerow([r["query_id"]] + [int(v) for v in row])

    # ---- submission sanity checks
    sample = os.path.join(D, "sample_submission.csv")
    if os.path.exists(sample):
        want = [row["query_id"] for row in csv.DictReader(open(sample, encoding="utf-8"))]
        got = [r["query_id"] for r in test]
        assert sorted(want) == sorted(got), "query_id set does not match sample_submission"
    assert len(set(r["query_id"] for r in test)) == len(test), "duplicate query_id"
    assert (counts >= 0).all() and (counts <= h_te[:, None]).all(), "count out of range"
    print("wrote %s (%d rows)" % (args.out, len(test)))


if __name__ == "__main__":
    sys.exit(main())
