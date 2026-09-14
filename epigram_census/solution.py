import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import normalize

N_GENRES = 6
GENRE_COLS = ["g%d" % i for i in range(N_GENRES)]
ALPHA_GRID = [0.1, 0.2, 0.3, 0.5, 0.8]
N_FOLDS = 5
SEED = 0
N_GROUP_SPLITS = 6
HOLDOUT_FRACTION = 0.25
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)
SVD_DIMS = 200
N_TREES = 300
MIN_LEAF = 2
BLEND_RIDGE = 0.5
ROUND_UP_FROM = 0.40

_GREEK_ONLY = re.compile(r"[^Ͱ-Ͽἀ-῿\s]")

csv.field_size_limit(1 << 30)


def has_split(data_dir, split):
    d = Path(data_dir)
    return (d / (split + ".jsonl")).exists() or (d / (split + ".csv")).exists()


def read_queries(data_dir, split):
    jsonl = Path(data_dir) / (split + ".jsonl")
    if jsonl.exists():
        with open(jsonl, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    rows = []
    with open(Path(data_dir) / (split + ".csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(
                {
                    "query_id": row["query_id"],
                    "total_count": int(row["total_count"]),
                    "hidden_count": int(row["hidden_count"]),
                    "revealed_occurrences": json.loads(row["revealed_occurrences"]),
                }
            )
    return rows


def read_labels(data_dir, split):
    path = Path(data_dir) / (split + "_labels.csv")
    out = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["query_id"]] = np.array([float(row[c]) for c in GENRE_COLS])
    return out


def normalise(text):
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = _GREEK_ONLY.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def docs_of(rows):
    return [[normalise(o["text"]) for o in r["revealed_occurrences"]] for r in rows]


def hidden_counts(rows):
    return np.array([r["hidden_count"] for r in rows], dtype=float)


def fit_vectoriser(doc_lists):
    return TfidfVectorizer(**TFIDF).fit([d for docs in doc_lists for d in docs])


def pool(vec, doc_lists):
    rows = [sp.csr_matrix(vec.transform(docs).mean(axis=0)) for docs in doc_lists]
    return normalize(sp.vstack(rows).tocsr())


def fit_models(X_fit, R_fit, alpha):
    ridge = Ridge(alpha=alpha).fit(X_fit, R_fit)
    svd = TruncatedSVD(SVD_DIMS, random_state=SEED).fit(X_fit)
    trees = ExtraTreesRegressor(
        N_TREES, random_state=SEED, n_jobs=-1, min_samples_leaf=MIN_LEAF
    ).fit(svd.transform(X_fit), R_fit)
    return ridge, svd, trees


def apply_models(models, X_query):
    ridge, svd, trees = models
    return BLEND_RIDGE * ridge.predict(X_query) + (1.0 - BLEND_RIDGE) * trees.predict(
        svd.transform(X_query)
    )


def predict_rates(L_fit, R_fit, L_query, alpha, ridge_only=False):
    vec = fit_vectoriser(L_fit)
    X_fit = pool(vec, L_fit)
    X_query = pool(vec, L_query)
    if ridge_only:
        return Ridge(alpha=alpha).fit(X_fit, R_fit).predict(X_query)
    return apply_models(fit_models(X_fit, R_fit, alpha), X_query)


def to_counts(rates, hidden):
    scaled = np.clip(rates, 0.0, 1.0) * hidden[:, None]
    counts = np.floor(scaled + (1.0 - ROUND_UP_FROM))
    return np.clip(counts, 0.0, hidden[:, None])


def bray_curtis(p, y):
    denom = p.sum() + y.sum()
    return 1.0 - np.abs(p - y).sum() / denom if denom > 0 else 1.0


def mean_bray_curtis(P, Y):
    return float(np.mean([bray_curtis(P[i], Y[i]) for i in range(len(Y))]))


def text_groups(rows):
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    seen = {}
    for i, r in enumerate(rows):
        for o in r["revealed_occurrences"]:
            t = normalise(o["text"])
            if t in seen:
                a, b = find(i), find(seen[t])
                if a != b:
                    parent[a] = b
            else:
                seen[t] = i
    return np.array([find(i) for i in range(len(rows))])


def size_bucket(h):
    return 0 if h <= 3 else (1 if h <= 8 else 2)


def reweighted_bray_curtis(P, Y, hidden, target):
    b = np.array([size_bucket(x) for x in hidden])
    source = np.array([(b == k).mean() for k in range(3)])
    w = np.array([target[k] / max(source[k], 1e-9) for k in b])
    w = w / w.sum()
    s = np.array([bray_curtis(P[i], Y[i]) for i in range(len(Y))])
    return float((w * s).sum())


def group_holdout_score(L, R, hidden, Y, groups, alpha, target, ridge_only=False):
    splitter = GroupShuffleSplit(
        N_GROUP_SPLITS, test_size=HOLDOUT_FRACTION, random_state=SEED
    )
    scores = []
    for fit_idx, held_idx in splitter.split(np.arange(len(L)), groups=groups):
        pred = predict_rates(
            [L[i] for i in fit_idx],
            R[fit_idx],
            [L[i] for i in held_idx],
            alpha,
            ridge_only=ridge_only,
        )
        counts = to_counts(pred, hidden[held_idx])
        scores.append(
            reweighted_bray_curtis(counts, Y[held_idx], hidden[held_idx], target)
        )
    return float(np.mean(scores)), float(np.std(scores))


def select_alpha(L, R, hidden, Y, groups, target):
    scores = {}
    for alpha in ALPHA_GRID:
        m, sd = group_holdout_score(
            L, R, hidden, Y, groups, alpha, target, ridge_only=True
        )
        scores[alpha] = m
        print("  alpha %-5s group-holdout Bray-Curtis %.4f +-%.4f" % (alpha, m, sd))
    best = max(scores, key=scores.get)
    print("  selected alpha %s" % best)
    return best


def write_submission(path, rows, counts):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id"] + GENRE_COLS)
        for row, values in zip(rows, counts):
            writer.writerow([row["query_id"]] + [int(v) for v in values])


def check_submission(rows, counts, hidden, data_dir):
    ids = [r["query_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query_id in submission")
    sample = Path(data_dir) / "sample_submission.csv"
    if sample.exists():
        with open(sample, encoding="utf-8") as fh:
            expected = [row["query_id"] for row in csv.DictReader(fh)]
        if sorted(expected) != sorted(ids):
            raise ValueError("submission query_id set does not match sample_submission")
    if counts.min() < 0 or (counts > hidden[:, None]).any():
        raise ValueError("genre count outside [0, hidden_count]")
    if not np.isfinite(counts).all():
        raise ValueError("non-finite genre count")


def main():
    if len(sys.argv) < 3:
        print("usage: python3 solution.py <public_dir> <submission_out>")
        return 2
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])

    train = read_queries(public_dir, "train")
    test = read_queries(public_dir, "test")
    y_train = read_labels(public_dir, "train")

    L_train, L_test = docs_of(train), docs_of(test)
    h_train, h_test = hidden_counts(train), hidden_counts(test)
    Y_train = np.array([y_train[r["query_id"]] for r in train])
    R_train = Y_train / h_train[:, None]

    use_valid = has_split(public_dir, "validation") and (
        Path(public_dir) / "validation_labels.csv"
    ).exists()
    if use_valid:
        valid = read_queries(public_dir, "validation")
        y_valid = read_labels(public_dir, "validation")
        L_valid = docs_of(valid)
        h_valid = hidden_counts(valid)
        Y_valid = np.array([y_valid[r["query_id"]] for r in valid])
        R_valid = Y_valid / h_valid[:, None]
    else:
        valid = []

    print("manuscripts: %d train, %d validation, %d test" % (len(train), len(valid), len(test)))

    target = np.array([size_bucket(r["hidden_count"]) for r in test])
    target = np.array([(target == k).mean() for k in range(3)])
    print("test hidden_count profile (<=3, 4-8, >8): %s" % np.round(target, 3))

    groups = text_groups(train)
    print("%d text-linked groups over %d train manuscripts" % (len(set(groups)), len(train)))
    print("selecting the ridge penalty on %d group-held-out splits, reweighted to the "
          "test size profile" % N_GROUP_SPLITS)
    alpha = select_alpha(L_train, R_train, h_train, Y_train, groups, target)

    m, sd = group_holdout_score(
        L_train, R_train, h_train, Y_train, groups, alpha, target
    )
    print("group-held-out Bray-Curtis (test-matched) %.4f +-%.4f" % (m, sd))
    if use_valid:
        valid_rates = predict_rates(L_train, R_train, L_valid, alpha)
        print("held-out validation Bray-Curtis %.4f"
              % mean_bray_curtis(to_counts(valid_rates, h_valid), Y_valid))
        L_fit = L_train + L_valid
        R_fit = np.vstack([R_train, R_valid])
    else:
        L_fit, R_fit = L_train, R_train

    test_rates = predict_rates(L_fit, R_fit, L_test, alpha)
    counts = to_counts(test_rates, h_test).astype(int)

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(submission_out, test, counts)
    check_submission(test, counts, h_test, public_dir)
    print("wrote %s (%d rows)" % (submission_out, len(test)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
