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
ALPHA_GRID = [0.1, 0.2, 0.3, 0.5, 0.8]
N_FOLDS = 5
SEED = 0
TFIDF = dict(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True)

_GREEK_ONLY = re.compile(r"[^Ͱ-Ͽἀ-῿\s]")


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def read_labels(path):
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
    rows = [
        sp.csr_matrix(normalize(vec.transform(docs)).mean(axis=0)) for docs in doc_lists
    ]
    return normalize(sp.vstack(rows).tocsr())


def predict_rates(L_fit, R_fit, L_query, alpha):
    vec = fit_vectoriser(L_fit)
    model = Ridge(alpha=alpha).fit(pool(vec, L_fit), R_fit)
    return model.predict(pool(vec, L_query))


def to_counts(rates, hidden):
    scaled = np.clip(rates, 0.0, 1.0) * hidden[:, None]
    return np.clip(np.rint(scaled), 0.0, hidden[:, None])


def bray_curtis(p, y):
    denom = p.sum() + y.sum()
    return 1.0 - np.abs(p - y).sum() / denom if denom > 0 else 1.0


def mean_bray_curtis(P, Y):
    return float(np.mean([bray_curtis(P[i], Y[i]) for i in range(len(Y))]))


def cross_val_rates(L, R, alpha, n_folds=N_FOLDS, seed=SEED):
    oof = np.zeros((len(L), N_GENRES))
    splitter = KFold(n_folds, shuffle=True, random_state=seed)
    for fit_idx, held_idx in splitter.split(np.arange(len(L))):
        oof[held_idx] = predict_rates(
            [L[i] for i in fit_idx], R[fit_idx], [L[i] for i in held_idx], alpha
        )
    return oof


def select_alpha(L, R, hidden, Y):
    scores = {}
    for alpha in ALPHA_GRID:
        oof = cross_val_rates(L, R, alpha)
        scores[alpha] = mean_bray_curtis(to_counts(oof, hidden), Y)
        print("  alpha %-5s cross-validated Bray-Curtis %.4f" % (alpha, scores[alpha]))
    best = max(scores, key=scores.get)
    print("  selected alpha %s" % best)
    return best, scores[best]


def write_submission(path, rows, counts):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_id"] + GENRE_COLS)
        for row, values in zip(rows, counts):
            writer.writerow([row["query_id"]] + [int(v) for v in values])


def check_submission(path, rows, counts, hidden, data_dir):
    ids = [r["query_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate query_id in submission")
    sample = os.path.join(data_dir, "sample_submission.csv")
    if os.path.exists(sample):
        with open(sample, encoding="utf-8") as fh:
            expected = [row["query_id"] for row in csv.DictReader(fh)]
        if sorted(expected) != sorted(ids):
            raise ValueError("submission query_id set does not match sample_submission")
    if counts.min() < 0 or (counts > hidden[:, None]).any():
        raise ValueError("genre count outside [0, hidden_count]")
    if not np.isfinite(counts).all():
        raise ValueError("non-finite genre count")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="submission.csv")
    parser.add_argument("--alpha", type=float, default=None)
    args = parser.parse_args()

    train = read_jsonl(os.path.join(args.data, "train.jsonl"))
    valid = read_jsonl(os.path.join(args.data, "validation.jsonl"))
    test = read_jsonl(os.path.join(args.data, "test.jsonl"))
    y_train = read_labels(os.path.join(args.data, "train_labels.csv"))
    y_valid = read_labels(os.path.join(args.data, "validation_labels.csv"))

    L_train, L_valid, L_test = docs_of(train), docs_of(valid), docs_of(test)
    h_train, h_valid, h_test = hidden_counts(train), hidden_counts(valid), hidden_counts(test)
    Y_train = np.array([y_train[r["query_id"]] for r in train])
    Y_valid = np.array([y_valid[r["query_id"]] for r in valid])
    R_train = Y_train / h_train[:, None]
    R_valid = Y_valid / h_valid[:, None]

    print("manuscripts: %d train, %d validation, %d test" % (len(train), len(valid), len(test)))

    print("selecting the ridge penalty by %d-fold cross-validation on train" % N_FOLDS)
    if args.alpha is None:
        alpha, cv_score = select_alpha(L_train, R_train, h_train, Y_train)
    else:
        alpha = args.alpha
        cv_score = mean_bray_curtis(
            to_counts(cross_val_rates(L_train, R_train, alpha), h_train), Y_train
        )
        print("  alpha %s cross-validated Bray-Curtis %.4f" % (alpha, cv_score))

    valid_rates = predict_rates(L_train, R_train, L_valid, alpha)
    valid_score = mean_bray_curtis(to_counts(valid_rates, h_valid), Y_valid)
    print("held-out validation Bray-Curtis %.4f" % valid_score)
    print("train cross-validated Bray-Curtis %.4f" % cv_score)

    L_fit = L_train + L_valid
    R_fit = np.vstack([R_train, R_valid])
    test_rates = predict_rates(L_fit, R_fit, L_test, alpha)
    counts = to_counts(test_rates, h_test).astype(int)

    write_submission(args.out, test, counts)
    check_submission(args.out, test, counts, h_test, args.data)
    print("wrote %s (%d rows)" % (args.out, len(test)))


if __name__ == "__main__":
    sys.exit(main())
