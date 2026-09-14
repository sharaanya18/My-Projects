import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import random
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

SEED = 0
TE_FOLD_SEED = 12345
TE_FOLDS = 5

random.seed(SEED)
np.random.seed(SEED)

LABELS = np.array([1, 2, 3])

TOKEN_FIELDS = [
    "report_tokens",
    "location_tokens",
    "problem_part_tokens",
    "tagged_problem_tokens",
    "problem_type_tokens",
]

COUNT_FIELDS = [
    "report_token_count",
    "location_token_count",
    "problem_part_token_count",
    "tagged_problem_token_count",
    "field_coverage",
]

TE_KEYS = [
    "report_tokens",
    "location_tokens",
    "problem_part_tokens",
    "tagged_problem_tokens",
    "report_x_location",
    "report_x_part",
]

MODEL_CONFIGS = [
    (0.10, 5.0, (1, 2), 2),
    (0.15, 10.0, (1, 2), 2),
    (0.20, 20.0, (1, 2), 2),
    (0.15, 10.0, (1, 1), 1),
    (0.15, 10.0, (1, 3), 3),
]

ORDINAL = 1.0 - np.abs(LABELS[:, None] - LABELS[None, :]) / 2.0

PREDICTION_KEY = "breadth"

def add_key_crossings(frame):
    frame = frame.copy()
    for field in TOKEN_FIELDS:
        frame[field] = frame[field].fillna("")
    frame["report_x_location"] = frame["report_tokens"] + " || " + frame["location_tokens"]
    frame["report_x_part"] = frame["report_tokens"] + " || " + frame["problem_part_tokens"]
    return frame

def load_data(public_dir):
    train = pd.read_csv(public_dir / "train.csv")
    targets = pd.read_csv(public_dir / "train_targets.csv")
    test = pd.read_csv(public_dir / "test.csv")
    train = train.merge(targets[["id", "target"]], on="id", how="inner", validate="one_to_one")
    return add_key_crossings(train), add_key_crossings(test)

class GroupTargetEncoder:

    def __init__(self, keys, smoothing):
        self.keys = list(keys)
        self.smoothing = float(smoothing)

    def fit(self, frame, y):
        self.priors_ = np.array([(y == c).mean() for c in LABELS])
        self.tables_ = {}
        for key in self.keys:
            levels, inverse = np.unique(frame[key].fillna("").to_numpy(), return_inverse=True)
            counts = np.zeros((len(levels), 3))
            np.add.at(counts, (inverse, y - 1), 1.0)
            self.tables_[key] = ({v: i for i, v in enumerate(levels)}, counts)
        return self

    def transform(self, frame, y=None):
        blocks = []
        for key in self.keys:
            lookup, counts = self.tables_[key]
            index = np.array([lookup.get(v, -1) for v in frame[key].fillna("").to_numpy()])
            observed = np.where((index >= 0)[:, None], counts[np.clip(index, 0, None)], 0.0)
            if y is not None:
                observed = observed - np.eye(3)[y - 1]
            observed = np.maximum(observed, 0.0)
            total = observed.sum(axis=1, keepdims=True)
            encoded = (observed + self.smoothing * self.priors_[None, :]) / (total + self.smoothing)
            blocks.append(np.hstack([encoded, np.log1p(total)]))
        return np.hstack(blocks)

def encode_targets(train, test, y_train, keys, smoothing):
    full = GroupTargetEncoder(keys, smoothing).fit(train, y_train)
    enc_test = full.transform(test)

    present = np.array([(y_train == c).sum() for c in LABELS])
    n_splits = int(min(TE_FOLDS, present[present > 0].min()))
    if n_splits < 2:
        return full.transform(train, y_train), enc_test

    enc_train = np.zeros((len(train), enc_test.shape[1]))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=TE_FOLD_SEED)
    for fit_idx, held_idx in splitter.split(train, y_train):
        inner = GroupTargetEncoder(keys, smoothing).fit(train.iloc[fit_idx], y_train[fit_idx])
        enc_train[held_idx] = inner.transform(train.iloc[held_idx])
    return enc_train, enc_test

def build_matrices(train, test, y_train, ngram_report, min_df, smoothing):
    train_blocks, test_blocks = [], []

    for field in TOKEN_FIELDS:
        ngram = ngram_report if field == "report_tokens" else (1, 1)
        vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",
            ngram_range=ngram,
            min_df=min_df,
            sublinear_tf=True,
        )
        try:
            train_blocks.append(vec.fit_transform(train[field]))
        except ValueError:
            continue
        test_blocks.append(vec.transform(test[field]))

    counts_train = train[COUNT_FIELDS].to_numpy(float)
    counts_test = test[COUNT_FIELDS].to_numpy(float)
    count_scaler = StandardScaler().fit(counts_train)
    train_blocks.append(sparse.csr_matrix(count_scaler.transform(counts_train)))
    test_blocks.append(sparse.csr_matrix(count_scaler.transform(counts_test)))

    enc_train, enc_test = encode_targets(train, test, y_train, TE_KEYS, smoothing)
    enc_scaler = StandardScaler().fit(enc_train)
    train_blocks.append(sparse.csr_matrix(enc_scaler.transform(enc_train)))
    test_blocks.append(sparse.csr_matrix(enc_scaler.transform(enc_test)))

    return sparse.hstack(train_blocks).tocsr(), sparse.hstack(test_blocks).tocsr()

def align_columns(model, probabilities):
    aligned = np.zeros((probabilities.shape[0], len(LABELS)))
    for column, label in enumerate(model.classes_):
        aligned[:, int(label) - 1] = probabilities[:, column]
    return aligned

def predict_proba(train, test, y_train):
    probabilities = np.zeros((len(test), len(LABELS)))
    for C, smoothing, ngram_report, min_df in MODEL_CONFIGS:
        X_train, X_test = build_matrices(train, test, y_train, ngram_report, min_df, smoothing)
        model = LogisticRegression(max_iter=5000, C=C, random_state=SEED)
        model.fit(X_train, y_train)
        probabilities += align_columns(model, model.predict_proba(X_test))
    return probabilities / len(MODEL_CONFIGS)

def decide(probabilities, priors):
    utility = (
        0.75 / 3.0 * (probabilities / np.maximum(priors, 1e-9)[None, :])
        + 0.25 * (probabilities @ ORDINAL.T)
    )
    return LABELS[np.argmax(utility, axis=1)]

def fallback_proba(train, test, y_train):
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", sublinear_tf=True)
    X_train = vec.fit_transform(train["report_tokens"])
    X_test = vec.transform(test["report_tokens"])
    model = LogisticRegression(max_iter=5000, C=0.15, random_state=SEED)
    model.fit(X_train, y_train)
    return align_columns(model, model.predict_proba(X_test))

def decode_prediction(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} predictions must be JSON objects")
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} predictions must be valid JSON") from exc
    if not isinstance(decoded, dict) or set(decoded) != {PREDICTION_KEY}:
        raise ValueError(f'{name} predictions must have exactly the "breadth" key')
    raw = decoded[PREDICTION_KEY]
    if isinstance(raw, bool) or not isinstance(raw, (int, np.integer)):
        raise ValueError(f'{name} "breadth" must be an integer')
    breadth = int(raw)
    if breadth not in LABELS:
        raise ValueError(f"{name} breadth must be one of {LABELS.tolist()}")
    return breadth

def build_submission(ids, predictions):
    return pd.DataFrame(
        {
            "id": np.asarray(ids),
            "prediction": [
                json.dumps({PREDICTION_KEY: int(b)}, separators=(",", ":")) for b in predictions
            ],
        }
    )

def write_submission(submission, test_ids, out_path):
    for value in submission["prediction"]:
        decode_prediction(value, "submission")
    if submission["id"].duplicated().any():
        raise ValueError("submission contains duplicate ids")
    if list(submission["id"]) != list(test_ids):
        raise ValueError("submission ids do not match the test ids")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=str(out_path.parent), suffix=".csv")
    os.close(handle)
    submission.to_csv(temp_name, index=False)
    os.replace(temp_name, out_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("public_dir", type=Path)
    parser.add_argument("submission_out", type=Path)
    args = parser.parse_args()

    test_ids = pd.read_csv(args.public_dir / "test.csv")["id"].to_numpy()
    write_submission(build_submission(test_ids, np.ones(len(test_ids), int)), test_ids, args.submission_out)

    train, test = load_data(args.public_dir)
    y = train["target"].to_numpy(int)
    priors = np.array([(y == c).mean() for c in LABELS])
    print(f"train={train.shape[0]} rows  test={test.shape[0]} rows  priors={np.round(priors, 4)}")

    try:
        probabilities = predict_proba(train, test, y)
    except Exception:
        traceback.print_exc()
        print("main pipeline failed, falling back to report tokens only")
        try:
            probabilities = fallback_proba(train, test, y)
        except Exception:
            traceback.print_exc()
            print("fallback failed, keeping the placeholder submission")
            return

    predictions = decide(probabilities, priors)
    write_submission(build_submission(test["id"].to_numpy(), predictions), test_ids, args.submission_out)

    counts = pd.Series(predictions).value_counts().sort_index()
    print(f"wrote {args.submission_out} ({len(predictions)} rows)")
    print("predicted breadth distribution:", {int(k): int(v) for k, v in counts.items()})

if __name__ == "__main__":
    main()
