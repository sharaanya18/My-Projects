"""Privacy-Preserving Maintenance Work-Breadth Estimation.

Predicts a breadth code (1, 2 or 3+ distinct corrective components) for each
held-out problem report from salted-token problem-side features only.

Approach
--------
1. Features: per-field TF-IDF over the opaque token identifiers (word analyzer
   on whitespace, so hashed tokens stay intact; 1-2 grams on the report body
   to keep some compositional signal), the five numeric count columns, and
   leave-one-out smoothed target encodings of the categorical key fields.
   The token normalisation collapses numerals and side markers, so distinct
   reports often share an identical token string; the group encodings turn
   that collision structure into usable evidence without any source-text
   lookup.
2. Model: multinomial logistic regression, trained *unweighted* so its
   outputs are true posteriors. Probabilities are averaged over a handful of
   regularisation / smoothing / n-gram variants to reduce variance.
3. Decision: labels are not argmax. The grader's utility is known in closed
   form, so each report is assigned the label maximising expected score:

       U(k) = 0.75/3 * P(k|x) / N_k + 0.25/N * sum_c P(c|x) * (1 - |k-c|/2)

   with N_k the expected number of true class-k reports in the evaluation set
   (train priors x test size; the split is stratified). The 1/N_k term is what
   buys minority-class recall, and the ordinal term discourages 1<->3 flips.
   This rule is parameter-free -- a tuned variant was tried and lost under
   nested CV (0.629 vs 0.639), so the closed form is used as-is.

Reproducible from the supplied public files alone; no external data or
network calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path("dataset/public")
OUT_PATH = Path("working/submission.csv")

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
# Categorical keys for group target encoding: the four informative annotation
# fields plus two crossings with the report body.
TE_KEYS = [
    "report_tokens",
    "location_tokens",
    "problem_part_tokens",
    "tagged_problem_tokens",
    "report_x_location",
    "report_x_part",
]

# (C, target-encoding smoothing, report n-gram range, min_df)
MODEL_CONFIGS = [
    (0.10, 5.0, (1, 2), 2),
    (0.15, 10.0, (1, 2), 2),
    (0.20, 20.0, (1, 2), 2),
    (0.15, 10.0, (1, 1), 1),
    (0.15, 10.0, (1, 3), 3),
]

# Ordinal credit matrix: ORDINAL[k, c] is the utility of predicting k when the
# truth is c (1.0 exact, 0.5 adjacent, 0.0 for confusing 1 with 3).
ORDINAL = 1.0 - np.abs(LABELS[:, None] - LABELS[None, :]) / 2.0


def load_data(data_dir: Path = DATA_DIR):
    """Load train features, the target sidecar and test features."""
    train = pd.read_csv(data_dir / "train.csv")
    targets = pd.read_csv(data_dir / "train_targets.csv")
    test = pd.read_csv(data_dir / "test.csv")
    # Join on id -- row order is not assumed to match.
    train = train.merge(targets[["id", "target"]], on="id", how="inner", validate="one_to_one")
    return add_key_crossings(train), add_key_crossings(test)


def add_key_crossings(frame: pd.DataFrame) -> pd.DataFrame:
    """Fill missing token fields and add crossed categorical keys."""
    frame = frame.copy()
    for field in TOKEN_FIELDS:
        frame[field] = frame[field].fillna("")
    frame["report_x_location"] = frame["report_tokens"] + " || " + frame["location_tokens"]
    frame["report_x_part"] = frame["report_tokens"] + " || " + frame["problem_part_tokens"]
    return frame


class GroupTargetEncoder:
    """Smoothed per-class target encoding of categorical keys.

    Training rows are encoded leave-one-out so a report never sees its own
    label; unseen keys fall back to the class priors.
    """

    def __init__(self, keys, smoothing: float):
        self.keys = list(keys)
        self.smoothing = float(smoothing)

    def fit(self, frame: pd.DataFrame, y: np.ndarray) -> "GroupTargetEncoder":
        self.priors_ = np.array([(y == c).mean() for c in LABELS])
        self.tables_ = {}
        for key in self.keys:
            values = frame[key].fillna("").to_numpy()
            levels, inverse = np.unique(values, return_inverse=True)
            counts = np.zeros((len(levels), 3))
            np.add.at(counts, (inverse, y - 1), 1.0)
            self.tables_[key] = ({v: i for i, v in enumerate(levels)}, counts)
        return self

    def transform(self, frame: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
        blocks = []
        for key in self.keys:
            lookup, counts = self.tables_[key]
            index = np.array([lookup.get(v, -1) for v in frame[key].fillna("").to_numpy()])
            seen = index >= 0
            observed = np.where(seen[:, None], counts[np.clip(index, 0, None)], 0.0)
            if y is not None:  # leave-one-out on the fitting frame
                observed = observed - np.eye(3)[y - 1]
            observed = np.maximum(observed, 0.0)
            total = observed.sum(axis=1, keepdims=True)
            encoded = (observed + self.smoothing * self.priors_[None, :]) / (total + self.smoothing)
            blocks.append(np.hstack([encoded, np.log1p(total)]))
        return np.hstack(blocks)


def build_matrices(train, test, y_train, ngram_report, min_df, smoothing):
    """TF-IDF blocks + scaled counts + target encodings for one config."""
    train_blocks, test_blocks = [], []
    for field in TOKEN_FIELDS:
        ngram = ngram_report if field == "report_tokens" else (1, 1)
        vec = TfidfVectorizer(
            analyzer="word",
            token_pattern=r"\S+",  # hashed tokens are whitespace-delimited
            ngram_range=ngram,
            min_df=min_df,
            sublinear_tf=True,
        )
        train_blocks.append(vec.fit_transform(train[field]))
        test_blocks.append(vec.transform(test[field]))

    count_scaler = StandardScaler().fit(train[COUNT_FIELDS].to_numpy(float))
    train_blocks.append(sparse.csr_matrix(count_scaler.transform(train[COUNT_FIELDS].to_numpy(float))))
    test_blocks.append(sparse.csr_matrix(count_scaler.transform(test[COUNT_FIELDS].to_numpy(float))))

    encoder = GroupTargetEncoder(TE_KEYS, smoothing).fit(train, y_train)
    enc_train = encoder.transform(train, y_train)
    enc_test = encoder.transform(test)
    enc_scaler = StandardScaler().fit(enc_train)
    train_blocks.append(sparse.csr_matrix(enc_scaler.transform(enc_train)))
    test_blocks.append(sparse.csr_matrix(enc_scaler.transform(enc_test)))

    return sparse.hstack(train_blocks).tocsr(), sparse.hstack(test_blocks).tocsr()


def predict_proba(train, test, y_train) -> np.ndarray:
    """Posterior probabilities averaged over the model configs."""
    probabilities = np.zeros((len(test), 3))
    for C, smoothing, ngram_report, min_df in MODEL_CONFIGS:
        X_train, X_test = build_matrices(train, test, y_train, ngram_report, min_df, smoothing)
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train, y_train)
        probabilities += model.predict_proba(X_test)
    return probabilities / len(MODEL_CONFIGS)


def decide(probabilities: np.ndarray, priors: np.ndarray, n_eval: int) -> np.ndarray:
    """Assign the breadth code maximising expected grader utility."""
    expected_counts = np.maximum(priors * n_eval, 1e-9)
    utility = (
        0.75 / 3.0 * (probabilities / expected_counts[None, :])
        + 0.25 / n_eval * (probabilities @ ORDINAL.T)
    )
    return LABELS[np.argmax(utility, axis=1)]


def grader_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Reimplementation of the official metric, for local validation."""
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    per_class = []
    for c in LABELS:
        mask = y_true == c
        per_class.append(float((y_pred[mask] == c).mean()) if mask.any() else 0.0)
    balanced = float(np.mean(per_class))
    ordinal = float(np.clip(1.0 - np.abs(y_true - y_pred) / 2.0, 0.0, 1.0).mean())
    return 0.75 * balanced + 0.25 * ordinal, balanced, ordinal, per_class


def cross_validate(train, y, priors, seeds=(0, 1, 2), n_splits=5) -> None:
    """Repeated stratified CV under the official metric."""
    scores = []
    for seed in seeds:
        oof = np.zeros((len(y), 3))
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fit_idx, val_idx in splitter.split(train, y):
            oof[val_idx] = predict_proba(train.iloc[fit_idx], train.iloc[val_idx], y[fit_idx])
        total, balanced, ordinal, per_class = grader_score(y, decide(oof, priors, len(y)))
        scores.append(total)
        print(
            f"  seed {seed}: score={total:.4f} balanced={balanced:.4f} "
            f"ordinal={ordinal:.4f} per-class={[round(p, 3) for p in per_class]}"
        )
    print(f"  mean score over {len(seeds)} seed(s): {np.mean(scores):.4f} (sd {np.std(scores):.4f})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--cv", type=int, default=0, help="number of CV seeds to run before fitting (0 skips)")
    args = parser.parse_args()

    train, test = load_data(args.data_dir)
    y = train["target"].to_numpy(int)
    priors = np.array([(y == c).mean() for c in LABELS])
    print(f"train={train.shape[0]} rows  test={test.shape[0]} rows  priors={np.round(priors, 4)}")

    if args.cv:
        print(f"cross-validating ({args.cv} seed(s))...")
        cross_validate(train, y, priors, seeds=tuple(range(args.cv)))

    probabilities = predict_proba(train, test, y)
    predictions = decide(probabilities, priors, len(test))

    submission = pd.DataFrame(
        {
            "id": test["id"].to_numpy(),
            "prediction": [json.dumps({"breadth": int(b)}, separators=(",", ":")) for b in predictions],
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.out, index=False)

    counts = pd.Series(predictions).value_counts().sort_index()
    print(f"wrote {args.out} ({len(submission)} rows)")
    print("predicted breadth distribution:", {int(k): int(v) for k, v in counts.items()})


if __name__ == "__main__":
    main()
