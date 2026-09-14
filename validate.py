import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import solution as sol

SEED = 0
HOLDOUT_SEED = 20240601
random.seed(SEED)
np.random.seed(SEED)

LABELS = np.array([1, 2, 3], dtype=int)
REQUIRED_COLUMNS = ["id", "prediction"]
PREDICTION_KEY = "breadth"

def _decode_prediction(value, name):
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

def _validated_predictions(frame, name):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"{name} must be a pandas DataFrame")
    if set(frame.columns) != set(REQUIRED_COLUMNS):
        raise ValueError(f"{name} must contain exactly {REQUIRED_COLUMNS}")
    if len(frame) == 0:
        raise ValueError(f"{name} must not be empty")
    if frame["id"].isna().any() or frame["id"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{name} contains a blank id")
    if frame["id"].duplicated().any():
        raise ValueError(f"{name} contains duplicate ids")
    breadths = [_decode_prediction(value, name) for value in frame["prediction"]]
    checked = frame[["id"]].copy()
    checked["_breadth"] = breadths
    return checked

def _balanced_breadth_coverage(y_true, y_pred):
    per_breadth = []
    for breadth in LABELS:
        mask = y_true == breadth
        per_breadth.append(float(np.mean(y_pred[mask] == breadth)) if mask.any() else 0.0)
    return float(np.mean(per_breadth))

def grade(submission, answers):
    submitted = _validated_predictions(submission, "submission")
    answer_columns = [column for column in REQUIRED_COLUMNS if column in answers.columns]
    if answer_columns != REQUIRED_COLUMNS:
        raise ValueError("answers must contain id and prediction columns")
    expected = _validated_predictions(answers[REQUIRED_COLUMNS], "answers")
    if len(submitted) != len(expected) or set(submitted["id"]) != set(expected["id"]):
        raise ValueError("submission ids must match answers ids exactly")
    aligned = expected.merge(submitted, on="id", how="inner", validate="one_to_one")
    y_true = aligned["_breadth_x"].to_numpy(dtype=int)
    y_pred = aligned["_breadth_y"].to_numpy(dtype=int)
    balanced_coverage = _balanced_breadth_coverage(y_true, y_pred)
    ordinal_utility = np.clip(1.0 - np.abs(y_true - y_pred) / 2.0, 0.0, 1.0).mean()
    score = 0.75 * balanced_coverage + 0.25 * float(ordinal_utility)
    return float(np.clip(score, 0.0, 1.0))

def score_predictions(ids, y_true, y_pred):
    to_frame = lambda values: pd.DataFrame(
        {"id": ids, "prediction": [json.dumps({PREDICTION_KEY: int(v)}, separators=(",", ":")) for v in values]}
    )
    total = grade(to_frame(y_pred), to_frame(y_true))
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    recall = [float((y_pred[y_true == c] == c).mean()) if (y_true == c).any() else 0.0 for c in LABELS]
    confusion = np.zeros((3, 3), int)
    for t, p in zip(y_true, y_pred):
        confusion[t - 1, p - 1] += 1
    distribution = {int(c): int((y_pred == c).sum()) for c in LABELS}
    return total, recall, confusion, distribution

ANNOTATION_KEYS = [
    "location_tokens",
    "problem_part_tokens",
    "tagged_problem_tokens",
]
EXTRA_CROSS_KEYS = ["part_x_type", "part_x_tagged", "location_x_part"]

SHARED_FIELDS = [
    "location_tokens",
    "problem_part_tokens",
    "tagged_problem_tokens",
    "problem_type_tokens",
]

def add_extra_crossings(frame):
    frame = frame.copy()
    frame["part_x_type"] = frame["problem_part_tokens"] + " || " + frame["problem_type_tokens"]
    frame["part_x_tagged"] = frame["problem_part_tokens"] + " || " + frame["tagged_problem_tokens"]
    frame["location_x_part"] = frame["location_tokens"] + " || " + frame["problem_part_tokens"]
    return frame

def structural_features(frame):
    columns = []
    report_tokens = [str(s).split() for s in frame["report_tokens"]]
    report_sets = [set(t) for t in report_tokens]
    for field in sol.TOKEN_FIELDS:
        tokens = [str(s).split() for s in frame[field]]
        columns.append([len(set(t)) for t in tokens])
        columns.append([1.0 if t else 0.0 for t in tokens])
    columns.append([(len(t) - len(set(t))) / len(t) if t else 0.0 for t in report_tokens])
    for field in SHARED_FIELDS:
        other = [set(str(s).split()) for s in frame[field]]
        columns.append([len(a & b) for a, b in zip(report_sets, other)])
    return np.asarray(columns, dtype=float).T

BACKOFF_PARENTS = {
    "report_x_location": "report_tokens",
    "report_x_part": "report_tokens",
    "report_tokens": None,
    "location_tokens": None,
    "problem_part_tokens": None,
    "tagged_problem_tokens": None,
    "report_cluster": None,
}


class BackoffTargetEncoder:
    def __init__(self, keys, smoothing, parents=None, with_counts=False):
        self.keys = list(keys)
        self.smoothing = float(smoothing)
        self.parents = dict(parents or BACKOFF_PARENTS)
        self.with_counts = with_counts

    def fit(self, frame, y):
        self.priors_ = np.array([(y == c).mean() for c in LABELS])
        self.tables_ = {}
        for key in self.keys:
            levels, inverse = np.unique(frame[key].fillna("").to_numpy(), return_inverse=True)
            counts = np.zeros((len(levels), 3))
            np.add.at(counts, (inverse, y - 1), 1.0)
            self.tables_[key] = ({v: i for i, v in enumerate(levels)}, counts)
        return self

    def _observed(self, frame, key, y):
        lookup, counts = self.tables_[key]
        index = np.array([lookup.get(v, -1) for v in frame[key].fillna("").to_numpy()])
        observed = np.where((index >= 0)[:, None], counts[np.clip(index, 0, None)], 0.0)
        if y is not None:
            observed = observed - np.eye(3)[y - 1]
        return np.maximum(observed, 0.0)

    def transform(self, frame, y=None):
        observed = {key: self._observed(frame, key, y) for key in self.keys}
        estimates = {}
        blocks = []
        for key in self.keys:
            parent = self.parents.get(key)
            if parent is not None and parent in estimates:
                target = estimates[parent]
            else:
                target = np.repeat(self.priors_[None, :], len(frame), axis=0)
            total = observed[key].sum(axis=1, keepdims=True)
            estimate = (observed[key] + self.smoothing * target) / (total + self.smoothing)
            estimates[key] = estimate
            block = [estimate, np.log1p(total)]
            if self.with_counts:
                block.append(1.0 / (1.0 + total))
                block.append((total > 0).astype(float))
            blocks.append(np.hstack(block))
        return np.hstack(blocks)


def backoff_encode(train, evaluate, y_train, keys, smoothing, with_counts=False, hierarchical=True):
    parents = None if hierarchical else {k: None for k in keys}
    make = lambda: BackoffTargetEncoder(keys, smoothing, parents=parents, with_counts=with_counts)
    full = make().fit(train, y_train)
    enc_eval = full.transform(evaluate)
    present = np.array([(y_train == c).sum() for c in LABELS])
    n_splits = int(min(5, present[present > 0].min()))
    if n_splits < 2:
        return full.transform(train, y_train), enc_eval
    enc_train = np.zeros((len(train), enc_eval.shape[1]))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=sol.TE_FOLD_SEED)
    for fit_idx, held_idx in splitter.split(train, y_train):
        inner = make().fit(train.iloc[fit_idx], y_train[fit_idx])
        enc_train[held_idx] = inner.transform(train.iloc[held_idx])
    return enc_train, enc_eval


def encode_targets(train, evaluate, y_train, keys, smoothing, mode):
    if mode == "backoff":
        return backoff_encode(train, evaluate, y_train, keys, smoothing)
    if mode == "backoff_counts":
        return backoff_encode(train, evaluate, y_train, keys, smoothing, with_counts=True)
    if mode == "oof_counts":
        return backoff_encode(train, evaluate, y_train, keys, smoothing,
                              with_counts=True, hierarchical=False)
    if mode == "oof":
        return sol.encode_targets(train, evaluate, y_train, keys, smoothing)
    if mode.startswith("oof_multi"):
        levels = [2.0, 10.0, 50.0]
        parts = [sol.encode_targets(train, evaluate, y_train, keys, lvl) for lvl in levels]
        return np.hstack([p[0] for p in parts]), np.hstack([p[1] for p in parts])
    if mode.startswith("oof_bag"):
        repeats = int(mode.split("_bag")[1] or 5)
        full = sol.GroupTargetEncoder(keys, smoothing).fit(train, y_train)
        enc_eval = full.transform(evaluate)
        stacked = []
        for offset in range(repeats):
            enc_train = np.zeros((len(train), enc_eval.shape[1]))
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=sol.TE_FOLD_SEED + offset)
            for fit_idx, held_idx in splitter.split(train, y_train):
                inner = sol.GroupTargetEncoder(keys, smoothing).fit(train.iloc[fit_idx], y_train[fit_idx])
                enc_train[held_idx] = inner.transform(train.iloc[held_idx])
            stacked.append(enc_train)
        return np.mean(stacked, axis=0), enc_eval
    full = sol.GroupTargetEncoder(keys, smoothing).fit(train, y_train)
    return full.transform(train, y_train), full.transform(evaluate)

def assign_clusters(train, evaluate, threshold=0.8, max_posting=200):
    train_sets = [set(str(v).split()) for v in train["report_tokens"]]
    postings = {}
    for i, tokens in enumerate(train_sets):
        for t in tokens:
            postings.setdefault(t, []).append(i)
    parent = list(range(len(train_sets)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def candidates(tokens):
        found = set()
        for t in tokens:
            posting = postings.get(t)
            if posting is not None and len(posting) <= max_posting:
                found.update(posting)
        return found

    def similarity(a, b):
        union = len(a | b)
        return len(a & b) / union if union else 0.0

    for i, tokens in enumerate(train_sets):
        for j in candidates(tokens):
            if j <= i:
                continue
            if similarity(tokens, train_sets[j]) >= threshold:
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)

    train_labels = np.array([f"c{find(i)}" for i in range(len(train_sets))])
    eval_labels = []
    for value in evaluate["report_tokens"]:
        tokens = set(str(value).split())
        best, best_score = "", threshold
        for j in candidates(tokens):
            score = similarity(tokens, train_sets[j])
            if score >= best_score:
                best, best_score = train_labels[j], score
        eval_labels.append(best)
    return train_labels, np.array(eval_labels)


class FoldFeatures:

    def __init__(self, train, evaluate, y_train, variant):
        self.variant = variant
        self.te_keys = list(variant["te_keys"])
        threshold = variant.get("cluster")
        if threshold:
            train = train.copy()
            evaluate = evaluate.copy()
            train["report_cluster"], evaluate["report_cluster"] = assign_clusters(
                train, evaluate, threshold
            )
            self.te_keys.append("report_cluster")
        self.train = train
        self.evaluate = evaluate
        self.y_train = y_train
        self._tfidf = {}
        self._encodings = {}

    def tfidf(self, ngram_report, min_df):
        key = (ngram_report, min_df)
        if key not in self._tfidf:
            train_blocks, eval_blocks = [], []
            for field in sol.TOKEN_FIELDS:
                ngram = ngram_report if field == "report_tokens" else (1, 1)
                vec = TfidfVectorizer(
                    analyzer="word", token_pattern=r"\S+", ngram_range=ngram,
                    min_df=min_df, sublinear_tf=True,
                )
                try:
                    train_blocks.append(vec.fit_transform(self.train[field]))
                except ValueError:
                    continue
                eval_blocks.append(vec.transform(self.evaluate[field]))
            self._tfidf[key] = (train_blocks, eval_blocks)
        return self._tfidf[key]

    def encodings(self, smoothing):
        if smoothing not in self._encodings:
            self._encodings[smoothing] = encode_targets(
                self.train, self.evaluate, self.y_train,
                self.te_keys, smoothing, self.variant["te_mode"],
            )
        return self._encodings[smoothing]

    def dense(self, smoothing):
        enc_train, enc_eval = self.encodings(smoothing)
        train_parts = [enc_train, self.train[sol.COUNT_FIELDS].to_numpy(float)]
        eval_parts = [enc_eval, self.evaluate[sol.COUNT_FIELDS].to_numpy(float)]
        if self.variant.get("structural"):
            train_parts.append(structural_features(self.train))
            eval_parts.append(structural_features(self.evaluate))
        return np.hstack(train_parts), np.hstack(eval_parts)

    def matrices(self, ngram_report, min_df, smoothing):
        train_blocks, eval_blocks = self.tfidf(ngram_report, min_df)
        dense_train, dense_eval = self.dense(smoothing)
        scaler = StandardScaler().fit(dense_train)
        return (
            sparse.hstack(train_blocks + [sparse.csr_matrix(scaler.transform(dense_train))]).tocsr(),
            sparse.hstack(eval_blocks + [sparse.csr_matrix(scaler.transform(dense_eval))]).tocsr(),
        )

def logistic_probabilities(features, y_train, n_eval):
    probabilities = np.zeros((n_eval, 3))
    for C, smoothing, ngram_report, min_df in sol.MODEL_CONFIGS:
        X_train, X_eval = features.matrices(ngram_report, min_df, smoothing)
        model = LogisticRegression(max_iter=5000, C=C, random_state=SEED)
        model.fit(X_train, y_train)
        probabilities += sol.align_columns(model, model.predict_proba(X_eval))
    return probabilities / len(sol.MODEL_CONFIGS)

def lightgbm_probabilities(features, y_train, smoothing=10.0, n_components=100):
    import lightgbm as lgb

    dense_train, dense_eval = features.dense(smoothing)
    vec = TfidfVectorizer(analyzer="word", token_pattern=r"\S+", min_df=2, sublinear_tf=True)
    report_train = vec.fit_transform(features.train["report_tokens"])
    report_eval = vec.transform(features.evaluate["report_tokens"])
    svd = TruncatedSVD(n_components=min(n_components, report_train.shape[1] - 1), random_state=SEED)
    dense_train = np.hstack([dense_train, svd.fit_transform(report_train)])
    dense_eval = np.hstack([dense_eval, svd.transform(report_eval)])
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=3, n_estimators=400, learning_rate=0.05,
        num_leaves=15, min_child_samples=30, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.7, reg_lambda=5.0, random_state=SEED, n_jobs=1,
        deterministic=True, force_row_wise=True, verbose=-1,
    )
    model.fit(dense_train, y_train)
    return sol.align_columns(model, model.predict_proba(dense_eval))

def two_stage_probabilities(features, y_train, n_eval, stage2_C=None):
    upper = y_train > 1
    if upper.sum() < 2 or len(np.unique(y_train[upper])) < 2:
        return logistic_probabilities(features, y_train, n_eval)
    p_upper = np.zeros(n_eval)
    p_top = np.zeros(n_eval)
    for C, smoothing, ngram_report, min_df in sol.MODEL_CONFIGS:
        X_train, X_eval = features.matrices(ngram_report, min_df, smoothing)
        first = LogisticRegression(max_iter=5000, C=C, random_state=SEED)
        first.fit(X_train, upper.astype(int))
        p_upper += first.predict_proba(X_eval)[:, 1]
        second = LogisticRegression(max_iter=5000, C=C if stage2_C is None else stage2_C, random_state=SEED)
        second.fit(X_train[upper], (y_train[upper] == 3).astype(int))
        p_top += second.predict_proba(X_eval)[:, 1]
    n = len(sol.MODEL_CONFIGS)
    p_upper /= n
    p_top /= n
    return np.column_stack([1.0 - p_upper, p_upper * (1.0 - p_top), p_upper * p_top])


def fold_probabilities(train, evaluate, y_train, variant):
    features = FoldFeatures(train, evaluate, y_train, variant)
    if variant.get("model", "lr") == "two_stage":
        return two_stage_probabilities(features, y_train, len(evaluate), variant.get("stage2_C"))
    if variant.get("model", "lr") == "two_stage_blend":
        direct = logistic_probabilities(features, y_train, len(evaluate))
        staged = two_stage_probabilities(features, y_train, len(evaluate), variant.get("stage2_C"))
        return 0.5 * direct + 0.5 * staged
    probabilities = logistic_probabilities(features, y_train, len(evaluate))
    if variant.get("model") == "lr+lgbm":
        probabilities = 0.5 * probabilities + 0.5 * lightgbm_probabilities(features, y_train)
    return probabilities

BASE_KEYS = sol.TE_KEYS

VARIANTS = {
    "baseline": dict(te_mode="loo", te_keys=BASE_KEYS, model="lr"),
    "oof_te": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr"),
    "loo_annot": dict(te_mode="loo", te_keys=ANNOTATION_KEYS, model="lr"),
    "oof_annot": dict(te_mode="oof", te_keys=ANNOTATION_KEYS, model="lr"),
    "oof_struct": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr", structural=True),
    "oof_cross": dict(te_mode="oof", te_keys=BASE_KEYS + EXTRA_CROSS_KEYS, model="lr"),
    "oof_struct_cross": dict(
        te_mode="oof", te_keys=BASE_KEYS + EXTRA_CROSS_KEYS, model="lr", structural=True
    ),
    "backoff": dict(te_mode="backoff", te_keys=BASE_KEYS, model="lr"),
    "oof_counts": dict(te_mode="oof_counts", te_keys=BASE_KEYS, model="lr"),
    "backoff_counts": dict(te_mode="backoff_counts", te_keys=BASE_KEYS, model="lr"),
    "oof_cluster": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr", cluster=0.8),
    "backoff_cluster": dict(te_mode="backoff", te_keys=BASE_KEYS, model="lr", cluster=0.8),
    "oof_cluster90": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr", cluster=0.9),
    "oof_cluster95": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr", cluster=0.95),
    "two_stage": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage"),
    "two_stage_blend": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage_blend"),
    "two_stage_blend_c05": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage_blend", stage2_C=0.05),
    "two_stage_c05": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage", stage2_C=0.05),
    "two_stage_c10": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage", stage2_C=0.10),
    "two_stage_c30": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage", stage2_C=0.30),
    "two_stage_c100": dict(te_mode="oof", te_keys=BASE_KEYS, model="two_stage", stage2_C=1.00),
    "oof_multi": dict(te_mode="oof_multi", te_keys=BASE_KEYS, model="lr"),
    "oof_bag3": dict(te_mode="oof_bag3", te_keys=BASE_KEYS, model="lr"),
    "oof_bag5": dict(te_mode="oof_bag5", te_keys=BASE_KEYS, model="lr"),
    "oof_bag10": dict(te_mode="oof_bag10", te_keys=BASE_KEYS, model="lr"),
    "oof_lgbm": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr+lgbm"),
    "oof_struct_lgbm": dict(te_mode="oof", te_keys=BASE_KEYS, model="lr+lgbm", structural=True),
}

def load_dev_holdout(public_dir):
    train, _ = sol.load_data(public_dir)
    train = add_extra_crossings(train).reset_index(drop=True)
    y = train["target"].to_numpy(int)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=HOLDOUT_SEED)
    dev_idx, holdout_idx = next(splitter.split(train, y))
    return train.iloc[np.sort(dev_idx)].reset_index(drop=True), train.iloc[np.sort(holdout_idx)].reset_index(drop=True)

def _one_fold(dev, y, variant, fit_idx, val_idx):
    keys = dev["report_tokens"].to_numpy()
    matched = np.isin(keys[val_idx], keys[fit_idx])
    return val_idx, fold_probabilities(dev.iloc[fit_idx], dev.iloc[val_idx], y[fit_idx], variant), matched

def cross_validate(dev, variant, seeds, n_jobs):
    y = dev["target"].to_numpy(int)
    priors = np.array([(y == c).mean() for c in LABELS])
    per_seed = []
    for seed in seeds:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_fold)(dev, y, variant, fit_idx, val_idx)
            for fit_idx, val_idx in splitter.split(dev, y)
        )
        oof = np.zeros((len(y), 3))
        matched = np.zeros(len(y), bool)
        for val_idx, probabilities, match in results:
            oof[val_idx] = probabilities
            matched[val_idx] = match
        predictions = sol.decide(oof, priors)
        split_recall = {}
        for label, mask in (("matched", matched), ("unmatched", ~matched)):
            split_recall[label] = [
                float((predictions[(y == c) & mask] == c).mean()) if ((y == c) & mask).sum() else np.nan
                for c in LABELS
            ]
        per_seed.append((score_predictions(dev["id"].to_numpy(), y, predictions), oof, split_recall))
    return per_seed

def holdout_score(dev, holdout, variant):
    y = dev["target"].to_numpy(int)
    priors = np.array([(y == c).mean() for c in LABELS])
    probabilities = fold_probabilities(dev, holdout, y, variant)
    return score_predictions(
        holdout["id"].to_numpy(), holdout["target"].to_numpy(int), sol.decide(probabilities, priors)
    )

def report_calibration(dev, per_seed):
    y = dev["target"].to_numpy(int)
    oof = np.mean([r[1] for r in per_seed], axis=0)
    for c in LABELS:
        observed = float((y == c).mean())
        print(
            f"    class {c}: mean predicted={oof[:, c - 1].mean():.4f} observed={observed:.4f}"
            f"  ratio={oof[:, c - 1].mean() / max(observed, 1e-9):.3f}"
        )
        for low, high in ((0.0, 0.2), (0.2, 0.5), (0.5, 1.01)):
            band = (oof[:, c - 1] >= low) & (oof[:, c - 1] < high)
            if band.sum():
                print(
                    f"      p in [{low},{high}): n={int(band.sum()):5d} "
                    f"mean p={oof[band, c - 1].mean():.3f} actual={float((y[band] == c).mean()):.3f}"
                )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("public_dir", type=Path)
    parser.add_argument("--variants", default="baseline")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--holdout", action="store_true")
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    dev, holdout = load_dev_holdout(args.public_dir)
    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    print(f"dev={len(dev)} rows  holdout={len(holdout)} rows")

    seeds = tuple(range(args.seeds))
    baseline_scores = None
    for name in names:
        variant = VARIANTS[name]
        per_seed = cross_validate(dev, variant, seeds, args.n_jobs)
        scores = np.array([r[0][0] for r in per_seed])
        recall = np.mean([r[0][1] for r in per_seed], axis=0)
        confusion = np.sum([r[0][2] for r in per_seed], axis=0)
        distribution = per_seed[-1][0][3]
        matched_recall = np.nanmean([r[2]["matched"] for r in per_seed], axis=0)
        unmatched_recall = np.nanmean([r[2]["unmatched"] for r in per_seed], axis=0)
        line = f"{name:26s} cv={scores.mean():.4f} +/-{scores.std():.4f}"
        if baseline_scores is None:
            baseline_scores = scores
        else:
            gain = scores - baseline_scores
            line += f"  gain={gain.mean():+.4f} +/-{gain.std():.4f}"
        print(line)
        print(f"    per-seed cv={np.round(scores, 4)}")
        if baseline_scores is not scores:
            print(f"    per-seed gain={np.round(scores - baseline_scores, 4)}")
        print(f"    per-class recall={np.round(recall, 3)}  predicted={distribution}")
        print(f"    recall | matched  ={np.round(matched_recall, 3)}")
        print(f"    recall | unmatched={np.round(unmatched_recall, 3)}")
        print(f"    confusion (rows=true) =\n{confusion}")
        if args.calibration:
            report_calibration(dev, per_seed)
        if args.holdout:
            total, hrecall, hconf, hdist = holdout_score(dev, holdout, variant)
            print(f"    holdout={total:.4f}  recall={np.round(hrecall, 3)}  predicted={hdist}")

if __name__ == "__main__":
    main()
