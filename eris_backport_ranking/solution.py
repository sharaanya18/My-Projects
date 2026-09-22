from __future__ import annotations

import os

for _k, _v in (
    ("PYTHONHASHSEED", "0"),
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ[_k] = _v

import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import ndtri
from scipy.stats import rankdata
from sklearn.cluster import KMeans

SEED = 20240921
TOP_K = 20

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

DEVICE = torch.device("cpu")

GAIN_TABLE = torch.tensor([0.0, 1.0, 3.0], dtype=torch.float32)

REQUIRED_FILES = ("train.csv", "test.csv", "train_targets.csv")


def ndcg_at_k(truth, prediction, k: int = TOP_K) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size == 0:
        raise ValueError("The answer set cannot be empty.")
    cutoff = min(k, truth.size)
    gains = 2.0 ** truth - 1.0
    order = np.argsort(-prediction, kind="mergesort")[:cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    dcg = float(np.sum(gains[order] * discounts))
    ideal_dcg = float(np.sum(np.sort(gains)[::-1][:cutoff] * discounts))
    return 0.0 if ideal_dcg <= 0.0 else dcg / ideal_dcg


def _self_check_metric() -> None:
    d = 1.0 / np.log2(np.arange(2, 5, dtype=float))
    assert abs(ndcg_at_k([2, 1, 0], [3, 2, 1], 3) - 1.0) < 1e-12
    reversed_ratio = (3 * d[2] + 1 * d[1]) / (3 * d[0] + 1 * d[1])
    assert abs(ndcg_at_k([2, 1, 0], [1, 2, 3], 3) - reversed_ratio) < 1e-12
    assert ndcg_at_k([0, 0, 0], [3, 2, 1], 3) == 0.0
    assert abs(ndcg_at_k([2, 0, 0], [1, 1, 1], 3) - 1.0) < 1e-12


def find_data_dir() -> Path:
    here = Path.cwd().resolve()
    candidates: list[Path] = []

    override = os.environ.get("ERIS_DATA_DIR")
    if override:
        candidates.append(Path(override))

    for base in (here, here.parent):
        candidates.extend(
            [base, base / "data", base / "input", base / "public", base / "public_data",
             base / "dataset", base / "data" / "public", base / "input" / "public"]
        )
    kaggle = Path("/kaggle/input")
    if kaggle.is_dir():
        candidates.append(kaggle)
        candidates.extend(sorted(p for p in kaggle.iterdir() if p.is_dir()))

    for cand in candidates:
        if all((cand / name).is_file() for name in REQUIRED_FILES):
            return cand

    for depth in (1, 2, 3):
        for cand in sorted(here.glob("/".join(["*"] * depth))):
            if cand.is_dir() and all((cand / name).is_file() for name in REQUIRED_FILES):
                return cand

    raise FileNotFoundError(
        "could not locate train.csv / test.csv / train_targets.csv; "
        "set ERIS_DATA_DIR to the directory holding them"
    )


def output_path() -> Path:
    override = os.environ.get("ERIS_OUTPUT")
    out = Path(override) if override else Path.cwd() / "working" / "submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def parse_profile(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns or "profile_text" not in df.columns:
        raise ValueError("expected 'id' and 'profile_text' columns")
    records = []
    for text in df["profile_text"].astype(str):
        record = {}
        for token in text.split():
            key, sep, value = token.partition("=")
            if sep:
                record[key] = value
        records.append(record)
    parsed = pd.DataFrame.from_records(records)
    parsed.insert(0, "id", df["id"].to_numpy())
    return parsed


def split_field_kinds(parsed: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric, categorical = [], []
    for col in parsed.columns:
        if col == "id":
            continue
        values = pd.to_numeric(parsed[col], errors="coerce")
        (numeric if values.notna().mean() > 0.9 else categorical).append(col)
    return sorted(numeric), sorted(categorical)


class FieldEncoder:

    def __init__(self, numeric_fields: list[str], categorical_fields: list[str],
                 transform: str = "rank"):
        self.numeric_fields = numeric_fields
        self.categorical_fields = categorical_fields
        self.transform_kind = transform

    @staticmethod
    def _numeric_block(parsed: pd.DataFrame, fields: list[str]) -> np.ndarray:
        cols = [
            np.log1p(pd.to_numeric(parsed[f], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float))
            for f in fields
        ]
        return np.column_stack(cols) if cols else np.zeros((len(parsed), 0))

    def fit(self, parsed: pd.DataFrame) -> "FieldEncoder":
        block = self._numeric_block(parsed, self.numeric_fields)
        self.mu_ = block.mean(axis=0)
        self.sd_ = block.std(axis=0) + 1e-6
        self.reference_ = np.sort(block, axis=0)
        self.vocab_ = {
            f: {v: i + 1 for i, v in enumerate(sorted(parsed[f].astype(str).unique()))}
            for f in self.categorical_fields
        }
        self.cardinality_ = {f: len(self.vocab_[f]) + 1 for f in self.categorical_fields}
        return self

    def _rank_transform(self, block: np.ndarray) -> np.ndarray:
        n_ref = len(self.reference_)
        out = np.empty_like(block)
        for j in range(block.shape[1]):
            column = self.reference_[:, j]
            low = np.searchsorted(column, block[:, j], side="left")
            high = np.searchsorted(column, block[:, j], side="right")
            quantile = (low + high) / (2.0 * n_ref)
            quantile = np.clip(quantile, 1.0 / (2.0 * n_ref), 1.0 - 1.0 / (2.0 * n_ref))
            out[:, j] = ndtri(quantile)
        return out

    def transform(self, parsed: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        block = self._numeric_block(parsed, self.numeric_fields)
        if self.transform_kind == "rank":
            numeric = self._rank_transform(block)
        else:
            numeric = (block - self.mu_) / self.sd_
        numeric = np.clip(np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)
        cat_cols = [
            parsed[f].astype(str).map(self.vocab_[f]).fillna(0).to_numpy(dtype=np.int64)
            for f in self.categorical_fields
        ]
        categorical = np.column_stack(cat_cols) if cat_cols else np.zeros((len(parsed), 0), dtype=np.int64)
        return numeric.astype(np.float32), categorical.astype(np.int64)


class FieldRanker(nn.Module):

    def __init__(self, n_numeric, cardinalities, emb_dim=6, hidden=96, depth=2,
                 dropout=0.25, n_out=3):
        super().__init__()
        self.numeric_scale = nn.Parameter(torch.ones(n_numeric, emb_dim))
        self.numeric_shift = nn.Parameter(torch.zeros(n_numeric, emb_dim))
        self.embeddings = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cardinalities])
        self.emb_dim = emb_dim
        width = emb_dim * (n_numeric + len(cardinalities))
        self.width = width
        self.skip = nn.Linear(width, n_out)
        nn.init.zeros_(self.skip.bias)
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = hidden
        layers.append(nn.Linear(width, n_out))
        self.trunk = nn.Sequential(*layers)

    def _fields(self, numeric, categorical):
        num_tokens = numeric.unsqueeze(-1) * self.numeric_scale + self.numeric_shift
        parts = [num_tokens.flatten(1)]
        for j, emb in enumerate(self.embeddings):
            parts.append(emb(categorical[:, j]))
        return torch.cat(parts, dim=1)

    def forward(self, numeric, categorical):
        fields = self._fields(numeric, categorical)
        return self.skip(fields) + self.trunk(fields)


def train_field_ranker(numeric, categorical, labels, *, mode, seed, epochs, lr,
                       weight_decay, hidden, depth, dropout, emb_dim, batch_size,
                       cardinalities):
    torch.manual_seed(seed)
    model = FieldRanker(
        n_numeric=numeric.shape[1], cardinalities=cardinalities, emb_dim=emb_dim,
        hidden=hidden, depth=depth, dropout=dropout, n_out=(3 if mode == "eg" else 1),
    ).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    xn = torch.tensor(numeric, device=DEVICE)
    xc = torch.tensor(categorical, device=DEVICE)
    y = torch.tensor(np.asarray(labels, dtype=np.int64), device=DEVICE)
    gains = GAIN_TABLE.to(DEVICE)[y]

    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(y)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(n, generator=generator).to(DEVICE)
        for start in range(0, n, batch_size):
            index = order[start:start + batch_size]
            if len(index) < 16:
                continue
            out = model(xn[index], xc[index])
            if mode == "eg":
                loss = nn.functional.cross_entropy(out, y[index], label_smoothing=0.05)
            else:
                scores = out.squeeze(-1)
                target = gains[index]
                if float(target.sum()) <= 0.0:
                    continue
                loss = -(torch.softmax(target, dim=0) * torch.log_softmax(scores, dim=0)).sum()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        schedule.step()
    model.eval()
    return model


def score_with(model, numeric, categorical, mode) -> np.ndarray:
    with torch.no_grad():
        out = model(torch.tensor(numeric, device=DEVICE), torch.tensor(categorical, device=DEVICE))
        if mode == "eg":
            return (torch.softmax(out, dim=1) @ GAIN_TABLE.to(DEVICE)).cpu().numpy().astype(float)
        return out.squeeze(-1).cpu().numpy().astype(float)


def bagged_scores(parsed_fit, labels_fit, parsed_apply, *, mode, params, n_seeds,
                  numeric_fields, categorical_fields):
    model_params = dict(params)
    encoder = FieldEncoder(
        numeric_fields, categorical_fields, transform=model_params.pop("transform")
    ).fit(parsed_fit)
    xn_fit, xc_fit = encoder.transform(parsed_fit)
    xn_apply, xc_apply = encoder.transform(parsed_apply)
    cardinalities = [encoder.cardinality_[f] for f in categorical_fields]
    total = np.zeros(len(parsed_apply), dtype=float)
    for k in range(n_seeds):
        model = train_field_ranker(
            xn_fit, xc_fit, labels_fit, mode=mode, seed=SEED + 101 * k,
            cardinalities=cardinalities, **model_params
        )
        total += _to_ranks(score_with(model, xn_apply, xc_apply, mode)) / n_seeds
    return total


def _to_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(values, dtype=float), method="average")


N_REGIONS = 10
PURGE_RADIUS = 2.0
MIN_VALID_ROWS = 120
MIN_TRAIN_ROWS = 400


def region_disjoint_splits(standardised: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    regions = KMeans(n_clusters=N_REGIONS, n_init=10, random_state=SEED).fit_predict(standardised)
    splits = []
    for region in range(N_REGIONS):
        valid = np.where(regions == region)[0]
        if len(valid) < MIN_VALID_ROWS:
            continue
        train = np.where(regions != region)[0]
        nearest = _min_distance(standardised[train], standardised[valid])
        kept = train[nearest > PURGE_RADIUS]
        if len(kept) < MIN_TRAIN_ROWS:
            kept = train[np.argsort(-nearest)[:MIN_TRAIN_ROWS]]
        splits.append((np.sort(kept), np.sort(valid)))
    if not splits:
        raise RuntimeError("no usable validation region; check the training file")
    return splits


def _min_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a), dtype=float)
    b_sq = (b ** 2).sum(axis=1)
    for start in range(0, len(a), 512):
        chunk = a[start:start + 512]
        d2 = (chunk ** 2).sum(axis=1)[:, None] + b_sq[None, :] - 2.0 * chunk @ b.T
        out[start:start + 512] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def standardise(parsed: pd.DataFrame, numeric_fields: list[str]) -> np.ndarray:
    block = FieldEncoder._numeric_block(parsed, numeric_fields)
    return (block - block.mean(axis=0)) / (block.std(axis=0) + 1e-9)


SEARCH_SPACE = [
    dict(transform="rank", epochs=90,  lr=3e-3, weight_decay=3e-2, hidden=64, depth=1, dropout=0.35, emb_dim=4, batch_size=256),
    dict(transform="rank", epochs=140, lr=2e-3, weight_decay=3e-2, hidden=64, depth=2, dropout=0.40, emb_dim=4, batch_size=256),
    dict(transform="rank", epochs=140, lr=2e-3, weight_decay=1e-1, hidden=96, depth=2, dropout=0.30, emb_dim=6, batch_size=256),
    dict(transform="log",  epochs=90,  lr=3e-3, weight_decay=3e-2, hidden=64, depth=1, dropout=0.35, emb_dim=4, batch_size=256),
    dict(transform="log",  epochs=140, lr=2e-3, weight_decay=3e-2, hidden=64, depth=2, dropout=0.40, emb_dim=4, batch_size=256),
]
SEARCH_SEEDS = 2
FINAL_SEEDS = 10
BLEND_GRID = np.linspace(0.0, 1.0, 11)
SELECTION_TOLERANCE = 0.005


def robust_fold_score(fold_scores: list[float]) -> float:
    ordered = sorted(float(s) for s in fold_scores)
    half = max(1, (len(ordered) + 1) // 2)
    return float(np.mean(ordered[:half]))


def search_head(mode, parsed, labels, splits, numeric_fields, categorical_fields, log):
    results = []
    for params in SEARCH_SPACE:
        oof = np.full(len(labels), np.nan)
        fold_scores = []
        for train_idx, valid_idx in splits:
            scores = bagged_scores(
                parsed.iloc[train_idx].reset_index(drop=True), labels[train_idx],
                parsed.iloc[valid_idx].reset_index(drop=True),
                mode=mode, params=params, n_seeds=SEARCH_SEEDS,
                numeric_fields=numeric_fields, categorical_fields=categorical_fields,
            )
            oof[valid_idx] = scores / len(valid_idx)
            fold_scores.append(ndcg_at_k(labels[valid_idx], scores))
        score = robust_fold_score(fold_scores)
        log.append(
            f"  [{mode}] {params} -> worst-half {score:.4f}  mean {np.mean(fold_scores):.4f}"
        )
        results.append((params, score, oof))
    best_score = max(score for _, score, _ in results)
    for params, score, oof in results:
        if score >= best_score - SELECTION_TOLERANCE:
            log.append(f"  [{mode}] selected (within {SELECTION_TOLERANCE} of best {best_score:.4f})")
            return params, score, oof
    raise RuntimeError("unreachable: no configuration met its own best score")


def learn_blend_weight(oof_eg, oof_listnet, labels, splits, log):
    scored = []
    for w in BLEND_GRID:
        fold_scores = []
        for _, valid_idx in splits:
            mixed = w * oof_eg[valid_idx] + (1.0 - w) * oof_listnet[valid_idx]
            fold_scores.append(ndcg_at_k(labels[valid_idx], mixed))
        scored.append((float(w), robust_fold_score(fold_scores)))
    best_score = max(s for _, s in scored)
    near_best = [w for w, s in scored if s >= best_score - SELECTION_TOLERANCE]
    best_w = min(near_best, key=lambda w: (abs(w - 0.5), w))
    log.append(
        f"  head mixture: w(expected-gain)={best_w:.2f}, chosen from {len(near_best)} weights within "
        f"{SELECTION_TOLERANCE} of the best -> worst-half region-disjoint NDCG@20 {best_score:.4f}"
    )
    return best_w, best_score


def check_submission(submission: pd.DataFrame, test_ids: pd.Series) -> None:
    assert list(submission.columns) == ["id", "prediction"], f"columns={list(submission.columns)}"
    assert len(submission) == len(test_ids), f"{len(submission)} rows, expected {len(test_ids)}"
    assert submission["id"].duplicated().sum() == 0, "duplicate ids"
    assert set(submission["id"]) == set(test_ids), "id set does not match test.csv"
    assert (submission["id"].to_numpy() == test_ids.to_numpy()).all(), "test id order changed"
    assert pd.api.types.is_numeric_dtype(submission["prediction"]), "prediction is not numeric"
    values = submission["prediction"].to_numpy(dtype=float)
    assert np.isfinite(values).all(), "non-finite prediction"
    assert submission["prediction"].notna().all(), "missing prediction"
    assert submission["prediction"].nunique() > 1, "a constant score carries no ranking"


def main() -> int:
    started = time.time()
    _self_check_metric()

    data_dir = find_data_dir()
    train_raw = pd.read_csv(data_dir / "train.csv")
    test_raw = pd.read_csv(data_dir / "test.csv")
    targets = pd.read_csv(data_dir / "train_targets.csv")

    parsed_train = parse_profile(train_raw)
    parsed_test = parse_profile(test_raw)

    labelled = parsed_train.merge(targets[["id", "target"]], on="id", how="inner")
    if len(labelled) != len(parsed_train):
        raise ValueError(f"{len(parsed_train) - len(labelled)} training rows have no target")
    labels = labelled.pop("target").to_numpy(dtype=float)
    if not set(np.unique(labels)).issubset({0.0, 1.0, 2.0}):
        raise ValueError("target values outside {0, 1, 2}")
    labelled = labelled.reset_index(drop=True)

    numeric_fields, categorical_fields = split_field_kinds(labelled)
    for field in numeric_fields + categorical_fields:
        if field not in parsed_test.columns:
            parsed_test[field] = "0" if field in numeric_fields else "__unseen__"

    log: list[str] = [
        f"data directory            : {data_dir}",
        f"train rows / test rows    : {len(labelled)} / {len(parsed_test)}",
        f"numeric / categorical     : {len(numeric_fields)} / {len(categorical_fields)} profile fields",
    ]

    splits = region_disjoint_splits(standardise(labelled, numeric_fields))
    log.append(
        f"region-disjoint folds     : {len(splits)} of {N_REGIONS} regions usable  "
        + "(train->valid: " + ", ".join(f"{len(tr)}->{len(va)}" for tr, va in splits) + ")"
    )

    params_eg, score_eg, oof_eg = search_head(
        "eg", labelled, labels, splits, numeric_fields, categorical_fields, log)
    params_listnet, score_listnet, oof_listnet = search_head(
        "listnet", labelled, labels, splits, numeric_fields, categorical_fields, log)
    weight, blend_score = learn_blend_weight(oof_eg, oof_listnet, labels, splits, log)

    final_eg = bagged_scores(
        labelled, labels, parsed_test, mode="eg", params=params_eg, n_seeds=FINAL_SEEDS,
        numeric_fields=numeric_fields, categorical_fields=categorical_fields)
    final_listnet = bagged_scores(
        labelled, labels, parsed_test, mode="listnet", params=params_listnet, n_seeds=FINAL_SEEDS,
        numeric_fields=numeric_fields, categorical_fields=categorical_fields)

    n_test = len(parsed_test)
    predictions = (weight * final_eg + (1.0 - weight) * final_listnet) / n_test

    submission = pd.DataFrame({"id": parsed_test["id"].to_numpy(), "prediction": predictions})
    check_submission(submission, test_raw["id"])
    destination = output_path()
    submission.to_csv(destination, index=False)

    print("\n".join(log))
    print()
    print("model                     : FieldRanker (learned field embeddings + MLP trunk)")
    print(f"  expected-gain head      : {params_eg}")
    print(f"                            worst-half region-disjoint NDCG@20 {score_eg:.4f}")
    print(f"  listwise (ListNet) head : {params_listnet}")
    print(f"                            worst-half region-disjoint NDCG@20 {score_listnet:.4f}")
    print(f"  head mixture            : {weight:.2f} expected-gain / {1 - weight:.2f} listwise")
    print(f"validation NDCG@20        : {blend_score:.4f}  (worst half of the region-disjoint folds)")
    print(f"prediction range          : [{predictions.min():.6f}, {predictions.max():.6f}]")
    print(f"distinct predictions      : {submission['prediction'].nunique()} of {n_test}")
    print(f"output path               : {destination}")
    print(f"runtime                   : {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
