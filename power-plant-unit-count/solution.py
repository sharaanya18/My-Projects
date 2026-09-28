"""Power Plant Unit Count -- end-to-end, from-scratch solution (CPU only).

Task
----
One row = 72 consecutive hours x 6 plant-level totals (gross load, heat input,
steam, SO2, NOx, CO2), summed over a plant's monitored units. Predict how many
of those units ran during the window: a whole number in 1..16.

Score = max(0, 1 - MAE / MAE(answer 3 everywhere)). MAE over integers is
minimised by the MEDIAN of the predictive distribution, so every model below
outputs cumulative probabilities P(units > k), k = 1..15, and the final count
is the median of the blended distribution: 1 + #{k : P(units > k) > 0.5}.
That decode rule follows from the metric itself; nothing in it is tuned.

Models (everything is fitted inside this script, from train.csv rows only)
------------------------------------------------------------------------
A. LightGBM multiclass over ~170 per-window features computed by
   `window_features` (step sizes, load-level histograms, heat-input vs load
   curve, emission intensities, autocorrelation, spectra). A unit coming online
   is a simultaneous step in load, heat and emissions whose size is that
   unit's size; the features describe those steps, the trees learn what they
   mean for the count.
B. `UnitNet`, a dilated residual 1-D CNN over the raw hourly sequence,
   randomly initialised, trained with an ordinal (cumulative BCE) head and the
   augmentations the challenge allows (time reversal, whole-window scaling).

The blend weight between A and B is picked on plant-grouped out-of-fold
predictions from `validation_folds.json`, inside this run. LightGBM's round
count comes from early stopping on an inner plant-disjoint fold, also inside
this run. All other settings are fixed defaults written before looking at
any data.

Compliance summary
------------------
* No pretrained weights, no external data, no network access.
* Test rows are used only for per-row inference: no statistic, scaler,
  vocabulary or calibration is ever fitted on them, and they are never
  grouped by plant.
* Identifiers and directory names are only used to find each row's file.
* Deterministic: fixed seeds, fixed thread counts, deterministic LightGBM and
  torch algorithms.

Usage:  python solution.py [DATA_DIR] [OUT_CSV]
        defaults: ./dataset/public  ./submission.csv
"""
import os

N_THREADS = 8
# Thread counts are set explicitly (not setdefault) before numpy/torch import,
# so an inherited environment cannot change floating-point reduction order.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = str(N_THREADS)

import json
import random
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F_nn

SEED = 42
N_CLASSES = 16          # counts 1..16
REF_COUNT = 3           # the reference answer the score is normalised by
TIME_BUDGET_S = 3000    # stop optional work well before the 1.5 h limit

CNN_EPOCHS = 30
CNN_BATCH = 256
CNN_FINAL_SEEDS = 3

LGB_PARAMS = dict(
    objective="multiclass", num_class=N_CLASSES, learning_rate=0.03,
    num_leaves=31, min_data_in_leaf=40, feature_fraction=0.7,
    bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    num_threads=N_THREADS, seed=SEED, deterministic=True,
    force_row_wise=True, verbose=-1,
)
LGB_MAX_ROUNDS = 3000

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
OUT_PATH = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./submission.csv")

T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


torch.set_num_threads(N_THREADS)
torch.use_deterministic_algorithms(True)
seed_everything(SEED)


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def row_path(evidence_dir):
    p = Path(evidence_dir)
    if p.parts[0] != "payload":          # spec: "relative path under payload/"
        p = Path("payload") / p
    return DATA_DIR / p / "plant_totals.npy"


def load_windows(df):
    X = np.empty((len(df), 72, 6), dtype=np.float32)
    for i, d in enumerate(df["evidence_dir"]):
        a = np.load(row_path(d))
        assert a.shape == (72, 6), (d, a.shape)
        X[i] = a
    bad = ~np.isfinite(X)
    neg = X < 0
    if bad.any() or neg.any():
        log(f"cleaned {int(bad.sum())} non-finite and {int(neg.sum())} negative values")
    return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), 0, None)


def write_submission(ids, counts, path):
    counts = np.asarray(counts)
    assert len(ids) == len(counts)
    assert counts.dtype.kind in "iu" and counts.min() >= 1 and counts.max() <= N_CLASSES
    sub = pd.DataFrame({"id": ids, "units": counts.astype(int)})
    assert sub["id"].is_unique
    tmp = Path(str(path) + ".tmp")
    sub.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# Metric and decode
# --------------------------------------------------------------------------
def score(y_true, y_pred):
    e = np.abs(y_true - y_pred).mean()
    b = np.abs(y_true - REF_COUNT).mean()
    return max(0.0, 1.0 - e / b)


def decode_median(cum):
    """cum[:, k-1] = P(units > k). Median of the implied distribution."""
    cum = np.minimum.accumulate(np.clip(cum, 0, 1), axis=1)   # enforce monotone
    return (1 + (cum > 0.5).sum(1)).astype(int)


def class_probs_to_cum(p):
    """p[:, j] = P(units == j+1)  ->  P(units > k) for k = 1..15."""
    tail = np.cumsum(p[:, ::-1], axis=1)[:, ::-1]      # tail[:, j] = P(units >= j+1)
    return tail[:, 1:]


# --------------------------------------------------------------------------
# Hand-computed window features (model A input)
# --------------------------------------------------------------------------
COLS = ("load", "heat", "steam", "so2", "nox", "co2")


def _nanstat(fn, a, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(a, *args, **kw)


def window_features(X):
    X = X.astype(np.float64)
    eps = 1e-9
    F = {}
    mx = X.max(1)                                        # (N, 6)
    L, H, S = X[:, :, 0], X[:, :, 1], X[:, :, 2]

    # level statistics, each column relative to its own window maximum
    for c, nm in enumerate(COLS):
        a, m = X[:, :, c], mx[:, c]
        r = a / (m[:, None] + eps)
        F[f"{nm}_logmax"] = np.log1p(m)
        F[f"{nm}_logmean"] = np.log1p(a.mean(1))
        F[f"{nm}_on"] = (a > 0).mean(1)
        for q in (5, 25, 50, 75, 95):
            F[f"{nm}_q{q}"] = np.percentile(r, q, axis=1)
        F[f"{nm}_cv"] = a.std(1) / (a.mean(1) + eps)

    # step structure: a unit starting or stopping is a jump whose size is the unit
    for c, nm in ((0, "load"), (1, "heat"), (5, "co2")):
        r = X[:, :, c] / (mx[:, c][:, None] + eps)
        d = np.diff(r, axis=1)
        ad = np.abs(d)
        for t in (0.02, 0.05, 0.1, 0.2, 0.35):
            F[f"{nm}_nstep_{t}"] = (ad > t).sum(1)
        F[f"{nm}_tv"] = ad.sum(1)
        top = -np.sort(-ad, axis=1)
        for k in (0, 1, 2, 4, 8):
            F[f"{nm}_step_top{k}"] = top[:, k]
        up = -np.sort(-np.where(d > 0, d, 0), axis=1)
        dn = -np.sort(-np.where(d < 0, -d, 0), axis=1)
        F[f"{nm}_up1"], F[f"{nm}_up3"] = up[:, 0], up[:, :3].mean(1)
        F[f"{nm}_dn1"], F[f"{nm}_dn3"] = dn[:, 0], dn[:, :3].mean(1)
        F[f"{nm}_inv_up1"] = 1.0 / (up[:, 0] + 0.02)
        F[f"{nm}_flat"] = (ad < 0.01).mean(1)
        F[f"{nm}_changes"] = ((ad[:, 1:] >= 0.02) & (ad[:, :-1] < 0.02)).sum(1)
        F[f"{nm}_acc"] = np.abs(np.diff(d, axis=1)).mean(1)
        # level histogram: how many distinct operating levels the window visits
        b = np.clip((r * 10).astype(int), 0, 9)
        hist = np.stack([(b == k).mean(1) for k in range(10)], 1)
        for k in range(10):
            F[f"{nm}_hist{k}"] = hist[:, k]
        F[f"{nm}_hist_entropy"] = -(hist * np.log(hist + eps)).sum(1)
        F[f"{nm}_hist_nbins"] = (hist > 0.02).sum(1)

    # heat input vs load: the intercept is no-load heat, which grows with the
    # number of committed units; the slope is incremental heat rate
    on = L > 0.05 * mx[:, 0][:, None]
    n_on = on.sum(1)
    F["n_on_hours"] = n_on
    hr = np.where(on, H / (L + eps), np.nan)
    hr_med = _nanstat(np.nanmedian, hr, axis=1)
    F["heat_rate_log"] = np.log(hr_med + eps)
    q75, q25 = _nanstat(np.nanpercentile, hr, [75, 25], axis=1)
    F["heat_rate_iqr"] = (q75 - q25) / (hr_med + eps)
    w = on.astype(np.float64)
    nn_ = np.maximum(n_on, 1)
    Ln, Hn = L / (mx[:, 0][:, None] + eps), H / (mx[:, 1][:, None] + eps)
    mL, mH = (w * Ln).sum(1) / nn_, (w * Hn).sum(1) / nn_
    cLH = (w * (Ln - mL[:, None]) * (Hn - mH[:, None])).sum(1)
    vL = (w * (Ln - mL[:, None]) ** 2).sum(1)
    vH = (w * (Hn - mH[:, None]) ** 2).sum(1)
    slope = cLH / (vL + 1e-9)
    icpt = mH - slope * mL
    F["hl_slope"], F["hl_icpt"] = slope, icpt
    F["hl_r2"] = cLH ** 2 / (vL * vH + 1e-12)
    resid = Hn - (icpt[:, None] + slope[:, None] * Ln)
    F["hl_resid"] = np.sqrt((w * resid ** 2).sum(1) / nn_)

    # emission intensities per unit heat: fuel / control mix of the running
    # units; shifts inside the window mean different units came and went
    hon = H > 0.05 * mx[:, 1][:, None]
    for c, nm in ((3, "so2"), (4, "nox"), (5, "co2")):
        z = np.where(hon, np.log((X[:, :, c] + 1e-3) / (H + 1e-3)), np.nan)
        F[f"{nm}_int_med"] = _nanstat(np.nanmedian, z, axis=1)
        F[f"{nm}_int_std"] = _nanstat(np.nanstd, z, axis=1)
        p90, p10 = _nanstat(np.nanpercentile, z, [90, 10], axis=1)
        F[f"{nm}_int_range"] = p90 - p10
        zc = z - _nanstat(np.nanmean, z, axis=1)[:, None]
        lc = np.where(hon, Ln - (Ln * hon).sum(1)[:, None] / np.maximum(hon.sum(1), 1)[:, None], np.nan)
        num = _nanstat(np.nansum, zc * lc, axis=1)
        den = np.sqrt(_nanstat(np.nansum, zc ** 2, axis=1) * _nanstat(np.nansum, lc ** 2, axis=1))
        F[f"{nm}_int_load_corr"] = num / (den + eps)

    F["heat_no_load"] = ((H > 0) & (L == 0)).mean(1)
    F["load_no_heat"] = ((L > 0) & (H == 0)).mean(1)
    F["steam_present"] = (S > 0).mean(1)
    F["steam_per_load"] = np.log1p(S.sum(1)) - np.log1p(L.sum(1))

    # temporal structure of the normalised load curve
    Lc = Ln - Ln.mean(1, keepdims=True)
    v = (Lc ** 2).sum(1) + eps
    for lag in (1, 3, 12, 24):
        F[f"load_acf{lag}"] = (Lc[:, lag:] * Lc[:, :-lag]).sum(1) / v
    spec = np.abs(np.fft.rfft(Lc, axis=1))[:, 1:10]
    spec = spec / (spec.sum(1, keepdims=True) + eps)
    for k in range(spec.shape[1]):
        F[f"load_fft{k + 1}"] = spec[:, k]

    # cross-column agreement
    def corr(a, b):
        a = a - a.mean(1, keepdims=True)
        b = b - b.mean(1, keepdims=True)
        return (a * b).sum(1) / (np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + eps)
    F["corr_load_heat"] = corr(L, H)
    F["corr_load_co2"] = corr(L, X[:, :, 5])
    F["corr_heat_so2"] = corr(H, X[:, :, 3])
    F["corr_heat_nox"] = corr(H, X[:, :, 4])

    return pd.DataFrame(F).astype(np.float32)


def rule_of_thumb(feats):
    """Largest single load jump ~ one unit, so max/jump ~ unit count.
    Logged only, to show how much the trained models add (strip-the-ML check);
    never used for predictions."""
    return np.clip(np.round(1.0 / np.maximum(feats["load_up1"].to_numpy(), 1 / 16)), 1, 16).astype(int)


# --------------------------------------------------------------------------
# Model A: LightGBM
# --------------------------------------------------------------------------
def lgb_fit(Xf, y, rounds=None, Xes=None, yes=None):
    ds = lgb.Dataset(Xf, label=y - 1, free_raw_data=False)
    if rounds is None:
        dv = lgb.Dataset(Xes, label=yes - 1, reference=ds)
        m = lgb.train(LGB_PARAMS, ds, num_boost_round=LGB_MAX_ROUNDS, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        return m, m.best_iteration
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds), rounds


def lgb_cum(model, Xf):
    return class_probs_to_cum(model.predict(Xf, num_iteration=model.best_iteration or None))


# --------------------------------------------------------------------------
# Model B: dilated 1-D CNN with an ordinal head, random init
# --------------------------------------------------------------------------
def build_input(xb):
    """xb: (B, 72, 6) raw totals -> (B, 28, 72) sequence channels, (B, 6) scale."""
    m = xb.amax(1, keepdim=True)
    r = xb / (m + 1e-6)
    lg = (torch.log1p(xb) - torch.log1p(m)).clamp(min=-12.0)
    pos = (xb > 0).float()
    d = torch.diff(r, dim=1, prepend=r[:, :1])
    L, H = xb[..., 0:1], xb[..., 1:2]
    ratios = torch.cat([torch.log((xb[..., 3:6] + 1e-3) / (H + 1e-3)),
                        torch.log((H + 1e-3) / (L + 1e-3))], -1)
    seq = torch.cat([r, lg, pos, d, ratios], -1)
    return seq.transpose(1, 2), torch.log1p(m.squeeze(1))


class Block(nn.Module):
    def __init__(self, ch, dil):
        super().__init__()
        self.c1 = nn.Conv1d(ch, ch, 3, padding=dil, dilation=dil)
        self.b1 = nn.BatchNorm1d(ch)
        self.c2 = nn.Conv1d(ch, ch, 3, padding=1)
        self.b2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        h = F_nn.gelu(self.b1(self.c1(x)))
        return F_nn.gelu(x + self.b2(self.c2(h)))


class UnitNet(nn.Module):
    def __init__(self, mu_s, sd_s, mu_g, sd_g, ch=64):
        super().__init__()
        # input standardisation, fitted on the training rows of this model only
        for k, v in (("mu_s", mu_s), ("sd_s", sd_s), ("mu_g", mu_g), ("sd_g", sd_g)):
            self.register_buffer(k, v)
        self.stem = nn.Conv1d(28, ch, 3, padding=1)
        self.blocks = nn.Sequential(*[Block(ch, d) for d in (1, 2, 4, 8, 16)])
        self.head = nn.Sequential(nn.Linear(2 * ch + 6, 128), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(128, N_CLASSES - 1))

    def forward(self, xb):
        seq, glob = build_input(xb)
        seq = (seq - self.mu_s[None, :, None]) / self.sd_s[None, :, None]
        glob = (glob - self.mu_g) / self.sd_g
        h = self.blocks(self.stem(seq))
        h = torch.cat([h.mean(2), h.amax(2), glob], 1)
        return self.head(h)                               # logits of P(units > k)


def input_stats(X):
    with torch.no_grad():
        seq, glob = build_input(torch.from_numpy(X))
        mu_s = seq.mean((0, 2))
        sd_s = seq.std((0, 2)).clamp(min=1e-3)
        return mu_s, sd_s, glob.mean(0), glob.std(0).clamp(min=1e-3)


def cnn_fit(X, y, seed, epochs=CNN_EPOCHS):
    seed_everything(seed)
    g = torch.Generator().manual_seed(seed)
    model = UnitNet(*input_stats(X))
    Xt = torch.from_numpy(X)
    Yt = (torch.from_numpy(y)[:, None] > torch.arange(1, N_CLASSES)[None, :]).float()
    steps = (len(X) + CNN_BATCH - 1) // CNN_BATCH
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3,
                                                total_steps=epochs * steps, pct_start=0.2)
    model.train()
    for ep in range(epochs):
        if time.time() - T0 > TIME_BUDGET_S:
            log(f"time budget reached, CNN stops at epoch {ep}")
            break
        perm = torch.randperm(len(X), generator=g)
        for i in range(steps):
            idx = perm[i * CNN_BATCH:(i + 1) * CNN_BATCH]
            xb = Xt[idx].clone()
            # allowed augmentations: reverse time; scale the whole window
            flip = torch.rand(len(idx), generator=g) < 0.5
            xb[flip] = xb[flip].flip(1)
            xb = xb * torch.exp((torch.rand(len(idx), 1, 1, generator=g) - 0.5) * 0.4)
            loss = F_nn.binary_cross_entropy_with_logits(model(xb), Yt[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def cnn_cum(model, X):
    out = []
    for i in range(0, len(X), 1024):
        xb = torch.from_numpy(X[i:i + 1024])
        # per-row test-time augmentation: average the window and its time reverse
        p = (torch.sigmoid(model(xb)) + torch.sigmoid(model(xb.flip(1)))) / 2
        out.append(p.numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    # a valid (score-0) submission exists before any modelling starts
    write_submission(test["id"].tolist(), np.full(len(test), REF_COUNT), OUT_PATH)

    folds_meta = json.loads((DATA_DIR / "validation_folds.json").read_text())
    fold = train["id"].map(folds_meta["fold_of"]).to_numpy()
    assert not np.isnan(fold.astype(float)).any(), "every train row needs a fold"
    fold = fold.astype(int)
    n_folds = int(folds_meta["n_folds"])
    y = train["units"].to_numpy().astype(int)
    assert y.min() >= 1 and y.max() <= N_CLASSES
    log(f"train {len(train)} rows, test {len(test)} rows, {n_folds} plant-disjoint folds")

    Xtr, Xte = load_windows(train), load_windows(test)
    Ftr, Fte = window_features(Xtr), window_features(Xte)
    log(f"loaded windows and computed {Ftr.shape[1]} window features")

    # ---- out-of-fold predictions on plant-disjoint folds ----
    oof_lgb = np.zeros((len(train), N_CLASSES - 1))
    oof_cnn = np.zeros((len(train), N_CLASSES - 1))
    best_iters = []
    for f in range(n_folds):
        va = fold == f
        es_fold = (f + 1) % n_folds             # inner plant-disjoint early-stop fold
        es = fold == es_fold
        fit = ~va & ~es
        m, it = lgb_fit(Ftr[fit], y[fit], Xes=Ftr[es], yes=y[es])
        best_iters.append(it)
        oof_lgb[va] = lgb_cum(m, Ftr[va])
        net = cnn_fit(Xtr[~va], y[~va], seed=SEED + f)
        oof_cnn[va] = cnn_cum(net, Xtr[va])
        log(f"fold {f}: lgb {score(y[va], decode_median(oof_lgb[va])):.4f} "
            f"(iters {it}) | cnn {score(y[va], decode_median(oof_cnn[va])):.4f}")

    s_lgb = score(y, decode_median(oof_lgb))
    s_cnn = score(y, decode_median(oof_cnn))
    grid = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    blend_scores = [score(y, decode_median(w * oof_lgb + (1 - w) * oof_cnn)) for w in grid]
    w_lgb = float(grid[int(np.argmax(blend_scores))])
    log(f"OOF score  lgb {s_lgb:.4f}  cnn {s_cnn:.4f}  "
        f"blend {max(blend_scores):.4f} at w_lgb={w_lgb}")
    log(f"OOF score of the hand rule max/largest-step (no trained model): "
        f"{score(y, rule_of_thumb(Ftr)):.4f}")
    per_fold = [score(y[fold == f], decode_median(w_lgb * oof_lgb[fold == f]
                                                  + (1 - w_lgb) * oof_cnn[fold == f]))
                for f in range(n_folds)]
    log("blend per fold: " + " ".join(f"{s:.4f}" for s in per_fold)
        + f"  (mean {np.mean(per_fold):.4f}, std {np.std(per_fold):.4f})")

    # ---- final models on all training rows ----
    final_rounds = int(np.mean(best_iters) * 1.15)
    m_all, _ = lgb_fit(Ftr, y, rounds=final_rounds)
    te_lgb = lgb_cum(m_all, Fte)
    log(f"final lgb fitted on all rows, {final_rounds} rounds")

    te_cnn = np.zeros((len(test), N_CLASSES - 1))
    n_done = 0
    for s in range(CNN_FINAL_SEEDS):
        if n_done > 0 and time.time() - T0 > TIME_BUDGET_S:
            log("time budget reached, skipping remaining CNN seeds")
            break
        net = cnn_fit(Xtr, y, seed=SEED + 100 + s)
        te_cnn += cnn_cum(net, Xte)
        n_done += 1
    te_cnn /= n_done
    log(f"final cnn fitted on all rows, {n_done} seeds")

    pred = decode_median(w_lgb * te_lgb + (1 - w_lgb) * te_cnn)
    write_submission(test["id"].tolist(), pred, OUT_PATH)
    dist = np.bincount(pred, minlength=N_CLASSES + 1)[1:]
    log(f"wrote {OUT_PATH} ({len(pred)} rows); predicted count histogram 1..16: {dist.tolist()}")


if __name__ == "__main__":
    main()
