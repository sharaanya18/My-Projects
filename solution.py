"""
How Cold Was It -- predict air temperature (deg C) from three views of a falling snowflake.

Run:  python3 solution.py <public_dir> <submission_out>
      (defaults: ./dataset/public  ./working/submission.csv)

Approach (everything is fitted inside this script, from the released training rows only):
  * A small CNN trained FROM SCRATCH. One conv encoder is shared by the three views; the
    three pooled embeddings are concatenated in camera order (so each camera keeps its own
    slot) and fed to an MLP head.
  * A handful of hand-computed per-view statistics (size, brightness, edge energy, border
    contact, ...) are standardised and fed into the same head as extra inputs. They are only
    an additional input to the CNN, never a stand-alone predictor.
  * L1 loss (the metric is a mean absolute error), AdamW + one-cycle schedule, EMA weights,
    flip / right-angle-rotation / brightness / small-shift augmentation.
  * Week-disjoint validation: model A is trained without fold f0 of validation_folds.json and
    scored on it (reports the pooled-style score against the training-median baseline).
    Model B is trained on every training row. The submission is the mean of A and B,
    each averaged over dihedral test-time augmentation of a single row (no use of other test rows).
  * Fixed seeds, fixed epoch counts, CPU only, fixed thread count: no branch depends on wall-clock time,
    hardware or file presence.
"""
import json
import os
import random
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- config
SEED = 42
EPOCHS = 16            # epochs per model (fixed; sized so both models finish well inside 1.5 h on CPU)
BATCH = 128
LR = 2e-3
WEIGHT_DECAY = 2e-2
EMA_DECAY = 0.998
WIDTHS = (16, 16, 32, 64, 96)   # stem, stage1, stage2, stage3, stage4 channels
HOLDOUT_FOLD = "f0"
N_THREADS = 10          # fixed (the graded machine has 10 cores); never derived from the host at runtime


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.set_num_threads(N_THREADS)
torch.use_deterministic_algorithms(True)   # CPU-only, fixed plan: no backend / device / worker-dependent branches
print(f"cpu threads={N_THREADS}", flush=True)

# --------------------------------------------------------------------------- paths / data
public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")

train = pd.read_csv(public_dir / "train.csv")
test = pd.read_csv(public_dir / "test.csv")


def load_views(df, fname):
    """Rows of `df` -> uint8 array (n, 3, 80, 80) using the `evidence` column (file.npy:row)."""
    arr = np.load(public_dir / fname, mmap_mode="r")
    rows = df["evidence"].str.split(":").str[1].astype(int).to_numpy()
    return np.ascontiguousarray(arr[rows])


Xtr = load_views(train, "train_images.npy")
Xte = load_views(test, "test_images.npy")
ytr = train["temperature_c"].to_numpy(dtype=np.float64)
print(f"train {Xtr.shape} test {Xte.shape}", flush=True)

Y_MEDIAN = float(np.median(ytr))          # reference value of the metric (-4.8 on the released data)
Y_LO, Y_HI = float(ytr.min()), float(ytr.max())
Y_MEAN, Y_STD = float(ytr.mean()), float(ytr.std())

fold_of = json.load(open(public_dir / "validation_folds.json"))["fold_of"]   # week-disjoint folds shipped with the data
folds = train["id"].map(fold_of).to_numpy()
is_hold = folds == HOLDOUT_FOLD


# --------------------------------------------------------------------------- hand-made per-view features
def hand_features(X):
    """Per-view size / brightness / texture statistics, (n, 3*K) float32. Computed row by row in chunks."""
    out = []
    yy, xx = np.mgrid[0:80, 0:80].astype(np.float32)
    yy -= 39.5
    xx -= 39.5
    rr = np.sqrt(yy ** 2 + xx ** 2)
    for s in range(0, len(X), 1024):
        v = X[s:s + 1024].astype(np.float32) / 255.0             # (b,3,80,80)
        m = v > 0.02
        area = m.sum((2, 3)).astype(np.float32)
        area_safe = np.maximum(area, 1.0)
        tot = v.sum((2, 3))
        mean_on = tot / area_safe
        sq = (v ** 2).sum((2, 3)) / area_safe
        rmax = (m * rr).max((2, 3))
        rmean = (m * rr).sum((2, 3)) / area_safe
        border = (m[:, :, 0, :].sum(2) + m[:, :, -1, :].sum(2) + m[:, :, :, 0].sum(2) + m[:, :, :, -1].sum(2)).astype(np.float32)
        gx = np.abs(np.diff(v, axis=3)).sum((2, 3))
        gy = np.abs(np.diff(v, axis=2)).sum((2, 3))
        grad = (gx + gy) / area_safe                              # edge energy per lit pixel
        # second moments of the lit region (elongation: needles/columns vs plates)
        cy = (m * yy).sum((2, 3)) / area_safe
        cx = (m * xx).sum((2, 3)) / area_safe
        syy = (m * (yy - cy[..., None, None]) ** 2).sum((2, 3)) / area_safe
        sxx = (m * (xx - cx[..., None, None]) ** 2).sum((2, 3)) / area_safe
        sxy = (m * (yy - cy[..., None, None]) * (xx - cx[..., None, None])).sum((2, 3)) / area_safe
        tr = syy + sxx
        det = syy * sxx - sxy ** 2
        disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
        l1, l2 = tr / 2 + disc, np.maximum(tr / 2 - disc, 1e-3)
        elong = np.sqrt(l1 / l2)
        # rows/cols spanned by the flake (bounding extent)
        rows = m.any(3).sum(2).astype(np.float32)
        cols = m.any(2).sum(2).astype(np.float32)
        hi = (v > 0.4).sum((2, 3)).astype(np.float32) / area_safe  # fraction of very bright pixels
        feats = [np.log1p(area), tot / 100.0, mean_on, sq, v.max((2, 3)), rmax, rmean, np.log1p(border),
                 grad, np.log(l1 + 1.0), np.log(l2 + 1.0), np.log(elong), rows, cols, hi,
                 area / np.maximum(rows * cols, 1.0)]           # fill ratio of the bounding box
        out.append(np.concatenate([f for f in feats], axis=1).astype(np.float32))
    return np.concatenate(out, axis=0)


Ftr = hand_features(Xtr)
Fte = hand_features(Xte)
F_MU, F_SD = Ftr.mean(0), Ftr.std(0) + 1e-6                        # scaler fitted on training rows only
Ftr = np.clip((Ftr - F_MU) / F_SD, -6, 6).astype(np.float32)
Fte = np.clip((Fte - F_MU) / F_SD, -6, 6).astype(np.float32)
N_HAND = Ftr.shape[1]
print(f"hand features: {N_HAND}", flush=True)


# --------------------------------------------------------------------------- model
def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Encoder(nn.Module):
    """Shared per-view CNN; avg+max pooled embedding."""

    def __init__(self, w=WIDTHS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, w[0], 5, 2, 2, bias=False), nn.BatchNorm2d(w[0]), nn.ReLU(inplace=True),   # 40x40
            conv_bn(w[0], w[1]),
            conv_bn(w[1], w[2], 2), conv_bn(w[2], w[2]),                                            # 20x20
            conv_bn(w[2], w[3], 2), conv_bn(w[3], w[3]),                                            # 10x10
            conv_bn(w[3], w[4], 2),                                                                 # 5x5
        )
        self.out_dim = 2 * w[4]

    def forward(self, x):
        f = self.net(x)
        return torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)


class SnowNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = Encoder()
        d = self.enc.out_dim
        self.head = nn.Sequential(
            nn.Linear(3 * d + N_HAND, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.25),
            nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))

    def forward(self, x, h):                       # x: (b,3,80,80) float, h: (b,N_HAND)
        b = x.shape[0]
        z = self.enc(x.reshape(b * 3, 1, 80, 80).contiguous(memory_format=torch.channels_last))
        z = z.reshape(b, -1)                       # camera order preserved in the concat
        return self.head(torch.cat([z, h], 1)).squeeze(1)


# --------------------------------------------------------------------------- augmentation (train only)
def dihedral(x, k):
    """x (b,3,H,W); k in 0..7: rotation by 90*(k%4) with optional horizontal flip."""
    if k >= 4:
        x = x.flip(3)
    return torch.rot90(x, k % 4, (2, 3))


def augment(x):
    """Independent random dihedral op per (sample, view), brightness scaling, small shift."""
    b = x.shape[0]
    ops = torch.randint(0, 8, (b, 3))
    out = torch.empty_like(x)
    for k in range(8):
        idx = (ops == k).nonzero(as_tuple=True)
        if len(idx[0]):
            out[idx] = dihedral(x[idx[0], idx[1]].unsqueeze(1), k).squeeze(1)
    out = out * (1.0 + 0.15 * (torch.rand(b, 3, 1, 1) * 2 - 1))
    sx, sy = np.random.randint(-3, 4, 2)
    out = torch.roll(out, shifts=(int(sy), int(sx)), dims=(2, 3))
    return out.clamp_(0, 1.5)


# --------------------------------------------------------------------------- train / predict
def to_float(xb):
    return torch.from_numpy(xb).float().div_(255.0)


@torch.no_grad()
def predict(model, X, Fh, tta=4, bs=256):
    """Row-wise prediction (deg C); averaged over `tta` dihedral variants of each row."""
    model.eval()
    res = np.zeros(len(X), dtype=np.float64)
    for s in range(0, len(X), bs):
        xb = to_float(X[s:s + bs])
        hb = torch.from_numpy(Fh[s:s + bs])
        p = 0
        for k in range(tta):
            p = p + model(dihedral(xb, k), hb)
        res[s:s + bs] = (p / tta).numpy()
    return np.clip(res * Y_STD + Y_MEAN, Y_LO, Y_HI)


def train_model(idx, seed, tag, X_val=None, F_val=None, y_val=None):
    set_seed(seed)
    model = SnowNet().to(memory_format=torch.channels_last)
    ema = SnowNet().to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    steps_per_epoch = len(idx) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=EPOCHS * steps_per_epoch, pct_start=0.25)
    yn = ((ytr - Y_MEAN) / Y_STD).astype(np.float32)
    rng = np.random.RandomState(seed)
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(idx)
        tot = 0.0
        for bi in range(steps_per_epoch):
            bidx = np.sort(perm[bi * BATCH:(bi + 1) * BATCH])
            xb = augment(to_float(Xtr[bidx]))
            hb = torch.from_numpy(Ftr[bidx]) + 0.05 * torch.randn(len(bidx), N_HAND)
            yb = torch.from_numpy(yn[bidx])
            loss = F.smooth_l1_loss(model(xb, hb), yb, beta=0.05)   # ~ L1, matches the MAE metric
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            with torch.no_grad():                                    # EMA of weights (and BN buffers)
                d = min(EMA_DECAY, (1 + ep * steps_per_epoch + bi) / (10 + ep * steps_per_epoch + bi))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            tot += loss.item()
        msg = f"[{tag}] epoch {ep + 1}/{EPOCHS} train_loss {tot / steps_per_epoch:.4f}"
        if X_val is not None and (ep % 4 == 3 or ep == EPOCHS - 1):
            pv = predict(ema, X_val, F_val, tta=1)
            msg += f" | holdout MAE {np.abs(pv - y_val).mean():.3f}"
        print(msg, flush=True)
    return ema


all_idx = np.arange(len(train))
hold_idx = all_idx[is_hold]
fit_idx = all_idx[~is_hold]

# Model A: trained without the held-out weeks -> honest week-disjoint estimate of the score.
model_a = train_model(fit_idx, SEED, "A", Xtr[hold_idx], Ftr[hold_idx], ytr[hold_idx])
pa_hold = predict(model_a, Xtr[hold_idx], Ftr[hold_idx])
e = np.abs(pa_hold - ytr[hold_idx]).mean()
b = np.abs(Y_MEDIAN - ytr[hold_idx]).mean()
print(f"holdout fold {HOLDOUT_FOLD}: MAE {e:.3f}  baseline MAE {b:.3f}  score {max(0.0, 1 - e / b):.3f}", flush=True)

# Model B: trained on every training row, different seed.
model_b = train_model(all_idx, SEED + 1, "B")

# --------------------------------------------------------------------------- submission
pred = 0.5 * predict(model_a, Xte, Fte) + 0.5 * predict(model_b, Xte, Fte)
submission = pd.DataFrame({"id": test["id"], "temperature_c": np.round(pred, 2)})

assert len(submission) == len(test), "row count mismatch"
assert submission["id"].is_unique and set(submission["id"]) == set(test["id"]), "id mismatch"
assert np.isfinite(submission["temperature_c"]).all(), "non-finite prediction"
out_path.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(out_path, index=False)
print(f"wrote {out_path} {submission.shape}; mean {pred.mean():.2f} std {pred.std():.2f}", flush=True)
