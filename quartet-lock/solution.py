#!/usr/bin/env python3
"""
Quartet Lock: One Record Each -- end-to-end solution.

Run:  python3 solution.py <public_dir> <submission_out>
Both paths are read from sys.argv, exactly as the platform starter does.

TASK
  Each batch holds 4 image sets (3 photographs of pinned insects each) and 4
  collection records (year, day_of_year, latitude, longitude).  A hidden
  perfect matching assigns one record to each set.  For every set we output a
  strict ranking of its 4 candidate positions.
  Metric: S = 0.6 * top1 + 0.4 * MRR.

DATA FACTS USED (checked on the released files)
  * train.csv: batch_id, set_id, image_names (JSON list of 3 files),
    candidates (JSON list of 4 records), ranking (true position first).
    test.csv has the same columns without ranking.
  * 376 complete train batches (1504 sets), 181 test batches (724 sets).
    All rows of a batch share one candidate list, and the 4 candidates of a
    batch are exactly the true records of its 4 sets.
  * Images: 576x576 grey canvas with the specimen photo pasted unscaled, so
    pixel extent is physical extent.  Records are British (lat 50-59).

APPROACH (all learning happens inside this script, every run)
  1. Images.  Pixel size is physical size, so no image is rescaled on its
     own: one fixed-size window (same native size for every image, derived
     from train images) is cut around the specimen and all windows are
     resized by the same factor.  Morphometrics measured at native resolution
     (area, bbox, elongation, colour) enter the network next to CNN features.
  2. Dual encoder, fine-tuned end to end.
       set tower    : pretrained ConvNeXt-T (ImageNet-22k weights from timm /
                      Hugging Face) per image + morphometrics -> projection
                      -> permutation-invariant mean/max pooling over 3 images.
       record tower : MLP on a multi-scale random-Fourier encoding of the
                      point on the sphere (resolves the >25 km separation),
                      cyclic day-of-year harmonics and year RBFs.
     Primary loss: exact log-likelihood of the true matching among all 24
     matchings of a batch's 4x4 score matrix ("one record each" is trained,
     not only decoded).  Auxiliary losses: per-row cross-entropy over the 4
     candidates and an in-minibatch CLIP loss (easy geographic negatives).
  3. Learned-embedding neighbour atlas.  Labelled train sets form an atlas:
     for a query set we retrieve the most similar train images / sets in the
     fine-tuned embedding space and score each candidate record with a kernel
     density of the neighbours' true records (space x season x year).
     Bandwidths are selected on out-of-fold predictions (HPO in the script).
  4. Blend + Bayes-optimal decode.  A 3-weight conditional-logit blend
     (network logit, image atlas, set atlas) is fitted by minimising the
     out-of-fold matching NLL.  Per batch we enumerate the 24 matchings, take
     exact marginals P(set i owns record j) and sort each row by marginal:
     expected score is linear in the marginals with decreasing rank weights,
     so this is optimal for both the top-1 and the reciprocal-rank terms.

VALIDATION / ENSEMBLE
  5 seeded folds over batches (a batch is the unit of the task and is never
  split).  Collection sites chain every train batch into one connected
  component, so site-disjoint folds are impossible here.  Each fold model uses
  a fixed cosine schedule with a fixed epoch count (no early stopping, so the
  out-of-fold score is not selected on its own validation data).  Atlas
  kernels and blend weights are fitted on all out-of-fold predictions and the
  blend score is also reported cross-fitted (fit on 4 folds, scored on the
  5th).  Test logits are the average of the 5 fold models' blended logits.

FIXED, DETERMINISTIC RUNTIME PLAN
  * Every quantity of work is a constant: 5 folds x N_EPOCHS epochs, fixed
    batch size, fixed image size, fixed TTA, fixed hyperparameter grid.
    Wall-clock time is only printed; it never changes what is computed.
    The fixed plan is sized to finish well inside one hour on the A10G.
  * One device path (CUDA, fp16 autocast), no fallbacks, fixed worker count.
  * Seeds (42) for python / numpy / torch; cudnn deterministic, TF32 off,
    deterministic algorithms requested, CUBLAS workspace config set.  Ops
    whose CUDA backward is non-deterministic (index-add / scatter from
    advanced indexing, gather, adaptive pooling) are replaced by one-hot
    matmuls and plain means.

COMPLIANCE
  * Pretrained weights: generic ImageNet-22k ConvNeXt-T via timm (HF hub).
    No self-fine-tuned or externally hosted task weights.
  * Data: only train.csv, test.csv, sample_submission.csv and the released
    images.  No external datasets, no synthetic or generated training data.
  * Test usage: inference only.  Test images are embedded in eval mode in
    fixed-size minibatches (ConvNeXt uses LayerNorm: no batch statistics),
    the atlas holds TRAIN sets only, and decoding uses only the other rows of
    the same published batch (explicitly allowed).  No pseudo-labels, no
    test-time adaptation, no statistic computed over the evaluation set;
    normalisation statistics come from training images only.
  * Identifiers, row order, candidate order and file metadata are not used as
    features (ids only join rows to images, batches and the output).
  * Removing the learned model breaks the pipeline: the atlas searches the
    fine-tuned embedding space and the blend is fitted on network output.
"""

import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "42"

import io
import itertools
import json
import math
import random
import sys
import time
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

PUBLIC_DIR = Path(sys.argv[1])
SUB_OUT = Path(sys.argv[2])

# --------------------------------------------------------------------------- #
# Fixed configuration (no value below depends on runtime conditions)
# --------------------------------------------------------------------------- #
SEED = 42
DEVICE = torch.device("cuda")      # platform hardware: one NVIDIA A10G
BACKBONE = "convnext_tiny.fb_in22k"
IMG_SIZE = 256                     # every window is resized by the same factor
EMB_DIM = 256
N_SIZE_FEATS = 12
BATCHES_PER_STEP = 16              # 16 batches x 4 sets x 3 images = 192 images
N_FOLDS = 5
N_EPOCHS = 20
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
RFF_SCALES = [1, 3, 10, 30, 100, 300, 1000]   # radians^-1 on the unit sphere
KNN_K = 32
TTA = 4                            # identity, hflip, vflip, hvflip
N_WORKERS = 4                      # image decoding; results are order-preserving
EMBED_BS = 128
N_IMG = 3                          # photographs per set

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=True)

T_START = time.time()


def log(msg):  # elapsed time is telemetry only
    print(f"[{time.time() - T_START:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Metric (Set Provenance Ranking Score)
# --------------------------------------------------------------------------- #
def provenance_score(rankings, truths):
    pos = np.array([list(r).index(t) for r, t in zip(rankings, truths)])
    top1 = float(np.mean(pos == 0))
    mrr = float(np.mean(1.0 / (1.0 + pos)))
    return 0.6 * top1 + 0.4 * mrr, top1, mrr


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
def parse_candidates(s):
    """JSON list of 4 dicts -> (4, 4) array [year, day_of_year, latitude, longitude]."""
    return [[float(d["year"]), float(d["day_of_year"]), float(d["latitude"]), float(d["longitude"])]
            for d in json.loads(s)]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def doy_diff(a, b):
    d = np.abs(a - b) % 365.25
    return np.minimum(d, 365.25 - d)


class RecordFeaturizer:
    """Fixed (non-learned) encoding of a record; the MLP on top is learned."""

    def __init__(self, train_recs):
        rng = np.random.RandomState(SEED)
        self.B = np.concatenate([rng.normal(0.0, s, size=(3, 16)) for s in RFF_SCALES], 1)
        y = train_recs[:, 0]
        self.ymu, self.ysd = float(y.mean()), float(max(y.std(), 1.0))
        self.ycent = np.linspace(y.min(), y.max(), 8)
        self.yw = float(max((y.max() - y.min()) / 7.0, 1.0))

    def __call__(self, recs):
        recs = np.asarray(recs, dtype=np.float64)
        yr, doy, lat, lon = recs[..., 0], recs[..., 1], recs[..., 2], recs[..., 3]
        la, lo = np.radians(lat), np.radians(lon)
        xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)
        pr = xyz @ self.B
        t = 2 * np.pi * doy / 365.25
        doyf = np.stack([f(k * t) for k in (1, 2, 3) for f in (np.sin, np.cos)], -1)
        yf = np.concatenate([((yr - self.ymu) / self.ysd)[..., None],
                             np.exp(-0.5 * ((yr[..., None] - self.ycent) / self.yw) ** 2)], -1)
        return np.concatenate([xyz, np.sin(pr), np.cos(pr), doyf, yf], -1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Images: located on disk or inside a released zip archive (read in place)
# --------------------------------------------------------------------------- #
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def build_image_index(public_dir):
    """name -> ("file", path) or ("zip", archive, member). Deterministic (sorted walk)."""
    index = {}
    for p in sorted(public_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            index.setdefault(p.name, ("file", str(p)))
    for z in sorted(public_dir.rglob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for m in sorted(zf.namelist()):
                name = Path(m).name
                if Path(m).suffix.lower() in IMG_EXTS and name not in index:
                    index[name] = ("zip", str(z), m)
    return index


def _load(src):
    if src[0] == "file":
        return Image.open(src[1]).convert("RGB")
    with zipfile.ZipFile(src[1]) as zf:
        return Image.open(io.BytesIO(zf.read(src[2]))).convert("RGB")


def _fg_stats(a):
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]], 0)
    bg = np.median(border, axis=0)
    diff = np.abs(a.astype(np.int16) - bg.astype(np.int16)).max(-1)
    return bg, diff > 12


def _bbox_side(src):
    a = np.asarray(_load(src))
    _, fg = _fg_stats(a)
    ys, xs = np.nonzero(fg)
    if len(ys) < 10:
        return 0, a.shape[0], a.shape[1]
    return max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1), a.shape[0], a.shape[1]


def _process_image(args):
    src, win, size = args
    im = _load(src)
    a = np.asarray(im)
    H, W = a.shape[:2]
    bg, fg = _fg_stats(a)
    ys, xs = np.nonzero(fg)
    feats = np.zeros(N_SIZE_FEATS, np.float32)
    if len(ys) >= 10:
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx = (y0 + y1) / 2.0, (x0 + x1) / 2.0
        cov = np.cov(np.stack([ys, xs]).astype(np.float64))
        ev = np.sort(np.clip(np.linalg.eigvalsh(cov), 1e-6, None))
        px = a[fg].astype(np.float32) / 255.0
        sat = px.max(1) - px.min(1)
        feats[:] = [np.log1p(len(ys)), np.log1p(y1 - y0 + 1), np.log1p(x1 - x0 + 1),
                    0.5 * np.log(ev[1] / ev[0]), *px.mean(0), *px.std(0),
                    float((px.mean(1) < 0.25).mean()), float(sat.mean())]
    else:
        cy, cx = H / 2.0, W / 2.0
    left, top = int(round(cx - win / 2.0)), int(round(cy - win / 2.0))
    canvas = Image.new("RGB", (win, win), tuple(int(v) for v in bg))
    sx0, sy0 = max(left, 0), max(top, 0)
    sx1, sy1 = min(left + win, W), min(top + win, H)
    if sx1 > sx0 and sy1 > sy0:
        canvas.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - left, sy0 - top))
    out = canvas.resize((size, size), Image.BILINEAR, reducing_gap=2.0)
    return np.asarray(out, dtype=np.uint8), feats


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class QuartetNet(nn.Module):
    def __init__(self, rec_dim):
        super().__init__()
        self.backbone = timm.create_model(BACKBONE, pretrained=True, num_classes=0, drop_path_rate=0.1)
        d = self.backbone.num_features
        self.feat_norm = nn.LayerNorm(d)
        self.img_proj = nn.Sequential(nn.Linear(d + N_SIZE_FEATS, 512), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(512, EMB_DIM))
        self.set_mlp = nn.Sequential(nn.Linear(2 * EMB_DIM, 512), nn.GELU(), nn.Dropout(0.1),
                                     nn.Linear(512, EMB_DIM))
        self.rec_mlp = nn.Sequential(nn.Linear(rec_dim, 512), nn.GELU(), nn.LayerNorm(512),
                                     nn.Linear(512, 512), nn.GELU(), nn.Linear(512, EMB_DIM))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_images(self, x, size_feats):
        # global mean instead of adaptive pooling (deterministic backward on CUDA)
        f = self.feat_norm(self.backbone.forward_features(x).float().mean((2, 3)))
        z = self.img_proj(torch.cat([f, size_feats], 1))
        return f, z

    def encode_sets(self, z):  # z: (n_sets, N_IMG, EMB)
        return F.normalize(self.set_mlp(torch.cat([z.mean(1), z.max(1).values], 1)), dim=-1)

    def encode_records(self, r):
        return F.normalize(self.rec_mlp(r), dim=-1)

    def scale(self):
        return self.logit_scale.clamp(max=math.log(100.0)).exp()


PERMS_NP = np.array(list(itertools.permutations(range(4))))          # (24, 4)
PERM_ONEHOT = np.eye(4)[PERMS_NP]                                     # (24, 4, 4)


def perm_nll_torch(S, onehot_lab):
    """Exact matching NLL. S (G,4,4) logits (rows = sets), onehot_lab (G,4,4)."""
    P = torch.as_tensor(PERM_ONEHOT, dtype=S.dtype, device=S.device)
    tot = torch.einsum("gij,pij->gp", S, P)                             # score of each matching
    true = (S * onehot_lab).sum((1, 2))
    return torch.logsumexp(tot, 1) - true


def perm_marginals_np(S):
    tot = np.einsum("gij,pij->gp", S, PERM_ONEHOT)
    tot = tot - tot.max(1, keepdims=True)
    p = np.exp(tot)
    p /= p.sum(1, keepdims=True)
    return np.einsum("gp,pij->gij", p, PERM_ONEHOT)


def perm_nll_np(S, labels):
    tot = np.einsum("gij,pij->gp", S, PERM_ONEHOT)
    m = tot.max(1, keepdims=True)
    lse = (m + np.log(np.exp(tot - m).sum(1, keepdims=True)))[:, 0]
    true = np.take_along_axis(S, labels[:, :, None], 2)[..., 0].sum(1)
    return lse - true


def knn_logdens(sims, nb_rec, cand, tau, hg, hd, hy):
    """Atlas score. sims (Q,K), nb_rec (Q,K,4), cand (Q,4,4) -> log density (Q,4)."""
    w = np.exp((sims - sims.max(1, keepdims=True)) / tau)
    w /= w.sum(1, keepdims=True)
    c, n = cand[:, :, None, :], nb_rec[:, None, :, :]
    dg = haversine_km(c[..., 2], c[..., 3], n[..., 2], n[..., 3])
    dd = doy_diff(c[..., 1], n[..., 1])
    dy = np.abs(c[..., 0] - n[..., 0])
    ker = np.exp(-0.5 * ((dg / hg) ** 2 + (dd / hd) ** 2 + (dy / hy) ** 2))
    return np.log((ker * w[:, None, :]).sum(-1) + 1e-4)


KNN_GRID = [(tau, hg, hd, hy) for tau in (0.03, 0.07) for hg in (15, 40, 100, 300, 1000)
            for hd in (20, 1e3) for hy in (5, 1e3)]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
    log(f"train {train.shape}  test {test.shape}  sample {sample.shape}")

    tr_cands = np.array([parse_candidates(v) for v in train["candidates"]])     # (n,4,4)
    te_cands = np.array([parse_candidates(v) for v in test["candidates"]])
    labels = np.array([int(json.loads(r)[0]) for r in train["ranking"]])         # truth is listed first
    tr_true = tr_cands[np.arange(len(train)), labels]

    # ---- images -------------------------------------------------------------------
    index = build_image_index(PUBLIC_DIR)
    tr_names = [json.loads(v) for v in train["image_names"]]
    te_names = [json.loads(v) for v in test["image_names"]]
    names = sorted({n for lst in tr_names + te_names for n in lst})
    img_of = {n: i for i, n in enumerate(names)}
    missing = [n for n in names if n not in index]
    assert not missing, f"{len(missing)} images not found, e.g. {missing[:3]}"
    assert all(len(l) == N_IMG for l in tr_names + te_names), "every set must hold 3 photographs"
    tr_imgs = np.array([[img_of[n] for n in l] for l in tr_names])
    te_imgs = np.array([[img_of[n] for n in l] for l in te_names])
    log(f"{len(names)} images indexed")

    # fixed physical window from train bbox statistics (same window for every image)
    u_tr = np.unique(tr_imgs)
    probe = np.random.RandomState(SEED).choice(u_tr, size=min(600, len(u_tr)), replace=False)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        sides = list(ex.map(_bbox_side, [index[names[i]] for i in probe], chunksize=16))
    canvas = max(max(h, w) for _, h, w in sides)
    win = int(np.clip(1.15 * np.percentile([s for s, _, _ in sides], 97), 32, canvas))
    log(f"canvas {canvas}px, fixed window {win}px -> {IMG_SIZE}px (uniform scale {IMG_SIZE / win:.3f})")

    IMGS = np.empty((len(names), IMG_SIZE, IMG_SIZE, 3), np.uint8)
    SIZE = np.zeros((len(names), N_SIZE_FEATS), np.float32)
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        jobs = [(index[n], win, IMG_SIZE) for n in names]
        for i, (arr, feats) in enumerate(ex.map(_process_image, jobs, chunksize=32)):
            IMGS[i] = arr
            SIZE[i] = feats
    log("images preprocessed")

    featurizer = RecordFeaturizer(tr_true)

    # ---- groups = published batches (candidate list shared by all 4 rows) ----------
    def make_groups(df, cands, lab):
        groups = []
        for _, rows in df.groupby("batch_id", sort=True).indices.items():
            rows = np.sort(rows)
            ref = cands[rows[0]]
            assert len(rows) == 4 and all(np.array_equal(cands[r], ref) for r in rows), "malformed batch"
            groups.append({"rows": rows, "cands": ref,
                           "labels": lab[rows] if lab is not None else np.zeros(4, int)})
        return groups

    tr_groups = make_groups(train, tr_cands, labels)
    te_groups = make_groups(test, te_cands, None)
    fold_of = np.empty(len(tr_groups), int)
    fold_of[np.random.RandomState(SEED).permutation(len(tr_groups))] = np.arange(len(tr_groups)) % N_FOLDS
    log(f"train batches {len(tr_groups)}  test batches {len(te_groups)}  fold sizes {np.bincount(fold_of).tolist()}")

    def pack(groups, set_imgs):
        rows = np.stack([g["rows"] for g in groups])                    # (G,4)
        return dict(rows=rows, glab=np.stack([g["labels"] for g in groups]),
                    imgs=set_imgs[rows.reshape(-1)], cands=np.stack([g["cands"] for g in groups]))

    mean_t = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
    std_t = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

    def to_input(img_idx, aug, tta_k=0):
        x = torch.from_numpy(IMGS[img_idx]).to(DEVICE).permute(0, 3, 1, 2).float().div_(255.0)
        if aug:
            # augmentation is load-bearing with ~1.2k training sets per fold.
            # No scale augmentation: pixel size is a physical measurement.
            n = x.shape[0]
            flip_h = torch.rand(n, device=DEVICE) < 0.5
            flip_v = torch.rand(n, device=DEVICE) < 0.5
            x = torch.where(flip_h[:, None, None, None], x.flip(3), x)
            x = torch.where(flip_v[:, None, None, None], x.flip(2), x)
            b = 1.0 + 0.1 * (torch.rand(n, 1, 1, 1, device=DEVICE) - 0.5)
            c = 1.0 + 0.1 * (torch.rand(n, 1, 1, 1, device=DEVICE) - 0.5)
            m = x.mean((1, 2, 3), keepdim=True)
            x = ((x - m) * c + m * b).clamp(0, 1)
            dy, dx = [int(v) for v in torch.randint(-8, 9, (2,))]
            x = torch.roll(x, shifts=(dy, dx), dims=(2, 3))    # uniform grey canvas: roll is safe
        else:
            if tta_k & 1:
                x = x.flip(3)
            if tta_k & 2:
                x = x.flip(2)
        return ((x - mean_t) / std_t).contiguous(memory_format=torch.channels_last)

    def size_tensor(img_idx, norm):
        return torch.from_numpy(((SIZE[img_idx] - norm[0]) / norm[1]).astype(np.float32)).to(DEVICE)

    # ---- training: fixed schedule, identical for every fold ------------------------
    def train_model(groups, fold_seed):
        torch.manual_seed(fold_seed)
        rng = np.random.RandomState(fold_seed)
        rec_dim = featurizer(np.zeros((1, 4))).shape[-1]
        model = QuartetNet(rec_dim).to(DEVICE).to(memory_format=torch.channels_last)
        u_img = np.unique(tr_imgs[np.concatenate([g["rows"] for g in groups])])
        norm = (SIZE[u_img].mean(0), SIZE[u_img].std(0) + 1e-6)          # training images only
        bb = list(model.backbone.parameters())
        bb_ids = {id(p) for p in bb}
        head = [p for p in model.parameters() if id(p) not in bb_ids]
        opt = torch.optim.AdamW([{"params": bb, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}],
                                weight_decay=WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler()
        steps_per_epoch = int(math.ceil(len(groups) / BATCHES_PER_STEP))
        total = steps_per_epoch * N_EPOCHS
        warm = max(1, int(0.05 * total))
        base_lrs = [LR_BACKBONE, LR_HEAD]
        rfeats = featurizer(np.stack([g["cands"] for g in groups]))
        eye4 = torch.eye(4, device=DEVICE)
        probe_p = bb[len(bb) // 2]
        probe0 = probe_p.detach().clone()
        step = 0
        for ep in range(N_EPOCHS):
            order = rng.permutation(len(groups))
            model.train()
            ep_loss = []
            for bi in range(steps_per_epoch):
                sel = order[bi * BATCHES_PER_STEP:(bi + 1) * BATCHES_PER_STEP]
                G = len(sel)
                pk = pack([groups[i] for i in sel], tr_imgs)
                lr_mult = step / warm if step < warm else 0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for pg, b in zip(opt.param_groups, base_lrs):
                    pg["lr"] = b * lr_mult
                flat = pk["imgs"].reshape(-1)
                x = to_input(flat, aug=True)
                sz = size_tensor(flat, norm)
                R = torch.from_numpy(rfeats[sel]).to(DEVICE)
                lab1h = eye4[torch.from_numpy(pk["glab"]).to(DEVICE)]            # (G,4,4) constant
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _, z = model.encode_images(x, sz)
                    U = model.encode_sets(z.float().view(G * 4, N_IMG, -1)).view(G, 4, -1)
                    V = model.encode_records(R.view(G * 4, -1)).view(G, 4, -1)
                U, V = U.float(), V.float()
                sc = model.scale()
                S = sc * torch.einsum("gie,gje->gij", U, V)                    # (G, sets, records)
                pnll = perm_nll_torch(S, lab1h).mean()                          # primary: exact matching NLL
                row_ce = -(F.log_softmax(S, -1) * lab1h).sum(-1).mean()
                # auxiliary CLIP over the minibatch's true records (identical records masked)
                Vt = torch.einsum("gij,gje->gie", lab1h, V).reshape(G * 4, -1)
                Uf = U.reshape(G * 4, -1)
                Gm = sc * Uf @ Vt.t()
                raw = torch.from_numpy(np.take_along_axis(pk["cands"], pk["glab"][:, :, None], 1).reshape(-1, 4)).to(DEVICE)
                eye = torch.eye(G * 4, device=DEVICE)
                same = ((raw[:, None, :] - raw[None, :, :]).abs().sum(-1) < 1e-6).float() - eye
                Gm = Gm - 1e4 * same
                clip = -0.5 * ((F.log_softmax(Gm, 1) * eye).sum(1).mean() + (F.log_softmax(Gm, 0) * eye).sum(0).mean())
                loss = pnll + 0.5 * row_ce + 0.5 * clip
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                ep_loss.append(float(loss))
                step += 1
            if ep == 0 or (ep + 1) % 5 == 0:
                log(f"  epoch {ep + 1}/{N_EPOCHS} loss {np.mean(ep_loss):.4f}")
        # the backbone must actually have been fine-tuned
        assert float((probe_p.detach() - probe0).abs().max()) > 0, "backbone weights did not change"
        return model, norm

    @torch.no_grad()
    def set_outputs(model, norm, groups, set_imgs):
        """Network logits (G,4,4), per-image features (G*4,3,d), set embeddings (G*4,E)."""
        model.eval()
        pk = pack(groups, set_imgs)
        flat = pk["imgs"].reshape(-1)
        fs, zs = [], []
        for i in range(0, len(flat), EMBED_BS):
            idx = flat[i:i + EMBED_BS]
            sz = size_tensor(idx, norm)
            f_acc, z_acc = 0, 0
            for k in range(TTA):
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    f, z = model.encode_images(to_input(idx, aug=False, tta_k=k), sz)
                f_acc = f_acc + F.normalize(f.float(), dim=-1)
                z_acc = z_acc + z.float()
            fs.append(F.normalize(f_acc, dim=-1))
            zs.append(z_acc / TTA)
        f, z = torch.cat(fs), torch.cat(zs)
        G = len(groups)
        U = model.encode_sets(z.view(G * 4, N_IMG, -1)).float()
        R = torch.from_numpy(featurizer(pk["cands"])).to(DEVICE)
        V = model.encode_records(R.view(G * 4, -1)).view(G, 4, -1).float()
        L = model.scale() * torch.einsum("gie,gje->gij", U.view(G, 4, -1), V)
        return pk, L.cpu().numpy(), f.view(G * 4, N_IMG, -1).cpu().numpy(), U.cpu().numpy()

    def knn_search(Q, B):
        Qt = torch.from_numpy(np.ascontiguousarray(Q)).to(DEVICE)
        Bt = torch.from_numpy(np.ascontiguousarray(B)).to(DEVICE)
        v, j = (Qt @ Bt.t()).topk(KNN_K, dim=1)
        return v.cpu().numpy(), j.cpu().numpy()

    def atlas_query(bank, q):
        """Neighbours of query sets in the train-only atlas (raw arrays for the kernel grid)."""
        pk_b, _, f_b, u_b = bank
        rec_b = np.take_along_axis(pk_b["cands"], pk_b["glab"][:, :, None], 1).reshape(-1, 4)  # true record per bank set
        pk_q, L_q, f_q, u_q = q
        si, ii = knn_search(f_q.reshape(-1, f_q.shape[-1]), f_b.reshape(-1, f_b.shape[-1]))
        ss, is_ = knn_search(u_q, u_b)
        cand = np.repeat(pk_q["cands"], 4, 0)                                  # (G*4,4,4)
        return dict(glab=pk_q["glab"], L=L_q, si=si, inb=rec_b[ii // N_IMG], ss=ss, snb=rec_b[is_], cand=cand)

    def atlas_img(r, p):
        ld = knn_logdens(r["si"], r["inb"], np.repeat(r["cand"], N_IMG, 0), *p)
        return ld.reshape(-1, N_IMG, 4).sum(1).reshape(-1, 4, 4)

    def atlas_set(r, p):
        return knn_logdens(r["ss"], r["snb"], r["cand"], *p).reshape(-1, 4, 4)

    # ---- K-fold training: out-of-fold predictions + per-fold test predictions ------
    oof, test_parts = [], []
    for k in range(N_FOLDS):
        fit_g = [tr_groups[i] for i in np.where(fold_of != k)[0]]
        val_g = [tr_groups[i] for i in np.where(fold_of == k)[0]]
        log(f"fold {k}: training on {len(fit_g)} batches, validating on {len(val_g)}")
        model, norm = train_model(fit_g, SEED + k)
        bank = set_outputs(model, norm, fit_g, tr_imgs)             # atlas = this fold's training sets
        r = atlas_query(bank, set_outputs(model, norm, val_g, tr_imgs))
        oof.append(r)
        test_parts.append(atlas_query(bank, set_outputs(model, norm, te_groups, te_imgs)))
        M = perm_marginals_np(r["L"])
        s = provenance_score(np.argsort(-M.reshape(-1, 4), 1), r["glab"].reshape(-1))
        log(f"fold {k}: network-only S {s[0]:.4f} (top1 {s[1]:.3f}, MRR {s[2]:.3f})")
        del model, bank
        torch.cuda.empty_cache()

    # ---- atlas kernels and blend weights fitted on out-of-fold predictions ----------
    def blend_nll(glab, feats, w):
        return perm_nll_np(sum(wi * f for wi, f in zip(w, feats)), glab).mean() + 1e-4 * float(np.sum(np.square(w)))

    def fit_weights(glab, feats, w0):
        res = minimize(lambda w: blend_nll(glab, feats, w), np.asarray(w0, float), method="L-BFGS-B")
        return np.maximum(res.x, 0.0), res.fun

    def evaluate(glab, S):
        M = perm_marginals_np(S)
        return provenance_score(np.argsort(-M.reshape(-1, 4), 1), glab.reshape(-1))

    glab_o = np.concatenate([r["glab"] for r in oof])
    best = {}
    for name, fn in (("img", atlas_img), ("set", atlas_set)):
        scored = [(fit_weights(glab_o, [np.concatenate([fn(r, p) for r in oof])], [1.0])[1], p) for p in KNN_GRID]
        best[name] = min(scored)[1]

    def feats_of(r):
        return [r["L"], atlas_img(r, best["img"]), atlas_set(r, best["set"])]

    per_fold = [feats_of(r) for r in oof]
    feats_o = [np.concatenate([pf[j] for pf in per_fold]) for j in range(3)]
    W, _ = fit_weights(glab_o, feats_o, [1.0, 0.5, 0.5])
    log(f"atlas kernels (tau, km, days, years): img={best['img']} set={best['set']}; blend weights {np.round(W, 3)}")
    for name, S in (("network only", feats_o[0]), ("image atlas only", feats_o[1]),
                    ("set atlas only", feats_o[2]), ("blend (in-sample)", sum(w * f for w, f in zip(W, feats_o)))):
        s = evaluate(glab_o, S)
        log(f"OOF {name:18s}: S {s[0]:.4f}  top1 {s[1]:.4f}  MRR {s[2]:.4f}")
    cf, n_sets = 0.0, 0
    for j in range(N_FOLDS):          # cross-fitted: weights from the other folds
        others = [i for i in range(N_FOLDS) if i != j]
        w_j, _ = fit_weights(np.concatenate([oof[i]["glab"] for i in others]),
                             [np.concatenate([per_fold[i][f] for i in others]) for f in range(3)], [1.0, 0.5, 0.5])
        s = evaluate(oof[j]["glab"], sum(w * f for w, f in zip(w_j, per_fold[j])))
        cf += s[0] * oof[j]["glab"].size
        n_sets += oof[j]["glab"].size
    log(f"OOF blend (cross-fitted): S {cf / n_sets:.4f}")

    # ---- test: average the fold models' blended logits, decode each batch ------------
    S_t = np.mean([sum(w * f for w, f in zip(W, feats_of(r))) for r in test_parts], 0)   # (G,4,4)
    M = perm_marginals_np(S_t)            # exact marginals over the 24 matchings of each batch
    set_ids = test["set_id"].astype(str).str.strip().values
    rank_by_id = {}
    for gi, g in enumerate(te_groups):
        for t, row in enumerate(g["rows"]):
            rank_by_id[set_ids[row]] = [int(c) for c in np.argsort(-(M[gi, t] + 1e-9 * S_t[gi, t]), kind="stable")]

    # ---- submission: exactly the columns set_id, ranking; every test id once ----------
    out_ids = sample["set_id"].astype(str).str.strip().tolist()
    assert set(out_ids) == set(set_ids) and len(out_ids) == len(set_ids), "sample/test id mismatch"
    submission = pd.DataFrame({"set_id": out_ids,
                               "ranking": [json.dumps(rank_by_id[i]) for i in out_ids]})
    assert submission["set_id"].is_unique
    assert all(sorted(json.loads(r)) == [0, 1, 2, 3] for r in submission["ranking"])
    submission_out = SUB_OUT
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    log(f"wrote {len(submission)} rows to {submission_out}")


if __name__ == "__main__":
    main()
