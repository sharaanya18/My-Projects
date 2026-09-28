"""
solution.py - Visible Flood-Failure Mechanism Recognition (Project Eris)

CHALLENGE: flood-failure-mechanisms
DOMAIN:    Computer Vision, multi-label image classification (4 labels)
METRIC:    mean over labels of site-weighted average precision (w_i = 1 / n_site)

Compliance header (maps onto the challenge rules and the Eris guidebook)
------------------------------------------------------------------------
Hardware / runtime (fixed plan)
  - Requires one CUDA GPU (graded on an Nvidia A10G) and always runs the same
    plan: 5 folds x EPOCHS epochs, the same batch size, worker count, image size
    and fp16 autocast. Nothing depends on wall-clock time, CPU count or device
    detection. Elapsed time is printed for logging only and never changes what
    is trained or predicted.
  - The plan was sized to fit the challenge's 30-minute cap: the same script
    measured about 17 minutes end to end on a slower Kaggle T4, and an A10G is
    faster.
Pretrained weights
  - Only a general-purpose ImageNet backbone (timm "convnext_tiny.fb_in22k_ft_in1k")
    is downloaded from the public timm / Hugging Face hub. No self-hosted or
    previously fine-tuned weights are loaded. All fine-tuning happens in this
    script, from the raw photographs, on every run.
Data sources
  - Reads only train.csv, train_targets.csv, test.csv, sample_submission.csv
    and the JPEGs under images/. No external datasets, inspection records,
    coordinates, web lookups of sites, or external annotations are used.
Training data / labels
  - Only the 841 public training photographs and their image-level labels fit
    the model. No synthetic images are generated. No mixup or cutmix is used,
    because blending images would also blend image-level visual evidence.
Test-set usage
  - Test images are scored one image at a time (horizontal-flip TTA only, per
    image). There is no pseudo-labelling, no test-time adaptation, no
    calibration to the test distribution, and no pooling of predictions across
    photographs that share a site prefix. Each prediction describes only the
    evidence visible in that single photograph.
  - The site prefix of an id is used only on TRAIN rows: to build grouped CV
    folds and to weight the loss like the metric. It is never a model input.
Model selection
  - Site-grouped 5-fold CV (StratifiedGroupKFold on the site prefix) runs inside
    this script. It reports the exact challenge metric out of fold, and the five
    fold models are averaged for the test predictions.
Determinism
  - Seeds are fixed at 42 for python, numpy and torch, and cuDNN runs in
    deterministic mode with torch.use_deterministic_algorithms(True). DataLoader
    workers have fixed seeds and a fixed count, so the same inputs give the
    same outputs on every run.
"""

import json
import os
import random
import sys
import time
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

GLOBAL_START = time.time()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
import timm
import torchvision.transforms.v2 as T

# -- Reproducibility --------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic cuBLAS
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
# One fixed backend: the script needs a CUDA GPU and fails loudly without one,
# instead of silently switching to a different (CPU / fp32) plan.
assert torch.cuda.is_available(), "This solution requires a CUDA GPU (graded on an A10G)"
DEVICE = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)


def elapsed() -> float:
    # Logging only; never used to change the training or inference plan.
    return time.time() - GLOBAL_START


# -- Paths: the platform may pass <public_dir> <submission_out> positionally ------
DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dataset/public"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "./working/submission.csv"
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

LABELS = [
    "support_scour",
    "debris_obstruction_or_impact",
    "approach_or_embankment_washout",
    "structural_displacement_or_collapse",
]

# -- Config -----------------------------------------------------------------------
BACKBONE = "convnext_tiny.fb_in22k_ft_in1k"  # strong small ImageNet-22k backbone
# 3:2 landscape input. The audit found ~93% of photos are landscape and most are
# 960x640 (3:2), with the rest 4:3 or portrait. Letterboxing keeps the
# aspect ratio, so tilt and displacement angles are not distorted, and keeps
# the whole frame (a random crop can cut away the only visible evidence).
IMG_H, IMG_W = 384, 576
CACHE_SCALE = 1.1  # cache slightly larger so random crops still cover the scene
N_FOLDS = 5
EPOCHS = 12
BATCH = 16
LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.05
DROP_PATH = 0.1
WARMUP_EPOCHS = 1
EMA_DECAY = 0.99  # weight averaging smooths small-data fine-tuning; 0 disables
NUM_WORKERS = 2  # fixed, not derived from the machine's CPU count


# -- Metric (verbatim logic from the challenge page) -------------------------------
def site_weighted_macro_ap(ids, y_true, y_score):
    groups = [str(i).split("-", 1)[0] for i in ids]
    counts = Counter(groups)
    w = np.asarray([1.0 / counts[g] for g in groups])
    aps = []
    for c in range(len(LABELS)):
        yt = y_true[:, c]
        if yt.sum() == 0 or yt.sum() == len(yt):
            aps.append(np.nan)  # a fold with no positives for a label is not scorable
            continue
        aps.append(average_precision_score(yt, y_score[:, c], sample_weight=w))
    return float(np.nanmean(aps)), aps


# -- Load tables --------------------------------------------------------------------
train = pd.read_csv(f"{DATA_DIR}/train.csv")
targets = pd.read_csv(f"{DATA_DIR}/train_targets.csv")
test = pd.read_csv(f"{DATA_DIR}/test.csv")
sample_sub = pd.read_csv(f"{DATA_DIR}/sample_submission.csv")

tgt = targets["target"].apply(json.loads)
for lab in LABELS:
    targets[lab] = tgt.apply(lambda d: int(d[lab]))
train = train.merge(targets[["id"] + LABELS], on="id", how="left", validate="one_to_one")
assert train[LABELS].notnull().all().all(), "Every training image needs all four labels"
Y = train[LABELS].values.astype(np.float32)
train["site"] = train["id"].astype(str).str.split("-", n=1).str[0]

print(f"Train {train.shape}, Test {test.shape}, train sites {train['site'].nunique()}")
print("Label prevalence (image level):", dict(zip(LABELS, Y.mean(0).round(3))))


# -- Decode every photo once into RAM (small dataset, avoids repeated JPEG decoding) --
def letterbox(path, h, w):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")  # respect camera orientation
    img = ImageOps.pad(img, (w, h), method=Image.BICUBIC, color=(0, 0, 0))
    return np.asarray(img, dtype=np.uint8)


CH, CW = int(IMG_H * CACHE_SCALE), int(IMG_W * CACHE_SCALE)
train_imgs = [letterbox(f"{DATA_DIR}/{p}", CH, CW) for p in train["image_path"]]
test_imgs = [letterbox(f"{DATA_DIR}/{p}", CH, CW) for p in test["image_path"]]
print(f"Images cached at {CH}x{CW} | elapsed {elapsed():.0f}s", flush=True)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
# Geometric augmentation is deliberately mild. Horizontal flip is physically
# valid. Vertical flips and large rotations are excluded because "tilted" and
# "collapsed" are defined relative to gravity, so rotating an intact pier could
# make it look tilted. Crops keep >= 70% of the frame so evidence survives.
train_tf = T.Compose([
    T.ToImage(),
    T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.7, 1.0), ratio=(1.35, 1.65), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.25, 0.25, 0.15, 0.03)], p=0.8),
    T.RandomApply([T.GaussianBlur(3)], p=0.1),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([
    T.ToImage(),
    T.Resize((IMG_H, IMG_W), antialias=True),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(MEAN, STD),
])


class PhotoDS(Dataset):
    def __init__(self, imgs, y=None, w=None, tf=None):
        self.imgs, self.y, self.w, self.tf = imgs, y, w, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(self.imgs[i])
        if self.y is None:
            return x
        return x, torch.from_numpy(self.y[i]), torch.tensor(self.w[i], dtype=torch.float32)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_model():
    return timm.create_model(BACKBONE, pretrained=True, num_classes=len(LABELS), drop_path_rate=DROP_PATH)


def site_weights(sites):
    # Mirror the metric: each training site contributes equal total weight, so
    # heavily photographed sites do not dominate. Normalised to mean 1.
    c = Counter(sites)
    w = np.asarray([1.0 / c[s] for s in sites], dtype=np.float32)
    return w / w.mean()


@torch.no_grad()
def predict(model, imgs):
    # Scores each image independently. Horizontal-flip TTA touches one sample at a time.
    model.eval()
    dl = DataLoader(PhotoDS(imgs, tf=eval_tf), batch_size=BATCH * 2, shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x in dl:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            p = torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float())
        out.append((p / 2).cpu().numpy())
    return np.concatenate(out)


def train_one(tr_idx, fold):
    y_tr = Y[tr_idx]
    w_tr = site_weights(train["site"].values[tr_idx])
    # pos_weight is computed on this fold's training split only. It is capped
    # because AP is a ranking metric and only needs rare positives to get enough
    # gradient, not full re-balancing.
    pos = y_tr.sum(0).clip(min=1)
    pos_weight = torch.tensor(np.sqrt((len(y_tr) - pos) / pos).clip(1, 4), dtype=torch.float32, device=DEVICE)

    g = torch.Generator()
    g.manual_seed(SEED + fold)
    dl = DataLoader(
        PhotoDS([train_imgs[i] for i in tr_idx], y_tr, w_tr, train_tf),
        batch_size=BATCH, shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker, generator=g, pin_memory=True,
    )
    torch.manual_seed(SEED + fold)
    model = make_model().to(DEVICE)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": LR_BACKBONE}, {"params": head, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY
    )
    total = EPOCHS * len(dl)
    warm = WARMUP_EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY) if EMA_DECAY > 0 else None

    for epoch in range(EPOCHS):
        model.train()
        run = 0.0
        for x, y, w in dl:
            x, y, w = x.to(DEVICE, non_blocking=True), y.to(DEVICE), w.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = model(x).float()
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
            loss = (loss.mean(1) * w).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            run += loss.item()
        print(f"  fold {fold} epoch {epoch + 1}/{EPOCHS} loss {run / len(dl):.4f} | elapsed {elapsed():.0f}s", flush=True)
    return ema.module if ema is not None else model


# -- Site-grouped CV: honest OOF metric + fold ensemble for test -------------------------
# Stratify on the 4-bit label pattern so every fold sees each mechanism, and group
# on the site prefix so no site appears on both sides (this mirrors the held-out-site test).
strat = (Y * np.array([1, 2, 4, 8])).sum(1).astype(int)
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(np.zeros(len(train)), strat, groups=train["site"].values))
for tr_idx, va_idx in folds:
    assert not set(train["site"].values[tr_idx]) & set(train["site"].values[va_idx]), "site leak"

oof = np.full(Y.shape, np.nan, dtype=np.float64)
test_pred = np.zeros((len(test), len(LABELS)), dtype=np.float64)
for fold, (tr_idx, va_idx) in enumerate(folds):
    model = train_one(tr_idx, fold)
    oof[va_idx] = predict(model, [train_imgs[i] for i in va_idx])
    fs, _ = site_weighted_macro_ap(train["id"].values[va_idx], Y[va_idx].astype(int), oof[va_idx])
    print(f"Fold {fold} site-weighted macro AP: {fs:.4f} | elapsed {elapsed():.0f}s", flush=True)
    test_pred += predict(model, test_imgs)
    del model
    torch.cuda.empty_cache()

test_pred /= N_FOLDS

cv, per_label = site_weighted_macro_ap(train["id"].values, Y.astype(int), oof)
print()
print(f"OOF site-weighted macro AP ({N_FOLDS} folds): {cv:.4f}")
for lab, ap, prev in zip(LABELS, per_label, Y.mean(0)):
    print(f"  {lab:38s} AP {ap:.4f}  (prevalence {prev:.3f})")

# -- Build submission --------------------------------------------------------------
test_pred = np.clip(test_pred, 0.0, 1.0)
submission = pd.DataFrame({
    "id": test["id"].values,
    "prediction": [json.dumps({lab: round(float(p[k]), 6) for k, lab in enumerate(LABELS)}) for p in test_pred],
})

assert len(submission) == len(test), f"row count {len(submission)} != {len(test)}"
assert submission["id"].is_unique, "duplicate ids"
assert set(submission["id"]) == set(sample_sub["id"]), "ids differ from sample_submission"
assert np.isfinite(test_pred).all() and (test_pred >= 0).all() and (test_pred <= 1).all()
for s in submission["prediction"].head(3):
    assert set(json.loads(s)) == set(LABELS)

submission.to_csv(OUT_PATH, index=False)
print()
print(f"Submission written: {OUT_PATH} {submission.shape}")
print(submission.head(3).to_string())
print(f"Total runtime: {elapsed():.0f}s")
