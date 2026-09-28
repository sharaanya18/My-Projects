"""
solution.py - Visible Flood-Failure Mechanism Recognition (Project Eris)

CHALLENGE: flood-failure-mechanisms
DOMAIN:    Computer Vision, multi-label image classification (4 labels)
METRIC:    mean over labels of site-weighted average precision (w_i = 1 / n_site)

Approach: fine-tuned ConvNeXt + trained heads on pretrained encoders
---------------------------------------------------------------------
The training set is small (841 photos, 54 sites) and several labels have
positives at only 14-28 sites. Site-grouped out-of-fold experiments showed
that (a) heavily regularised heads trained on strong general-purpose
encoders transfer to unseen sites at least as well as end-to-end
fine-tuning, (b) different encoders carry different mechanisms (DINOv2 ->
collapse/washout, SigLIP/CLIP -> scour/debris), and (c) a ConvNeXt
fine-tuned end to end on full frames is complementary to both (OOF
correlation ~0.6), so all of them are blended.

  Members (backbones run in fp16 on the GPU; every head is trained here):
    P1  SigLIP-B/16-384   mean embedding of 3 tiles              -> logistic regression
    P2  DINOv2-B/14       [CLS, mean-patch] of full+flip+tiles   -> logistic regression
    P3  CLIP ViT-B/16     mean embedding of 3 tiles              -> logistic regression
    A   SigLIP-so400m-384 mean embedding of 3 tiles              -> text-anchored linear head
        logit = s*<x, t> + <x, delta> + b, where t is the direction from
        "intact scene" to the label's codebook description in SigLIP's
        joint text-image space; s, b and delta are fitted on the training
        labels with an L2 pull of delta toward 0. An unanchored probe on the
        same features overfits site noise (OOF 0.253 vs 0.277 anchored).
    F   ConvNeXt-Tiny (ImageNet-22k) fine-tuned end to end on letterboxed
        full frames: 10 epochs, warmup + cosine, body LR 3e-5 / head LR 1e-3,
        EMA 0.99, capped pos_weight BCE, mild colour/blur/scale augmentation,
        horizontal-flip TTA.
  frozen = 0.4 * z(A) + 0.6 * mean_k z(logit P_k)
  final  = sigmoid(0.7 * z(frozen) + 0.3 * z(logit F))
  z() standardises each member with the mean/std of its OUT-OF-FOLD training
  scores, so every test photo is scored on its own. Weights were chosen on
  grouped OOF and sit on a flat plateau (0.3-0.5 for F give the same AP).

  Views: each photo is resized so its short side equals the model's native
  size (aspect preserved, so tilt angles are not distorted), then three
  overlapping native-size squares (left/centre/right, or top/centre/bottom
  for portrait photos) cover the whole frame at native resolution. Tile-mean
  embeddings beat full-frame embeddings for every encoder tested, consistent
  with failure evidence occupying a small part of the frame. No rotations
  or vertical flips are used ("tilted"/"collapsed" are defined by gravity).

  Heads: per-label, sample weights 1/n_site so training matches the
  site-weighted metric; logistic regressions use C=0.003 on standardised
  features. Both choices raised grouped OOF AP for every encoder tested.

Compliance header (maps onto the challenge rules and the Eris guidebook)
------------------------------------------------------------------------
Hardware / runtime (fixed plan)
  - Requires one CUDA GPU (graded on an Nvidia A10G). Always the same plan:
    4 frozen encoders with fixed views, 5 folds of heads, and 5 folds x 10
    epochs of ConvNeXt fine-tuning with fixed batch size and worker count.
    Nothing depends on wall-clock time, CPU count or device detection, and
    there is no fallback path. Measured on a Kaggle T4: ~6.5 min for the
    frozen encoders + ~17 min for ConvNeXt; an A10G is roughly 2x faster.
Pretrained weights
  - Only general-purpose public checkpoints from Hugging Face:
    google/siglip-base-patch16-384, google/siglip-so400m-patch14-384,
    facebook/dinov2-base, openai/clip-vit-base-patch16, and timm
    convnext_tiny.fb_in22k_ft_in1k. No self-hosted or previously fine-tuned
    weights; all fine-tuning happens in this script on every run.
Data sources
  - Reads only train.csv, train_targets.csv, test.csv, sample_submission.csv
    and the JPEGs under images/. No external datasets, inspection records,
    coordinates, web lookups or external annotations. The text prompts
    paraphrase the challenge's own label definitions and only set the
    anchor direction of head A.
Training data / labels
  - Only the 841 public training photographs and their labels fit the heads
    and the fine-tuned ConvNeXt.
    No synthetic images, no mixup/cutmix, no external labels.
Test-set usage
  - Each test photo is scored independently: no pseudo-labelling, no
    test-time adaptation, no rank/normalisation computed over the test
    set, no pooling across photos that share a site prefix.
  - The site prefix is used only on TRAIN rows: to build grouped folds and
    as a 1/n_site training weight. It is never a model input feature.
Model selection
  - Site-grouped 5-fold CV runs inside this script with folds balanced on
    sites, images, positive images and positive SITES per label (no site
    on both sides). It reports the exact challenge metric out of fold,
    per label, per fold and per member.
Determinism
  - Fixed seeds, deterministic cuDNN/cuBLAS (torch.use_deterministic_algorithms),
    eager attention, fixed image order, DataLoader shuffling from a seeded
    generator with a fixed worker count and seeded workers, and
    deterministic L-BFGS solvers for every head.
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
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from scipy.optimize import minimize
from transformers import AutoModel, AutoTokenizer

# -- Reproducibility --------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for deterministic cuBLAS
torch.use_deterministic_algorithms(True)
# One fixed backend: CUDA + fp16 autocast. The script fails loudly without a GPU
# instead of switching to a different plan.
assert torch.cuda.is_available(), "This solution requires a CUDA GPU (graded on an A10G)"
DEVICE = "cuda"
print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)


def elapsed() -> float:
    return time.time() - GLOBAL_START  # logging only


DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dataset/public"
OUT_PATH = sys.argv[2] if len(sys.argv) > 2 else "./working/submission.csv"
os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)

LABELS = [
    "support_scour",
    "debris_obstruction_or_impact",
    "approach_or_embankment_washout",
    "structural_displacement_or_collapse",
]

# Prompts paraphrase the challenge's codebook definitions (zero-shot prior only).
PROMPTS = {
    "support_scour": [
        "a photo of scour around a bridge pier foundation",
        "a photo of eroded riverbed exposing a bridge abutment foundation",
        "a photo of soil washed away from around a bridge support",
        "a photo of an exposed bridge pier footing after a flood",
        "a photo of a bridge abutment undermined by erosion",
        "a photo of loss of bed material at the base of a bridge column",
    ],
    "debris_obstruction_or_impact": [
        "a photo of flood debris piled against a bridge",
        "a photo of logs and trees jammed against bridge piers",
        "a photo of debris blocking the opening under a bridge",
        "a photo of driftwood lodged on a bridge railing",
        "a photo of a pile of branches and trash caught on a bridge",
        "a photo of uprooted trees stuck against a bridge deck",
    ],
    "approach_or_embankment_washout": [
        "a photo of a road embankment washed out by a flood",
        "a photo of a bridge approach road collapsed and eroded",
        "a photo of a washed out road next to a bridge",
        "a photo of a breached embankment with the fill washed away",
        "a photo of a road undercut by flood water leaving a gap",
        "a photo of asphalt pavement broken where the ground beneath washed away",
    ],
    "structural_displacement_or_collapse": [
        "a photo of a collapsed bridge",
        "a photo of a broken bridge deck that has fallen into the river",
        "a photo of a tilted and displaced bridge pier",
        "a photo of a bridge span knocked off its supports",
        "a photo of a fractured concrete bridge girder",
        "a photo of a destroyed bridge with twisted wreckage",
    ],
}
NEG_PROMPTS = [
    "a photo of an intact bridge over a river",
    "a photo of a river",
    "a photo of a road",
    "a photo of an undamaged bridge",
    "a photo of a muddy landscape after a flood",
]

# -- Config (fixed plan) ------------------------------------------------------------
N_FOLDS = 5
PROBE_C = 0.003
ANCHOR_LAMBDA = 1e-3  # L2 pull of the anchored probe toward the codebook text direction
ANCHOR_WEIGHT = 0.4  # blend weight of the anchored probe; the rest is split over probes
FINETUNE_WEIGHT = 0.3  # weight of the fine-tuned ConvNeXt vs the frozen-encoder blend
BATCH = 32

CLIP_NORM = ([0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758])
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
HALF_NORM = ([0.5] * 3, [0.5] * 3)

ENCODERS = {
    # name: (hf id, native size, patch, normalisation, views)
    "siglipb": ("google/siglip-base-patch16-384", 384, 16, HALF_NORM, ("tiles",)),
    "dinob": ("facebook/dinov2-base", 224, 14, IMAGENET_NORM, ("full", "flip", "tiles")),
    "clipb16": ("openai/clip-vit-base-patch16", 224, 16, CLIP_NORM, ("tiles",)),
    "siglipso": ("google/siglip-so400m-patch14-384", 384, 14, HALF_NORM, ("tiles",)),
}
PROBE_MEMBERS = ["siglipb", "dinob", "clipb16"]
ANCHOR_MEMBER = "siglipso"


# -- Metric (verbatim logic from the challenge page) -------------------------------
def site_weights(ids):
    groups = [str(i).split("-", 1)[0] for i in ids]
    counts = Counter(groups)
    return np.asarray([1.0 / counts[g] for g in groups])


def per_label_ap(ids, y_true, y_score):
    w = site_weights(ids)
    out = []
    for c in range(len(LABELS)):
        yt = y_true[:, c]
        out.append(np.nan if yt.min() == yt.max() else average_precision_score(yt, y_score[:, c], sample_weight=w))
    return np.array(out)


def site_weighted_macro_ap(ids, y_true, y_score):
    return float(np.nanmean(per_label_ap(ids, y_true, y_score)))


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
Y = train[LABELS].values.astype(int)
SITE = train["id"].astype(str).str.split("-", n=1).str[0].values
IDS = train["id"].values
N_TR = len(train)
print(f"Train {train.shape}, Test {test.shape}, train sites {len(set(SITE))}")
print("Positive images:", dict(zip(LABELS, Y.sum(0).tolist())))


# -- Site-grouped folds balanced on positive SITES ------------------------------------
def balanced_group_folds(k=N_FOLDS, seed=0, n_trials=4000):
    """Search random site->fold assignments and keep the most even one in
    images, sites, positive images per label and positive sites per label.
    StratifiedGroupKFold on the 4-bit pattern left some folds with only 2
    positive sites for a label; the metric is site-weighted, so positive-site
    balance is what keeps every fold's AP informative."""
    rng = np.random.default_rng(seed)
    sites = np.array(sorted(set(SITE)))
    si = np.searchsorted(sites, SITE)
    n_img = np.bincount(si, minlength=len(sites))
    pos_img = np.stack([np.bincount(si, weights=Y[:, c], minlength=len(sites)) for c in range(4)], 1)
    feats = np.concatenate([n_img[:, None], np.ones((len(sites), 1)), pos_img, 3 * (pos_img > 0)], 1)
    target = feats.sum(0) / k
    best, best_cost = None, np.inf
    for _ in range(n_trials):
        a = rng.permutation(len(sites)) % k
        tot = np.stack([feats[a == f].sum(0) for f in range(k)])
        cost = (((tot - target) / (target + 1e-9)) ** 2).sum()
        if cost < best_cost:
            best, best_cost = a, cost
    fa = best[si]
    return [(np.where(fa != f)[0], np.where(fa == f)[0]) for f in range(k)]


FOLDS = balanced_group_folds()
for f, (tr_idx, va_idx) in enumerate(FOLDS):
    assert not set(SITE[tr_idx]) & set(SITE[va_idx]), "site leak"
    print(f"fold {f}: {len(va_idx)} imgs, {len(set(SITE[va_idx]))} sites, positive sites "
          f"{[len(set(SITE[va_idx][Y[va_idx, c] == 1])) for c in range(4)]}")

# -- Decode every photo once ----------------------------------------------------------
paths = [f"{DATA_DIR}/{p}" for p in list(train["image_path"]) + list(test["image_path"])]
IMAGES = [ImageOps.exif_transpose(Image.open(p)).convert("RGB") for p in paths]
print(f"Decoded {len(IMAGES)} photos | elapsed {elapsed():.0f}s", flush=True)


def make_views(img, S, P, mean, std):
    """Aspect-preserving resize (short side = S, long side a multiple of the patch),
    plus 3 overlapping S x S tiles along the long side."""
    w, h = img.size
    if w >= h:
        nh, nw = S, max(S, int(round(w * S / h / P)) * P)
    else:
        nw, nh = S, max(S, int(round(h * S / w / P)) * P)
    r = img.resize((nw, nh), Image.BICUBIC)

    def to_t(im):
        x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - mean) / std

    L = max(nw, nh)
    offs = [0, (L - S) // 2, L - S]
    tiles = [to_t(r.crop((o, 0, o + S, S)) if w >= h else r.crop((0, o, S, o + S))) for o in offs]
    return to_t(r), tiles


@torch.no_grad()
def extract(name):
    hf, S, P, (mean, std), views = ENCODERS[name]
    model = AutoModel.from_pretrained(hf, attn_implementation="eager").eval().to(DEVICE)
    mean_t, std_t = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)

    if name.startswith("dino"):
        def embed(x):
            h = model(pixel_values=x).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], 1)
    elif name.startswith("siglip"):
        def embed(x):
            return model.vision_model(pixel_values=x).pooler_output
    else:  # CLIP
        def embed(x):
            return model.visual_projection(model.vision_model(pixel_values=x).pooler_output)

    def run(batch):
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            return embed(torch.stack(batch).to(DEVICE)).float().cpu()

    tile_feats, full_feats, flip_feats = [], [], []
    buf_t, buf_f = [], []  # tiles are all S x S -> large batches
    for i, img in enumerate(IMAGES):
        full, tiles = make_views(img, S, P, mean_t, std_t)
        buf_t.extend(tiles)
        if "full" in views:  # full frames differ in shape by orientation -> one image at a time
            e = run([full, torch.flip(full, dims=[2])])
            full_feats.append(e[0]); flip_feats.append(e[1])
        if len(buf_t) >= BATCH * 3 or i == len(IMAGES) - 1:
            tile_feats.append(run(buf_t)); buf_t = []
    tiles = torch.cat(tile_feats).view(len(IMAGES), 3, -1)
    out = {"tiles": F.normalize(tiles, dim=-1).numpy()}
    if "full" in views:
        out["full"] = F.normalize(torch.stack(full_feats), dim=-1).numpy()
        out["flip"] = F.normalize(torch.stack(flip_feats), dim=-1).numpy()

    if name == ANCHOR_MEMBER:
        tok = AutoTokenizer.from_pretrained(hf)

        def text(texts):
            t = tok(texts, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(DEVICE)
            e = model.text_model(**t).pooler_output  # SigLIP text embedding (head applied)
            return F.normalize(F.normalize(e.float(), dim=-1).mean(0), dim=0).cpu().numpy()

        out["text_pos"] = np.stack([text(PROMPTS[lab]) for lab in LABELS])
        out["text_neg"] = text(NEG_PROMPTS)
    del model
    torch.cuda.empty_cache()
    print(f"  features {name}: {out['tiles'].shape} | elapsed {elapsed():.0f}s", flush=True)
    return out


def l2(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


FEATS = {name: extract(name) for name in ENCODERS}


def probe_matrix(name):
    """Tile-mean embedding; DINOv2 additionally averages full frame and its flip."""
    f = FEATS[name]
    tmean = l2(f["tiles"].mean(1))
    if "full" in f:
        return l2(f["full"] + f["flip"] + tmean)
    return tmean


# -- Probes: out-of-fold for evaluation, fold-averaged for test ------------------------
SW = site_weights(IDS)  # 1 / n_site on TRAIN rows only


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


oof_logit, test_logit = {}, {}
for name in PROBE_MEMBERS:
    X = probe_matrix(name)
    Xtr, Xte = X[:N_TR], X[N_TR:]
    oof = np.zeros((N_TR, 4))
    te = np.zeros((len(Xte), 4))
    for tr_idx, va_idx in FOLDS:
        sc = StandardScaler().fit(Xtr[tr_idx])
        w = SW[tr_idx] * len(tr_idx) / SW[tr_idx].sum()
        for c in range(4):
            m = LogisticRegression(C=PROBE_C, max_iter=5000)
            m.fit(sc.transform(Xtr[tr_idx]), Y[tr_idx, c], sample_weight=w)
            oof[va_idx, c] = m.predict_proba(sc.transform(Xtr[va_idx]))[:, 1]
            te[:, c] += m.predict_proba(sc.transform(Xte))[:, 1] / N_FOLDS
    oof_logit[name], test_logit[name] = logit(oof), logit(te)

# -- Text-anchored probe on SigLIP-so400m ----------------------------------------------
# logit_c(x) = s_c * <x, t_c> + <x, delta_c> + b_c, with t_c = text(label c) - text(intact).
# s_c, b_c and delta_c are all fitted on the training labels (site-weighted BCE) with an
# L2 penalty ANCHOR_LAMBDA * ||delta_c||^2 that keeps the learned direction close to the
# codebook text direction. A free probe on these features overfits site-specific noise
# (grouped OOF 0.253); anchoring it to the text direction generalises better (0.277).
def fit_anchored(X, y, w, t):
    a = X @ t

    def objective(p):
        s, b, d = p[0], p[1], p[2:]
        z = s * a + X @ d + b
        loss = np.sum(w * (np.logaddexp(0.0, z) - y * z)) / w.sum() + ANCHOR_LAMBDA * d @ d
        g = w * (1.0 / (1.0 + np.exp(-z)) - y) / w.sum()
        return loss, np.concatenate([[g @ a, g.sum()], X.T @ g + 2 * ANCHOR_LAMBDA * d])

    p0 = np.zeros(X.shape[1] + 2)
    p0[0] = 10.0  # start from a scaled zero-shot direction
    return minimize(objective, p0, jac=True, method="L-BFGS-B", options=dict(maxiter=500)).x


zf = FEATS[ANCHOR_MEMBER]
X = l2(zf["tiles"].mean(1))
Xtr, Xte = X[:N_TR], X[N_TR:]
oof = np.zeros((N_TR, 4))
te = np.zeros((len(Xte), 4))
for tr_idx, va_idx in FOLDS:
    mu = Xtr[tr_idx].mean(0)  # centred with training-fold statistics only
    for c in range(4):
        t = zf["text_pos"][c] - zf["text_neg"]
        p = fit_anchored(Xtr[tr_idx] - mu, Y[tr_idx, c], SW[tr_idx], t)
        score = lambda Z: p[0] * ((Z - mu) @ t) + (Z - mu) @ p[2:] + p[1]
        oof[va_idx, c] = score(Xtr[va_idx])
        te[:, c] += score(Xte) / N_FOLDS
oof_logit["anchored_" + ANCHOR_MEMBER], test_logit["anchored_" + ANCHOR_MEMBER] = oof, te


# -- Fine-tuned member: ConvNeXt-Tiny trained end to end on the photos ------------------
# Full-frame letterboxed photos (aspect kept, whole frame visible) complement the
# tile-based frozen heads. Recipe: short warmup, cosine decay, lower LR for the
# pretrained body than the new head, EMA weights, capped positive weights, and
# mild photometric / scale augmentation. Horizontal flip is the only geometric
# flip: "tilted" and "collapsed" are defined relative to gravity.
FT_H, FT_W = 384, 576
FT_CACHE_H, FT_CACHE_W = 422, 633  # letterbox cache, cropped/resized to FT_H x FT_W
FT_EPOCHS, FT_BATCH, FT_WARMUP = 10, 16, 1
FT_LR_BODY, FT_LR_HEAD, FT_EMA = 3e-5, 1e-3, 0.99
NUM_WORKERS = 2  # fixed, not derived from the machine


def letterbox(img, h, w):
    return np.asarray(ImageOps.pad(img, (w, h), method=Image.BICUBIC, color=(0, 0, 0)), dtype=np.uint8)


FT_IMAGES = [letterbox(img, FT_CACHE_H, FT_CACHE_W) for img in IMAGES]
ft_train_tf = T.Compose([
    T.ToImage(),
    T.RandomResizedCrop((FT_H, FT_W), scale=(0.7, 1.0), ratio=(1.35, 1.65), antialias=True),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.ColorJitter(0.25, 0.25, 0.15, 0.03)], p=0.8),
    T.RandomApply([T.GaussianBlur(3)], p=0.1),
    T.ToDtype(torch.float32, scale=True),
    T.Normalize(*IMAGENET_NORM),
])
ft_eval_tf = T.Compose([T.ToImage(), T.Resize((FT_H, FT_W), antialias=True),
                        T.ToDtype(torch.float32, scale=True), T.Normalize(*IMAGENET_NORM)])


class PhotoDS(Dataset):
    def __init__(self, imgs, y=None, tf=None):
        self.imgs, self.y, self.tf = imgs, y, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(self.imgs[i])
        return x if self.y is None else (x, torch.from_numpy(self.y[i]))


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


@torch.no_grad()
def ft_predict(model, idx):
    """Mean sigmoid over the photo and its horizontal flip (per-photo TTA)."""
    model.eval()
    dl = DataLoader(PhotoDS([FT_IMAGES[i] for i in idx], tf=ft_eval_tf), batch_size=FT_BATCH * 2,
                    shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x in dl:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            p = torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float())
        out.append((p / 2).cpu().numpy())
    return np.concatenate(out)


def ft_train(tr_idx, fold):
    y_tr = Y[tr_idx].astype(np.float32)
    pos = y_tr.sum(0).clip(min=1)
    pos_weight = torch.tensor(np.sqrt((len(y_tr) - pos) / pos).clip(1, 4), dtype=torch.float32, device=DEVICE)
    g = torch.Generator()
    g.manual_seed(SEED + fold)
    dl = DataLoader(PhotoDS([FT_IMAGES[i] for i in tr_idx], y_tr, ft_train_tf), batch_size=FT_BATCH,
                    shuffle=True, drop_last=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                    generator=g, pin_memory=True)
    torch.manual_seed(SEED + fold)
    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=4,
                              drop_path_rate=0.1).to(DEVICE)
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": body, "lr": FT_LR_BODY, "weight_decay": 0.05},
                             {"params": head, "lr": FT_LR_HEAD, "weight_decay": 0.0}])
    total, warm = FT_EPOCHS * len(dl), FT_WARMUP * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda")
    ema = timm.utils.ModelEmaV3(model, decay=FT_EMA)
    for epoch in range(FT_EPOCHS):
        model.train()
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x).float()
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
    return ema.module


oof = np.zeros((N_TR, 4))
te = np.zeros((len(test), 4))
test_idx = np.arange(N_TR, len(IMAGES))
for fold, (tr_idx, va_idx) in enumerate(FOLDS):
    model = ft_train(tr_idx, fold)
    oof[va_idx] = ft_predict(model, va_idx)
    te += ft_predict(model, test_idx) / N_FOLDS
    print(f"  convnext fold {fold}: macro AP {site_weighted_macro_ap(IDS[va_idx], Y[va_idx], oof[va_idx]):.4f} "
          f"| elapsed {elapsed():.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
oof_logit["convnext_ft"], test_logit["convnext_ft"] = logit(oof), logit(te)

# -- Blend: standardise each member with its OOF training statistics ------------------
def zscore(name, v):
    return (v - STATS[name][0]) / STATS[name][1]


def frozen_score(scores):
    probes = np.mean([zscore(m, scores[m]) for m in PROBE_MEMBERS], 0)
    return ANCHOR_WEIGHT * zscore("anchored_" + ANCHOR_MEMBER, scores["anchored_" + ANCHOR_MEMBER]) \
        + (1 - ANCHOR_WEIGHT) * probes


def blend(scores):
    """scores: dict member -> (n,4). Every standardisation uses OUT-OF-FOLD train
    statistics only, so a test photo's score never depends on other test photos."""
    fz = frozen_score(scores)
    fz = (fz - FROZEN_STATS[0]) / FROZEN_STATS[1]
    z = (1 - FINETUNE_WEIGHT) * fz + FINETUNE_WEIGHT * zscore("convnext_ft", scores["convnext_ft"])
    return 1.0 / (1.0 + np.exp(-z))


STATS = {m: (v.mean(0), v.std(0) + 1e-9) for m, v in oof_logit.items()}
_fz = frozen_score(oof_logit)
FROZEN_STATS = (_fz.mean(0), _fz.std(0) + 1e-9)
oof_final = blend(oof_logit)
final = blend(test_logit)

# -- Report ---------------------------------------------------------------------------
print()
for name, P in list(oof_logit.items()) + [("ENSEMBLE", oof_final)]:
    pooled = per_label_ap(IDS, Y, P)
    folds = [site_weighted_macro_ap(IDS[v], Y[v], P[v]) for _, v in FOLDS]
    print(f"[{name:12s}] OOF pooled {np.mean(pooled):.4f} | fold mean {np.mean(folds):.4f} "
          f"± {np.std(folds):.4f} (min {np.min(folds):.4f}) | per label {np.round(pooled, 3)}")
np.save(os.path.join(os.path.dirname(OUT_PATH) or ".", "oof_predictions.npy"), oof_final)

# -- Build submission -------------------------------------------------------------------
final = np.clip(final, 0.0, 1.0)
submission = pd.DataFrame({
    "id": test["id"].values,
    "prediction": [json.dumps({lab: round(float(p[k]), 6) for k, lab in enumerate(LABELS)}) for p in final],
})
assert len(submission) == len(test), f"row count {len(submission)} != {len(test)}"
assert submission["id"].is_unique, "duplicate ids"
assert set(submission["id"]) == set(sample_sub["id"]), "ids differ from sample_submission"
assert np.isfinite(final).all() and (final >= 0).all() and (final <= 1).all()
for s in submission["prediction"]:
    assert set(json.loads(s)) == set(LABELS)

submission.to_csv(OUT_PATH, index=False)
print()
print(f"Submission written: {OUT_PATH} {submission.shape}")
print(submission.head(3).to_string())
print(f"Total runtime: {elapsed():.0f}s")
