"""
solution.py - Visible Flood-Failure Mechanism Recognition (Project Eris)

CHALLENGE: flood-failure-mechanisms
DOMAIN:    Computer Vision, multi-label image classification (4 labels)
METRIC:    mean over labels of site-weighted average precision (w_i = 1 / n_site)

Approach
  Two image models are fine-tuned end to end on the training photographs and
  their probabilities are averaged with fixed equal weights:
    A. ConvNeXt-Tiny (ImageNet-22k) with a 4-way sigmoid head.
    B. CLIP ViT-B/16 image encoder whose 4 label prototypes are INITIALISED from
       CLIP text embeddings of each label's codebook definition, then trained
       together with the whole image encoder. The text only sets a starting
       point; every weight is updated by training on the labelled photographs.
       This start helps because the labels have few positives (95-233 per
       label) and are noisy.

Compliance header (maps onto the challenge rules and the Eris guidebook)
------------------------------------------------------------------------
Hardware / runtime (fixed plan)
  - Requires one CUDA GPU (graded on an Nvidia A10G) and always runs the same
    plan: 5 folds x 2 models x a fixed number of epochs, with the same batch
    sizes, worker count, image sizes and fp16 autocast. Nothing depends on
    wall-clock time, CPU count or device detection. Elapsed time is printed
    for logging only and never changes what is trained or predicted.
  - The plan is sized for the challenge's 30-minute cap on an A10G.
Pretrained weights
  - Only general-purpose public checkpoints are downloaded: timm
    "convnext_tiny.fb_in22k_ft_in1k" and Hugging Face "openai/clip-vit-base-patch16".
    No self-hosted or previously fine-tuned weights are loaded. All
    fine-tuning happens in this script, from the raw photographs, on every run.
Data sources
  - Reads only train.csv, train_targets.csv, test.csv, sample_submission.csv
    and the JPEGs under images/. No external datasets, inspection records,
    coordinates, web lookups of sites, or external annotations are used. The
    CLIP prompts below paraphrase the challenge's own label definitions.
Training data / labels
  - Only the 841 public training photographs and their image-level labels fit
    the models. No synthetic images are generated. No mixup or cutmix is used,
    because blending images would also blend image-level visual evidence.
Test-set usage
  - Test images are scored one image at a time (horizontal-flip TTA only, per
    image). There is no pseudo-labelling, no test-time adaptation, no
    calibration or rank-normalisation over the test set, and no pooling of
    predictions across photographs that share a site prefix. Each prediction
    describes only the evidence visible in that single photograph.
  - The site prefix of an id is used only on TRAIN rows, to build grouped CV
    folds. It is never a model input.
Model selection
  - Site-grouped 5-fold CV (StratifiedGroupKFold on the site prefix) runs inside
    this script. It reports the exact challenge metric out of fold for each
    model and for the ensemble. The five fold models of each architecture are
    averaged for the test predictions, and the two architectures are combined
    50/50.
Determinism
  - Seeds are fixed at 42 for python, numpy and torch, and cuDNN runs in
    deterministic mode with torch.use_deterministic_algorithms(True). Attention
    uses the eager implementation. DataLoader workers have fixed seeds and a
    fixed count, so the same inputs give the same outputs on every run.
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
from transformers import CLIPModel, CLIPTokenizer

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

# Label prompts paraphrase the challenge's codebook definitions. They only
# initialise model B's label prototypes, which are then trained on the photos.
PROMPTS = {
    "support_scour": [
        "a photo of scour around a bridge pier foundation",
        "a photo of eroded riverbed exposing a bridge abutment foundation",
        "a photo of soil washed away from around a bridge support",
    ],
    "debris_obstruction_or_impact": [
        "a photo of flood debris piled against a bridge",
        "a photo of logs and trees jammed against bridge piers",
        "a photo of debris blocking the opening under a bridge",
    ],
    "approach_or_embankment_washout": [
        "a photo of a road embankment washed out by a flood",
        "a photo of a bridge approach road collapsed and eroded",
        "a photo of a washed out road next to a bridge",
    ],
    "structural_displacement_or_collapse": [
        "a photo of a collapsed bridge",
        "a photo of a broken bridge deck that has fallen into the river",
        "a photo of a tilted and displaced bridge pier",
    ],
}
NEG_PROMPTS = ["a photo of an intact bridge over a river", "a photo of a river", "a photo of a road"]

# -- Config -----------------------------------------------------------------------
N_FOLDS = 5
NUM_WORKERS = 2  # fixed, not derived from the machine's CPU count
WARMUP_EPOCHS = 1
EMA_DECAY = 0.99  # weight averaging smooths small-data fine-tuning
# Cache 3:2 letterboxed images once. The audit found ~93% of photos are
# landscape, mostly 960x640 (3:2). Letterboxing keeps the aspect ratio, so tilt
# and displacement angles are not distorted, and keeps the whole frame.
CACHE_H, CACHE_W = 422, 633

IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLIP_MEAN, CLIP_STD = [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]
CLIP_NAME = "openai/clip-vit-base-patch16"

MODEL_SPECS = [
    # name, input size (3:2), epochs, batch, lr for the pretrained body / new head
    dict(name="convnext", h=384, w=576, epochs=10, batch=16, lr_body=3e-5, lr_head=1e-3,
         mean=IMAGENET_MEAN, std=IMAGENET_STD),
    # 336x512 is a multiple of the 16-px patch; position embeddings are interpolated.
    dict(name="clip", h=336, w=512, epochs=6, batch=16, lr_body=1e-5, lr_head=2e-4,
         mean=CLIP_MEAN, std=CLIP_STD),
]
ENSEMBLE_WEIGHTS = {"convnext": 0.5, "clip": 0.5}  # fixed, not tuned on test


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


train_imgs = [letterbox(f"{DATA_DIR}/{p}", CACHE_H, CACHE_W) for p in train["image_path"]]
test_imgs = [letterbox(f"{DATA_DIR}/{p}", CACHE_H, CACHE_W) for p in test["image_path"]]
print(f"Images cached at {CACHE_H}x{CACHE_W} | elapsed {elapsed():.0f}s", flush=True)


def make_transforms(spec):
    # Geometric augmentation is deliberately mild. Horizontal flip is physically
    # valid. Vertical flips and large rotations are excluded because "tilted" and
    # "collapsed" are defined relative to gravity, so rotating an intact pier
    # could make it look tilted. Crops keep >= 70% of the frame so evidence survives.
    size = (spec["h"], spec["w"])
    train_tf = T.Compose([
        T.ToImage(),
        T.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(1.35, 1.65), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.25, 0.25, 0.15, 0.03)], p=0.8),
        T.RandomApply([T.GaussianBlur(3)], p=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(spec["mean"], spec["std"]),
    ])
    eval_tf = T.Compose([
        T.ToImage(),
        T.Resize(size, antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(spec["mean"], spec["std"]),
    ])
    return train_tf, eval_tf


class PhotoDS(Dataset):
    def __init__(self, imgs, y=None, tf=None):
        self.imgs, self.y, self.tf = imgs, y, tf

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(self.imgs[i])
        if self.y is None:
            return x
        return x, torch.from_numpy(self.y[i])


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


# -- Model B: CLIP image encoder + text-initialised label prototypes -----------------
_clip_tok = CLIPTokenizer.from_pretrained(CLIP_NAME)


class ClipPrototypeClassifier(nn.Module):
    """logit_k = s * (cos(img, pos_k) - cos(img, neg_k)) + b_k.

    pos_k starts as the mean CLIP text embedding of label k's prompts, and neg_k
    as the mean embedding of generic "intact scene" prompts. Every parameter
    (image encoder, projection, prototypes, bias) is trained on the photos.
    """

    def __init__(self):
        super().__init__()
        clip = CLIPModel.from_pretrained(CLIP_NAME, attn_implementation="eager")
        with torch.no_grad():
            def text_emb(texts):
                tok = _clip_tok(texts, padding=True, return_tensors="pt")
                e = clip.text_projection(clip.text_model(**tok).pooler_output)
                return F.normalize(F.normalize(e, dim=-1).mean(0), dim=-1)

            pos = torch.stack([text_emb(PROMPTS[lab]) for lab in LABELS])
            neg = text_emb(NEG_PROMPTS).unsqueeze(0).repeat(len(LABELS), 1)
        self.vision = clip.vision_model
        self.proj = clip.visual_projection
        self.pos = nn.Parameter(pos)
        self.neg = nn.Parameter(neg)
        self.bias = nn.Parameter(torch.zeros(len(LABELS)))
        self.scale = 50.0  # fixed temperature so initial logits are a few units wide
        del clip  # the text tower is only needed for the initialisation above

    def head_parameters(self):
        return [self.pos, self.neg, self.bias]

    def forward(self, x):
        f = self.vision(pixel_values=x, interpolate_pos_encoding=True).pooler_output
        f = F.normalize(self.proj(f), dim=-1)
        pos = F.normalize(self.pos, dim=-1)
        neg = F.normalize(self.neg, dim=-1)
        return self.scale * (f @ pos.T - f @ neg.T) + self.bias  # (batch, labels)


class ConvNeXtClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=len(LABELS), drop_path_rate=0.1
        )

    def head_parameters(self):
        return list(self.net.get_classifier().parameters())

    def forward(self, x):
        return self.net(x)


BUILDERS = {"convnext": ConvNeXtClassifier, "clip": ClipPrototypeClassifier}


@torch.no_grad()
def predict(model, imgs, eval_tf, batch):
    # Scores each image independently. Horizontal-flip TTA touches one sample at a time.
    model.eval()
    dl = DataLoader(PhotoDS(imgs, tf=eval_tf), batch_size=batch * 2, shuffle=False, num_workers=NUM_WORKERS)
    out = []
    for x in dl:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            p = torch.sigmoid(model(x).float()) + torch.sigmoid(model(torch.flip(x, dims=[3])).float())
        out.append((p / 2).cpu().numpy())
    return np.concatenate(out)


def train_one(spec, train_tf, tr_idx, fold):
    y_tr = Y[tr_idx]
    # pos_weight is computed on this fold's training split only. It is capped
    # because AP is a ranking metric and only needs rare positives to get enough
    # gradient, not full re-balancing.
    pos = y_tr.sum(0).clip(min=1)
    pos_weight = torch.tensor(np.sqrt((len(y_tr) - pos) / pos).clip(1, 4), dtype=torch.float32, device=DEVICE)

    g = torch.Generator()
    g.manual_seed(SEED + fold)
    dl = DataLoader(
        PhotoDS([train_imgs[i] for i in tr_idx], y_tr, train_tf),
        batch_size=spec["batch"], shuffle=True, drop_last=True, num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker, generator=g, pin_memory=True,
    )
    torch.manual_seed(SEED + fold)
    model = BUILDERS[spec["name"]]().to(DEVICE)
    head = model.head_parameters()
    head_ids = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in head_ids]
    opt = torch.optim.AdamW(
        [{"params": body, "lr": spec["lr_body"], "weight_decay": 0.05},
         {"params": head, "lr": spec["lr_head"], "weight_decay": 0.0}]
    )
    total = spec["epochs"] * len(dl)
    warm = WARMUP_EPOCHS * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm)))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY)

    for epoch in range(spec["epochs"]):
        model.train()
        run = 0.0
        for x, y in dl:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
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
            run += loss.item()
        print(f"  [{spec['name']}] fold {fold} epoch {epoch + 1}/{spec['epochs']} loss {run / len(dl):.4f} "
              f"| elapsed {elapsed():.0f}s", flush=True)
    return ema.module


# -- Site-grouped CV: honest OOF metric + fold ensemble for test -------------------------
# Stratify on the 4-bit label pattern so every fold sees each mechanism, and group
# on the site prefix so no site appears on both sides (this mirrors the held-out-site test).
strat = (Y * np.array([1, 2, 4, 8])).sum(1).astype(int)
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(np.zeros(len(train)), strat, groups=train["site"].values))
for tr_idx, va_idx in folds:
    assert not set(train["site"].values[tr_idx]) & set(train["site"].values[va_idx]), "site leak"

ids = train["id"].values
oof = {s["name"]: np.zeros(Y.shape, dtype=np.float64) for s in MODEL_SPECS}
test_pred = {s["name"]: np.zeros((len(test), len(LABELS)), dtype=np.float64) for s in MODEL_SPECS}
for spec in MODEL_SPECS:
    train_tf, eval_tf = make_transforms(spec)
    for fold, (tr_idx, va_idx) in enumerate(folds):
        model = train_one(spec, train_tf, tr_idx, fold)
        oof[spec["name"]][va_idx] = predict(model, [train_imgs[i] for i in va_idx], eval_tf, spec["batch"])
        fs, _ = site_weighted_macro_ap(ids[va_idx], Y[va_idx].astype(int), oof[spec["name"]][va_idx])
        print(f"[{spec['name']}] fold {fold} site-weighted macro AP: {fs:.4f} | elapsed {elapsed():.0f}s", flush=True)
        test_pred[spec["name"]] += predict(model, test_imgs, eval_tf, spec["batch"]) / N_FOLDS
        del model
        torch.cuda.empty_cache()

oof["ensemble"] = sum(ENSEMBLE_WEIGHTS[k] * oof[k] for k in ENSEMBLE_WEIGHTS)
final = sum(ENSEMBLE_WEIGHTS[k] * test_pred[k] for k in ENSEMBLE_WEIGHTS)

print()
for name, P in oof.items():
    cv, per_label = site_weighted_macro_ap(ids, Y.astype(int), P)
    fold_scores = [site_weighted_macro_ap(ids[v], Y[v].astype(int), P[v])[0] for _, v in folds]
    print(f"[{name}] OOF site-weighted macro AP {cv:.4f} | fold mean {np.mean(fold_scores):.4f} "
          f"| per label {np.round(per_label, 3)}")
np.save(os.path.join(os.path.dirname(OUT_PATH) or ".", "oof_predictions.npy"), np.stack([oof[k] for k in oof]))

# -- Build submission --------------------------------------------------------------
final = np.clip(final, 0.0, 1.0)
submission = pd.DataFrame({
    "id": test["id"].values,
    "prediction": [json.dumps({lab: round(float(p[k]), 6) for k, lab in enumerate(LABELS)}) for p in final],
})

assert len(submission) == len(test), f"row count {len(submission)} != {len(test)}"
assert submission["id"].is_unique, "duplicate ids"
assert set(submission["id"]) == set(sample_sub["id"]), "ids differ from sample_submission"
assert np.isfinite(final).all() and (final >= 0).all() and (final <= 1).all()
for s in submission["prediction"].head(3):
    assert set(json.loads(s)) == set(LABELS)

submission.to_csv(OUT_PATH, index=False)
print()
print(f"Submission written: {OUT_PATH} {submission.shape}")
print(submission.head(3).to_string())
print(f"Total runtime: {elapsed():.0f}s")
