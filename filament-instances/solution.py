#!/usr/bin/env python3
"""Filament Instances Across Scales: end-to-end solution.

Usage:  python3 solution.py <public_dir> <submission_out>

Pipeline (see APPROACH.md for the reasoning behind each step):
  1. Load every image into memory. Measure the filament width of each training crop
     from its own instance masks.
  2. Train a U-Net from scratch on crops rescaled so the filaments are 4.3 to 8.5 px
     wide. Masks are re-rasterised from the normalised polygons with the task's exact
     pixel-centre / even-odd rule at every new size, so they are never resampled.
     Two output channels: filament and "junction" (where two instances touch or cross).
  3. Build a proxy set from the domain A and B crops of held-out objects, rendered
     at domain C width (5 to 7.5 px). On it, search the threshold, junction threshold,
     fragment filter and test-time-scale set with the official metric.
  4. Predict every evaluation crop at its native size with dihedral TTA (one image at
     a time), split instances with the junction channel, and write run-length strings.

No pretrained weights, no external data. Evaluation crops are used for inference only:
no self-training, no pseudo-labels, no statistics taken across the test set.
"""
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

T0 = time.time()
# Wall-clock guard (seconds since start). Training stops at TRAIN_END so that there is
# room for the validation search and test inference within the one-hour budget.
TRAIN_END = float(os.environ.get("FIL_TRAIN_END", 2700))
HPO_END = float(os.environ.get("FIL_HPO_END", 3150))
SEED = 2024
TARGET_W = 6.0            # domain C filament width in pixels, from the task statement
AUG_W = (4.3, 8.5)        # range of filament widths seen during training
PROXY_W = (5.0, 7.5)      # widths the held-out proxy crops are rendered at
CROP = 256
BATCH = 16
STORE_W = 9.0             # domain A images are stored pre-shrunk to this filament width

cv2.setNumThreads(1)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = torch.bfloat16 if (DEVICE.type == "cuda" and torch.cuda.is_bf16_supported()) else None
STRUCT8 = np.ones((3, 3), dtype=bool)


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- RLE / raster
def encode_rle(mask):
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if not flat.any():
        return ""
    padded = np.concatenate(([False], flat, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return " ".join(f"{s} {e - s}" for s, e in zip(edges[0::2], edges[1::2]))


def decode_rle(rle, width, height):
    mask = np.zeros(width * height, dtype=bool)
    parts = str(rle).split()
    for start, length in zip(parts[0::2], parts[1::2]):
        s, n = int(start), int(length)
        mask[max(s, 0):min(s + n, width * height)] = True
    return mask.reshape(height, width)


def rasterize_one(poly, width, height):
    """Vectorised copy of the task's reference rasteriser for one polygon.

    Same arithmetic in the same order (float64), so the output matches the reference
    exactly: a pixel is inside when its centre (x+0.5, y+0.5) is inside, even-odd rule.
    """
    mask = np.zeros((height, width), dtype=bool)
    px = np.clip(poly[:, 0], 0.0, 1.0) * width
    py = np.clip(poly[:, 1], 0.0, 1.0) * height
    x0, y0, x1, y1 = px, py, np.roll(px, -1), np.roll(py, -1)
    r0 = max(int(np.floor(py.min() - 0.5)), 0)
    r1 = min(int(np.ceil(py.max())), height - 1)
    if r1 < r0:
        return mask
    rows = np.arange(r0, r1 + 1)
    yc = (rows.astype(np.float64) + 0.5)[:, None]
    crosses = ((y0 <= yc) & (y1 > yc)) | ((y1 <= yc) & (y0 > yc))
    if not crosses.any():
        return mask
    dy = np.where(y1 == y0, 1.0, y1 - y0)
    xi = np.where(crosses, x0 + (yc - y0) * (x1 - x0) / dy, np.inf)
    xi.sort(axis=1)
    k = int(crosses.sum(axis=1).max())
    k -= k % 2
    if k == 0:
        return mask
    starts, stops = xi[:, 0:k:2], xi[:, 1:k:2]
    ok = np.isfinite(starts) & np.isfinite(stops)
    ri, ci = np.nonzero(ok)
    left = np.ceil(starts[ri, ci] - 0.5).astype(np.int64)
    right = np.ceil(stops[ri, ci] - 0.5).astype(np.int64)
    left, right = np.clip(left, 0, width), np.clip(right, 0, width)
    keep = right > left
    ri, left, right = ri[keep], left[keep], right[keep]
    diff = np.zeros((len(rows), width + 1), dtype=np.int32)
    np.add.at(diff, (ri, left), 1)       # spans of one row are sorted and disjoint,
    np.add.at(diff, (ri, right), -1)     # so XOR (reference) == OR here
    mask[r0:r1 + 1] = np.cumsum(diff, axis=1)[:, :width] > 0
    return mask


def parse_vertices(s):
    v = np.array(str(s).split(), dtype=np.float64)
    return v.reshape(-1, 2)


def filament_width(mask):
    """Width in pixels of one filament mask: 2 * median(distance to background on the
    skeleton) - 0.5. The distance at the skeleton of a k-pixel strip is about (k+1)/2
    for odd k and k/2 for even k, so the 0.5 offset gives an unbiased estimate on average."""
    ys, xs = np.nonzero(mask)
    if len(ys) < 3:
        return np.nan
    sub = np.pad(mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 1)
    sk = skeletonize(sub)
    if not sk.any():
        return np.nan
    d = ndi.distance_transform_edt(sub)[sk]
    return 2.0 * float(np.median(d)) - 0.5


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def resize(img, w, h, s, rng=None):
    if s < 1.0:
        interp = cv2.INTER_AREA
    elif rng is None:
        interp = cv2.INTER_CUBIC
    else:
        interp = cv2.INTER_LINEAR if rng.random() < 0.5 else cv2.INTER_CUBIC
    return cv2.resize(img, (w, h), interpolation=interp)


def render(item, target_w, rng=None):
    """Rescale a training crop so its filaments are target_w px wide. Returns the image
    and the instance masks, re-rasterised from the polygons at the new size."""
    img = item["img"]
    s = target_w / item["w_eff"]
    h0, w0 = img.shape[:2]
    w, h = max(int(round(w0 * s)), 16), max(int(round(h0 * s)), 16)
    im = resize(img, w, h, s, rng) if (w, h) != (w0, h0) else img.copy()
    insts = [m for m in (rasterize_one(p, w, h) for p in item["polys"]) if m.any()]
    return im, insts


def targets_from_instances(insts, h, w):
    """Channel 0: union of filaments. Channel 1: junction, i.e. pixels within 2 px of
    two or more instances (crossings, contacts, near-contacts)."""
    union = np.zeros((h, w), dtype=bool)
    count = np.zeros((h, w), dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for m in insts:
        union |= m
        count += cv2.dilate(m.astype(np.uint8), ker) > 0
    junction = (count >= 2) & cv2.dilate(union.astype(np.uint8), ker).astype(bool)
    return union, junction


def normalise(img):
    x = img.astype(np.float32)
    mean = x.reshape(-1, 3).mean(0)
    std = x.reshape(-1, 3).std(0) + 1.0
    return (x - mean) / std


# ----------------------------------------------------------------------------- augmentation
def photometric(img, rng):
    x = img.astype(np.float32)
    if rng.random() < 0.8:                                  # contrast / brightness / colour cast
        x = x * rng.uniform(0.7, 1.3) + rng.uniform(-25, 25)
        x = x * rng.uniform(0.9, 1.1, size=3)[None, None, :]
    if rng.random() < 0.15:                                 # grayscale
        x[:] = x.mean(axis=2, keepdims=True)
    x = np.clip(x, 0, 255)
    if rng.random() < 0.5:                                  # gamma
        x = 255.0 * (x / 255.0) ** rng.uniform(0.7, 1.4)
    if rng.random() < 0.3:                                  # optical blur of another camera
        x = cv2.GaussianBlur(x, (0, 0), rng.uniform(0.3, 1.1))
    if rng.random() < 0.3:                                  # sensor noise
        x = x + rng.normal(0, rng.uniform(2, 8), size=x.shape)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                  # JPEG re-compression
        ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(45, 95))])
        if ok:
            x = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return x


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, items, length):
        self.items = items
        self.length = length
        self.by_dom = {d: [i for i, it in enumerate(items) if it["domain"] == d] for d in ("A", "B")}
        self.rng = np.random.default_rng(SEED)

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        rng = self.rng
        dom = "A" if rng.random() < 0.5 else "B"            # balance the two scales
        pool = self.by_dom[dom] or self.by_dom["B" if dom == "A" else "A"]
        item = self.items[pool[int(rng.integers(len(pool)))]]
        tw = math.exp(rng.uniform(math.log(AUG_W[0]), math.log(AUG_W[1])))
        im, insts = render(item, tw, rng)
        h, w = im.shape[:2]
        union, junc = targets_from_instances(insts, h, w)
        im = normalise(photometric(im, rng))
        # pad to at least CROP (reflected image, zero loss weight on the padding)
        ph, pw = max(CROP - h, 0), max(CROP - w, 0)
        weight = np.ones((h, w), dtype=np.float32)
        if ph or pw:
            pads = ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2))
            im = np.pad(im, pads + ((0, 0),), mode="symmetric")
            union, junc = np.pad(union, pads), np.pad(junc, pads)
            weight = np.pad(weight, pads)
            h, w = union.shape
        if rng.random() < 0.7 and union.any():              # centre most crops on a filament
            ys, xs = np.nonzero(union)
            j = int(rng.integers(len(ys)))
            y0 = int(np.clip(ys[j] - CROP // 2 + rng.integers(-64, 65), 0, h - CROP))
            x0 = int(np.clip(xs[j] - CROP // 2 + rng.integers(-64, 65), 0, w - CROP))
        else:                                                # background edges are negatives
            y0, x0 = int(rng.integers(h - CROP + 1)), int(rng.integers(w - CROP + 1))
        sl = (slice(y0, y0 + CROP), slice(x0, x0 + CROP))
        x = im[sl].transpose(2, 0, 1)
        t = np.stack([union[sl], junc[sl]]).astype(np.float32)
        wt = weight[sl][None]
        k = int(rng.integers(8))                             # random dihedral transform
        x, t, wt = (np.rot90(a, k % 4, axes=(1, 2)) for a in (x, t, wt))
        if k >= 4:
            x, t, wt = (a[:, :, ::-1] for a in (x, t, wt))
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(t)),
                torch.from_numpy(np.ascontiguousarray(wt)))


def worker_init(wid):
    ds = torch.utils.data.get_worker_info().dataset
    ds.rng = np.random.default_rng(SEED + 1000 * (wid + 1))


# ----------------------------------------------------------------------------- model
def block(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
        nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Plain U-Net trained from scratch. Four poolings give a 16x-downsampled
    bottleneck, enough context to tell a filament from a background edge at ~6 px width."""

    def __init__(self, chs=(32, 64, 128, 256, 320), n_out=2):
        super().__init__()
        self.enc = nn.ModuleList()
        ci = 3
        for c in chs:
            self.enc.append(block(ci, c))
            ci = c
        self.dec = nn.ModuleList(block(chs[i + 1] + chs[i], chs[i]) for i in reversed(range(len(chs) - 1)))
        self.head = nn.Conv2d(chs[0], n_out, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for dec, skip in zip(self.dec, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], 1))
        return self.head(x)


def loss_fn(logits, tgt, wt):
    """BCE + soft Dice per channel, masked by the padding weight. Dice keeps the
    gradient meaningful when filaments are only a few percent of the pixels."""
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, tgt, reduction="none")
    bce = (bce * wt).sum((0, 2, 3)) / wt.sum().clamp(min=1.0)
    p = torch.sigmoid(logits) * wt
    inter = (p * tgt).sum((0, 2, 3))
    dice = 1.0 - (2 * inter + 1.0) / (p.sum((0, 2, 3)) + (tgt * wt).sum((0, 2, 3)) + 1.0)
    per_ch = bce + dice
    return per_ch[0] + 0.5 * per_ch[1]


# ----------------------------------------------------------------------------- inference
@torch.no_grad()
def predict_probs(model, img, scales):
    """Filament and junction probabilities at the image's native grid. For each scale,
    resize, pad to a square multiple of 32, run all 8 dihedral variants in one batch,
    undo them, average, and resize back. Each image is handled on its own."""
    H, W = img.shape[:2]
    acc = np.zeros((2, H, W), dtype=np.float32)
    for s in scales:
        if s == 1.0:
            im = img
        else:
            im = resize(img, max(int(round(W * s)), 16), max(int(round(H * s)), 16), s)
        h, w = im.shape[:2]
        S = int(math.ceil(max(h, w) / 32) * 32)
        x = np.pad(normalise(im), ((0, S - h), (0, S - w), (0, 0)), mode="symmetric")
        x = torch.from_numpy(x.transpose(2, 0, 1).copy())[None].to(DEVICE)
        views = [torch.rot90(x, k % 4, dims=(2, 3)) for k in range(4)]
        views += [torch.flip(v, dims=(3,)) for v in views]
        batch = torch.cat(views)
        if AMP_DTYPE is not None:
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                out = model(batch)
        else:
            out = model(batch)
        out = torch.sigmoid(out.float())
        back = []
        for k in range(8):
            o = out[k:k + 1]
            if k >= 4:
                o = torch.flip(o, dims=(3,))
            back.append(torch.rot90(o, -(k % 4), dims=(2, 3)))
        p = torch.cat(back).mean(0)[:, :h, :w].cpu().numpy()
        if (h, w) != (H, W):
            p = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in p])
        acc += p
    return acc / len(scales)


def image_width(P):
    m = P > 0.5
    if m.sum() < 20:
        return TARGET_W
    w = filament_width(m)
    return float(np.clip(w, 3.0, 12.0)) if np.isfinite(w) else TARGET_W


def seg_dist(py, px, a, b):
    ay, ax = a
    by, bx = b
    vy, vx = by - ay, bx - ax
    L2 = vy * vy + vx * vx
    t = np.zeros_like(py) if L2 < 1e-9 else np.clip(((py - ay) * vy + (px - ax) * vx) / L2, 0, 1)
    return np.hypot(py - (ay + t * vy), px - (ax + t * vx))


def instance_supports(P, J, t_low, t_j):
    """Split the thresholded filament map into instance supports (flat pixel indices).

    Arms are connected components of the filament mask with junction pixels removed.
    At each junction, arms leaving in near-opposite directions are joined (a filament
    passing through a crossing), and junction pixels go to the joined filament whose
    straight path covers them, otherwise to the nearest arm. Crossing pixels can
    belong to two instances, like the annotations."""
    H, W = P.shape
    w_img = image_width(P)
    support = P > t_low
    core = support & (J < t_j)
    arms, n_arms = ndi.label(core, structure=STRUCT8)
    if n_arms == 0:
        return [], w_img
    parent = list(range(n_arms + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    extra = {}                                   # arm id -> list of flat junction indices
    jl, n_j = ndi.label(support & ~core, structure=STRUCT8)
    R = int(round(3 * w_img)) + 3
    half = w_img / 2.0 + 0.5
    for j, sl in enumerate(ndi.find_objects(jl), start=1):
        if sl is None:
            continue
        y0, y1 = max(sl[0].start - R, 0), min(sl[0].stop + R, H)
        x0, x1 = max(sl[1].start - R, 0), min(sl[1].stop + R, W)
        jm = jl[y0:y1, x0:x1] == j
        al = arms[y0:y1, x0:x1]
        ring = ndi.binary_dilation(jm, STRUCT8, iterations=2)
        adj = [int(a) for a in np.unique(al[ring & (al > 0)])]
        if not adj:
            continue
        jy, jx = np.nonzero(jm)
        cy, cx = jy.mean(), jx.mean()
        yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0]
        near = np.hypot(yy - cy, xx - cx) <= R
        ends, dirs, leaving = {}, {}, []
        for a in adj:
            am = al == a
            ey, ex = np.nonzero(am & ring)
            ends[a] = (ey.mean(), ex.mean(), ey, ex)
            ny, nx = np.nonzero(am & near)
            v = np.array([ny.mean() - cy, nx.mean() - cx]) if len(ny) else np.zeros(2)
            if np.linalg.norm(v) < 1e-6:
                v = np.array([ends[a][0] - cy, ends[a][1] - cx])
            dirs[a] = v / (np.linalg.norm(v) + 1e-9)
            # an arm "leaves" the junction when its own axis points away from it; an arm
            # lying alongside a contact band (parallel touching strands) does not
            if len(ny) >= 3:
                evals, evecs = np.linalg.eigh(np.cov(np.stack([ny, nx]).astype(float)))
                if abs(float(evecs[:, -1] @ dirs[a])) > 0.7:
                    leaving.append(a)
        pairs = []
        if len(leaving) == 2 and len(adj) == 2:
            a, b = leaving
            if float(dirs[a] @ dirs[b]) < 0.0:          # continuing, not a V
                pairs.append((a, b))
        elif len(leaving) >= 2:
            cand = sorted((float(dirs[a] @ dirs[b]), a, b)
                          for i, a in enumerate(leaving) for b in leaving[i + 1:])
            used = set()
            for c, a, b in cand:
                if c < -0.5 and a not in used and b not in used:
                    pairs.append((a, b))
                    used.update((a, b))
        assigned = np.zeros(len(jy), dtype=bool)
        flat = (jy + y0) * W + (jx + x0)
        for a, b in pairs:
            parent[find(a)] = find(b)
            on = seg_dist(jy.astype(float), jx.astype(float), ends[a][:2], ends[b][:2]) <= half
            extra.setdefault(a, []).append(flat[on])
            assigned |= on
        if (~assigned).any():
            ry, rx = jy[~assigned], jx[~assigned]
            dists = np.stack([np.min(np.hypot(ry[:, None] - ends[a][2][None], rx[:, None] - ends[a][3][None]), axis=1)
                              for a in adj])
            nearest = np.argmin(dists, axis=0)
            for i, a in enumerate(adj):
                extra.setdefault(a, []).append(flat[~assigned][nearest == i])
    groups = {}
    for a in range(1, n_arms + 1):
        groups.setdefault(find(a), []).append(a)
    arm_flat = arms.ravel()
    order = np.argsort(arm_flat, kind="stable")
    bounds = np.searchsorted(arm_flat[order], np.arange(n_arms + 2))
    out = []
    for members in groups.values():
        parts = [order[bounds[a]:bounds[a + 1]] for a in members]
        parts += [e for a in members for e in extra.get(a, [])]
        out.append(np.unique(np.concatenate(parts)))
    return out, w_img


def finalize(supports, w_img, P_flat, t_m, min_area_k):
    """Keep the pixels of each support above the mask threshold (this sets the width),
    then drop fragments smaller than min_area_k * width^2."""
    keep = []
    min_area = min_area_k * w_img * w_img
    for s in supports:
        s = s[P_flat[s] > t_m]
        if len(s) >= min_area:
            keep.append(s)
    return keep


def evaluate(pred_instances, true_instances):
    """Official metric (copied from the task statement)."""
    thresholds = [0.5 + 0.05 * k for k in range(10)]
    totals = {t: [0, 0, 0] for t in thresholds}
    for image_id, trues in true_instances.items():
        preds = pred_instances.get(image_id, [])
        iou = np.zeros((len(preds), len(trues)))
        for i, p in enumerate(preds):
            for j, q in enumerate(trues):
                inter = np.logical_and(p, q).sum()
                iou[i, j] = inter / (p.sum() + q.sum() - inter) if inter else 0.0
        matched = []
        used_p, used_t = set(), set()
        for flat in np.argsort(-iou, axis=None, kind="stable"):
            i, j = divmod(int(flat), max(len(trues), 1))
            if iou.size == 0 or iou[i, j] <= 0:
                break
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j); matched.append(iou[i, j])
        for t in thresholds:
            tp = sum(m >= t for m in matched)
            totals[t][0] += tp; totals[t][1] += len(preds) - tp; totals[t][2] += len(trues) - tp
    f1 = [2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0 for tp, fp, fn in totals.values()]
    return float(np.mean(f1))


def fast_score(pred_flat, true_flat):
    """Same metric on flat index sets, for the parameter search."""
    thresholds = [0.5 + 0.05 * k for k in range(10)]
    tot = np.zeros((10, 3))
    for key, trues in true_flat.items():
        preds = pred_flat.get(key, [])
        iou = np.zeros((len(preds), len(trues)))
        for i, p in enumerate(preds):
            for j, q in enumerate(trues):
                inter = len(np.intersect1d(p, q, assume_unique=True))
                if inter:
                    iou[i, j] = inter / (len(p) + len(q) - inter)
        matched, used_p, used_t = [], set(), set()
        for flat in np.argsort(-iou, axis=None, kind="stable"):
            i, j = divmod(int(flat), max(len(trues), 1))
            if iou.size == 0 or iou[i, j] <= 0:
                break
            if i in used_p or j in used_t:
                continue
            used_p.add(i); used_t.add(j); matched.append(iou[i, j])
        matched = np.array(matched)
        for k, t in enumerate(thresholds):
            tp = int((matched >= t - 1e-12).sum())
            tot[k] += (tp, len(preds) - tp, len(trues) - tp)
    f1 = [2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0 for tp, fp, fn in tot]
    return float(np.mean(f1))


# ----------------------------------------------------------------------------- main
def main():
    public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
    submission_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
    log(f"device={DEVICE} amp={AMP_DTYPE}")

    train = pd.read_csv(public_dir / "train.csv")
    meta = pd.read_csv(public_dir / "train_meta.csv")
    inst = pd.read_csv(public_dir / "train_instances.csv")
    polys = pd.read_csv(public_dir / "train_polygons.csv")
    test = pd.read_csv(public_dir / "test.csv", dtype={"image_id": str})
    train = train.merge(meta[["image_id", "domain", "object_id"]], on="image_id", how="left")
    poly_map = {k: [parse_vertices(v) for v in g.sort_values("instance_index")["vertices"]]
                for k, g in polys.groupby("image_id")}
    inst_map = {k: list(g.sort_values("instance_index")["rle"]) for k, g in inst.groupby("image_id")}

    # --- 1. per-crop filament width, measured on the published native instance masks
    widths = {}
    for r in train.itertuples():
        ws = [filament_width(decode_rle(rle, r.width, r.height)) for rle in inst_map.get(r.image_id, [])]
        ws = [w for w in ws if np.isfinite(w)]
        widths[r.image_id] = float(np.median(ws)) if ws else np.nan
    train["w_raw"] = train["image_id"].map(widths)
    dom_med = train.groupby("domain")["w_raw"].median().to_dict()
    for d, m in dom_med.items():
        q = train.loc[train.domain == d, "w_raw"].quantile([0.1, 0.9]).round(2).tolist()
        log(f"domain {d}: median filament width {m:.2f} px (10-90%: {q})")
    # shrink each crop's estimate towards its domain median (geometric mean) and clip,
    # so one noisy measurement cannot send a crop far outside the training width range
    def shrink(r):
        m = dom_med[r.domain]
        w = r.w_raw if np.isfinite(r.w_raw) else m
        return float(np.clip(math.sqrt(w * m), 0.75 * m, 1.33 * m))
    train["w_crop"] = train.apply(shrink, axis=1)

    # --- load images into memory; domain A is pre-shrunk (its filaments stay >= STORE_W px)
    items = []
    for r in train.itertuples():
        img = load_rgb(public_dir / "train" / "images" / f"{r.image_id}.jpg")
        w_eff = r.w_crop
        if w_eff > STORE_W:
            s = STORE_W / w_eff
            img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))),
                             interpolation=cv2.INTER_AREA)
            w_eff = r.w_crop * img.shape[1] / r.width
        items.append(dict(image_id=r.image_id, domain=r.domain, object_id=r.object_id,
                          img=img, w_eff=w_eff, polys=poly_map.get(r.image_id, [])))
    log(f"loaded {len(items)} training crops")

    # --- object-disjoint hold-out: all crops of an object (both domains, all dates) together
    rng = np.random.default_rng(SEED)
    objects = np.array(sorted(train.object_id.unique()))
    rng.shuffle(objects)
    val_obj = set(objects[:max(1, int(round(0.15 * len(objects))))])
    tr_items = [it for it in items if it["object_id"] not in val_obj]
    va_items = [it for it in items if it["object_id"] in val_obj]
    log(f"train crops {len(tr_items)}, held-out crops {len(va_items)} ({len(val_obj)} objects)")

    # proxy for domain C: held-out crops rendered at a width drawn from PROXY_W
    prng = np.random.default_rng(SEED + 7)
    proxy = []
    for it in va_items:
        tw = float(prng.uniform(*PROXY_W))
        im, insts = render(it, tw, None)
        proxy.append(dict(key=it["image_id"], domain=it["domain"], img=im, insts=insts))

    # --- 2. training, stopped by the wall clock
    model = UNet().to(DEVICE).to(memory_format=torch.channels_last)
    ema = UNet().to(DEVICE).to(memory_format=torch.channels_last)
    ema.load_state_dict(model.state_dict())
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    n_workers = min(8, os.cpu_count() or 1)
    loader = torch.utils.data.DataLoader(
        TrainSet(tr_items, 10_000_000), batch_size=BATCH, num_workers=n_workers,
        worker_init_fn=worker_init, pin_memory=DEVICE.type == "cuda", drop_last=True,
        persistent_workers=n_workers > 0)
    t_start = time.time() - T0
    span = max(TRAIN_END - t_start, 1.0)
    step, run_loss, last_log = 0, 0.0, time.time()
    model.train()
    for x, t, wt in loader:
        frac = (time.time() - T0 - t_start) / span
        if frac >= 1.0:
            break
        # lr keyed on elapsed time: 3% warm-up then cosine, so the schedule completes
        # whatever the hardware speed
        lr = 2e-3 * (frac / 0.03 if frac < 0.03 else 0.5 * (1 + math.cos(math.pi * (frac - 0.03) / 0.97)))
        for g in opt.param_groups:
            g["lr"] = lr
        x = x.to(DEVICE, non_blocking=True).to(memory_format=torch.channels_last)
        t, wt = t.to(DEVICE, non_blocking=True), wt.to(DEVICE, non_blocking=True)
        if AMP_DTYPE is not None:
            with torch.autocast("cuda", dtype=AMP_DTYPE):
                out = model(x)
        else:
            out = model(x)
        loss = loss_fn(out, t, wt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        decay = min(0.999, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        step += 1
        run_loss += loss.item()
        if time.time() - last_log > 60:
            log(f"step {step} lr {lr:.2e} loss {run_loss / max(step, 1):.4f}")
            last_log = time.time()
    del loader
    log(f"training stopped after {step} steps")
    ema.eval()

    # --- 3. parameter search on the proxy, scored with the official metric
    scale_sets = {"1.0": [1.0], "0.8-1.0-1.25": [0.8, 1.0, 1.25]}
    true_flat = {p["key"]: [np.flatnonzero(m.ravel()) for m in p["insts"]] for p in proxy}
    grid_low, grid_j = [0.3, 0.45], [0.35, 0.5, 0.65]
    grid_m = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    grid_area = [1.5, 3.0, 5.0, 8.0]
    best = (-1.0, dict(scales="1.0", t_low=0.45, t_j=0.5, t_m=0.5, area=3.0))
    results = []
    for sname, scales in scale_sets.items():
        if time.time() - T0 > HPO_END:
            log("search time guard reached")
            break
        probs = {p["key"]: predict_probs(ema, p["img"], scales) for p in proxy}
        for t_low in grid_low:
            for t_j in grid_j:
                if time.time() - T0 > HPO_END:
                    break
                sup = {k: instance_supports(v[0], v[1], t_low, t_j) for k, v in probs.items()}
                for t_m in grid_m:
                    if t_m < t_low:
                        continue
                    for a in grid_area:
                        pred = {k: finalize(s, w, probs[k][0].ravel(), t_m, a) for k, (s, w) in sup.items()}
                        sc = fast_score(pred, true_flat)
                        cfg = dict(scales=sname, t_low=t_low, t_j=t_j, t_m=t_m, area=a)
                        results.append((sc, cfg))
                        if sc > best[0]:
                            best = (sc, cfg)
        log(f"scales {sname}: best proxy score so far {best[0]:.4f} {best[1]}")
    results.sort(key=lambda r: -r[0])
    for sc, cfg in results[:5]:
        log(f"  top proxy {sc:.4f} {cfg}")
    cfg = best[1]

    # per-domain breakdown of the chosen configuration (official evaluate for a check)
    scales = scale_sets[cfg["scales"]]
    for dom in ("A", "B"):
        sub = [p for p in proxy if p["domain"] == dom]
        if not sub or time.time() - T0 > HPO_END + 120:
            continue
        pred, true = {}, {}
        for p in sub:
            P, J = predict_probs(ema, p["img"], scales)
            s, w = instance_supports(P, J, cfg["t_low"], cfg["t_j"])
            h, wd = P.shape
            pred[p["key"]] = [np.isin(np.arange(h * wd), f).reshape(h, wd)
                              for f in finalize(s, w, P.ravel(), cfg["t_m"], cfg["area"])]
            true[p["key"]] = p["insts"]
        log(f"proxy from domain {dom} ({len(sub)} crops): {evaluate(pred, true):.4f}")

    # --- 4. inference on the evaluation crops at native resolution
    rows = []
    for r in test.itertuples():
        field = ""
        try:
            img = load_rgb(public_dir / "test" / "images" / f"{r.image_id}.jpg")
            P, J = predict_probs(ema, img, scales)
            s, w = instance_supports(P, J, cfg["t_low"], cfg["t_j"])
            h, wd = P.shape
            masks = []
            for f in finalize(s, w, P.ravel(), cfg["t_m"], cfg["area"])[:300]:
                m = np.zeros(h * wd, dtype=bool)
                m[f] = True
                masks.append(encode_rle(m.reshape(h, wd)))
            field = ";".join(e for e in masks if e)
        except Exception as e:  # never lose a row: an empty field is a valid "no filament"
            log(f"inference failed on {r.image_id}: {e!r}")
        rows.append((r.image_id, field))
    sub = pd.DataFrame(rows, columns=["image_id", "rle"])
    assert len(sub) == len(test) and sub.image_id.is_unique and set(sub.image_id) == set(test.image_id)
    n_inst = sub.rle.map(lambda s: len([x for x in s.split(";") if x]))
    log(f"test crops {len(sub)}, predicted instances {int(n_inst.sum())} "
        f"(mean {n_inst.mean():.2f}/crop, empty crops {int((n_inst == 0).sum())})")
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)
    log(f"wrote {submission_out}")


if __name__ == "__main__":
    main()
