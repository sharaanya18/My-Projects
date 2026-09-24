#!/usr/bin/env python3
"""Checks of solution.py against the real training labels (needs only the CSV files).

Usage: python3 tools/check_labels.py <public_dir>

1. rasterize_one(polygon) reproduces every published instance mask exactly.
2. Filament width per domain (the statement says A 15-20 px, B 4-5 px).
3. How many crops have crossing / touching filaments.
4. Ceiling of the instance step: crops rendered at 5-7.5 px width, perfect probability
   maps, official metric. Also with maps 1 px too fat, to show the cost of width errors.
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from solution import (decode_rle, fast_score, filament_width, finalize, instance_supports,  # noqa: E402
                      parse_vertices, rasterize_one, targets_from_instances)

pub = Path(sys.argv[1])
t = pd.read_csv(pub / "train.csv").merge(pd.read_csv(pub / "train_meta.csv"))
ins = pd.read_csv(pub / "train_instances.csv")
pol = pd.read_csv(pub / "train_polygons.csv")
df = ins.merge(pol[["instance_id", "vertices"]]).merge(t[["image_id", "width", "height", "domain"]])

bad, widths = 0, []
for r in df.itertuples():
    m = decode_rle(r.rle, r.width, r.height)
    bad += bool((rasterize_one(parse_vertices(r.vertices), r.width, r.height) != m).any())
    widths.append(filament_width(m))
df["w"] = widths
print(f"1. rasteriser mismatches: {bad} / {len(df)}")
print("2. filament width per domain:")
print(df.groupby("domain").w.describe(percentiles=[.1, .5, .9]).round(2))

multi = overlap = touch = 0
for _, g in df.groupby("image_id"):
    if len(g) < 2:
        continue
    multi += 1
    W, H = g.width.iloc[0], g.height.iloc[0]
    ms = [decode_rle(x, W, H).astype(np.uint8) for x in g.rle]
    overlap += bool((sum(ms) >= 2).any())
    rad = max(1, int(round(2 * g.w.median() / 6)))           # 2 px at test scale
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    touch += bool((sum((cv2.dilate(m, k) > 0).astype(np.uint8) for m in ms) >= 2).any())
print(f"3. crops with >=2 filaments {multi}, overlapping {overlap}, touching at test scale {touch}")

t["w_raw"] = t.image_id.map(df.groupby("image_id").w.median())
med = t.groupby("domain").w_raw.median().to_dict()
t["w_crop"] = [float(np.clip(math.sqrt(x * med[d]), .75 * med[d], 1.33 * med[d])) for x, d in zip(t.w_raw, t.domain)]
polys = {k: [parse_vertices(v) for v in g.sort_values("instance_index").vertices] for k, g in pol.groupby("image_id")}
rng = np.random.default_rng(0)
rendered = {}
for r in t.itertuples():
    s = rng.uniform(5, 7.5) / r.w_crop
    W, H = max(int(round(r.width * s)), 16), max(int(round(r.height * s)), 16)
    insts = [m for m in (rasterize_one(p, W, H) for p in polys[r.image_id]) if m.any()]
    u, j = targets_from_instances(insts, H, W)
    rendered[r.image_id] = (r.domain, insts, u.astype(np.float32), j.astype(np.float32))
print("4. instance step ceiling (official metric):")
for name, fn in [("perfect maps", lambda u: u), ("maps 1px too fat", lambda u: cv2.dilate(u, np.ones((3, 3), np.uint8)))]:
    for dom in ["A", "B", "all"]:
        pred, true = {}, {}
        for key, (d, insts, u, j) in rendered.items():
            if dom != "all" and d != dom:
                continue
            P = fn(u)
            s, wi = instance_supports(P, j, 0.45, 0.5)
            pred[key] = finalize(s, wi, P.ravel(), 0.5, 1.5)
            true[key] = [np.flatnonzero(m.ravel()) for m in insts]
        print(f"   {name:18s} domain {dom:3s}: {fast_score(pred, true):.3f}")
