# Filament Instances Across Scales: approach

`solution.py` runs the whole pipeline: `python3 solution.py <public_dir> <submission_out>`.
It trains from scratch and predicts in a single run of about 55 minutes on one A10G.

## 1. What the problem really asks

| Fact | Consequence for the design |
|---|---|
| Scored per filament with mask IoU at 0.50 to 0.95 | The mask must have the right **width** (one pixel too fat on each side gives IoU ≈ 0.7) and the right **number of pieces** (a fused pair or a split strand loses the match at every threshold). |
| Labels at ~17 px (A) and ~4.5 px (B); test at ~6 px (C) | Train at the **test scale**. Rescale every training crop so its filaments are about 6 px wide, and add scale jitter to cover the unseen camera. |
| Masks use a pixel-centre, even-odd rasteriser; `cv2.fillPoly` is off by 10 to 30 % | Never resize the masks. **Re-rasterise the normalised polygons** at every new size with an exact copy of the reference rule. |
| 456 objects, the same object appears in both domains and on several dates | Validate on **whole held-out objects** only. |
| No C labels | Tune thresholds on a **proxy**: held-out A and B crops rendered at C width. |

## 2. Data preparation (fitted inside the script)

1. **Width per training crop.** For each published instance mask:
   `width = 2·median(distance-to-background on the skeleton) − 0.5`.
   Each crop's value is shrunk towards its domain median and clipped to ±30 %, so one noisy estimate
   can't mis-scale a crop. The log prints both domain medians, which should be close to 15 to 20 and 4 to 5.
2. **Exact rasteriser.** `rasterize_one` is a vectorised version of the reference function (same
   float64 arithmetic in the same order). It matched the reference pixel for pixel on 3,000 random
   polygons, including self-intersecting ones and vertices exactly on pixel centres. It is fast enough
   (about 1 ms per polygon) to re-rasterise every training sample on the fly, so any scale can be used.
3. **Memory.** All images are cached in RAM. Domain A crops are stored pre-shrunk to a 9 px filament
   width (this only shrinks further later), so the cache stays under 1 GB.

## 3. Model and training

* **Architecture:** a plain U-Net from scratch (32 to 320 channels, 4 poolings, BatchNorm), with **two
  output channels**:
  * `filament`: union of all instances,
  * `junction`: pixels within 2 px of two or more instances (crossings, contacts, near-contacts).
    This is the extra signal that lets touching filaments be separated later.
* **Scale handling:** each sample picks a target width from 4.3 to 8.5 px (log-uniform, centred near 6)
  and rescales the image to it (`INTER_AREA` down, bilinear or bicubic up). The masks are re-rasterised
  from polygons at that exact size. A is shrunk about 2.9×, B is enlarged about 1.3×. A and B are
  sampled 50/50.
* **Camera-gap augmentation:** contrast, brightness, colour cast, gamma, grayscale, Gaussian blur (for
  the softness of enlarged B), sensor noise, JPEG re-compression, and the 8 dihedral flips and
  rotations. Each image is normalised by its own mean and std, which is also done at test time.
* **Crops:** 256×256. 70 % are centred near a filament. The other 30 % are uniform, so background edges
  and seams are seen as negatives. Padding is reflected and gets zero loss weight.
* **Loss:** BCE + soft Dice on each channel (junction weighted 0.5). Dice handles the few-percent
  positive rate.
* **Optimiser:** AdamW, lr 2e-3, 3 % warm-up, then cosine decay. The schedule is **keyed on elapsed
  wall-clock time**, so it completes on any hardware. Training stops at 45 min (`FIL_TRAIN_END`).
  The EMA of the weights is used for inference. bf16 autocast on GPU.

## 4. Turning probabilities into instances

1. `support = P > t_low` (low threshold, so a strand isn't cut where the model is unsure);
   `core = support & (J < t_j)`.
2. **Arms** = connected components of `core`. **Junctions** = `support & ~core`.
3. At each junction, look at the arms touching it. An arm counts as *leaving* the junction when its own
   principal axis (PCA of its pixels near the junction) points away from the junction centre.
   * Arms that leave in **opposite directions** are joined: a filament passing through a crossing, or a
     strand cut by a false junction.
   * Two arms that form a V, or that lie **alongside** a contact band (parallel touching strands), stay
     separate.
4. Junction pixels go to each joined filament whose straight path through the junction covers them.
   Pixels at a crossing can belong to both filaments, as in the annotations. Any remaining pixels go to
   the nearest arm.
5. **Width calibration:** the final instance = its support ∩ `P > t_m`. `t_m` alone sets how fat the
   masks are.
6. **Fragment filter:** drop instances smaller than `k · w²` pixels, where `w` is the filament width
   measured on that image's own prediction.

Checks with perfect probability maps: X crossing, three-way crossing, T contact, parallel strands
touching along their whole length, and a single strand with a spurious junction. Every case gives the
correct instances with a score of 1.000.

## 5. Validation and parameter search (in the script, not hard-coded)

* **Split:** 15 % of `object_id`s are held out. All their crops, from both domains and all dates, are
  excluded from training, which also keeps the `pairs.csv` partners together.
* **Proxy for domain C:** each held-out crop is rendered at a width drawn from 5.0 to 7.5 px. Masks are
  re-rasterised from polygons, never resampled. Native-resolution crops are never used for tuning.
* **Search** with the official metric (micro-F1 averaged over 10 IoU thresholds):
  test-time scale set `{1.0}` vs `{0.8, 1.0, 1.25}` × `t_low ∈ {0.3, 0.45}` × `t_j ∈ {0.35, 0.5, 0.65}` ×
  `t_m ∈ {0.35 … 0.70}` × `k ∈ {1.5, 3, 5, 8}`. The best configuration is used for test. The top 5 and the
  score split by source domain (A proxy vs B proxy) are logged, to check that it doesn't rely on one scale.

## 6. Test inference

Each evaluation crop is predicted **at its native size**, one image at a time, with 8-way dihedral TTA
(and the multi-scale set if the search chose it). It goes through the same instance step, is encoded
with the published RLE, and the file is checked (452 unique ids) before it is written. If one image
fails, its row is left empty instead of the whole file being lost.

## 7. Rules followed (challenge statement + Solver Guidebook)

* Trained from scratch on the public files only: no pretrained weights, no external or synthetic
  training data.
* **The evaluation crops are only used for per-image inference.** The challenge text allows
  self-training on them, but the Guidebook (§4.2.5) forbids pseudo-labelling, test-time adaptation and
  test-set statistics, so none of that is done. TTA is per image, which the Guidebook allows.
* No ids, `capture_group`, file order or metadata are used as features.
* Everything is fitted inside the script (widths, thresholds, fragment filter, TTA choice). The only
  constants come from the task statement (C width ≈ 6 px) or are standard defaults.
* Wall-clock guards: training stops at 45 min and the search at 52.5 min, which leaves room for inference.

## 8. Why this should generalise (and what could still go wrong)

* It scores **0.26+** only if the model separates filaments from background edges at ~6 px and the
  width is right. The two main risks are the camera gap (C was captured in a later campaign) and the
  width calibration. Both are addressed by generic means: scale and photometric jitter, and a
  threshold tuned on an object-disjoint proxy at the target width. Nothing is fitted to the test set.
* If the log's A proxy and B proxy scores are far apart, the downscaled-A proxy is the closer match to a
  native camera. Consider weighting it more rather than tuning further.
* Possible extensions that keep to the rules: a second seed or fold for an ensemble (time permitting),
  a skeleton-distance channel for sharper widths, and a cross-view consistency loss on `pairs.csv` if a
  rough A↔B alignment can be learned.

The number above is a target, not a measured result. Here the script was only run on a small toy
dataset to check that it runs end to end and writes a valid file; the real data wasn't available.
Before submitting, run it locally and read the logged proxy score.
