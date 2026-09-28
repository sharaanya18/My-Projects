# Visible Flood-Failure Mechanism Recognition: Solution Approach

Companion to [`solution.py`](solution.py). This document explains what was analysed, which rules constrain the solution, and the model and validation design. It also lays out how to reach a site-weighted macro AP of **≥ 0.50 on held-out sites without overfitting**.

> **Status.** The uploaded `shipd-eris-env-final.zip` holds the solver workspace (rules, skills, templates, scripts) but **no dataset**. `solution.py` has been smoke-tested end to end on a random-pixel fixture: it completes on CPU in 47 s, the output passes format checks, and the compliance scan shows no errors. It has **not** been scored on the real photographs yet. Every score in this document is a target or an estimate until the data is loaded and the CV in §5 is run.

---

## 1. What was analysed

| Source | Findings that shape the solution |
|---|---|
| **Solver Guidebook (PDF)** | Real training must happen inside the script. Pretrained backbones from timm/HF are allowed (§4.1). No external data, synthetic data, pseudo-labels, test-time adaptation or test-distribution calibration (§4.2). TTA is allowed only when applied one sample at a time. **A challenge-specific runtime cap under 1 h overrides the default (§4.4).** CV challenges must use a real CNN/ViT, never a tabular model on pixels (§5.2). |
| **`CLAUDE.md` / skills** | Seeds fixed at 42. Relative I/O through `./dataset/public` → `./working/submission.csv`. Assert the row count before writing. `StratifiedGroupKFold` whenever a group exists, with k ≥ 5. The CV metric must match the challenge metric exactly. |
| **`LEARNINGS.md`** | (a) The platform may call `python3 solution.py <public_dir> <submission_out>`, so read positional args and fall back to relative paths. (b) Keep a detailed compliance header: stripping it caused "Check Errors". (c) With about 100–200 groups a single GroupKFold is noisy, so **repeat the split with several shuffles** before trusting a decision. (d) Entity-held-out splits shift the **label prior**; compute a prior-only baseline. (e) Avoid environment-dependent `try/except` model branches. |
| **`TROUBLESHOOTING.md`** | Most bad scores come from a lying CV harness (group leakage, wrong metric), a missing time guard, or a stale submission file. |
| **Challenge description** | 841 train / 215 test images. Multi-label (4). The metric is the mean over labels of AP weighted by `1/n_site`. Test sites are **unseen**. **30-minute A10G limit.** Labels are per-image visible evidence, not site-level condition. |

## 2. Contract

```
Input        dataset/public/{train.csv, train_targets.csv (JSON target), test.csv, images/}
Output       working/submission.csv   columns: id, prediction (JSON string, 4 keys, floats in [0,1])
Rows         215, every test id exactly once
Metric       mean_k AP_k(y, p, sample_weight = 1 / n_site(i))   higher is better
Group key    id.split("-", 1)[0]  (pseudonymous site)
Runtime      <= 30 min on one A10G (challenge rule; overrides the 3100 s guidebook default)
```

## 3. Rule mapping: decisions forced by the rules

| Rule | How the solution complies |
|---|---|
| 30-minute cap | `TRAIN_BUDGET_SEC = 24 min` measured from script start, checked before every epoch and fold. Inference and CSV writing need about 1 min. The expected full run is ~8–10 min (§7). |
| Training inside the script | 5 fold models are fine-tuned from an ImageNet backbone on every run. Nothing is cached. |
| Pretrained weights | Only the public `timm` checkpoint `convnext_tiny.fb_in22k_ft_in1k`. **See the open question in §9.** |
| Train data and labels only | No external images, inspection records, coordinates or web lookups. |
| No synthetic data | Standard photometric and mild geometric augmentation only. **No mixup or cutmix**: blending two photos also blends their image-level evidence and makes labels wrong, and it sits close to "synthetic data". |
| Per-image targets | **No aggregation of test predictions across photos that share a site prefix.** That would copy site-level evidence onto each photo, which the challenge forbids, and it needs whole-test-set visibility, which guidebook §4.2.5 forbids. |
| Test usage | One image at a time with h-flip TTA (§4.1.3). ConvNeXt uses LayerNorm, so predictions don't depend on batch mates. |
| Site key | Used on train rows only, for grouped folds and metric-matched loss weights. It is never a model input, and test sites are unseen anyway. |
| Determinism | Seeds fixed at 42, `cudnn.deterministic=True`, seeded DataLoader workers and generator. |

## 4. Data audit (first step once the data is loaded)

Run these checks and record the results in `CHALLENGE_NOTES.md`:

1. **Per-label prevalence**, both image-level and **site-weighted**. The site-weighted prevalence is roughly the AP of a random ranking, so it sets the floor. If prevalences are 10–30%, a random submission scores about 0.2 and the 0.5 target is about 2.5× chance.
2. **Label co-occurrence matrix.** Collapse, debris and washout often co-occur. A shared backbone with one 4-way head exploits this for free.
3. **Photos per site**: count, median, max. This decides how much site-weighting matters and how noisy the grouped CV will be.
4. **Image sizes, aspect ratios and EXIF orientation.** This sets the 4:3 letterbox input and confirms `exif_transpose` is needed.
5. **Label consistency within a site.** If labels vary a lot inside a site, that confirms per-image evidence matters and site-level shortcuts would hurt.
6. **Visual review of about 20 positives per label** to see how small the evidence is. Small evidence, such as scour at a pier base, argues for higher resolution (Experiment E2).

## 5. Validation design: how "no overfitting" is enforced

The test set is built from **unseen sites**, so validation must be as well.

- **`StratifiedGroupKFold(5)` grouped on the site prefix**, stratified on the 4-bit label pattern. `solution.py` asserts that no site lands on both sides of a split.
- **The exact challenge metric** (site-weighted macro AP) runs on out-of-fold predictions. It is reported pooled and per fold, and per label with prevalence.
- **Repeated CV for every modelling decision.** Run 3 shuffles × 5 folds (seeds 42/43/44) and compare mean ± std, following LEARNINGS (c). A single split can flip the ranking between two configs.
- **Keep/revert rule.** Keep a change only if the repeated-CV mean improves by **≥ 0.01** *and* it doesn't lose on 2 of 3 repeats. This is stricter than the workspace's generic +0.003, because with about 840 images and site weighting the fold std is likely 0.03–0.06.
- **Limit the number of decisions.** Run at most about 6–8 experiments (§8). Each extra comparison against the same OOF adds optimism.
- **No early stopping on the validation fold.** The schedule has a fixed epoch count, so OOF scores are not inflated by picking the best epoch per fold.
- **Prior-only sanity check.** Score a constant-prior submission under the metric. The trained model must clearly beat it, and it shows how far the label mix of held-out sites can move the score.
- **Treat Public LB as a soft signal only** (guidebook §1.2). Upload the CSV check first, which is free, and compare it with CV before spending a script credit.

## 6. Model design (as implemented in `solution.py`)

| Component | Choice | Why |
|---|---|---|
| Backbone | `convnext_tiny.fb_in22k_ft_in1k` (28M params) | ImageNet-22k features transfer well to outdoor structural scenes. It is small enough for 5 folds in minutes on an A10G, and its LayerNorm is batch-independent. With 841 images, fine-tuning a strong pretrained backbone is far more reliable than the reference from-scratch CNN. |
| Head | Linear, 4 logits, sigmoid | Multi-label. The shared representation captures label co-occurrence. |
| Input | 384×512 (4:3), **letterboxed** | Preserves aspect ratio, so tilt and displacement angles aren't distorted, and keeps the whole frame, so edge evidence isn't cropped away. |
| Augmentation | RandomResizedCrop keeping 70–100% of the frame, h-flip, colour jitter, light blur | Label-preserving. **No vertical flips or rotations**, because "tilted" or "collapsed" is defined relative to gravity: rotating an intact pier fakes the label. |
| Loss | BCE with **site weights `1/n_site`** (mean 1) × capped `pos_weight = √(neg/pos)` in [1, 4] | The loss mirrors the metric, so each training site counts equally. The capped pos_weight gives rare labels gradient without fully re-balancing, since AP only cares about ranking. Both are computed on the fold's training split only. |
| Optimiser | AdamW, lr 1e-4 backbone / 1e-3 head, wd 0.05, drop-path 0.1, 1-epoch warmup + cosine, grad-clip 2, AMP fp16 | A standard, stable recipe for small-data fine-tuning. The low backbone LR limits forgetting and overfitting. |
| Epochs | 12, fixed | Enough to converge on ~670 images per fold. Confirmed or tuned in E5 with repeated CV. |
| Ensemble | Average of the 5 fold models' test probabilities | Reduces variance and reuses the CV models, with no extra full-data run. |
| TTA | Original + h-flip, averaged per image | Allowed per guidebook §4.1.3. |
| Output | Probabilities clipped to [0, 1], JSON with the 4 keys | Matches the submission schema exactly. Assertions cover ids, uniqueness, keys and finiteness. |

## 7. Runtime budget (A10G, estimated; check on the first real run)

| Stage | Estimate |
|---|---|
| Decode and letterbox 1,056 JPEGs into RAM | 20–40 s |
| Weight download (~110 MB) | 5–20 s |
| 5 folds × 12 epochs × ~670 images at 384×512, fp16 | ~5–7 min |
| OOF and test inference with flip TTA | < 1 min |
| **Total** | **~7–10 min**, well inside 30 min. The 24-minute guard is a safety net. |

The headroom is deliberate. It pays for E2 (higher resolution) or E4 (second backbone) if they earn their place.

## 8. Experiment plan toward ≥ 0.50

Run each experiment with 3× repeated grouped CV and keep it only under the rule in §5.

| # | Hypothesis | Change | Expected effect |
|---|---|---|---|
| E0 | Floor | Prior-only and random scores under the metric | Calibrates what 0.5 means for these prevalences |
| E1 | Baseline | `solution.py` as written | Target ≳ 0.45–0.55. This is an estimate and depends on prevalences. |
| E2 | Evidence is small (scour, lodged debris) | 448×597 or 512×683 input | + for scour and debris if E1 per-label AP is weakest there |
| E3 | Backbone capacity or inductive bias | `convnext_small.fb_in22k_ft_in1k`, `tf_efficientnetv2_s.in21k_ft_in1k`, `eva02_small_patch14_336` | Pick one by repeated CV |
| E4 | Architecturally diverse models make different errors | Rank-average 2 backbones, each with 5 folds, only if OOF correlation < 0.9 | Typically +0.01–0.03 AP. Must still fit in about 20 min. |
| E5 | Schedule | Epochs {8, 12, 20}, backbone LR {5e-5, 1e-4, 2e-4} | Small; guards against under- or over-training |
| E6 | Label imbalance | Asymmetric loss (γ⁻=2) vs capped BCE | Helps the rarest label |
| E7 | Weight averaging | EMA of weights (decay ≈ 0.99) | Smoother small-data fine-tuning |

**Error analysis between experiments.** Check per-label AP against prevalence, and list the lowest-AP held-out sites to see whether their photos are close-ups, night shots or drone views. Look at the top false positives per label: are they intact piers with water, which suggests a shortcut on "water present"? Choose the next experiment from this evidence, not from instinct.

**If CV stays below 0.5 after E1–E4**, try the next steps in this order. Raise resolution further; train a two-view model (full frame plus a centre 2× zoom, features concatenated); run LLRD with more epochs. Do not add data, pseudo-labels or site pooling. Those are banned, and they are also the usual sources of CV-to-private-LB collapse.

## 9. Open question: pretrained weights

The description says the *reference* trains a compact CNN **from scratch** and says to "use only the provided training photographs and labels to fit the model". It does **not** say pretrained backbones are banned or that this is a From-Scratch challenge, and guidebook §4.1 allows general-purpose timm/HF backbones by default. Guidebook §4.4 says to **ask a reviewer when unsure**, so confirm on Discord before the first script submission.

- If pretrained is allowed, use this plan.
- If the challenge is From-Scratch, set `pretrained=False` and move to a ResNet-18/26-width CNN at 256×341, 60–80 epochs, stronger photometric augmentation and dropout, lr 1e-3 with a one-cycle schedule. Expect a noticeably lower score, and 0.5 may not be reachable, but beating the reference is the payout bar.

## 10. Pre-submission checklist

- [ ] `python scripts/compliance_scan.py solution.py` shows no errors. The single "pseudo" warning comes from the header sentence that *forbids* pseudo-labelling.
- [ ] A fresh run regenerates `working/submission.csv`, so no stale file is submitted.
- [ ] 215 rows, ids match `sample_submission.csv`, 4 keys in every row, values in [0, 1].
- [ ] The run log shows total runtime well under 1800 s and the time guard never fired.
- [ ] OOF repeated-CV mean ± std and per-label AP are recorded in `CHALLENGE_NOTES.md`.
- [ ] Upload the CSV check (free) → compare with CV → only then submit the script (1 credit).
