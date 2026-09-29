# Cotton-Module View Classification — Design & Approach

Challenge: classify each cotton-module frame as `0 = background`, `1 = end_view`
(circular wrapped end), or `2 = side_view` (cylindrical wrapped side).
Platform: Project Eris (shipd.ai), AI baseline score = **0.75**.

## 1. What the data actually looks like

| | value |
|---|---|
| Train | 2,430 images, 256×144 RGB JPEG |
| Test | 570 images, **one operational family absent from train** |
| Class balance | side_view 59.0%, background 31.7%, end_view 9.3% |
| Metric | `0.70·macro_F1 + 0.15·end_view_recall + 0.15·side_view_precision` |

Visual inspection of sample grids per class confirmed the task is genuinely
learnable from pixels: `end_view` shows a round/elliptical wrap-end (sometimes
filling the frame, sometimes a small distant shape among several modules),
`side_view` shows an elongated curved wrap surface dominating the frame, and
`background` is mostly heavy glare/overexposure or no module in view. This
rules out a naive "brightness threshold" heuristic — some end_view examples
are small and distant against complex yards — and argues for a real
spatially-aware CNN rather than global color statistics.

**Majority-class baseline** (always predict `side_view`) scores only **0.26**
on this metric — hitting 0.75 requires genuinely balanced 3-class performance,
not just overall accuracy.

## 2. The real risk: family shift, not in-distribution accuracy

The held-out 570 rows are a *complete* operational family with zero
representation in train — different camera/lighting/equipment, per the
problem statement. I validated this empirically before designing around it:

- A ResNet18 fine-tuned on a random 700-image split, validated on a random
  300-image split → **0.77** composite score.
- The same setup, but validated on a *pseudo-unseen family* (images
  clustered by color/lighting into 7 groups, one held out entirely) →
  **0.65** — a **0.12-point drop** from the same amount of training data.

That's the central design constraint: a solution tuned to maximize
in-distribution CV will look great locally and underperform on the real
holdout. Everything below is chosen to close that gap, not to push the
in-fold number higher.

## 3. Approach

**Architecture** — `timm` EfficientNet-B0, ImageNet-pretrained, fully
fine-tuned, 3-way softmax head with dropout 0.3. Chosen over a larger backbone
(ResNet50/ViT) because train is small (2,430 images, only 226 `end_view`) —
a bigger model overfits the majority-family cues (specific backgrounds,
lighting) faster than it learns geometry.

**Domain-robust augmentation** (real transforms of real images — no
generation, nothing added to the dataset, fully compliant with the "no
synthetic data" rule):
- `RandomResizedCrop(scale=0.75–1.0)` — simulates the "modules at different
  distances" nuisance factor.
- Strong `ColorJitter` (brightness/contrast/saturation/hue) — directly targets
  the glare/overexposure/night-capture shift between families.
- Mild rotation, horizontal flip, occasional Gaussian blur — camera angle and
  motion-blur robustness.

**Class imbalance** — `WeightedRandomSampler` per fold so each epoch sees a
roughly balanced mix instead of drowning `end_view` (9.3% natural frequency)
under `side_view`. Label smoothing (0.05) as light additional regularization.

**Validation & ensembling** — 5-fold `StratifiedKFold` (no group column is
exposed in the data, by design — see "What Not To Use" in the problem
statement — so stratified is the correct, compliant choice per the platform's
own CV harness rules). All 5 fold models are ensembled at inference
(soft-voted), which is both a straightforward accuracy gain and a variance
reduction — a single fold's overfit quirks get averaged out, which matters
more than usual given the family-shift risk.

**Decision-rule tuning** — the metric explicitly rewards `end_view` recall and
`side_view` precision, not raw accuracy, so I grid-search a small per-class
probability multiplier against **out-of-fold training predictions** (never
the test set — this stays inside the "no test-time calibration" rule) to
directly maximize the competition's own formula rather than a plain argmax.
The search range is deliberately modest (0.9–1.8× for `end_view`, 0.85–1.2×
for `side_view`) after a debug run on a tiny subset showed a wide-open search
overfitting to sampling noise on the rare class.

**Test-time augmentation** — horizontal-flip TTA at inference, explicitly
listed as allowed (one-sample-at-a-time, no test-set visibility).

**Compliance** — real training/fine-tuning happens inside the script every
run (no cached artifacts, no self-hosted weights), fixed seeds throughout,
relative I/O paths, a 3100s time guard with graceful early stop, and a
shape/uniqueness assertion before writing the submission. No external data,
no synthetic images, no test-distribution calibration, no tabular-on-raw-
pixels shortcut.

## 4. Evidence this clears 0.75

A deliberately tiny/fast proof-of-concept (480 images, half resolution, 4
epochs, CPU, simple augmentation, *no* ensembling or threshold tuning) already
scored **0.80** on a random in-distribution split. The full design adds
5-fold ensembling, 22 epochs at native resolution, stronger domain-targeted
augmentation, and metric-aware threshold tuning — but the ~0.12-point
cross-family gap measured in Section 2 means the honest expectation on the
true holdout is closer to **0.70–0.82** than a flat "add 0.05 to the smoke
test." The design choices above (smaller backbone, heavier photometric
augmentation, ensembling, OOF-only tuning) are specifically the ones that move
the needle on the family-shift number, not the in-distribution one.

## 5. Files

- `solution.py` — the actual submission script. Run as
  `python3 solution.py <public_dir> <submission_out>` per the platform's
  contract.
- `validate.py` — local CV harness (same model/augmentation, OOF metrics
  only, no submission written) for iterating before spending a submission
  credit: `python3 validate.py ./dataset/public --quick` for a fast pass,
  drop `--quick` for a real read before submitting.

## 6. Suggested iteration path (matches the platform's own workflow)

1. **Sub 1** — this script as-is: correct pipeline, EfficientNet-B0, 5-fold
   ensemble, tuned thresholds.
2. **Sub 2–3** — if `validate.py`'s OOF score plateaus, try: a second backbone
   (e.g. `resnet34`) ensembled with EfficientNet-B0; feeding a hand-engineered
   brightness/exposure feature into the classifier head alongside the image
   (explicitly allowed — "extra signal for a genuine CV model"); heavier
   augmentation if the CV/public-LB gap is large.
3. **Sub 4–5** — multi-backbone ensemble, per-fold TTA at multiple crop
   scales, revisit the bias-search range with the full dataset's actual OOF
   stability rather than the conservative default above.

Submit the CSV first (free score check, no credit cost) before submitting the
script itself, and don't over-fit decisions to the Public LB — the
guidebook is explicit that the Private LB is what counts.
