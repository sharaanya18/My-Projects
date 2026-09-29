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
relative I/O paths, and a shape/uniqueness assertion before writing the
submission. No external data, no synthetic images, no test-distribution
calibration, no tabular-on-raw-pixels shortcut. Note: training uses a fixed
schedule (5 folds × 22 epochs, early stopping keyed on validation score only)
rather than a wall-clock time guard — see Section 4 for why.

## 4. Evidence this clears 0.75 (measured, not projected)

A deliberately tiny/fast proof-of-concept (480 images, half resolution, 4
epochs, CPU) scored 0.80 on a random in-distribution split, which motivated
building the full pipeline. That full pipeline was then **actually run on a
Kaggle GPU** (T4/P100) against the real 2,430/570 split, twice:

| Run | Total runtime | Per-fold best scores | OOF score (bias-tuned) |
|---|---|---|---|
| 1st (had wall-clock guard, never triggered) | 741s (~12.4 min) | 0.829 / 0.818 / 0.785 / 0.816 / 0.786 | **0.8088** |
| 2nd (guard removed, fully deterministic) | 775s (~12.9 min) | identical | **0.8088** |

The two runs producing bit-identical per-fold scores, chosen bias
`(1.0, 1.25, 1.05)`, and OOF score confirms the fix (removing wall-clock
branching, described below) didn't change behavior — it was dead code that
never fired, since the full schedule finishes in ~13 minutes against a
90-minute cap.

**0.8088 is in-distribution OOF, not the true held-out-family score** — the
~0.12-point cross-family gap measured in Section 2 still applies, so the
honest range for the real holdout is roughly **0.68–0.81**, comfortably
straddling the 0.75 baseline rather than guaranteed above it. One more signal
worth watching: the model's predicted test-set distribution (4.6% background
/ 10.4% end_view / 85.1% side_view) is quite different from train's (31.7% /
9.3% / 59.0%). That could be the unseen family genuinely looking different
(fewer glare/background frames, more side-on shots), or the model
over-committing to the majority class under distribution shift — there's no
way to tell without ground truth, so treat it as an open risk rather than a
red flag to "fix."

### A real compliance bug caught along the way

The initial version used a wall-clock time budget (`if elapsed() > BUDGET:
break`) to cut training short if it ran long — standard advice in a lot of
ML-competition boilerplate. Watching Project Eris's own pre-submission
checker reject an unrelated solution for exactly this pattern
("Deterministic Execution: your solution changes its training plan based on
runtime conditions") made clear this script had the same latent bug: which
fold/epoch gets kept would depend on how fast the GPU happened to run that
day, which isn't reproducible. Since the measured runtime (~13 min) has huge
margin against the cap, the fix was simply to remove the time-based branches
entirely and rely only on a fixed schedule plus validation-score-based early
stopping (deterministic given fixed seeds/data, independent of hardware
speed).

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
