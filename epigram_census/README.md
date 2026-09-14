# Unseen Epigram Census

Predict the six-count editorial genre census (`g0`..`g5`) of the *withheld*
epigrams of a manuscript, given the Greek text of the epigrams that *are*
shown plus `total_count` and `hidden_count`.

Scored by mean Bray-Curtis similarity `S = 1 - |p - y|_1 / (|p|_1 + |y|_1)`.

## Final model

1. **Normalisation** - lower-case, NFD, drop combining marks, keep Greek
   letters and whitespace. Folds away accentuation and editorial punctuation,
   which vary by transcription rather than by genre.
2. **Mean-pooled per-epigram TF-IDF** - `char_wb` 2-4 grams, sublinear tf.
   Each epigram is vectorised on its own and the manuscript vector is the
   mean, then L2-normalised. Pooling first and applying a linear map second
   makes the model a per-epigram genre scorer averaged over the manuscript -
   the mixture assumption the task describes - and stops long epigrams from
   dominating.
3. **Two estimators blended 50/50**: ridge on the pooled TF-IDF, and
   ExtraTrees (300 trees) on a 200-dim TruncatedSVD of the same matrix. Both
   regress the rate vector `y / hidden_count`.
4. **Counts** = `floor(rate * hidden_count + 0.60)`, i.e. round up from 0.40,
   clipped to `[0, hidden_count]`.

The ridge penalty is chosen by 5-fold CV on train alone, under the same
rounding rule the final model uses.

## Results

| model | train CV (3 seeds) | group-aware CV | validation |
| --- | --- | --- | --- |
| global genre rate x hidden_count | - | - | 0.563 |
| previous solution (ridge only, rint) | 0.6846 | 0.6569 | 0.7153 |
| **+ SVD-ExtraTrees blend** | 0.6912 | 0.6633 | 0.7228 |
| **+ round-up-from-0.40** | **0.6935** | **0.6744** | **0.7313** |

Both accepted changes improve all three estimators simultaneously. Runtime
2m05s on 4 CPU cores; two full runs produce byte-identical submissions.

## What produced the gain

**The blend (+0.007 val).** ExtraTrees on SVD features scores 0.7216 on
validation alone against ridge's 0.7153, and errs differently. Every blend
weight from 0.2 to 0.8 beats ridge on both group CV and validation - the gain
is flat across the whole range, which is what genuine model complementarity
looks like rather than a tuned artifact. The untuned 0.5 is used, so no
weight was fitted.

**The rounding threshold (+0.009 val).** Predicted total genre mass averaged
1.177 per hidden epigram while the labels average 1.25 (train) and 1.30
(validation): the model systematically under-predicts mass, and the
Bray-Curtis denominator makes under-prediction the costlier error. Rounding
up from 0.40 corrects exactly that bias - predicted mass becomes 1.306 - and
improves CV, group CV and validation together. This is a bias correction with
a measurable mechanism, not a search against the metric.

## Measured and rejected

Every row below was measured with per-fold refitting of the vectoriser, so no
held-out manuscript influences its own preprocessing.

| change | train CV | group CV | validation | verdict |
| --- | --- | --- | --- | --- |
| char 2-5 / 3-5 grams | 0.686 | 0.655 | 0.712 / 0.703 | worse |
| word 1-gram / 1-2 grams | 0.675 / 0.668 | 0.647 / 0.643 | 0.703 / 0.691 | worse |
| char + word concatenated | 0.6808 | 0.6562 | 0.7137 | no gain |
| whole-manuscript-document pooling | 0.6727 | 0.6529 | 0.7090 | worse |
| max pooling | 0.6816 | 0.6590 | 0.7125 | mixed, no gain |
| mean+max pooling blend | 0.6833 | - | 0.7128 | worse |
| no final L2 on the pooled vector | 0.6813 | 0.6527 | 0.7043 | worse |
| size features (total/hidden/revealed) | 0.6844 | 0.6634 | 0.7139 | worse |
| text-length features | 0.6859 | 0.6609 | 0.7112 | worse |
| all numeric features | 0.6869 | 0.6611 | 0.7147 | worse |
| sample weight = hidden_count | 0.6800 | 0.6506 | 0.6979 | worse |
| sample weight = sqrt(hidden_count) | 0.6818 | 0.6555 | 0.7074 | worse |
| shrinkage to prior (k=0.5..4) | 0.664 -> 0.617 | 0.646 -> 0.596 | 0.704 -> 0.646 | worse |
| per-genre ridge alpha | 0.6841 | 0.6602 | 0.7136 | see below |
| SVD-300 / SVD-150 + ridge | 0.6848 / 0.6804 | 0.6592 / 0.6630 | 0.7036 / 0.7014 | worse |
| kNN k=25 / k=50 | 0.6744 / 0.6498 | 0.6566 / 0.6322 | 0.7151 / 0.6864 | no gain |
| SVD-150 + HistGradientBoosting | 0.6637 | 0.6463 | 0.7015 | worse |
| occurrence-level clip-then-average | 0.6352 | 0.6271 | 0.6528 | much worse |

**Per-genre alpha is the instructive rejection.** It looked like +0.0040 on
CV, but that CV shared fold seeds with the selection. Re-selecting on seeds
0-1 and scoring on unseen seeds 2-4 cut the gain to **+0.0013**, and the
selected alphas themselves changed with the seed - two thirds of the apparent
gain was selection bias. Validation was -0.0017. Rejected.

**Occurrence-level scoring is the instructive failure.** Clipping each
epigram's predicted rate to [0,1] before averaging collapses the score to
0.6528. The per-epigram linear scores are not calibrated probabilities; the
averaging is what makes them meaningful, so the aggregate must stay the unit
of prediction.

## Distribution audit (why validation misled)

A leaderboard submission of the ridge/rint model scored **0.638** against its
0.7153 validation - below even the 0.657 group-CV floor. The audit explains it:

| | train | val | test |
| --- | --- | --- | --- |
| hidden_count <= 3 | 59.9% | 55.6% | **72.6%** |
| hidden_count > 8 | 12.2% | 19.4% | **4.5%** |
| max cosine similarity to train | 0.573 | **0.622** | 0.573 |
| shares a normalised text with train | - | **32.3%** | 18.5% |

Validation is closer to train than train is to itself, and skews to large
manuscripts; test is at normal distance and skews small. Small hidden_count
makes Bray-Curtis coarser, so test is intrinsically harder. Reveal fractions
are identical everywhere (0.475), and TF-IDF coverage is equal or better on
test (246 vs 242 nonzero n-grams per epigram, no empty rows), so the shift is
manuscript size and similarity - not sampling or vocabulary.

Model selection therefore uses **group-held-out splits reweighted to the test
size profile**, computed from the public test inputs only. On that estimator
the submitted model scores 0.6574, close to its actual 0.638 - a usable proxy
where validation was off by 0.077.

| model | natural group split | test-matched | std |
| --- | --- | --- | --- |
| ridge, rint (scored 0.638) | 0.6680 | 0.6574 | +-0.0181 |
| **blend, round-up-from-0.40** | **0.6842** | **0.6751** | **+-0.0139** |
| + shrink 0.15 / 0.30 to prior | - | 0.6583 / 0.6444 | worse |
| ExtraTrees alone | 0.6762 | 0.6664 | +-0.0200 |

The blend is both the best and the **lowest-variance** of 27 configurations
tested across 6 group splits; ExtraTrees alone is the most volatile, so
blending is what buys robustness. Shrinkage toward the prior hurts uniformly.
The 0.40 threshold still beats 0.45 and 0.50 after reweighting to the
small-manuscript profile.

## Validation protocol

Splits are manuscript-level throughout. Inside every CV fold the TF-IDF
vocabulary and IDF, the SVD basis, and both regressors are fitted on the
training fold only. Validation labels are never used to build features or fit
anything - only to evaluate after predictions are made from training data.
The final model refits on train + validation only after every choice was
fixed.

Three estimators are tracked because they disagree in an informative way:

* **plain 5-fold CV**, averaged over 3 seeds. Single-seed CV is unreliable
  here - seed 0 alone reports 0.6861 for the ridge baseline against 0.6801
  over two seeds.
* **group-aware CV**. Under the normalisation above, 118 of 509 train
  manuscripts (23.2%) share a text with another, largest linked group 24, so
  plain CV leaks. Grouping by shared normalised text (union-find) removes it.
* **held-out validation**, a separate manuscript group, like the test split.

Plain CV alone would have wrongly adopted char 2-5 grams, SVD+ridge and
per-genre alpha - each gains on plain CV while losing on both of the others.
Only changes that improve all three were accepted.

The organisers group by *exact* text and that holds (zero raw texts shared
across splits), but accent-stripping re-links manuscripts across their
boundary, so the graded test set is not overlap-free either: 19.7% of test
manuscripts share a normalised text with the fitting set, against 23.2%
simulated by plain CV and 0% by group CV. Group CV is therefore a
conservative floor, not the expected score.

## Usage

    python3 solution.py <public_dir> <submission_out>

The parent directory of the output is created if missing. Inputs are read
from `*.jsonl` when present and from the `*.csv` mirrors otherwise (they
parse to identical records). The validation split is optional.

Needs only numpy, scipy and scikit-learn - no network, no pandas, no
pretrained weights (no transformer weights are available offline in this
environment, so semantic embeddings were not testable). Peak memory is modest:
the pooled TF-IDF matrix is 509 x ~26k sparse and the SVD basis is 200-dim.

Output is validated before exit: exact query-id set against
`sample_submission.csv`, no duplicates, every count a finite integer within
`[0, hidden_count]`. No sum-to-hidden-count constraint is imposed; the
predicted mass (1.306 per hidden epigram) matches the label mass as a
consequence of the model, not a constraint.
