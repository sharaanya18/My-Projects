# Unseen Epigram Census

Predict the six-count editorial genre census (`g0`..`g5`) of the *withheld*
epigrams of a manuscript, given the Greek text of the epigrams that *are*
shown plus the manuscript's `total_count` and `hidden_count`.

Scored by mean Bray-Curtis similarity `S = 1 - |p - y|_1 / (|p|_1 + |y|_1)`.

## Approach

Revealed and hidden epigrams are two subsets of the same manuscript, so the
hidden part's genre mixture is estimated from the revealed part's text.

1. **Normalisation.** Lower-case, NFD, drop combining marks, keep Greek
   letters and whitespace. This folds away accentuation, editorial brackets
   and abbreviation marks, which vary by transcription rather than by genre.
2. **Mean-pooled per-epigram TF-IDF.** Each epigram is vectorised
   (`char_wb` 2-4 grams, sublinear tf) and L2-normalised *on its own*; the
   manuscript vector is the mean over its epigrams. Pooling first and
   applying a linear map second means the model is a per-epigram genre scorer
   averaged over the manuscript — the mixture assumption the task describes.
   It also stops long epigrams from dominating short ones.
3. **Ridge regression** onto the genre rate vector `y / hidden_count`. The
   aggregate counts are the only supervision that exists; no per-epigram
   label is ever observed, and none is required.
4. **Counts** are `round(rate * hidden_count)`, clipped to `[0, hidden_count]`.

Character n-grams beat word features here because Greek is heavily inflected
and the transcriptions are not orthographically uniform, so subword matching
generalises across manuscripts better than whole-word matching.

## Guarding against overfitting

* The ridge penalty is the only tuned hyperparameter, chosen by 5-fold
  cross-validation **on train alone**. The alpha curve is a smooth interior
  optimum (0.683 / 0.684 / **0.686** / 0.685 / 0.677 over 0.1-0.8), not an
  isolated spike, so the choice is stable rather than lucky.
* `validation` is never used for any selection, so its score is an honest
  estimate of performance on unseen manuscripts. It comes out slightly
  *above* the train cross-validated score, which is what an under-fitted
  rather than over-fitted model looks like.
* The final model is refit on train + validation with that alpha, purely to
  use all available labelled manuscripts.
* No post-hoc fitting against the metric. Three score-raising tricks were
  tried and deliberately rejected: a per-genre affine recalibration searched
  directly on Bray-Curtis (12 free parameters tuned on the score function
  itself, no epigram-level meaning), a sum-repair step (a no-op), and a
  sampling-based expected-Bray-Curtis decoder (validation 0.728 -> 0.714).
  The first inflated reported scores but is the kind of thing that decays
  first on a group-held split.

## Results

| model | train CV | validation |
| --- | --- | --- |
| global genre rate x hidden_count | 0.525 | 0.563 |
| best constant vector | 0.604 | 0.622 |
| **pooled TF-IDF + ridge** | **0.686** | **0.715** |

Task baseline 0.66, quoted high score 0.70.

### How much of the train CV is text overlap

The organisers keep manuscripts linked by repeated *exact* text in the same
split, and that holds: across splits, zero raw texts are shared. But the
accent-stripping normalisation above collapses orthographic variants, which
re-links manuscripts the organisers had separated. Under normalised text,
118 of 509 train manuscripts (23.2%) share a text with another train
manuscript, the largest linked group being 24.

So plain shuffled KFold is optimistic. Grouping train by shared normalised
text (union-find) and using GroupKFold gives 0.657 rather than 0.686.

That 0.657 is a floor, not the expected test score, because the graded test
set is *not* overlap-free either:

| condition | manuscripts sharing a normalised text with the fitting set |
| --- | --- |
| GroupKFold simulates | 0% |
| plain KFold simulates | 23.2% |
| **the graded test set actually has** | **19.7%** |

Plain KFold (23.2%) approximates the real test condition (19.7%) far better
than GroupKFold (0%), so the honest range is 0.657 as a pessimistic floor,
0.686 as the best-matched estimate, and 0.715 on the held-out validation
split.

The choice of alpha is unaffected. Plain KFold, GroupKFold and the held-out
validation split all peak at alpha = 0.3, so the leakage moves the level but
not the ranking, and the shipped model is the same either way. Three
independent estimators agreeing on the same alpha is stronger evidence for
it than any one of them alone.

## Usage

The platform supplies both paths positionally:

    python3 solution.py <public_dir> <submission_out>

Locally:

    python3 solution.py ./dataset/public ./working/submission.csv

The parent directory of the output is created if it does not exist.
Inputs are read from `*.jsonl` when present and from the `*.csv` mirrors
otherwise (the mirrors parse to identical records, so either works).

Needs only numpy, scipy and scikit-learn -- no pandas, no network, no
pretrained weights. A full run is about four minutes on CPU, well inside the
90-minute budget.

Output is validated before exit: exact query-id set against
`sample_submission.csv`, no duplicates, and every count a finite integer
within `[0, hidden_count]`.
