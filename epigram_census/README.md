# Unseen Epigram Census

Predict the six-count editorial genre census (`g0`..`g5`) of the *withheld*
epigrams of a Byzantine manuscript, given the Greek text of the epigrams that
*are* shown plus the manuscript's `total_count` / `hidden_count`.

Scoring is the mean Bray-Curtis similarity

    S = 1 - |p - y|_1 / (|p|_1 + |y|_1)

over manuscripts.

## Approach

Revealed and hidden epigrams are two subsets of the same manuscript, so the
hidden part's genre *mixture* is estimated from the revealed part's text:

1. **Mean-pooled per-epigram TF-IDF.** Each epigram is normalised (lower-case,
   diacritics and editorial punctuation stripped), vectorised and
   L2-normalised on its own; the manuscript vector is the mean. A linear map on
   that pooled vector is exactly a per-epigram linear genre scorer averaged
   over the manuscript, which is the right inductive bias here — and it needs
   only the aggregate targets, the sole supervision that exists (no per-epigram
   label is ever observed).
2. **Ridge ensemble** over several views (char 2-4 gram, word unigram, a joint
   view, one view with manuscript-size metadata) plus a similarity-weighted
   k-NN mixture estimator. The target is the rate vector `y / hidden_count`.
3. **Bray-Curtis calibration.** Ridge shrinks toward the mean, which is wrong
   for a rounded L1-style metric — Bray-Curtis rewards committing to the
   genres actually present. A per-genre affine map `r -> a_g*r + b_g` is fitted
   by coordinate search directly on out-of-fold Bray-Curtis.
4. Counts are `round(r_g * hidden_count)`, clipped to `[0, hidden_count]`.

## Results

| model | train OOF | validation |
| --- | --- | --- |
| global genre rate x hidden_count | 0.525 | 0.563 |
| Bray-Curtis-optimal constant vector | 0.604 | 0.622 |
| single ridge on pooled char TF-IDF | 0.686 | 0.715 |
| **ensemble + Bray-Curtis calibration** | **0.721** | **0.728** |

## Usage

    python solution.py --data /path/to/public --out submission.csv

Add `--use-validation` to fold the validation manuscripts into the final fit
(the calibration is then refitted on out-of-fold predictions over the enlarged
set). Runs on CPU in a few minutes; needs only numpy / scipy / scikit-learn.
