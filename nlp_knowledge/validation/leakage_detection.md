# Leakage Detection

Leakage is information in training that will not be available (or not in the
same form) at test time. It inflates CV and collapses on the leaderboard.

Phase 13 explicitly warns that Shipd benchmarks may contain deliberate
distractors and artifacts — so treat leakage hunting as a **standing task**, not
a one-time check.

## Types, and how to find each

| Type | Symptom | Test |
|---|---|---|
| **Duplicate rows across splits** | implausibly high CV | exact + near-duplicate hashing (MinHash / char-3gram cosine) |
| **Group leakage** | pair/query tasks especially | connected-component grouping; see `cross_validation.md` |
| **Target leakage in a feature** | one feature dominates importance | drop it → does CV collapse? |
| **ID leakage** | row id / hash correlates with target | plot target vs index and vs id |
| **Temporal leakage** | future information in features | split by time, compare |
| **Preprocessing leakage** | vectorizer/scaler fit on all data | use `Pipeline` |
| **Metadata leakage** | source/annotator perfectly separates classes | per-value target rate |
| **Test-set leakage** | test text appears in train | search test strings in train |

## Standing checklist

Run all of these on any new dataset:

1. **Exact duplicates** on the raw text, within and across splits.
2. **Near duplicates** — char 3-gram TF-IDF cosine > 0.95, or MinHash.
   Report how many, and whether duplicates share a label (if they don't, you
   have label noise too).
3. **Target rate by every categorical column.** A column where some value has a
   100% target rate is a suspect.
4. **Target vs row order.** If sorting by index recovers the label, the file was
   assembled by class.
5. **Feature importances** after a first GBDT. Anything dominant gets explained
   or dropped.
6. **An adversarial validation run** (below).
7. **Read the top coefficients of a linear model** (`tfidf_linear.md`). Leakage
   tokens show up here immediately and legibly — formatting artifacts, template
   strings, annotation markers.

## Adversarial validation

The single most informative 10-minute check. Label train rows 0 and test rows 1,
then train a classifier to tell them apart:

- **AUC ≈ 0.5** → train and test are drawn alike. Good; random CV is defensible.
- **AUC ≈ 1.0** → they are trivially separable. Inspect the top features: that
  is either leakage, a preprocessing difference, or a real distribution shift.
  See `distribution_shift.md`.
- **AUC in between** → partial shift. Use the classifier's probability to build
  a validation set that *looks like* test (sample the training rows most
  test-like), which is a much better proxy than a random split.

Implemented in `../code/validation.py::adversarial_validation`.

## The decision rule

When you find a suspiciously predictive feature, the question is **not** "is it
leakage?" but:

> **Will this feature exist, with the same meaning and the same relationship to
> the target, in the test set?**

- Yes → legitimate. Use it (and say why you believe it).
- No → leakage. Drop it.
- Unknown → **test it**: split by time or by group and see if the relationship
  survives. If it doesn't survive a structural split, it won't survive test.

Phase 13's warning applies directly: do not assume a feature correlated with the
training target remains useful in test. Prove it with a structural split, not
with a random one.

## After removing leakage

Expect CV to drop. **That is the correct outcome** — the earlier number was
fiction. Re-baseline everything against the clean split, and don't compare
post-fix numbers to pre-fix ones.
