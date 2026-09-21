# Playbook: Classification

Binary · multiclass · multilabel.

## 0. Audit (before any code)

- Prediction unit: what is one row of the submission?
- Metric: accuracy / macro-F1 / micro-F1 / AUC / logloss / MCC? **This decides
  everything downstream** — especially whether thresholds matter.
- Label structure: mutually exclusive? hierarchical? multilabel?
- Class balance, in train and (if inferable) in test.
- Text fields available, plus every metadata column.
- Duplicates, near-duplicates, group structure.
- Row counts, text lengths, compute budget.

Write it into `SHIPD_TASK_ANALYSIS.md`.

## 1. Validation first

Pick the scheme per `../validation/cross_validation.md`. Save the fold
assignment to disk. Run adversarial validation now — before modeling — so you
know whether test looks like train.

## 2. Baseline ladder

| Exp | Model | Stop and think if… |
|---|---|---|
| B0 | majority class | — (this is the floor; report it) |
| B1 | TF-IDF(1,2) + LogisticRegression | B1 ≈ B0 → the text may not carry the signal; re-read the task |
| B2 | + char_wb(3,5) union | large gain → noisy/morphological text; keep char features everywhere |
| B3 | LinearSVC (calibrated) | — |
| B4 | + `class_weight="balanced"` | large gain → imbalance is the real problem; go to thresholds |
| B5 | **threshold tuning on OOF** | often the biggest single jump. Do it before any bigger model |
| B6 | LightGBM on SVD + handcrafted features | gain → metadata/structure matters |
| B7 | frozen embeddings + linear head | needs HF access |
| B8 | fine-tuned transformer | needs GPU |
| B9 | blend, only if OOF correlation justifies it | see `../techniques/ensembling.md` |

**B5 before B7/B8.** Tuning a threshold costs seconds and routinely beats a
model upgrade that costs hours.

## 3. Error analysis (mandatory after B2 and again after B6)

1. Confusion matrix → pick the worst confused pair.
2. **Read 20 real errors.** Not summary statistics — the actual text.
3. Break errors down by length, class, source, language, negation presence.
4. Decide the diagnosis: **lexical** (unseen vocabulary → embeddings/char
   n-grams) or **semantic** (words present, composition wrong → transformer) or
   **label noise** (→ `../validation/noisy_labels.md`).
5. Read the linear model's top coefficients per class. Leakage and preprocessing
   bugs surface here immediately.

That diagnosis picks the next rung. Skipping it means guessing.

## 4. Shipd-specific checks

- Distractor classes / near-identical classes → look at the confusion matrix
  first, not the aggregate metric.
- Test classes absent from train → the model can never predict them; check the
  label vocabulary of both splits.
- Duplicate ids or missing rows in the submission template.
- Abstain/"none of the above" class → is it scored? Is it in train?
- A feature that looks too good → `../validation/leakage_detection.md`.

## 5. Finalize

- Refit on all data **or** average the k fold-models' test predictions (usually
  the latter — it is a free ensemble and needs no refit).
- Apply the **saved** threshold/calibrator from OOF, do not recompute on test.
- Submission checks: column names, row count, id set equality (not just count),
  ordering, dtypes, value range, no NaNs, label vocabulary matches the expected
  set.
- Assert on the prediction distribution — e.g. all-one-class output is the
  classic silent bug (see the `argmax` over a single logit in
  `../antipatterns.md`).
