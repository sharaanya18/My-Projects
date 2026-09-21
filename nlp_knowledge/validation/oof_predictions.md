# Out-of-Fold Predictions

OOF predictions are the backbone of honest ensembling, threshold tuning,
calibration, and error analysis. If you build one piece of infrastructure first,
build this.

## What they are

For each row, the prediction made by a model that **did not see that row in
training**. Fit on k-1 folds, predict the held-out fold, repeat. The result is a
full-length prediction vector where every entry is out-of-sample.

```python
from sklearn.model_selection import cross_val_predict
oof = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)
```

For anything beyond a plain sklearn estimator, write the loop yourself and save
per-fold artifacts. See `../code/validation.py::oof_predict`.

## What they are for

1. **An unbiased CV metric** — compute the score on the full OOF vector.
2. **Threshold tuning** — the only honest place to do it
   (`threshold_and_calibration.md`).
3. **Calibration** — fit the calibrator on OOF.
4. **Ensemble weights** — fit blend weights on OOF, never on test.
5. **Stacking** — the meta-model's training input.
6. **Error analysis** — every row has an out-of-sample prediction, so you can
   analyze errors over the *entire* training set, not just one fold.
7. **Model correlation** — the input to "is this ensemble worth it?"

## Rules

- **Same fold assignment for every model.** Save it to disk. Different folds →
  the OOF vectors aren't comparable and the blend weights are meaningless.
- **Save the per-fold test predictions too.** The standard pattern is: for each
  fold, predict the held-out fold (→ OOF) *and* the test set; average the k test
  predictions at the end. You get a free k-model ensemble and never have to
  refit on the full data.
- Store `oof.npy`, `test.npy`, the fold ids, the config, and the CV score
  together per experiment. `../code/experiment_log.py` does this.
- For classification, keep **probabilities**, not hard labels — you cannot
  recover probabilities later, and thresholds/blending need them.

## The stacking leak

Stacking is the most leak-prone technique in common use. The meta-model must be
trained on **out-of-fold** base predictions only. Two failure modes:

1. Training the base model on all data, then predicting the same data → the
   meta-model sees in-sample base predictions and learns to trust them far too
   much.
2. Evaluating the meta-model without a **nested** outer loop → the reported
   score is optimistic, because the meta-model's own hyperparameters were chosen
   on the same OOF vector it's scored on.

Keep the meta-model deliberately simple (LogisticRegression, or a
non-negative-weight average). A complex meta-model over k columns will overfit
the OOF vector, and that overfit is invisible without nesting.

## Sanity checks

- OOF metric should be **close to** the mean of per-fold metrics. A large gap
  means a bug in assembly (misaligned indices are the usual cause).
- OOF metric should be **lower** than any in-sample metric. If not, something
  leaked.
- Check `len(oof) == len(train)` and that no row was predicted twice or zero
  times.
