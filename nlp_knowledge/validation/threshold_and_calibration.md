# Threshold Optimization & Calibration

Often the largest single gain available, and frequently skipped entirely.

## Thresholds

A classifier outputs a score. The metric needs a decision. `0.5` is the default
only because it is the default — it is almost never optimal for F1, and it is
badly wrong under class imbalance.

**Procedure:**

1. Get OOF probabilities (`oof_predictions.md`).
2. Sweep the threshold over a fine grid (or over the sorted unique scores).
3. Pick the threshold maximizing the **competition metric**, not accuracy.
4. Check stability: plot metric vs threshold. If the optimum is a narrow spike,
   it is fitted to noise — prefer a threshold in the middle of a broad plateau,
   even at a slightly lower peak value. A robust threshold beats an optimal one.
5. Verify across folds: tune per-fold, look at the spread of chosen thresholds.
   Wide spread = unstable = don't trust it.

Implemented in `../code/validation.py::tune_threshold`.

**Multilabel:** tune **one threshold per label** — label frequencies differ by
orders of magnitude and a global threshold is a compromise that serves no label
well.

**Ranking/top-k metrics:** tune **k**, or the score cutoff, the same way.

**Never tune the threshold on the data you fit on**, and never on test.

## Calibration

Needed when: probabilities are consumed downstream (expected-value decisions,
blending across model families, cost-sensitive thresholds), or when the metric
is a proper scoring rule (log loss, Brier).

**Not** needed when the metric only cares about ranking (AUC, nDCG, MRR) —
monotone calibration leaves those unchanged. Don't spend time on it then.

Methods:

- **Platt scaling** (sigmoid) — 2 parameters, good for small data, assumes a
  sigmoid-shaped distortion.
- **Isotonic regression** — nonparametric, more flexible, needs more data
  (rule of thumb: ≥1000 samples), can overfit below that.
- `CalibratedClassifierCV(estimator, method=..., cv=...)` in sklearn.

Fit the calibrator on **OOF** predictions, not on training predictions.

**Diagnose with a reliability diagram** (predicted probability bucket vs
observed frequency) before and after. If the "before" curve is already on the
diagonal, skip calibration.

Known tendencies: SVMs and boosted trees are typically **over-confident** at the
extremes; heavily-regularized models and bagged ensembles tend to be
**under-confident**; neural networks with modern training are often
over-confident. Averaging several models also shifts calibration — recalibrate
**after** blending, not before.

## Order of operations

```
train → OOF probabilities → blend (weights from OOF)
      → calibrate the blend (on OOF) → tune threshold (on calibrated OOF)
      → apply to test
```

Getting this order wrong (e.g. tuning the threshold before blending) means
re-tuning afterwards anyway. And whatever you tune on OOF, **apply the exact
same fitted transform to test** — save the threshold and the calibrator, don't
recompute them from test predictions.
