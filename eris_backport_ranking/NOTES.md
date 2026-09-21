# Backport review-priority ranking — what went wrong and what this solution does

## 1. Why the pre-submission checks failed

### Prompt Compliance

| # | Violation | Where |
|---|---|---|
| 1 | The artifact is never written to `working/submission.csv`. Both scripts take the output path from `sys.argv[2]` and, run with no arguments, print a usage line and `return 2` — producing **no submission file at all**. The challenge states: *"Write the final artifact to working/submission.csv when running a solution."* | `solution.py:325-328`, `solution_1.py:325-330` |
| 2 | Guidebook §5.3 (ranking challenges): *"they hand-engineer features out of the data, and then run a classic ranking algorithm on top of those features. That's not enough."* `solution_1.py` is exactly that — 60 hand-built features into `HistGradientBoostingRegressor` + `RandomForestRegressor`, regressing the raw label. There is no trained ranking model in it. | `solution_1.py:204-239` |
| 3 | Guidebook §1.1: hyperparameters must be **searched inside the submission script**, not hardcoded from an offline search. `solution_1.py` pins `max_iter=400, learning_rate=0.03, max_depth=5, …` and the blend `HGB_WEIGHT=0.6 / RF_WEIGHT=0.4`. | `solution_1.py:199-223` |
| 4 | Guidebook §3.8: the script must be fully independent end-to-end. A script that requires two positional arguments is not. | both |

### Deterministic Execution

Everything below changes **which model gets built** depending on how fast the host is:

| # | Violation | Where |
|---|---|---|
| 1 | `TIME_BUDGET_S = 3000` plus `time_left()` gates on the hyperparameter grid, the feature screen, the grouped-CV views and the extrapolation views. A slower machine searches less and therefore ships a *different model*. | `solution.py:281, 364, 394, 400` |
| 2 | `raise RuntimeError('no validation views could be built within the time budget')` — the script can fail outright purely because of machine speed. | `solution.py:409` |
| 3 | `torch.use_deterministic_algorithms(True, warn_only=True)` — `warn_only` explicitly *permits* nondeterministic kernels instead of rejecting them. | `solution.py:196` |
| 4 | `LGBMRanker` built without `deterministic=True`, `force_row_wise=True` or a pinned `n_jobs`, so histogram construction follows the host's core count. | `solution.py:248` |
| 5 | No `PYTHONHASHSEED`, no BLAS thread pinning, so reduction order varies with the machine. | both |
| 6 | `RandomForestRegressor(..., n_jobs=-1)` — worker count taken from the host. | `solution_1.py:222` |

## 2. Why the scores were bad — the part the checks don't tell you

This matters more than either check.

**Random ranking on 586 rows scores 0.342 ± 0.101** (3000 draws). So 0.2258 is *below chance*, and 0.54 is about the 95th percentile of chance.

The training rows are not i.i.d. with the hidden set, and the public validation signal is almost entirely fake:

| Measurement | Value |
|---|---|
| Random 5-fold CV NDCG@20 (`solution_1`'s model) | **0.974** |
| Its actual project-disjoint test score | **0.54** |
| 1-nearest-neighbour label agreement inside train | **0.81** (chance 0.38) |
| Exact duplicate `profile_text` rows in train | 248 of 2418 |
| Random-KFold OOF accuracy | 0.839 (majority baseline 0.507) |
| Random-KFold OOF Spearman vs target | 0.786 |
| Strongest single-feature Spearman | 0.283 (`test_file_count`) |
| **Adversarial AUC, train vs test** | **0.968** |

Read together: rows from one project repeat almost verbatim and share a label, so a model scores 0.97 by recognising project neighbourhoods rather than by learning anything about review workload — no combination of features whose strongest marginal correlation is 0.28 produces an OOF Spearman of 0.79. Meanwhile the test projects genuinely look different: `body_structure` goes from plain-dominant (0.47) to mixed-dominant (0.51), `change_scope` shifts toward large, `test_term` flips from mostly absent to mostly present, mean `additions` goes 458 → 2844 and mean `docs_file_count` 1.2 → 4.6.

`solution.py` then *selected* its feature screen, hyperparameters and blend weight against that fake signal — its KMeans "pseudo-groups" are still same-project — and tuned itself below chance. That is the whole story of 0.2258.

## 3. The validation design this solution uses instead

There is no project id anywhere in the public files, so a true project-disjoint split cannot
be reconstructed. The closest honest approximation is to hold out whole **regions** of the
profile space — cluster the rows and train on the other clusters only — and to purge from
each training fold any row still sitting in a held-out row's neighbourhood.

Measured on 7 held-out regions (per-fold NDCG@20, not pooled):

| Model | mean | worst half | per-fold spread |
|---|---|---|---|
| `solution_1`'s HGB+RF (scored 0.54 on the real test) | 0.741 | 0.600 | 0.50 – 1.00 |
| GBM depth 2 → expected gain | 0.724 | 0.580 | 0.42 – 0.96 |
| multinomial logit → expected gain | 0.572 | 0.425 | 0.25 – 0.93 |

The mean is still optimistic against the real 0.54, but the **hard folds bracket it**: the
hidden set behaves like one of the difficult regions, not like the average one. That is why
model selection here scores the **worst half of the folds** rather than the mean. Selecting
on the mean picks whichever model is best on the regions it already recognises, which is
the exact failure being designed around.

Note that low capacity is *not* the answer on its own — the logit is clearly the worst of
the three. What helps is a representation that survives the covariate shift, plus selection
that rewards the weak folds.

## 4. What the shipped solution does

**Model.** A trained neural ranker (`FieldRanker`). Every `key=value` token of the profile
becomes a field token: categorical fields get a learned embedding, numeric fields get a
learned per-field scale and shift. Those field vectors feed both a linear skip path and an
MLP trunk — the skip carries the broad per-field effects that survive a change of project,
the trunk adds the interactions. Two heads are trained and blended:

* **expected gain** — a 3-way softmax over the graded classes, scored as
  `E[2**y - 1] = 3·P(active) + 1·P(light)`. That is exactly the quantity NDCG's gain
  mapping rewards, so it is the metric-aligned pointwise rule.
* **ListNet** — listwise softmax cross-entropy between the score distribution and the gain
  distribution over a sampled list, optimising the ordering directly.

**Shift-robust encoding.** Numeric fields can be mapped through their own *training*
empirical CDF to a normal score instead of log1p-and-standardise. With the test projects
several times larger on mean `additions` and `docs_file_count`, a per-field CDF mapping is
invariant to that rescaling where a standardised log1p value is not. Which of the two is
used is one of the things the in-script search decides.

**Everything is learned in-script.** The field representation, each head's hyperparameters
(a fixed 5-point grid, every point always evaluated), and the head mixture all come from
`train_targets.csv` via the region-disjoint folds. Nothing is carried in from an offline
search.

**Compliance fixes.**

* Writes `working/submission.csv`, with no required CLI arguments; the data directory is
  found by a fixed ordered search.
* Thread counts and `PYTHONHASHSEED` pinned before numpy/torch load; one fixed CPU device
  with no availability probe and no fallback; `torch.use_deterministic_algorithms(True)`
  without `warn_only`; every model seeded.
* **No wall-clock branch anywhere.** The work plan is a fixed number of fits, so the same
  inputs give the same submission on any host. The plan is sized to finish well inside the
  runtime budget rather than being cut short by a timer.
* Only `train.csv`, `test.csv` and `train_targets.csv` are read. Test rows get one forward
  pass each and never touch fitting, feature statistics, thresholds or calibration.
