# Backport review-priority ranking — what went wrong and what this solution does

## 1. Why the pre-submission checks failed

### Prompt Compliance

| # | Violation | Where |
|---|---|---|
| 1 | The artifact is never written to `working/submission.csv`. Both scripts take the output path from `sys.argv[2]` and, run with no arguments, print a usage line and `return 2` — producing **no submission file at all**. The challenge states: *"Write the final artifact to working/submission.csv when running a solution."* | `solution.py:324-329`, `solution_1.py:324-330` |
| 2 | Guidebook §5.3 (ranking challenges): *"they hand-engineer features out of the data, and then run a classic ranking algorithm on top of those features. That's not enough."* `solution_1.py` is exactly that — 60 hand-built features into `HistGradientBoostingRegressor` + `RandomForestRegressor`, regressing the raw label. There is no trained ranking model in it. | `solution_1.py:204-239` |
| 3 | Guidebook §1.1: hyperparameters must be **searched inside the submission script**, not hardcoded from an offline search. `solution_1.py` pins `max_iter=400, learning_rate=0.03, max_depth=5, …` and the blend `HGB_WEIGHT=0.6 / RF_WEIGHT=0.4`. | `solution_1.py:199-223` |
| 4 | Guidebook §3.8: the script must be fully independent end-to-end. A script that requires two positional arguments is not. | both |

### Deterministic Execution

Everything below changes **which model gets built** depending on how fast the host is:

| # | Violation | Where |
|---|---|---|
| 1 | `TIME_BUDGET_S = 3000` plus `time_left()` gates on the hyperparameter grid, the feature screen, the grouped-CV views and the extrapolation views. A slower machine searches less and therefore ships a *different model*. | `solution.py:8, 275-281, 364, 394, 401` |
| 2 | `raise RuntimeError('no validation views could be built within the time budget')` — the script can fail outright purely because of machine speed. | `solution.py:409` |
| 3 | `torch.use_deterministic_algorithms(True, warn_only=True)` — `warn_only` explicitly *permits* nondeterministic kernels instead of rejecting them. | `solution.py:196` |
| 4 | `LGBMRanker` built without `deterministic=True`, `force_row_wise=True` or a pinned `n_jobs`, so histogram construction follows the host's core count. | `solution.py:248` |
| 5 | No `PYTHONHASHSEED`, no BLAS thread pinning, so reduction order varies with the machine. | both |
| 6 | `RandomForestRegressor(..., n_jobs=-1)` — worker count taken from the host. | `solution_1.py:222` |

## 2. Why the scores were bad — the part the checks don't tell you

This matters more than either check.

**Random ranking on 586 rows scores 0.326 ± 0.101** (20 000 draws). So 0.2258 sits in the bottom 17% of what shuffling would give you, and 0.54 is the 97.7th percentile of chance — genuinely better than nothing, but not by much.

The training rows are not i.i.d. with the hidden set, and the public validation signal is almost entirely fake:

| Measurement | Value |
|---|---|
| Random ranking, 586 rows (20 000 draws) | **0.326 ± 0.101** |
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
the three. What helps is metric-aligned scoring, heavy seed ensembling, and selection that
rewards the weak folds.

Individual neural configurations land across 0.516–0.610 worst-half on these folds, i.e.
straddling the tree baselines rather than beating them; the shipped two-head blend reaches
0.633 (§6). Since no single family dominates and guidebook §5.3 is explicit that a ranking
challenge wants a genuinely trained ranking model rather than hand-engineered features fed
to an off-the-shelf ranker, the shipped solution is neural only — no GBM in the blend. To be
clear about what was and was not measured: the neural and tree families were each scored on
these folds, but a neural+GBM *blend* was never run, so that decision rests on §5.3 and on
neither family dominating, not on a blend experiment.

## 4. What the shipped solution does

**Model — a trained ranker as the major portion (§5.3).** `FieldRanker` is a neural network
over the profile's fields: each categorical field gets a learned embedding, each numeric
field a learned per-field scale and shift, and the field vectors feed a linear skip path
plus an MLP trunk. The skip carries the broad per-field effects that survive a change of
project; the trunk supplies the interactions. It is trained with **ListNet** — listwise
softmax cross-entropy between the score distribution and the gain distribution over a
sampled list — so the model optimises the ordering itself rather than a pointwise target.
The same network also supports an **expected-gain** head, `3·P(active) + 1·P(light)`, which
is exactly what NDCG's gain mapping rewards.

**Co-model.** A LightGBM **LambdaRank** model over the same encoded fields, weight learned
in-script and capped at `MAX_TREE_WEIGHT = 0.5` so the neural ranker stays the major portion
per §5.3. The two families disagree substantially on the hidden rows (Spearman ≈ 0.40
between their test scores) and the public files cannot settle which generalises better, so
the blend hedges rather than betting.

**Encoding is fixed, not searched.** Numeric fields go through their *training* empirical CDF
to a normal score. An earlier version let the search choose between this and standardised
log1p; it chose log1p — the held-out regions contain no magnitude shift, so the folds cannot
see why the bound matters — and scored 0.27. See §6 for what this does and does not explain.

**Everything else is learned in-script (§1.1).** Both hyperparameter grids and the mixture
weight come from `train_targets.csv` via the region-disjoint folds. Nothing is carried in
from an offline search.

**Compliance.**

* Writes `working/submission.csv`, no CLI arguments; data directory found by a fixed ordered
  search.
* §3.5 safeguard present: a valid submission after 9s, a 3000s deadline that falls through to
  a fallback fixed by grid position, and a plan sized at ~7 minutes so the guard cannot fire.
* §3.6 honoured: CUDA when present, with cuDNN determinism and a fixed cuBLAS workspace;
  identical hyperparameters, seeds and fit counts on either device.
* Threads and `PYTHONHASHSEED` pinned before the numeric libraries load; every model seeded;
  `torch.use_deterministic_algorithms(True)` without `warn_only`; LightGBM with
  `deterministic=True`, `force_row_wise=True`, `n_jobs=1`.
* Only the three supplied files are read. Test rows get one forward pass each and never touch
  fitting, feature statistics, thresholds or calibration.

## 5. How to actually clear 0.5 on the hidden set

**Calibrate your expectations first.** Random ranking scores 0.326 ± 0.101 here (20 000
draws of 586 rows); a shuffle clears 0.50 on 5.3% of draws and 0.54 on 2.3%. Only the top 20
of 586 rows count, so a single submission carries a lot of variance — 0.5 is about 1.7
standard deviations above chance on one draw. A model with real signal clears it most of the
time, not every time. Guidebook §1.2 applies: treat the public number as a soft signal, not
proof.

**The largest single lever is not modelling, it is not wrecking the model at selection
time.** The family `solution_1` used scores roughly 0.54–0.60 on the honest folds. `solution.py`
started from the same data and tuned itself down to 0.226 — below chance — purely by
selecting a feature screen, hyperparameters and a blend weight against a validation signal
that was measuring project memorisation. Deleting the bad selection recovers most of the gap
before any new modelling happens.

In rough order of payoff:

1. **Never select on random K-fold.** It reports 0.97 here and ranks models wrongly. Hold
   out whole regions and score the weak folds.
2. **Rank by expected gain**, `3·P(active) + 1·P(light)`, not by a regression on the raw
   label. It is the exact quantity NDCG's `2**rel - 1` mapping rewards, and it is free.
3. **Ensemble seeds and average ranks.** The top 20 rows of 586 are unstable; a single fit
   moves them around far more than the underlying model quality does.
4. **Blend a pointwise and a listwise head.** Worth about +0.03 worst-half here
   (0.552 / 0.553 separately → 0.583 blended).
5. **Keep the search small.** Five configurations in one regularised family, an Occam
   tie-break toward the simpler one, and a blend weight pulled toward balanced. Every extra
   selected knob is another chance to fit seven noisy folds.
6. **Do not reach for capacity, and do not run from it either.** A multinomial logit is the
   worst model tested (0.425 worst-half). Depth-2 GBMs, `solution_1`'s deeper trees and the
   neural ranker all land within noise of each other around 0.58–0.60. Representation and
   selection discipline decide this problem; architecture barely does.

**Things that look tempting and are not allowed.** Quantile-normalising features using the
test rows, calibrating scores to the test distribution, or pseudo-labelling would all lift
the score and are all prohibited by guidebook §4.2 — it is about realism, not labels. The
rank transform here is fitted on training rows only, and test rows get one forward pass each.

## 6. Second round: what the 0.27 submission taught us

The first version of this solution scored **0.27** on the hidden set and still failed
Prompt Compliance. Three things came out of that.

**I broke a stated requirement on purpose, and it was the wrong call.** Guidebook §3.5:
*"One thing we always ask solvers to do: build a safeguard into your code that automatically
stops training and moves to inference and submission once you hit somewhere around 50 to 55
minutes (3000 to 3300 seconds)."* I removed the safeguard to satisfy the Deterministic
Execution check and documented the removal — trading a stated requirement for a guess about
another checker. Both are satisfiable at once, and the script now does: the plan is sized to
finish in ~7 minutes, a valid submission is written from the first fitted model after 9s, and
the 3000s guard falls through to a fallback fixed *by position in the grid* rather than to
whatever was best so far. It cannot fire on a real host, so it cannot change the output.

**§3.6 says the environment is an A10G.** The first version hard-pinned
`torch.device("cpu")` with no GPU path at all, and "hardware" is named in the compliance
message. The device is now chosen once at startup, using CUDA when present with cuDNN
determinism and a fixed cuBLAS workspace, with identical hyperparameters either way.

**§5.3 is the real tension, and it cuts against a trees-only rebuild.** Verbatim: *"Feature
engineering plus an off-the-shelf ranking algorithm isn't the same as a model that's actually
learned to rank, and it won't hold up here... a major portion of your overall solution still
has to be a genuinely trained or fine-tuned model."* That is a precise description of
`solution_1` — which scored 0.54 **and** failed Prompt Compliance. So the tree family gets
the score and fails the check, while the neural family passes the check and scored below
chance. The shipped answer is a trained listwise neural ranker as the major portion with a
LambdaRank co-model capped at `MAX_TREE_WEIGHT = 0.5`, hedging across both.

### What is *not* established

The extrapolation story does not survive contact with the data. The 0.27 version put 7 of
its top 20 beyond the training maximum on some field; the tree family, which scored 0.54,
puts 10 of 20 there. Being out of range does not separate a good ranking from a bad one
here. With three hidden-set observations against a metric whose own noise is ±0.10, the
cause of 0.27 is **not identifiable from the public files**. The CDF encoding is kept
because unbounded extrapolation on a deliberately project-disjoint split is a defect either
way — after it, no encoded test value falls outside the encoded training range — not
because it is a proven fix.

### A bug worth recording

The first attempt at the LambdaRank co-model scored 0.209 worst-half, against 0.555 for the
network, and the blend gave it weight 0.00. The cause was mine: `lambdarank_truncation_level`
set to the metric's own cutoff of 20, with the whole training set as a single group, so
gradients flowed for only the top 20 rows of a ~2000-row list. Putting the truncation in the
searched grid fixed it — 0.540 at truncation 2000, and the search correctly rejects 20.
The metric's evaluation cutoff is not the right training truncation.

## 7. What the shipped solution scores

One full run (`python3 solution.py`, 445 s, CPU):

```
  listnet head       : worst-half 0.5550   (mean 0.6972)
  lambdarank co-model: worst-half 0.5403   (mean 0.7141)
  blend              : w(neural)=0.80 / w(tree)=0.20
validation NDCG@20   : 0.5900   (worst half of the region-disjoint folds)
```

Two runs produced byte-identical submissions (md5 `fc5e69f3327e7e397e12e73931703883`),
586 rows in test order, all finite, 538 distinct scores.

Treat 0.5900 as a proxy that has already been wrong once: it said 0.6334 for the version
that scored 0.27. It is built from same-project rows and cannot see the covariate shift
(adversarial AUC 0.968). The reason to expect better than 0.27 is the §5.3-compliant hedge
across two families plus the fixed co-model, not the validation number.

## 8. Running it

```
project/
  solution.py
  data/            # train.csv, test.csv, train_targets.csv (also found at ./, ./input,
                   # ./public, ../data, /kaggle/input/*, or via ERIS_DATA_DIR)
  working/         # created if missing; submission.csv is written here
```

```
python3 solution.py
```

No arguments. `ERIS_DATA_DIR` and `ERIS_OUTPUT` exist only to relocate the two paths for
local testing; both default to the layout above.

Requires numpy, pandas, scipy, scikit-learn, lightgbm and torch — all present in the Kaggle
image (guidebook §3.1).
