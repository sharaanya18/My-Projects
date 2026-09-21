# Experiment Summary

Every comparison is **paired** (same resampled 586-row subsets for both models)
and, where a decision depended on it, repeated across cluster seeds and
granularities. Deltas are NDCG@20; `w` is win-rate across repeats.

## Phase 7 — baselines (sibling/cluster-disjoint unless stated)

| id | model | NDCG@20 | note |
|---|---|---|---|
| R00 | random | 0.326 ± 0.101 | absolute floor, 4000 sims |
| R01 | constant | 0.319 ± 0.101 | all ties → id order |
| B1 | 1 feature: `test_file_count` | 0.511 | best single causal feature |
| B2 | 1 feature: `title_digit_count` | 0.426 | high Spearman, poor top-20 |
| T1 | TF-IDF(1g) + LogReg on raw text | 0.820 | |
| T2 | TF-IDF(1-2g) + LogReg | 0.867 | text view is strictly worse than parsing |
| E03 | LGBM, **random KFold** | 0.987 | **REJECTED as evidence — project leakage** |
| E04 | LGBM, group-disjoint | 0.903 | first honest reference |

## Phase 9 — feature ablations (paired vs ALL)

| variant | cluster-CV | shift holdout | verdict |
|---|---|---|---|
| drop mention (2) | +0.005 | +0.006 | **adopt** — wins every view |
| drop fingerprint (8) | −0.004 | −0.052 | reject |
| drop shifted (9) | +0.043 | −0.046 | reject — views disagree |
| drop both | +0.020 | −0.016 | reject |
| causal only (24) | −0.101 | −0.052 | reject |
| derived only (39) | +0.014 | −0.047 | reject |
| drop digits (4) | −0.038 (0/9) | −0.024 (1/9) | **reject — hypothesis disproved** |
| drop raw digits (2) | −0.020 (1/9) | −0.030 (0/9) | reject |

## Phase 8 — surrogate objective (paired vs raw-label regression)

| objective | cluster-CV | shift holdout | verdict |
|---|---|---|---|
| **regression on `2^r−1`** | **+0.007** | **+0.012** | **adopt** |
| classifier `P(r=2)` | −0.003 | −0.022 | reject |
| classifier `E[gain]` | −0.005 | −0.058 | reject |
| LambdaRank, default trunc=30 | −0.264 | −0.480 | broken (see below) |
| LambdaRank, trunc=4000 | +0.023 | −0.060 | see robustness |

**LambdaRank diagnosis.** `lambdarank_truncation_level` defaults to **30**. With
one group of ~1,900 rows only 30 rows ever receive gradient. Raising it to 4000
moved the score 0.668 → 0.940. The metric's cutoff (20) and the training
truncation are **different knobs** — a genuine trap.

**LambdaRank robustness, 3 seeds × 3 granularities × 2 fracs (18 runs):**
cluster-CV delta **+0.019 ± 0.021 (16/18 wins)** but shift holdout
**−0.018 ± 0.054 (7/18 wins)**. Its advantage exists only in the more
contaminated view → **rejected**.

## Phase 10 — regularisation (paired vs default, 9 runs each)

| config | cluster-CV | shift holdout |
|---|---|---|
| default `leaves=15, mcs=20` | — | — |
| **`leaves=7, mcs=40`** | −0.001 (4/9) | **+0.011 (6/9)** |
| `leaves=4, mcs=60` | −0.010 (2/9) | +0.007 (5/9) |
| stumps `depth=2` | −0.048 (2/9) | −0.002 (5/9) |
| `leaves=7 + L2=10` | +0.005 (4/9) | +0.004 (7/9) |
| `leaves=31, mcs=10` | +0.007 (6/9) | +0.004 (4/9) |

All within noise. Shallow adopted for robustness, and confirmed by the
overfitting gap (0.166 vs 0.224 Spearman) at equal OOF.

## Phase 12 — combination

| combination | result | verdict |
|---|---|---|
| regressor + LambdaRank, rank-blend | ρ = **0.955**; CV +0.017, holdout −0.003 | reject — not complementary |
| seed-avg ×10 on **ranks** | −0.014 / −0.008 / −0.116 / −0.068 (0/4) | **reject — harmful** |
| seed-avg ×10 on **raw scores** | +0.007 / −0.018 / +0.007 / +0.004 (3/4) | **adopt** |

Rank-averaging flattens each seed's score into a uniform grade and discards how
confident it was. For a top-20 metric that confidence is the whole signal.

## Phase 11 — error analysis

- OOF top-20 composition: **87% label-2**, 7.7% label-1, 5.2% label-0
  (base rates 24 / 25 / 51%).
- Only **130 distinct rows** ever reach a top-20 across 500 draws → the model
  concentrates on a small confident set.
- Prediction deciles are cleanly monotone in mean target: 0.087 → 1.750.
- Costly promotions (label-0 in top-20) are dominated by `body_structure=code`
  (57% vs 9% overall) — code blocks in the body over-promote.
- On the real test set the top-20 is driven by causal features:
  `test_term` present 85% (vs 52%), `changed_lines` median 368 (vs 80),
  `change_scope=large` 55% (vs 37%). Not obviously fingerprint-driven.
- Test predictions are compressed (std 0.577 vs train 1.053); only 1 of 586
  exceeds train's 90th percentile — the model is appropriately less confident
  off-distribution.

## Phase 10 — stress test (final model)

Pooled across held-out projects, which is what the grader does:

| sibling components | NDCG@20 |
|---|---|
| 160 | 0.916 ± 0.049 |
| 100 | 0.934 ± 0.040 |
| 50 | 0.857 ± 0.062 |
| 25 | 0.924 ± 0.049 |
| 12 (harshest) | 0.842 ± 0.063 |

Single-draw risk at the harshest setting: p1 0.679, p5 0.739, p50 0.851,
**P(< 0.60) = 0.03%**.

Leave-one-component-out (a single project alone — *not* the graded task):
mean 0.685, median 0.619, min 0.245, max 1.000, 9/18 below 0.60.
