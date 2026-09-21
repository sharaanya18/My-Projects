# FINAL_APPROACH — Backport Review-Capacity Ranking (NDCG@20)

**Final pipeline:** parse `profile_text` into its 51-field table → add 22
scale-free ratio features → drop the 2 fields with the most extreme train→test
drift → LightGBM regression on the metric's own gain `2^rel − 1`, shallow
(`num_leaves=7`, `min_child_samples=40`), averaged over 10 seeds on the raw
score scale.

71 features · 2,418 training rows · fits in **1.9 s on 4 CPU cores** · no GPU,
no network, no model downloads.

---

## 1. Why this and not something bigger

| Rejected | Why, with evidence |
|---|---|
| Sentence embeddings / BGE / E5 / Qwen | `profile_text` is a `key=value` record, not language. Embedding it would encode `file_count=6` as words. Also structurally impossible here: huggingface.co is blocked. The first reason stands on its own. |
| TF-IDF over `profile_text` | Measured: 0.867 vs the parsed table's 0.918 (cluster-CV). Tokenising a serialised record is a lossy one-hot of the parse. |
| Cross-encoder / reranking | No query, no candidate list, no pair. Wrong problem shape. |
| **LambdaRank (LGBMRanker)** | The nominally "correct" LTR objective. **+0.019 on cluster-CV (16/18 wins) but −0.018 on the shift-aware holdout (7/18 wins).** Its edge lives only in the view contaminated by project memorisation. Rejected on robustness. |
| Transformer fine-tuning | 2,418 tabular rows. Nothing to fine-tune on. |
| Blending regressor + ranker | OOF Spearman between them **0.955**. Holdout delta −0.003 ± 0.028. No complementarity. |
| Seed averaging **on ranks** | **Actively harmful**: −0.014 / −0.008 / −0.116 / −0.068, 0 of 4 wins. |

## 2. The central problem: the training data memorises projects

The feature→target map is sharply non-monotonic in a way no mechanism explains:

```
title_digit_count :  2→1.35   3→1.25   4→0.22   5→1.35
```

A 4-digit title attracting a quarter of the review of a 3- or 5-digit one is
not a cause, it is an **identifier**. Confirmed by holding out a feature's own
value-groups:

| feature | random split | value-group holdout | reading |
|---|---|---|---|
| `title_digit_count` | +0.523 | **−0.015** | pure fingerprint |
| `body_code_block_count` | +0.124 | **−0.218** | fingerprint, sign flips |
| `body_digit_count` | +0.450 | +0.182 | partly real |
| `body_line_count` | +0.553 | +0.511 | real |
| `test_file_count` | +0.232 | +0.195 | real, and causally sensible |

Coarse 4-tuples of `(base_channel, body_structure, title_word_count,
body_line_count)` behave like project ids: 93 tuples with n≥5 cover **48.6%**
of train with within-tuple target std **0.339** vs global **0.823**; 31 are
100% one label.

**The hidden test set is project-disjoint, so none of this transfers.** Random
K-fold reports 0.973. That number is fiction and was never used to decide
anything.

## 3. Validation design

No project key is supplied, so I built one and made it strict.

**Sibling groups** = connected components over the union of three relations:
exact duplicate profile ∪ near-duplicate profile (char-4gram cosine ≥ 0.97) ∪
same template cluster. No two related rows can land on opposite sides of a
split. Built **without the target**; used only to split, never as a feature.

Every decision had to survive **two independent views**:

1. **Sibling-disjoint GroupKFold** — project memorisation is worthless.
2. **Test-like cluster holdout** — train on the least test-like whole clusters,
   score the most test-like ones. Shift-aware *and* leakage-resistant.

and was judged by **paired resampling** (same 586-row subsets for both models,
800 draws, 95% CI on the difference). Unpaired mean ± std has no power here:
NDCG@20's sampling noise is ±0.10 and it *cancels* when paired.

Decisions were re-run across 3 cluster seeds × 3 granularities (and 4 × 3 where
it mattered) — a single split is never enough.

## 4. What the evidence actually said

| Decision | Cluster-CV | Shift holdout | Verdict |
|---|---|---|---|
| Drop mention features | +0.005 | +0.006 | **adopt** — only change to win all three views |
| `2^r−1` vs raw label | +0.007 | +0.012 | **adopt** |
| `P(r=2)` classifier | −0.003 | −0.022 | reject |
| `E[gain]` classifier | −0.005 | −0.058 | reject |
| LambdaRank | +0.019 (16/18) | −0.018 (7/18) | **reject — not robust** |
| Blend with ranker | +0.017 | −0.003 | reject (ρ=0.955) |
| Drop digit features | −0.038 | −0.024 | **reject — contradicted my own hypothesis** |
| Shallow vs default | −0.001 | +0.011 (6/9) | adopt, on robustness not CV |
| Regularise harder still | −0.048 (stumps) | −0.002 | reject |
| Seed-avg on ranks | — | 0/4 wins | reject |
| Seed-avg on raw scores | — | 3/4 wins | **adopt** |

Two results worth calling out because they went against expectation:

- **The KB's "regularize harder under shift" advice did not replicate.** Every
  config from stumps to `num_leaves=31` landed within its own std of the
  default. I tested it rather than assuming it.
- **Dropping the digit features hurt**, even though `title_digit_count` alone
  has zero transferable signal. The single-feature probe and the full-model
  ablation disagree, and the ablation is what governs the deployed model. The
  digit *rates* (`digits/length`) carry real signal the raw count does not.

## 5. Expected score — measured, not claimed

NDCG@20 on 586 rows is brutally noisy. Measured on 4,000 simulations:
**random predictions score 0.326 ± 0.101.** One extra `2` at rank 1 is worth
0.142. The hidden score is a single draw from a distribution ~0.05–0.10 wide.

Honest estimates, pooled across held-out projects — the structure the grader
actually uses (586 rows drawn from *several* unseen projects):

| sibling components | NDCG@20 |
|---|---|
| 160 | 0.916 ± 0.049 |
| 100 | 0.934 ± 0.040 |
| 50 | 0.857 ± 0.062 |
| 25 | 0.924 ± 0.049 |
| 12 (harshest) | **0.842 ± 0.063** |

At the harshest setting, the single-draw distribution is
p1 = 0.679, p5 = 0.739, p50 = 0.851, and **P(draw < 0.60) = 0.03%**.

**A different, harder question:** scoring one *single* held-out project in
isolation gives mean 0.685, median 0.619, and **9 of 18 components below
0.60** (range 0.245–1.000). That is not the graded task — the test set pools
586 rows from multiple unseen projects, which averages this variance out — but
it is the honest bound on how much per-project variation exists, and it is why
I will not promise a number.

**So: 0.60 sits far below every pooled estimate, and the measured probability
of falling under it is ~0.03% under my harshest honest design.** I expect to
clear it comfortably. I cannot guarantee it, because my pseudo-project clusters
are a proxy of unknown fidelity for the real project split, and because a
single NDCG@20 draw is high-variance by construction. Claiming a guaranteed
leaderboard number would be exactly the unmeasured claim the brief forbids.

## 6. Overfitting and sibling checks

| Check | Result |
|---|---|
| In-sample vs OOF Spearman, final model | 0.858 → 0.692, gap **0.166** |
| Same, deep model (`leaves=31`, 1000 trees) | 0.916 → 0.692, gap 0.224 — same OOF, more overfit. Rejected. |
| Exact profile overlap train↔test | **0** |
| Id overlap train↔test | **0** |
| Test rows with a ≥0.97-similar train profile | **0 of 586** |
| Max train↔test profile similarity | 0.845 (median 0.408) |
| Single-feature dependence (drop top-6) | worst −0.033; reliance is distributed |
| Row-order artifact | Spearman(index, target) = −0.007 |
| Opaque id artifact | Spearman(int(id,16), target) = −0.007 |

The test set contains **no siblings of any training row**, so the deployed
pipeline cannot exhibit sibling behaviour at inference time. The sibling risk
was entirely in *validation*, and that is what the sibling-disjoint splits
removed.

## 7. Guideline compliance

| Rule | How it is met |
|---|---|
| Train only on supplied public files | Only `train.csv`, `train_targets.csv`, `test.csv` are read. |
| No external retrieval | No network at train or inference. Verified: blocked in this container anyway. |
| No raw URL / PR number / author / source cohort | Not present in the public view; nothing of the kind is constructed. |
| No inference from row order, opaque id, or project hash | Both checked (ρ ≈ −0.007) as *audit items only*; neither is a feature. Grep the feature list: id never enters `build()`. |
| No synthetic examples or labels | None generated. |
| Leakage-resistant split respecting project-disjointness | Sibling-disjoint GroupKFold + test-like cluster holdout. Clusters are a **validation device**, never a feature. |
| Graded-relevance loss, calibration, ensembling allowed | Used `2^r−1` regression; tested and rejected ensembling on evidence. |
| Target is not a quality judgement | Framed throughout as review-capacity load. |
| Write to `working/submission.csv` | Done. |

## 8. Submission verification

`verify_submission.py` re-implements the grader's checks independently (it does
not import `final_solution.py`, so a shared bug cannot hide). **16/16 pass:**
columns exactly `id,prediction`; 586 rows; id set identical to both `test.csv`
and `sample_submission.csv`; test id order preserved; 12-char string ids; numeric
dtype; all finite; no NaN; 542 distinct values; scores not labels; sorting by id
is lossless; the official metric evaluates in [0,1].

Reproducibility: two consecutive runs produce byte-identical predictions
(max abs diff 0.0). `inference.py` reproduces `final_solution.py` exactly.

## 9. Known risks

| Risk | Assessment |
|---|---|
| Pseudo-project clusters imperfectly proxy real projects | The main threat to my estimates. Mitigated by testing 5 granularities, 3 seeds, and two independent split designs; all pooled results land 0.84–0.93. |
| NDCG@20 single-draw variance | Irreducible. p5 = 0.739 at the harshest setting. |
| Per-project difficulty varies hugely | 0.245–1.000 leave-one-component-out. Pooling across projects averages it out; a test set dominated by one hard project would score lower. |
| ≥1.6% contradictory duplicate labels | Hard ceiling below 1.0. Not chased. |
| Residual fingerprint reliance | Fingerprint features hold 20.9% of importance. Removing them measurably hurt, so they stay — but the test top-20 is driven by causal features (`test_term` 85% vs 52% baseline, `changed_lines` median 368 vs 80). |
| Shift beyond what train shows | Adversarial AUC 0.975. Mitigated by dropping the worst-drifting features and by preferring the shallow model. |

## 10. Files

```
SHIPD_TASK_ANALYSIS.md      Phase 6 audit (written before any modelling)
FINAL_APPROACH.md           this file
final_solution.py           fit + predict + submission  (python final_solution.py)
inference.py                fit-once / predict-later variant
verify_submission.py        independent grader-contract checks
requirements.txt            pinned, tested versions
src/{data,features,models,validate,metric}.py
exp/e0*.py                  the experiments, re-runnable
working/submission.csv      586 rows, verified
working/EXPERIMENT_LOG.md   experiment record
working/*.csv               per-experiment result tables
```
