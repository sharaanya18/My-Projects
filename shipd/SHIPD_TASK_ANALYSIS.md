# SHIPD_TASK_ANALYSIS — Backport Review-Capacity Ranking

Phase 6 audit. Written **before** any modeling. Every number here was measured
in this session; the commands are in `src/` and the audit log below.

Date: 2026-09-21 · Data: `raw/` (public.zip) · Metric: NDCG@20

---

## 1. Objective

- **Task:** rank 586 backport change-requests by how much human review capacity
  they will attract, so a triage desk that can inspect only 20 per cycle sees
  the heaviest ones first.
- **Prediction unit:** one change request (one `id`) → one continuous score.
- **Target:** graded relevance in {0, 1, 2} = {quiet, light, active} =
  {0, 1, ≥2} non-empty human review messages.
- **Metric:** NDCG@20 over the whole 586-row test set, gain `2^rel - 1`,
  discount `1/log2(rank+1)`, stable mergesort on ties.
- **Only relative order matters.** Only the **top 20 of 586 rows** affect the
  score at all — 566 rows are invisible to the grader except through which 20
  they displace.

## 2. Data structure

| | train | test |
|---|---|---|
| rows | 2,418 | 586 |
| columns | `id`, `profile_text` | same |
| labels | separate `train_targets.csv` (2,418 rows) | hidden |
| id overlap | **0** with test | — |
| duplicate ids | 0 | 0 |
| id format | 12 hex chars, sorted ascending in both files | same |

`profile_text` is **not prose** — it is a space-separated `key=value` record
with a **fixed schema of 51 keys present in every row of both files**, no
missing values, no keys unique to either split. Effectively a tabular dataset
in a text wrapper.

- **17 categorical** keys: `*_band`, `*_term`, `body_structure`,
  `base_channel`, `change_scope`.
- **34 numeric** keys: title/body counts, file counts, `additions`,
  `deletions`, `changed_lines`, `max_file_*`.

Many categoricals are *banded copies of a numeric column* (`title_length_band`
↔ `title_length`). Redundant, but the band boundaries are themselves
information about how the organizers binned the data.

**No project column.** The statement confirms a project hash exists upstream
and is deliberately withheld, so **no legitimate group key is supplied**.

## 3. Label structure

| label | meaning | count | share | gain `2^r-1` |
|---|---|---|---|---|
| 0 | quiet | 1,226 | 50.7% | 0 |
| 1 | light | 609 | 25.2% | 1 |
| 2 | active | 583 | 24.1% | 3 |

An "active" row is worth **3×** a "light" row. Getting 2s into the top 20 is
almost the entire game; a 1 is worth a third of a 2 at the same rank.

## 4. The metric is extremely noisy — measured, not assumed

4,000 simulations of **random** predictions on 586-row samples:

```
random NDCG@20:  mean 0.3260   std 0.1013   p5 0.1716   p95 0.5035   max 0.6882
```

Reference points, on the ideal DCG of 21.12 (twenty 2s):

- one extra `2` placed at rank 1 is worth **0.142** NDCG
- one extra `2` at rank 20 is worth **0.032**
- a perfect oracle score corrupted by `N(0, 1)` noise still scores 0.933

**Consequences that shape the whole approach:**

1. A single train/valid split's NDCG@20 is close to meaningless. Every
   comparison below uses **hundreds of resampled evaluations** and reports a
   standard deviation.
2. The hidden leaderboard score is itself a single draw from a
   ~0.05–0.10-wide distribution. Large LB/CV gaps are expected and are **not**
   evidence of a broken model.
3. The objective is **precision at the very top**, not global ranking quality.
   A model with better overall Spearman can lose if its top 20 is worse.

## 5. Distribution shift — severe, and the dominant risk

**Adversarial validation (LightGBM, 5-fold): AUC = 0.9748.** Train and test are
almost perfectly separable. Per `nlp_knowledge/validation/leakage_detection.md`
this is the "SEVERE" band.

27 of 34 numeric columns differ at p < 0.01 (KS). The largest moves:

| column | KS | train mean | test mean |
|---|---|---|---|
| `body_mention_count` | 0.443 | 2.39 | **0.66** |
| `body_heading_count` | 0.259 | 1.80 | 2.33 |
| `body_line_count` | 0.245 | 20.2 | 32.7 |
| `title_colon_count` | 0.228 | 0.93 | 0.55 |
| `other_file_count` | 0.206 | 1.10 | 2.09 |
| `body_checklist_item_count` | 0.180 | 0.69 | **3.98** |
| `file_count` | 0.180 | 11.3 | 17.0 |

Categorical shifts (total variation distance):

| column | TVD | note |
|---|---|---|
| `body_mention_band` | **0.443** | zero-mentions 42% → **87%** |
| `body_structure` | 0.323 | `headed` 23%→7%, `mixed` 21%→51% |
| `bug_term` | 0.223 | present 53%→75% |
| `change_scope` | 0.147 | shifts toward `large` |
| `title_length_band` | 0.082 | `band_4` **appears only in test** |

**`body_mention_count` is the single most dangerous feature.** It is a strong
train signal (`body_mention_band=band_1` → mean target 1.06 vs 0.36/0.65
elsewhere) and it is **nearly constant in test** (87% zero). Any model leaning
on it is optimizing a feature that barely varies where it will be scored.

## 6. Leakage — found, quantified, and it invalidates naive CV

### 6.1 Duplicate profiles

- 248 exactly duplicated `profile_text` rows in train; 39 in test.
- 178 duplicate groups covering **426 rows (17.6%)**; largest group 9 rows.
- **0** train profiles reappear in test — no cross-split duplicate leakage.
- 14 groups (38 rows) carry **conflicting labels** on identical features
  → **irreducible label ambiguity ≥ 1.6%**. Identical profiles genuinely
  receive different review outcomes; part of this target is not predictable
  from these features at all.

### 6.2 Project-template fingerprinting — the real problem

The feature→target mapping is sharply **non-monotonic** in a way no causal
story explains:

```
title_digit_count :  0→0.27  1→0.30  2→1.35  3→1.25  4→0.22  5→1.35  6→0.35 …
```

A title with 4 digits attracting a quarter of the review of one with 3 or 5
digits is not a mechanism — it is an **identifier**. A project whose bot emits
version strings with a fixed digit count is being memorized.

Confirmation: single-feature LightGBM reaches Spearman 0.52–0.55 on
`title_digit_count` / `body_line_count`, where their raw rank correlation with
the target is only 0.21 and 0.10. The model is fitting value-clusters, not a trend.

Coarse 4-tuples of `(base_channel, body_structure, title_word_count,
body_line_count)` behave like project ids:

- 93 tuples with n≥5 cover **48.6%** of train
- mean within-tuple target std **0.339** vs global **0.823**
- 31 tuples are **100% one label**

**The hidden test set is project-disjoint, so none of this transfers.**

### 6.3 What group-disjoint validation costs

LightGBM, identical features and hyperparameters, only the split changes:

| validation | Spearman | NDCG@20 |
|---|---|---|
| random StratifiedKFold | 0.785 | **0.973 ± 0.026** |
| GroupKFold on exact-duplicate profile | 0.775 | 0.964 ± 0.030 |
| GroupKFold on exact 4-tuple (982 groups) | 0.762 | 0.924 ± 0.042 |
| GroupKFold on 100 template clusters | 0.735 | 0.889 ± 0.062 |
| GroupKFold on 25 template clusters | **0.516** | 0.871 ± 0.058 |

Monotone collapse as groups coarsen. **The 0.973 from random K-fold is
fiction** and must never be used to choose anything.

### 6.4 Clean checks

- `spearman(row_index, target) = -0.007` — no row-order artifact.
- `spearman(int(id,16), target) = -0.007` — the opaque id carries nothing.
  (Checked as an audit item only; both are forbidden as features and are not used.)

## 7. Hidden variables

The features are a lossy projection of the real driver. Review volume is
plausibly determined by project review culture, reviewer availability,
and the change's actual semantic risk — **none of which is observable here**.
What the profile does carry is a *proxy*: template structure, size, and topic flags.

This is why the ≥1.6% contradictory-duplicate rate matters: the target is
partly a function of variables that were deliberately removed. There is a
ceiling well below 1.0, and the useful question is how much of the *transferable*
signal (size, structure, topic) can be extracted without the *non-transferable*
signal (project identity).

## 8. Signal strength

Strongest individual rank correlations with the target (train):

| feature | Spearman |
|---|---|
| `test_file_count` | **+0.283** |
| `docs_file_count` | **−0.274** |
| `body_code_block_count` | +0.222 |
| `body_digit_count` | +0.217 |
| `title_digit_count` | +0.211 |
| `body_link_count` | −0.155 |

Categorical effects that have a plausible **causal** reading and should
therefore transfer across projects:

- `test_term` present → 0.94 vs absent 0.60 (touching tests ⇒ more review)
- `docs_term` present → 0.53 vs absent 0.82 (docs-only ⇒ less review)
- `body_structure=code` → 1.39 vs `plain` 0.63 (code in the body ⇒ discussion)
- `base_channel=other_branch` → 0.83 vs `release_or_stable` 0.68
- `security_term` present → 0.81 vs 0.73

No single feature is strong. This is a **weak-signal, high-noise** problem.

## 9. Problem family (Phase 5 detector)

Ran `nlp_knowledge/code/task_detector.py` reasoning manually against the
taxonomy:

- **Primary: learning-to-rank with graded relevance, single global group.**
  The metric is NDCG@20 over one list. There is no query column — the *entire
  test set is one ranking group*.
- **Secondary: ordinal regression / graded classification.** Labels are ordered
  with an explicit non-linear gain, so pointwise regression on `2^r - 1`, or on
  `P(r=2)`, are natural surrogates.
- **Not** retrieval, not semantic similarity, not sentence-pair: there is no
  query, no corpus, no pair.
- **Not a text task in the usual sense.** `profile_text` is a serialized
  fixed-schema record. TF-IDF over it is only a clumsy one-hot encoding, and
  sentence embeddings would be actively wrong — the tokens are counts, not
  language. This is a **tabular ranking problem**.

Playbooks: `nlp_knowledge/playbooks/ranking.md` (with the single-group caveat)
and `nlp_knowledge/validation/distribution_shift.md`.

## 10. Validation design

No project key is supplied, so I construct a **leakage-resistant proxy** and
require every claim to survive it. Three views, all reported for every experiment:

1. **`GroupKFold` on template clusters** (primary). Groups are KMeans clusters
   over *template-identifying* features only — structure, line/heading/checklist
   counts, title shape. Built **without the target**. This is a validation
   device, never a feature.
2. **Test-like holdout** (shift view). Rank train rows by adversarial
   `p(test)`; validate on the most test-like rows, train on the rest. Simulates
   the 0.975-AUC shift.
3. **Repeated 586-row resampling** of whatever OOF vector results, ≥300 draws,
   reporting mean ± std — because a single NDCG@20 is worth little.

**Noise band:** an improvement is only claimed when it exceeds the resampling
std under *both* view 1 and view 2.

## 11. Constraints

- 4 CPU cores, 15 GB RAM, **no GPU** (measured).
- numpy 2.4.6, pandas 3.0.6, scikit-learn 1.9.1, LightGBM 4.7.0.
- **huggingface.co is blocked** in this container, so BGE/E5/Qwen/cross-encoder
  rungs are unreachable. This is fine: §9 argues they are the wrong tool here
  anyway, and that argument does not rest on the block.
- Dataset is tiny (2,418 × 51). Every experiment runs in seconds.

## 12. Submission contract

- `working/submission.csv`, columns exactly `id,prediction`, in that order.
- Exactly **586** rows, test ids preserved, each once.
- Finite numeric predictions; values arbitrary, only order matters.
- Verified against `sample_submission.csv`: id sets match exactly.

## 13. Risks

| risk | mitigation |
|---|---|
| Project memorization inflates CV and dies on test | group-disjoint validation; prefer causally-plausible features |
| `body_mention_count` shift (2.39→0.66) | ablate it; check dependence explicitly |
| NDCG@20 variance swamps model differences | repeated resampling + noise band; prefer simple, robust models |
| ≥1.6% contradictory labels | do not chase the last points; expect a ceiling |
| Overfitting 586-row top-20 behaviour | never tune on a single draw |
| Test-only category `title_length_band=band_4` | tree models handle unseen levels; verified no crash |

## 14. Plan

| rung | experiment | question it answers |
|---|---|---|
| 0 | constant / random | the floor |
| 1 | single causal feature | is there transferable signal at all? |
| 2 | TF-IDF on profile_text + linear | does the "text" view add anything over parsing? |
| 3 | LightGBM regression, all features | leaky reference |
| 4 | rung 3 under group-disjoint CV | the honest reference |
| 5 | feature ablations (drop fingerprint/shifted features) | which features transfer |
| 6 | LGBMRanker lambdarank, P(r=2) classifier, ordinal | which surrogate objective suits NDCG@20 |
| 7 | shift-robust reweighting / regularization | does shift-awareness help |
| 8 | ensemble, only if OOF-complementary | |
