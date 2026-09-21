# Compliance Audit and Results — v2

## 1. Why the previous submission scored 0.247

Measured: **uniform-random predictions score 0.300–0.330** on this task.
The previous model scored **0.247** — i.e. it had *no* transferable signal.
Its local validation said 0.85–0.93.

Root cause, established from the data:

- The feature→target map is sharply non-monotonic in a way no mechanism
  explains: `title_digit_count` maps 2→1.35, 3→1.25, **4→0.22**, 5→1.35.
  That is an identifier (a project bot's version format), not a cause.
- Holding out a feature's own value-groups collapses `title_digit_count`'s
  rank correlation from **+0.52 to −0.02**.
- Coarse 4-tuples of style fields act as project ids: 93 tuples with n≥5 cover
  48.6% of train with within-tuple target std 0.339 vs global 0.823.

**The decisive methodological finding:** I could not build *any* validation from
the training rows that reproduces 0.247. Sibling-disjoint grouping gave 0.93,
whole-feature-space-region holdout 0.91, extrapolation holdout 0.88. Within
train, the projects are present on both sides of every possible split, so
memorisation always keeps working. Local validation is structurally incapable
of detecting this failure.

So v2 is selected by **restricting capacity**, not by maximising a local score.

### Memorisation premium predicts the failure

"Premium" = score above a zero-selection composite; the part that comes from
fitting rather than from the causal relationship.

| model | region-OOF | in-sample gap | premium | actual |
|---|---|---|---|---|
| plain GBM, all features (**the 0.247 model**) | 0.852 | +0.147 | **+0.326** | **0.247** |
| plain GBM, 12 causal features | 0.626 | +0.368 | +0.100 | — |
| monotone GBM, causal | 0.529 | +0.324 | +0.003 | — |
| **v2 composite (shipped)** | **0.601** | **+0.031** | ~0 | — |

The 0.247 model's entire measured advantage was premium, and all of it
evaporated.

## 2. How the v2 model was chosen

Three criteria, none of which depends on a holdout that I have shown to be
unreliable:

**(a) Causal sign, fixed a priori.** Every feature enters with the direction a
mechanism argument demands. This matters: an *unconstrained* greedy search
reached **0.858** locally by choosing `file_count` NEGATIVE and
`body_question_count` NEGATIVE — implausible signs, and exactly the 0.247
failure one level up. My validation could not distinguish it from a real result.

**(b) Direction stability.** Each feature's rank correlation with the target was
measured inside ~58 independent regions of feature space. Only features whose
sign agrees in ≥70% of regions are used.

| feature | consistency | used |
|---|---|---|
| `test_file_count` | **0.905** | yes |
| `docs_file_count` | **0.863** | yes |
| `max_file_changes` | 0.748 | yes |
| `changed_lines` | 0.732 | yes |
| `additions` | 0.732 | yes |
| `max_file_additions` | 0.732 | yes |
| `body_code_block_count` | 0.726 | (dropped, see below) |
| `deletions` | 0.655 | **no** |
| `file_count` | 0.655 | **no** |
| `max_file_deletions` | 0.655 | **no — a greedy search had selected it** |
| `code_file_count` | 0.632 | **no** |
| `body_question_count` | 0.560, sign contradicts prior | **no** |

This screen rejected `max_file_deletions`, which the sign-locked greedy search
had chosen as its second feature purely because it raised the training score.

**(c) Near-zero fitted capacity.** The score is a fixed, sign-constrained sum of
standardised log features. **No label-derived quantity is stored.** The only
things estimated from training data at all are a log mean and standard deviation
per feature, used to put concepts on a common scale.

Features are grouped into **concepts** and averaged within a concept, so the
four correlated size measures contribute one unit of weight rather than four.

## 3. Candidates evaluated (8 independent region partitions, 4 extrapolation splits)

| candidate | region mean ± std | extrapolation | gap |
|---|---|---|---|
| uniform random | 0.300 | — | — |
| composite, all 12 causal, no selection | 0.526 | 0.566 | +0.015 |
| tests+docs | 0.586 ± 0.009 | 0.633 | +0.008 |
| tests+docs+size+discussion | 0.600 ± 0.015 | 0.597 | +0.014 |
| **tests+docs+size (SHIPPED)** | **0.601 ± 0.012** | **0.654** | **+0.031** |
| greedy-3 incl. unstable feature | 0.632 ± 0.008 | 0.684 | +0.027 |
| monotone GBM, 12 causal | 0.529 | 0.738 | +0.324 |
| plain GBM, all numeric (**0.247 model**) | 0.852 | 0.912 | +0.147 |

The greedy-3 scores higher but rests on `max_file_deletions` (consistency
0.655) and shares **90%** of its test top-20 with the shipped model — so the
principled choice costs almost nothing in practice.

`body_code_block_count` passed the stability screen but leave-one-concept-out
showed the "discussion" concept slightly *hurt* (0.600 with vs 0.601 without),
so it is excluded.

## 4. Metric verification

`ndcg_at_k` is the challenge's code verbatim, hand-verified:

- perfect ordering [2,1,0] → 1.0 (matches hand calculation to 1e-12)
- reversed → 0.5869 (matches hand calculation)
- all-zero truth → 0.0
- ties: stable mergesort keeps input order (constant score, truth [2,0,0] → 1.0;
  truth [0,0,2] → 0.5)
- **one global ranking, cutoff 20**: 586 rows with the twenty 2s ranked first
  → 1.0; ranked last → 0.0

There is **no query/group column**; the entire test set is one ranking list.

## 5. Compliance audit of `solution.py`

| Requirement | Status |
|---|---|
| `python3 solution.py <public_dir> <submission_out>` | `sys.argv[1]` / `sys.argv[2]`, no flags |
| No hardcoded paths or row counts | verified by grep; only the docstring example mentions a path |
| Uses `len(test)` not 586 | yes |
| Reads only from `public_dir` | `train.csv`, `train_targets.csv`, `test.csv` only |
| Writes only to `submission_out` | one `to_csv`; creates parent dirs |
| Imports | `sys`, `pathlib`, `numpy`, `pandas` — nothing else |
| Network / external data / pretrained models | none; no `requests`/`urllib`/`socket`/`subprocess` |
| Test data used in fitting | **no** — `.fit()` is called on training rows only; the model is purely inductive |
| Test labels / hidden answers | never read |
| Row order, opaque id, project hash as features | not used; labels joined **by id**, never by position |
| Synthetic examples or labels | none |
| Determinism | no RNG in the scoring path at all; two runs produce byte-identical output |
| Hardware | CPU only, ~1 s, numpy+pandas |

## 6. Robustness tests (all pass)

| Test | Result |
|---|---|
| Runs from a different cwd, arbitrary directory names | pass |
| Byte-identical output across two runs | pass |
| Creates a missing output directory | pass |
| Shuffled test row order | output follows input order; per-id predictions identical |
| Shuffled `key=value` order + an unknown key | predictions identical |
| A required feature absent from `test.csv` entirely | exits 0, degrades gracefully |

Submission checks (independent verifier, 16/16): exact columns, row count equal
to `len(test)`, no duplicate ids, id set matches `test.csv` and
`sample_submission.csv`, input order preserved, numeric dtype, all finite, no
NaN, non-constant.

## 7. What the new model ranks first

Top-20 of the test set versus the whole test set:

| | top-20 median | test median |
|---|---|---|
| `test_file_count` | **21.0** | 1.0 |
| `changed_lines` | **2314** | 80 |
| `additions` | **578** | 58 |
| `max_file_changes` | **374** | 47 |
| `file_count` | **38** | 4 |
| `test_term` present | **85%** | 52% |

Large backports that touch many test files — a defensible answer to "which
change requests will consume the most reviewer capacity".

Overlap with the 0.247 model's top-20: **2 of 20**. This is a genuinely
different selection, not a reparameterisation.

## 8. Honest expectation

Region-holdout says 0.60 ± 0.10 and extrapolation says 0.65. **I do not claim
either will be the hidden score**, because I have demonstrated that local
validation on this dataset overstates a memorising model by 0.6 NDCG.

What I can defend:

- The shipped model has **~zero memorisation premium** (train/holdout gap
  +0.031 vs the failed model's +0.147 on top of a +0.326 premium), so there is
  very little for the project-disjoint holdout to take away.
- Its two strongest features are the two most direction-stable in the entire
  schema (0.905, 0.863).
- Every sign is fixed by mechanism, so the model cannot have learned a
  project-specific inversion.
- Uniform random is 0.30; a rule this simple beat 0.55 on unfitted training
  evaluation, and unfitted evaluation is the one measurement memorisation
  cannot inflate.

The main residual risk is that review volume in the hidden projects is driven by
something these six features do not capture at all, in which case the score
regresses toward random. That risk is not removable with the supplied features.
