# Top-5 Leaderboard Solutions — Consolidated Analysis Notes

**Scope:** 20 solutions total — the top 5 leaderboard submissions for each of 4
separate competition problems. This file is the single consolidated write-up
covering all of them: per-problem technique breakdowns plus a cross-problem
synthesis at the end.

1. **Problem 1 — Terraform Resource Complement Ranking** (rank a withheld
   resource among 5 candidates)
2. **Problem 2 — Sichuan Wavefront Witness Reconstruction** (reconstruct a
   seismic waveform at an unobserved station)
3. **Problem 3 — Multi-Witness Question Recovery** (recover a hidden shared
   question and the records that answer it)
4. **Problem 4 — Conversation Intervention Placement and Response
   Assignment** (pick when to intervene in a conversation and which reply to
   send)

---

## Problem 1 — Terraform Resource Complement Ranking

Given a Terraform module's visible `context` (list of resource type keys)
with one resource withheld, plus 5 `candidate_i` resource keys (one is the
true withheld target, four are distractors), rank the candidates by
likelihood of being the true target. Scoring rewards rank-1 hits most, with
partial credit for rank 2/3 (an ACU@3-style metric).

### Shared architecture across all 5 solutions

1. **No leakage, ever.** No statistic is computed from the test set or from
   a row's own module. Every co-occurrence/association table is built from
   *other* training modules only.
2. **Group-safe cross-validation.** Folds are built so duplicate/near-
   duplicate modules never span the fit/holdout split.
3. **Feature families present in every solution:** co-occurrence stats
   (forward conditional, reverse conditional, PMI/NPMI, lift, naive-Bayes
   log-odds); lexical/token features from splitting `resource_name` on `_`
   (Jaccard overlap, common-prefix length, coarse "family" grouping);
   slate-relative features (row z-score, in-row rank, gap-to-row-max);
   provider-match/structural features.
4. **Final model** is almost always a listwise ranker over 5-way groups —
   most commonly **LightGBM `lambdarank`**.
5. **Heavy ensembling**: multiple seeds/folds combined by per-row z-score
   normalization then averaging.
6. **Determinism engineering** everywhere: fixed seeds, fixed thread
   counts, deterministic torch/cudnn flags, `PYTHONHASHSEED` pinning.

### Solution-by-solution

**Rank 1** — vectorized sparse co-occurrence (scipy.sparse incidence
matrices, fully vectorized feature extraction) + aggressive **proxy-
screening ensemble**: cheap linear-combination proxy scores decide which
training rows are kept for the expensive model (only rows where the true
target isn't already trivially top-3 under some proxy), tested under
several "short-context" variants (first/last/rarest/most-common/strongest
8-of-context) and stress-tested with Gaussian-noise-perturbed re-checks (24
draws) for a soft/probabilistic curriculum filter. Four LightGBM rankers
blended, with held-out utility logged for each before committing.

**Rank 2** — three-layer statistics: `Stats` (plain unsupervised
co-occurrence), `TargetStats` (supervised, directional
`P(target|context resource)`), and `SlateStats` — the novel piece: pairwise
"win rate" of candidate A vs B when both are candidates together, triple-
wise interactions, family/size priors, slate-signature win rates — directly
modeling distractor-specific hardness. A compact z-scored "compatibility
proxy" decides row eligibility for the simpler of two blended LightGBM
models (simple/extended `SlateStats` depth, blended 0.8/0.2).

**Rank 3** — simplest architecture: one small neural `Ranker` (learned
candidate/context embeddings + ~200 engineered stats, MLP, multi-class
cross-entropy). Standout: **`hard_training_masks`** — a cheap
forward/reverse-rate+NPMI proxy flags rows where the target isn't already
trivially best, and those rows get 64× loss upweighting — explicit hard-
example mining via sample weighting. No k-fold loop at all, yet still
ranked 3rd.

**Rank 4** — most complex: represents the 5 candidates as a **fully-
connected graph** (per-node co-occurrence/family/token features, 5×5
directed edge features including PMI-profile cosine similarity), a custom
message-passing `GraphRanker` (mean/max-pool incoming/outgoing messages per
node), plus a separately pretrained **BERT/MLM-style masked-resource-
completion model** whose log-probabilities feed in as extra node features.
Union-find (DSU) fold grouping on rare-key/Jaccard-similar modules — most
rigorous leakage control of the five. 25-model ensemble (5 folds × 5
seeds). Placed 4th despite the heaviest compute.

**Rank 5** — "kitchen sink" ensemble of classical recommender techniques:
**EASE** (closed-form ridge-regularized item-item model, 3 ridge
strengths), **PMI-based SVD embeddings**, **kNN collaborative filtering**
over modules, a masked-resource `CompletionNet`, and "hub"/slate-centrality
cosine-similarity features (hypothesizing the true target is most "central"
to the distractor set). Deterministic folds via **MD5 hash of row id mod
5**. 15-model LightGBM ensemble. Widest feature/technique arsenal of the
five — placed last.

### Cross-cutting takeaways

- Row-relative transforms (z-score/rank/gap-to-max within the 5-slate) are
  universal — the label is a relative comparison, not an absolute score.
- Strict leakage discipline and directional (forward/reverse/symmetric)
  co-occurrence statistics are universal.
- Lexical decomposition of the resource key into token sets and a coarse
  "family" is a cheap, universally used feature.
- **Hard-example curriculum is the clearest differentiator of the top
  solutions** — ranks 1–3 all explicitly identify "easy" rows and either
  filter or upweight around them; ranks 4–5 instead threw more model
  capacity at the whole set and placed last.
- LightGBM `lambdarank` over 5-way groups is a strong, efficient default —
  it's the final model in the three best solutions.

---

## Problem 2 — Sichuan Wavefront Witness Reconstruction

Given 4 "witness" seismic stations' 256-sample waveforms plus lat/lon
geometry of witnesses and a "receiver" station, reconstruct the 256-sample
waveform that would have been recorded at the receiver — spatial
interpolation/extrapolation of a seismic wavefield within the same
earthquake event.

### The scoring function (shapes everything)

```
w = clip(1 - sum((pred-target)^2)/sum(target^2), 0, 1)   # signed fit
a = clip(1 - sum((|pred|-|target|)^2)/sum(target^2), 0, 1) # magnitude fit
z = clip((sqrt(w*a) - floor) / (1 - floor), 0, 1)
score = expm1(1.2*z) / expm1(1.2)
```

A geometric mean of a signed-error term and a magnitude(envelope)-only
term — getting *phase/shape* right and getting *amplitude* right are both
necessary; you can't compensate one with the other.

### Shared structure across all 5 solutions

1. **Event/leakage grouping via byte-exact witness matching** — the same
   witness recording appears verbatim across multiple rows of the same
   event; every solution hashes waveforms and union-find-clusters rows,
   splitting train/validation by event.
2. **"Any station can be the target" symmetry** — four of five solutions
   exploit this physical interchangeability as free data augmentation
   (reassigning a different pool member as pseudo-target).
3. **Frequency-domain modeling dominates** (`rfft` spectra; several solve
   small closed-form linear systems per frequency bin — classic kriging /
   Wiener filtering).
4. **Metric-aware loss shaping** — every training loss adds an explicit
   `MSE(|pred|, |target|)` term matching the scoring function's magnitude
   component.
5. **Odd-symmetry / sign robustness** — hard-coded (`f(x)=(raw(x)-raw(-x))/2`)
   or trained via random polarity-flip augmentation.
6. **Physics baseline + neural correction, calibrated against the true
   metric** — the stronger solutions (ranks 1, 2, 4) blend a closed-form
   statistical predictor with a neural residual corrector, with blend
   weights/scale numerically optimized directly against the real score.
7. **Determinism engineering** universal; rank 1 asserts bit-identical
   repeat-inference before writing the submission.

### Solution-by-solution

**Rank 1** — `SpatialSpectrum`: per-frequency-bin spatial covariance
matrix across unique station locations (power-normalized, frequency-
smoothed), nearest-location substitution for unseen coordinates. `.predict()`
solves a 4×4 **kriging (MMSE)** linear system per frequency bin. Leave-one-
out training inputs (own-event contribution subtracted) feed two neural
residual correctors (`SymmetricCorrector`, `PartsCorrector`/`UNetParts`,
both with the antisymmetric sign trick and EMA-averaged weights). Final
blend + scale calibrated via `scipy.optimize.minimize_scalar` directly
against the real score. Most rigorous leakage/reproducibility discipline.

**Rank 2** — `PhaseNet`/`SitePhaseNet`: learned per-frequency complex
mixing weights (a *learned* Wiener/kriging filter) conditioned on cross-
spectral coherence and a nearest-3-site station embedding.
`receiver_augmentation` re-labels a different station as target — row-level
"any station can be target" augmentation, with a curriculum
(`refine()` continues training on augmented rows after real-row fitting).
A closed-form **Wiener kernel** baseline with eigenvalue-repaired
correlation matrices (`positive_kernel`), and an **out-of-fold kernel prior**
stacked into a second model. **`MagnitudeWaveNet`**: a wholly separate
network trained only on `|target|`, directly matching the metric's
magnitude term. Most deliberately metric-decomposed pipeline.

**Rank 3** — spectral-gain CNN (`SpectralWavefrontNet`, per-witness FiLM-
conditioned convolutions over frequency bins + cross-correlation lag
features) + a novel **`AxialWavefrontTransformer`** (axial attention across
time within each witness channel, alternating with attention across the 4
witness channels, rotary embeddings, adaptive FiLM norm). Most
architecturally novel, purely neural — no closed-form baseline. Ranked 3rd.

**Rank 4** — `StationNet`: attention-pooled residual encoder/decoder;
softmax attention weight per witness from geometry + a "Gram matrix" of
pairwise waveform cross-correlation; "any station can be target" folded in
as an **auxiliary loss term** rather than augmented rows. Closed-form
**Wiener filter** baseline (coherence-weighted witness combination).
Writes an immediate trivial fallback submission before training even
starts. Simplest neural architecture of the top 4.

**Rank 5** — `WaveNet`: dilated-conv encoder per witness with cheap cross-
witness mixing (mean-pool injected via 1×1 conv), fixed near-identity
linear FIR baseline + geometry-conditioned **adaptive FIR filter**
(`tapnet`) as residual. Combinatorial "any pool member can be target"
augmentation across 10 predefined distance-rank subset patterns. **8-member**
architecture-diverse ensemble (varying width/depth/kernel/seed). No
closed-form statistical baseline at all. Ranked last despite the largest
ensemble.

### Cross-cutting takeaways

- Event-level leakage control via exact-waveform hashing is universal.
- "Any station can be the target" is the single most reused structural
  insight — turned into free augmentation or an auxiliary loss by 4 of 5
  solutions.
- The scoring metric's two components (shape, magnitude) are explicitly
  targeted, not left to plain MSE.
- **Closed-form, physically-motivated baselines (kriging/Wiener) + a
  learned residual correction beat pure end-to-end neural approaches** —
  the top 3 all include such a baseline; the two purely-neural solutions
  (one novel transformer, one large ensemble) placed lower.
- Ensemble weights and global scale are calibrated directly against the
  true competition score on held-out, event-grouped data.

---

## Problem 3 — Multi-Witness Question Recovery

Each "ledger" holds `n` answer/evidence records; exactly `n-2` of them
("witnesses") were written for the same hidden shared question, and 2 are
unrelated singletons. Task: (1) select the witness subset (size `n-2`), and
(2) reconstruct the hidden question's text. Scoring gates the text-quality
score (baseline-adjusted chrF, order 1–6, recall-weighted) behind an
**exact witness-set match** — get the witness set wrong and the question
score for that row is zero, so witness selection is the higher-leverage
half of the task.

### Shared structure across all 5 solutions

1. **Two-stage pipelines dominate** (a witness-selector model, then a
   question-generator model conditioned on the selected witnesses) — every
   solution except rank 5, which fine-tunes **one** LLM to jointly emit both
   outputs from a single prompt.
2. **Pairwise cross-encoder selectors.** Ranks 2–4 all fine-tune a
   cross-encoder to score "do these two records share a question", then
   pick the subset of size `n-2` maximizing summed pairwise affinity
   (brute-force over `C(n, n-2)`, small enough to enumerate exactly) —
   effectively a MAP decode over a pairwise log-odds graph.
3. **Minimum-Bayes-Risk (MBR) decoding directly under the competition's own
   chrF metric** is universal for question generation: generate a pool of
   candidates (beam + sampled), score every candidate against the pool
   under chrF, and submit the one with highest expected similarity to the
   pool — decoding to *maximize the actual metric*, not just top-1
   likelihood.
4. **Group-disjoint (by paper) validation folds** — no paper's records
   split across train/validation.
5. **Determinism engineering** universal (fixed seeds, deterministic
   cudnn/backends).

### Solution-by-solution

**Rank 1** — *(no implementation was uploaded for this rank; the file
contains only a docstring placeholder describing the intended approach —
"a joint scientific-ledger repair model," CUDA + pinned Qwen weights, no
external data — but no runnable code was provided, so no technical
analysis is possible beyond that stated intent.)*

**Rank 2** — Pairwise selector: **BAAI/bge-reranker-large**, symmetrized
pair-logit matrix, MAP subset decode. Question generation: **two**
generators (flan-t5-large full fine-tune + Qwen2.5-1.5B-Instruct via LoRA),
beam candidates pooled from both, MBR-selected under chrF — the hypothesis
space includes every candidate **and every pairwise concatenation** of
candidates, widening the search beyond what either model alone proposed.

**Rank 3** — Selector: **ModernBERT-large**, trained with a **combined**
pairwise BCE + full-subset cross-entropy loss (directly optimizing both the
pair-level and whole-cluster decisions together, not just pairs in
isolation). Generator: flan-t5-large with explicit **combinatorial data
augmentation** — every gold cluster is also broken into smaller sub-cluster
views (up to size 4) as additional (partial-witness-set → question)
training pairs, teaching the model to answer from imperfect/partial witness
selections too. MBR with temperature-weighted candidate pooling plus
two-way candidate unions.

**Rank 4** — Selector: **RoBERTa-large-MNLI** repurposed as a binary
cross-encoder, trained across **5 group K-folds**, MAP-decoded via
exhaustive search over pairwise log-odds (the `n-2` constraint is
explicitly verified against training labels first). Generator: flan-t5-large
trained with **"label-preserving views"** — both the full gold cluster and
each individual witness record singly count as valid (records → question)
training examples, a lighter version of rank 3's augmentation. Most
rigorous decoding-strategy selection: beam vs. MBR and several length
penalties are grid-searched on a **group-disjoint training hold-out
against the official metric**, and the script explicitly logs an
"ESTIMATED leaderboard score" (`selector_accuracy × question_quality`)
before ever touching the test set.

**Rank 5** — Single **Qwen2.5-7B-Instruct** model fine-tuned with LoRA to
emit **both** the witness-letter line and the question text from one
prompt in one generation — the only solution that doesn't split witness
selection and question generation into separate models. Uses **two
snapshots from the same training run**: witness selection keeps improving
with more epochs, but question wording starts overfitting the small
(424-ledger) training set, so the LoRA weights are snapshotted earlier
(`SNAP_QUESTION` epoch) for question generation and later (final epoch) for
witness selection — getting two effectively different model behaviors
without training two models. Witness selection: scores **every** candidate
subset by the model's own log-probability of producing that subset's
letters, averaged over multiple random re-orderings of the records in the
prompt (removing any position bias). Question: beam + sampled candidates,
MBR-picked via chrF against the pool.

### Cross-cutting takeaways

- **The task's own scoring gate (exact witness-set match) shapes every
  solution's effort allocation** — all four solutions with actual code
  invest real engineering (K-fold ensembling, joint losses, group-safe
  validation) specifically in the witness-selection stage, since getting it
  wrong zeroes out the question score entirely for that row.
- Brute-force MAP decoding over exhaustively enumerated candidate subsets
  is viable and universal here because `n` is small — no need for greedy or
  beam search over subsets.
- MBR decoding **directly under the competition's own similarity metric**
  (chrF, not generic likelihood) is the single most reused idea for the
  generation half — every solution with code does some form of it, several
  widening the hypothesis space with pairwise concatenations of candidates.
- Data augmentation via **partial/sub-cluster views** (rank 3's
  combinatorial subsets, rank 4's singleton views) teaches the generator
  robustness to an imperfect upstream witness selection — a form of
  train/inference-mismatch hardening.
- Rank 5's single dual-purpose model (two snapshots of one LoRA run) is a
  notably resource-efficient alternative to the two-model pipelines used
  everywhere else, and still competitive — though without rank 1's code for
  comparison, it's hard to say whether a genuinely joint model needed to be
  beaten to reach the top.

---

## Problem 4 — Conversation Intervention Placement and Response Assignment

Each "board" holds several conversation `tracks` and a shared bank of 10
candidate replies. Every track exposes exactly 7 "moments" (decision
snapshots); the task is, per track: (1) select **which moment** is the
right time for an assistant to intervene (a calibrated probability
distribution over the moments, plus an argmax pick), and (2) **which of the
shared candidate replies** to assign to that track. Response assignment is
**injective per board** (no two tracks in the same board may be assigned
the same reply), so it's solved as a bipartite/rectangular assignment
problem, not an independent per-track choice.

### Shared structure across all 5 solutions

1. **Cross-encoder scoring of (track, candidate-reply) pairs**, producing a
   7-moment × 10-reply (or 7×N) energy matrix per track, is the universal
   backbone — every solution reduces to "score every reply at every moment,
   then decode."
2. **Hungarian algorithm (`scipy.optimize.linear_sum_assignment`)** on a
   per-board marginal match-score matrix is universal for the injective
   response assignment — every solution collapses its 7×10 per-track energy
   to a single scalar match score (via max or log-sum-exp over the moment
   axis) and solves board-wide assignment on that.
3. **Moments 0 and 6 (the first and last) are structurally illegal** — every
   solution that reasons about this explicitly excludes them from the
   legal/scoreable set (`kmask`, `LEGAL_BIAS`, or a data-derived legal-index
   set), leaving 5 genuinely competing moments per track.
4. **Negative/foil sampling** during training (only a handful of the 10
   replies are scored per training step, not all 10) — every solution
   trains with 1–5 sampled foils per positive rather than all 9 negatives,
   for compute reasons.
5. **Determinism engineering** universal (fixed seeds, deterministic
   cudnn/backends, disabled TF32).

### Solution-by-solution

**Rank 1** — Two-stage: a **`SlotModel`** cross-encoder (DeBERTa-v3-large)
that packs the full track with `[MASK]` tokens inserted at each of the 7
moment "slot" positions and reads off a score per slot from the mask's
hidden state, trained with sampled negatives (5-of-10) via a joint
(response×moment) softmax. Includes a genuinely clever engineering trick:
a **custom `_MonoGather` autograd function** that monkey-patches DeBERTa-v3's
relative-position attention gather with a cumulative-sum-based backward
pass, reducing memory for large-batch training under strict determinism.
Response assignment solved via log-sum-exp marginalization + Hungarian
algorithm on the SlotModel's scores. A **second, smaller `InsModel`**
(DeBERTa-v3-base) then *refines timing conditioned on the already-assigned
response* — it physically inserts the assigned reply's text into the
track at each candidate gap and scores insertion quality, blended
(`W_INS=0.5`) into the final timing distribution. The only solution with
this two-phase "assign response, then re-score timing given that response"
refinement.

**Rank 2** — Two fully **independent** models rather than one joint
cross-encoder: a `TimingModel` (DeBERTa-v3-large) that inserts a special
`[MDEC]` marker token after each moment's last utterance and classifies
over the resulting marker positions (restricting the legal output space to
moments actually observed as gold in training, a data-driven legal set
rather than a hard-coded rule); and a separate `ResponseModel`
(DeBERTa-v3-small) that mean-pools a cross-encoding of track vs. each reply
and picks via a straightforward 10-way softmax, with no conditioning on
timing at all. Timing model bagged over **4 seeds**; response model single-
shot. Simplest, most decoupled architecture among the top solutions —
still ranked 2nd.

**Rank 3** — Most architecturally ambitious: **reimplements the
ModernBERT-large backbone from scratch** in raw PyTorch (manual rotary
embeddings, banded local-attention windows alternating with global-attention
layers, loaded directly from safetensors) rather than using the HF model
class. Packs **all 5 tracks of a board and the full 10-reply bank into one
single sequence**, letting one global self-attention pass model
interactions between tracks and replies directly, rather than requiring a
separate forward pass per (track, reply) pair. Three prediction heads: a
timing head (with engineered "advance in/out" features — how many
utterances newly enter/leave the window around a moment — plus a
direct-assistant-address flag), a response head, and an explicit **joint
head** producing a learned bilinear 7×10 timing-response compatibility
score, trained with a combined loss over all three. Order-invariance
test-time augmentation: 4 random re-shufflings of track/reply order,
averaged. Most novel per-solution engineering of the four problems
reviewed, yet placed 3rd.

**Rank 4** — A single unified cross-encoder (`RCModel`, DeBERTa-v3-base)
that represents each moment from **both sides of its decision gap**
(the last utterance before, and the first utterance after, the point where
the withheld reply would slot in) plus a global mean-pooled track
representation, fused with learned position/direction/"advance" embeddings.
Trained with only **1 sampled negative per positive** (the lightest
negative sampling of the five). Notably outputs **hard one-hot "probabilities"**
at inference (`probs[selected]=1.0`, all else 0) rather than a calibrated
soft distribution like every other solution here — a choice that likely
costs it on a probability-calibration-sensitive metric, plausibly
contributing to its lower rank despite reasonable architecture.

**Rank 5** — Heaviest ensembling: **3 members with genuinely different
backbones** (roberta-base, electra-base-discriminator, roberta-base again
with a different seed), each snapshotted at **two different epochs**
(9 and 11 of 12) and all 6 resulting checkpoints ensembled together. Same
pre/post-gap + attention-weighted window pooling as rank 4, but with a
**three-term loss** (joint response×moment cross-entropy, a timing-only
term conditioned on the true response, and a response-only term via
log-sum-exp over moments) — explicitly supervising both the joint and
marginal decisions, unlike rank 4's joint-only loss. Writes an immediate
"safety" fallback submission (uniform probability over the legal moments,
an arbitrary but valid response assignment) **before any training begins**,
and **incrementally rewrites the real submission after every snapshot**
during training — defensive engineering ensuring a valid submission exists
at any interruption point. Despite the largest, most architecturally
diverse ensemble, placed last.

### Cross-cutting takeaways

- **Every solution reduces to the same two-step decode**: score a
  per-track energy matrix over (moment × reply), collapse it to a per-track
  scalar match score, then solve board-wide injective assignment via the
  Hungarian algorithm — this decode-time structure is effectively fixed
  across all five; the *learning* is what differs.
- **Joint (timing × response) modeling — whether via a genuinely joint head
  (rank 3), an explicit two-phase refinement (rank 1's SlotModel→InsModel),
  or a multi-term loss supervising both marginals (rank 5) — was more
  common among the top-ranked solutions** than the fully decoupled
  approach (rank 2, which still did well) or the joint-loss-only approach
  without marginal refinement (rank 4, ranked lowest of the four with full
  code).
- **Calibrated soft probability output matters**: the one solution that
  emitted hard one-hot "probabilities" instead of a real distribution
  (rank 4) ranked lowest among the four fully-coded solutions — a strong
  hint that the scoring metric rewards calibration, not just the argmax
  being correct.
- The illegal-endpoint-moment constraint (moments 0 and 6 never being gold)
  was independently discovered and encoded by multiple solutions —
  another instance of baking an empirically-discovered task constraint
  directly into the legal output space at decode time.
- As in problems 1 and 2, **the largest, most architecture-diverse ensemble
  (rank 5) and the most novel from-scratch architecture (rank 3) did not
  win** — a two-stage, explicitly joint-aware but comparatively modest
  architecture (rank 1) did.

---

## Cross-Problem Meta-Patterns (synthesis across all 20 solutions)

A few patterns repeat across all four, otherwise unrelated, problem
domains strongly enough to be worth naming as general lessons rather than
per-problem observations:

1. **Complexity and ensemble size do not track leaderboard rank — and
   often invert it.** In problem 1, the most complex solution (a graph
   neural network + pretrained masked-completion model, rank 4) and the
   widest technique arsenal (a classical-CF "kitchen sink," rank 5) placed
   last. In problem 2, the most novel architecture (a custom axial
   transformer, rank 3) and the largest ensemble (8 architecture-diverse
   members, rank 5) both placed in the bottom half. In problem 4, the same
   split repeats almost exactly: the most novel from-scratch architecture
   (rank 3) and the largest, most diverse ensemble (rank 5, 3 backbones × 2
   snapshots) both placed lowest among fully-coded solutions. The pattern
   is consistent enough across three independent problem domains to treat
   as a real signal, not coincidence: **targeted, domain-aware structure
   beats throwing more model capacity or more ensemble members at the
   problem.**

2. **Every top solution calibrates directly against the actual competition
   metric on held-out data, never trusting training loss alone.** Problem 1
   ranks 1–2 build cheap proxy scores and validate screening thresholds
   against held-out utility; problem 2's ranks 1, 2, 4 grid-search or
   numerically optimize blend weights and global scale against the true
   score; problem 3's ranks 2–4 all do MBR decoding directly under the
   competition's own chrF metric (not generic likelihood), and rank 4
   explicitly logs an estimated leaderboard score computed from held-out
   validation before touching the test set; problem 4's solutions fit
   decode temperature and legal-moment masks against validation
   performance. This discipline — decode/blend/threshold for the *actual*
   metric, not a proxy — is the most consistently present technique across
   all 20 solutions.

3. **Group-safe (leakage-safe) validation splitting is universal and
   independently reimplemented every time**, adapted to each problem's own
   notion of a "group": modules deduplicated by content (problem 1), events
   recovered via exact-waveform hashing and union-find (problem 2), papers
   via the `group` column (problem 3), and boards via `board_id` (problem
   4). No solution across all 20 skips this.

4. **Turning a task's structural symmetry into free training signal is a
   recurring winning move.** Problem 2's "any station can be the target"
   (exploited by 4 of 5 solutions as row augmentation or an auxiliary
   loss); problem 3's partial/sub-cluster witness views as extra
   (records→question) training pairs; problem 4's joint timing-response
   heads and two-phase assign-then-refine pipeline. In each case, solutions
   that discovered and exploited a symmetry specific to the task's own
   generative structure — rather than relying on generic data augmentation
   — tended to rank higher.

5. **Empirically-discovered task constraints get hard-coded into the legal
   output space at decode time, not left for the model to learn on its
   own.** Problem 1's proxy-screened "eligible" rows; problem 2's nearest-
   observed-site substitution for unseen coordinates; problem 3's verified
   `|witnesses| = n - 2` constraint enabling exhaustive MAP subset search;
   problem 4's exclusion of the structurally-illegal first/last moment.
   Baking a hard, verified constraint into inference is treated as free
   accuracy, not cheating.

6. **Determinism is engineered as a first-class requirement, not an
   afterthought, in every single one of the 20 solutions** — fixed seeds
   across every RNG source, deterministic cuDNN/backend flags, disabled
   TF32, pinned thread counts, and in a few cases explicit repeat-inference
   equality assertions or deterministic hash-based fold assignment instead
   of stored random state. This is strong evidence the underlying
   competition platform enforces exact reproducibility of submissions as a
   grading requirement, independent of problem domain.

7. **Two-stage pipelines (cheap/robust structure-selection first, richer
   model conditioned on that structure second) dominate over single
   end-to-end joint models** — problem 1's screen-then-rank, problem 3's
   select-witnesses-then-generate-question (4 of 5 solutions), problem 4's
   assign-response-then-refine-timing (rank 1). The exceptions (problem 3
   rank 5's single dual-purpose LLM with two training snapshots; problem
   4 rank 3's single globally-packed joint transformer) are genuinely
   interesting alternative designs but didn't clearly outperform the
   decoupled approach in either case — worth trying, but not a
   demonstrated free win in this leaderboard.

8. **Defensive "always have a valid submission on disk" engineering**
   appears independently in problems 2 and 4 (writing an immediate trivial/
   neutral fallback submission before training starts, then incrementally
   overwriting it as better models finish) — a practical hedge against
   time or resource limits that costs little and bounds the downside of a
   failed run.
