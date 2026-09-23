# Terraform Resource Complement Ranking — Top-5 Leaderboard Solution Notes

Competition task: each row gives a Terraform module's visible `context` (list of
resource type keys, `provider:resource_name`) with one resource withheld, plus
5 `candidate_i` resource keys (one is the true withheld target, four are
distractors). The model must output a permutation of `[0..4]` ranking the
candidates most-likely-target first. Scoring rewards getting the true target
into (especially) rank 1, with partial credit for rank 2/3 (an "ACU@3"-style
metric — see rank-1's `UTIL = [0, 1, 2/3, 1/3, 0, 0]` array).

Five independently-written solutions were analyzed (ranks 1–5 on the
leaderboard). All five converge on the same *shape* of pipeline but differ
sharply in modeling approach and complexity — and, notably, added model
complexity did **not** correlate with a better rank.

---

## 1. Shared architecture across all 5 solutions

1. **No leakage, ever.** No statistic is computed from the test set or from a
   row's own module. Every co-occurrence/association table is built from
   *other* training modules only (out-of-fold, or "exclude this row's own
   module" bookkeeping). Because the label is literally "which candidate
   completes this module," any global statistic touching the row itself
   leaks the answer.
2. **Group-safe cross-validation.** Folds are built so that duplicate or
   near-duplicate modules (same or highly overlapping resource sets) never
   span the fit/holdout split — otherwise a near-duplicate module would let
   the model "look up" the answer instead of generalizing. Solutions range
   from simple module-string dedup (rank 2) to full union-find clustering on
   low-document-frequency keys + Jaccard similarity ≥ 0.8 (rank 4, most
   rigorous).
3. **Feature families, present in every solution:**
   - **Co-occurrence stats** between candidate and each context resource:
     forward conditional `P(cand|ctx_item)`, reverse conditional
     `P(ctx_item|cand)`, joint/PMI, NPMI (normalized), lift, and a
     naive-Bayes-style summed log-odds across the whole context.
   - **Lexical/token features** from parsing the resource key
     `provider:resource_name` by splitting `resource_name` on `_`: Jaccard
     token overlap, common-prefix length, and a coarser "family" grouping
     (first 1–2 tokens) used for family-match indicators. This captures that
     e.g. `aws_instance` and `aws_instance_profile` are semantically related
     by naming convention.
   - **Slate-relative features.** Since the decision is "which of these 5 is
     right," raw scores are converted to row-relative z-scores, in-row
     ranks, and gap-to-row-max, for every base feature block. This is the
     single most universal trick — every one of the 5 solutions does it.
   - **Provider-match** and structural features (module size, context
     length, etc.).
4. **Final model is a listwise/pairwise ranker or classifier over 5-way
   groups**, most commonly **LightGBM `lambdarank`** with `group=[5,5,...]`
   and binary relevance labels (ranks 1, 2, 5 use this). Ranks 3 and 4 use
   neural nets instead (cross-entropy classification and a graph ranker,
   respectively).
5. **Heavy ensembling**: multiple seeds and/or folds (5 to 25 models),
   combined by per-row z-score normalization then averaging before the
   final `argsort`.
6. **Determinism engineering** is treated as a hard requirement everywhere:
   fixed seeds, fixed thread counts, deterministic torch/cudnn flags,
   deterministic fold assignment (e.g. hashing row IDs instead of random
   permutation), `PYTHONHASHSEED` pinning.

---

## 2. Solution-by-solution breakdown

### Rank 1 — vectorized sparse co-occurrence + aggressive proxy-screening ensemble
- Rebuilds all co-occurrence tables as **scipy.sparse incidence matrices**
  per fold-pool (`Pool` class) instead of per-row `Counter` loops — fast,
  fully vectorized feature extraction in chunks.
- Defines many small **"proxy" scores** (cheap linear combinations of a few
  features, z-scored and averaged) and uses them to **screen** training rows:
  a row is kept for the expensive model only if at least one proxy already
  ranks the true target in the top 3 of 5. This is repeated under several
  "short-context" variants (only using a first-8 / last-8 / rarest-8 /
  most-common-8 / strongest-8 subset of the visible context) to test whether
  the signal is robust to partial context — a form of stress-testing labels
  before training on them.
- Adds **Gaussian-noise-perturbed re-checks** of the same proxies (24 draws)
  so the screening threshold isn't a brittle hard cutoff — effectively a
  soft/probabilistic curriculum filter.
- Trains **four separate LightGBM rankers** (whole-pool / module-proxy-
  screened / exact-readings bagged ×5 seeds / noise-screened bagged ×5
  seeds) and prints held-out utility for each before committing to the final
  blend — very rigorous internal ablation logging.
- Most engineering-heavy and experiment-driven of the five.

### Rank 2 — three-layer statistics (unsupervised / target-directional / slate-vs-slate) + two-model blend
- Cleanly separates three levels of statistics:
  - `Stats` — plain **unsupervised** co-occurrence between any two resources
    seen together in a module (symmetric).
  - `TargetStats` — **supervised, directional**: specifically
    `P(target | context resource)`, since only the withheld resource is the
    "answer," not any co-occurring pair.
  - `SlateStats` — the most novel piece: statistics conditioned on the
    **exact competing slate**. It learns pairwise "win rates" (how often
    candidate A beats candidate B when both appear together as candidates),
    triple-wise interactions, family-vs-target priors, and even a
    slate-signature (exact 5-tuple of candidates) win rate. This directly
    models "hardness" of the specific distractor set, not just resource
    popularity.
- Builds a compact **compatibility proxy** (small weighted z-score blend of
  a handful of the above) used purely to decide which rows are "eligible"
  for the simpler of its two final models — same screening idea as rank 1,
  implemented more simply.
- Trains two LightGBM lambdarank models on different feature-set depths
  ("simple" vs "extended" `SlateStats`), blended 0.8/0.2 by z-score.
- Categorical features (candidate id, family id, provider/prefix id) are
  passed natively to LightGBM rather than hand-encoded.

### Rank 3 — GPU neural ranker + hard-example reweighting (simplest architecture)
- A single small `Ranker` network: learned embeddings for candidate and
  (mean-pooled) context resources, concatenated with ~200 hand-engineered
  statistical features (raw + relative + rank, for both candidate-level and
  whole-module-level co-occurrence), fed through an MLP, trained with
  **multi-class cross-entropy** over the 5 slots (not a pairwise ranker).
- The standout idea: **`hard_training_masks`**. It computes a cheap
  forward/reverse-rate + NPMI proxy, checks whether the true target is
  *already* trivially the best candidate under that proxy (strict-best test
  + margin test), and **upweights the loss 64× on rows where it is not** —
  explicit hard-negative/hard-example mining via sample weighting rather
  than dropping rows. Two variants of this proxy are used as two "modes,"
  each trained with 3 seeds (6 models total), averaged.
- No k-fold training loop at all (fits once on all train data per
  seed/mode) — by far the simplest pipeline of the five, yet still placed
  3rd, suggesting the hard-example weighting + feature quality mattered more
  than architectural sophistication.
- Heavily engineered for bit-exact reproducibility (CUDA required,
  deterministic algorithms enforced, disables TF32/reduced-precision
  matmul).

### Rank 4 — Graph Neural Network ranker + Masked-Language-Model pretraining (most complex)
- Represents each row's 5 candidates as a **fully-connected graph**: per-node
  features (co-occurrence stats, naive-Bayes log-odds, family/token
  overlap) plus **5×5 directed edge features** between every candidate pair
  (relative log-odds, cosine similarity of "profile" vectors, family match,
  scene co-occurrence).
- `Tables.prof`/`.cos`: builds lightweight **item-item embeddings** by
  keeping only the top-64 highest-PMI co-occurring resources per resource
  (sparse pseudo-embedding), then cosine-similarity between candidates from
  this — a cheap alternative to full SVD embeddings.
- Separately pretrains a **BERT/MLM-style `MaskedSet` model**: for every
  module, one resource is masked and predicted from the bag of the rest
  (with sub-token hashing for OOV robustness). Its log-probability output
  per candidate becomes extra node features (raw, z-score, rank, margin) fed
  into the graph ranker — an auxiliary self-supervised signal layered on
  top of the supervised graph model.
- `GraphRanker`: an explicit message-passing network over the fully
  connected 5-node graph — computes messages for every ordered pair using
  node+edge encodings, mean/max-pools incoming and outgoing messages per
  node, and scores each node from the pooled representation. This is
  effectively a small transformer/GNN hybrid purpose-built for 5-way slates.
- Fold grouping uses **union-find (DSU)** to cluster training rows whose
  modules share rare context keys or have ≥0.8 Jaccard similarity — the
  most rigorous leakage control among the five.
- Final ensemble: 5 folds × 5 seeds = 25 GraphRanker models (log-softmax
  averaged), plus the MLM pretraining per fold/seed — by far the most
  compute-intensive solution. Despite this, it ranked 4th, one spot from
  last.

### Rank 5 — collaborative-filtering "kitchen sink" ensemble + LightGBM
- Per fold, builds an entire suite of classical recommender-system item-item
  models on the co-occurrence matrix:
  - **EASE** (Embarrassingly Shallow Autoencoder — closed-form ridge-
    regularized item-item linear model) at three ridge strengths.
  - **PMI-based SVD embeddings** (128-dim, via `randomized_svd` on a
    positive-PMI matrix) — classic word2vec-style resource embeddings.
  - **kNN collaborative filtering** over modules (cosine similarity of
    resource-incidence vectors, weighted voting) at K=10/30/100.
  - A **`CompletionNet`** masked-resource neural net (same idea as rank 4's
    MLM, simpler architecture) contributing log-prob/logit/rank features.
  - **"Hub"/slate-centrality features**: cosine similarity of each candidate
    to the other 4 candidates in two embedding spaces, hypothesizing the
    true target is more "central" to the distractor set (since distractors
    are presumably chosen to be plausible/similar).
  - Standard co-occurrence + lexical/family/token overlap stats as in every
    other solution.
- Deterministic fold assignment via **MD5 hash of the row id modulo 5**
  (rather than a stored random permutation) — a clean, dependency-free way
  to get perfectly reproducible folds.
- Final LightGBM lambdarank, 3 seeds × 5 folds = 15 models, z-score blended.
- The widest feature/ensemble arsenal of the five (CF + embeddings + neural
  + engineered stats all feeding one GBM), but placed last — echoing rank
  4's result that breadth of modeling technique didn't translate into
  leaderboard position here.

---

## 3. Cross-cutting takeaways (what actually seems to matter)

1. **Row-relative transforms (z-score / rank / gap-to-max within the
   5-candidate slate) are non-negotiable** — literally every solution does
   this, because the label is a relative comparison, not an absolute score.
2. **Strict leakage discipline** (only out-of-fold / own-module-excluded
   statistics, module-aware fold grouping) is treated as a hard constraint,
   not an afterthought, in all five.
3. **Directional statistics matter**: forward conditional, reverse
   conditional, and symmetric PMI/NPMI are computed separately — each
   carries different signal, and every solution keeps all three.
4. **Lexical decomposition of the Terraform key** (`provider:resource_name`
   tokenized on `_`, grouped into a coarse "family") is a cheap, universally
   used feature — resource naming conventions are informative.
5. **Hard-example curriculum is the clearest differentiator of the top
   solutions.** Ranks 1, 2, and 3 all explicitly identify "easy" rows (where
   a cheap proxy already nails the answer) and either filter them out of
   training or drastically upweight the hard ones. Ranks 4 and 5, which
   instead threw more model capacity (GNN + MLM pretraining, or a CF
   "kitchen sink") at the *whole* training set, placed last. This strongly
   suggests that for this task, **teaching the model where it's already
   confident vs. where it needs to work harder beats adding modeling
   power.**
6. **LightGBM `lambdarank` over 5-way groups is a strong, efficient default**
   for this kind of "pick 1-of-5" ranking task — it's the final model in the
   three best solutions, while the more exotic neural rankers (GNN, graph
   message passing) underperformed it despite far greater compute cost.
7. **Heavy seed/fold ensembling with per-row z-score blending** before the
   final `argsort` is universal and cheap insurance against variance.
8. **Determinism is engineered in, not incidental**: pinned seeds, fixed
   thread counts, deterministic CUDA/cuDNN flags, and hash-based (not
   RNG-based) fold assignment appear across all five, implying the
   competition explicitly required exact reproducibility of submissions.

---

## 4. If reusing these ideas elsewhere

- Start with: co-occurrence stats (forward/reverse/NPMI) + lexical/family
  token overlap + row-relative (z-score/rank) transforms + LightGBM
  lambdarank on 5-way groups. This alone reproduces the core of ranks 1, 2,
  and 5.
- Add SlateStats-style pairwise "who-beats-whom among candidates" features
  (rank 2) for a meaningful lift — it captures distractor-specific
  difficulty that plain co-occurrence misses.
- Add hard-example upweighting (rank 3) using a cheap proxy-vs-actual-label
  margin check — this was a bigger win than switching to fancier
  architectures.
- Treat GNN/MLM-pretraining-style additions (ranks 4/5) as a research
  direction to validate carefully via held-out ablation before trusting it
  to beat the simpler pipeline — in this competition it didn't.
