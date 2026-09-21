# Semantic Similarity & Sentence-Pair Tasks

Covers STS (graded similarity), duplicate/paraphrase detection, sentence-pair
classification (NLI/entailment), and semantic matching.

Sources: `ElizaLo/Text Similarity` `[VERIFIED]`, `sentence-transformers`
`[VERIFIED]`, `reddy-123/` (= MS nlp-recipes) `[VERIFIED]`,
`SimonLaub/NLP_JobTrend` `[VERIFIED]`.

## First: which similarity problem is it?

These need different machinery. Getting this wrong is the most common error.

| Shape | Example | Right tool |
|---|---|---|
| Graded score 0–5 / 0–1 | STS-B | regression on pair features, or `CoSENTLoss` |
| Binary duplicate | Quora pairs | pair classifier, or `OnlineContrastiveLoss` |
| Pick the match from candidates | entity linking | ranking over candidates, not per-pair classification |
| Symmetric (A~B same type) | dedup, STS | same encoder & same prompt both sides |
| **Asymmetric** (query vs doc) | search, QA | **different prompt per side**; see `information_retrieval.md` |

Symmetric vs asymmetric is the decision that silently ruins runs: using one
prompt for both sides of an asymmetric task, or mining negatives without
prompts, costs real points with no error message. `mine_hard_negatives` exposes
`query_prompt` and `corpus_prompt` precisely for this.

## Metric choice, settled

From `ElizaLo/Text Similarity` `[VERIFIED]` — its reasoning is correct and worth
keeping:

- **Cosine over Euclidean for text.** Euclidean is length-sensitive and behaves
  badly on sparse high-dimensional text vectors; cosine is invariant to length,
  so a long document isn't penalized for being long.
- **Jaccard ignores repetition** — the right choice when repetition is noise
  (product descriptions, tag sets), the wrong choice when term frequency carries
  signal.
- **Distance → similarity: use `1/exp(d)`, not min-max.** Min-max normalization
  is outlier-sensitive: a few huge distances squash everything else. Useful
  whenever you turn a distance into a bounded model feature.
- Word Mover's Distance is principled but O(n³)-ish and rarely worth it now that
  sentence encoders exist.

## The ladder

| # | Approach | Notes |
|---|---|---|
| 0 | Length difference, token overlap only | the floor; sometimes shockingly strong |
| 1 | TF-IDF cosine (word + char n-gram) | fit on the *combined* corpus of both sides |
| 2 | Lexical feature block + GBDT | see below — usually the best cost/benefit rung |
| 3 | Off-the-shelf sentence embeddings + cosine | needs HF access |
| 4 | Embedding cosine **as a feature** alongside rung 2 | almost always > rung 3 alone |
| 5 | Fine-tuned bi-encoder (`CoSENT`/`MNRL`) | needs labels + GPU |
| 6 | Cross-encoder on the pair | ceiling for pair scoring; O(n) pairs only |

**Rung 4 is the point most people skip.** Cosine similarity from a frozen
encoder is *one number*. Feeding it into a GBDT alongside lexical features
routinely beats thresholding it directly, because the model learns when to trust
it.

## The lexical feature block

Ported from `ElizaLo`'s `percent_shared` `[VERIFIED]` and extended. Implemented
and tested in `../code/text_features.py`. For a pair (A, B):

- `|A∩B| / |A∪B|` (Jaccard), `|A∩B|` (raw overlap count)
- `|A∩B| / |A|` and `|A∩B| / |B|` — **asymmetric coverage, keep both**; the
  difference between them is itself informative (containment vs equality)
- same three on **bigrams** and on **character 4-grams**
- IDF-weighted overlap: sum of IDF of shared terms / sum of IDF of union
  (a shared rare term means far more than a shared stopword)
- length features: `len(A)`, `len(B)`, `|len(A)-len(B)|`, `len(A)/len(B)`
- normalized edit distance / longest common subsequence ratio
- counts of shared numbers, shared capitalized tokens, shared out-of-vocabulary
  tokens (numbers and proper nouns are high-signal, and embeddings handle them
  poorly)
- TF-IDF cosine, plus cosine on char n-grams

That block, plus LightGBM, is a serious baseline for any pair task and runs on
CPU in seconds.

## Losses for fine-tuning a bi-encoder

From the `sentence-transformers` loss table `[VERIFIED]` — match the loss to the
data you actually have:

| Your data | Loss |
|---|---|
| `(a, b)` + float score 0–1 | **`CoSENTLoss`** or `AnglELoss` (modern drop-ins that beat `CosineSimilarityLoss`) |
| `(anchor, positive)`, no labels | **`MultipleNegativesRankingLoss`** ★ (InfoNCE) |
| `(anchor, pos, neg)` triplets | `MultipleNegativesRankingLoss` ★ / `TripletLoss` |
| `(a, b)` + binary 0/1 | `ContrastiveLoss` / `OnlineContrastiveLoss` |
| `(a, b)` + class label | `SoftmaxLoss` (the original SBERT-NLI setup) |
| single texts + class | `BatchHardTripletLoss` family |

The library's own note: `CosineSimilarityLoss` is traditional, but `CoSENTLoss`
and `AnglELoss` are superior drop-in replacements. Prefer them.

**MNRL's hidden assumption:** every other in-batch item is a true negative. If
your data contains duplicates or near-duplicates, you are training the model to
push apart things that are actually the same. **Deduplicate before using
in-batch negatives** — this is a real, silent failure.

## Evaluating similarity properly

- STS → **Spearman** correlation (rank-based, robust to monotone rescaling).
  Pearson alone rewards a calibration you probably don't need.
- Binary duplicate → AUC for ranking quality, then F1 **at a tuned threshold**.
  Report both: a model can win on AUC and lose on F1 purely via threshold.
- **Group your CV folds by entity.** If sentence X appears in pairs (X,A) and
  (X,B), a random split puts one in train and one in valid and the model
  memorizes X. Use `GroupKFold` on a connected-component id over the pair graph.
  This is the #1 leakage source in pair tasks — see
  `../validation/leakage_detection.md`.

## Robustness probe (steal this)

From `SimonLaub/NLP_JobTrend` `[VERIFIED]`: run the same matching over original
text **and** over deliberately perturbed text, commit both result sets, and diff
them. If similarity rankings collapse under paraphrase, the method is riding
lexical overlap, not meaning — which matters enormously if Shipd's test set is
paraphrased or adversarial. Cheap to run, and it is the concrete form of the
Phase 13 robustness check.

Same repo, same lesson, different failure: a monolingual English encoder applied
to non-English text **still returns embeddings and still ranks them**. Check the
language of every field before trusting any embedding score.
