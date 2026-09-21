# Ranking & Learning-to-Rank

The PDF cites two sources for ranking and **neither contains ranking content**
(verified): `sebastianruder/NLP-progress` has no ranking task file among its 38
English tasks, and `ElizaLo/Models and Algorithms` covers Levenshtein/CRF/NMF/
LSA/LDA. The one real ranking resource in the list is
`sentence-transformers`'s cross-encoder LTR losses `[VERIFIED]`.

## Recognizing a ranking task

You have a ranking task, not a classification task, when:

- rows are grouped — each **query/group** has several candidates,
- the metric is computed **per group then averaged** (nDCG, MRR, MAP, Recall@k),
- only the **order within a group** matters; absolute scores don't.

Consequence: a per-row classifier optimizing logloss is optimizing the wrong
thing. It spends capacity on being calibrated across groups when only
within-group order is scored.

**This changes validation**: fold on the **group**, never on the row. See
`../validation/cross_validation.md`.

## Three formulations

| | Trains on | Optimizes | Use when |
|---|---|---|---|
| **Pointwise** | one (q,d) at a time | per-item score | fastest, simplest; fine baseline; ignores group structure |
| **Pairwise** | (q, d⁺, d⁻) | P(d⁺ ranked above d⁻) | ordering matters, graded labels absent |
| **Listwise** | (q, [d₁..d_n], [s₁..s_n]) | the whole permutation | metric is nDCG; graded relevance available |

Start pointwise. Move up only when the group metric says so — pairwise/listwise
cost more data plumbing and are easier to get wrong.

## LightGBM `lambdarank` — the CPU workhorse

Since GPU/HF access is not guaranteed, this is the practical ranking rung:

```python
import lightgbm as lgb
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[5, 10],
    label_gain=[0, 1, 3, 7],   # MUST match your label scale (0..3 here)
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
)
ranker.fit(X_tr, y_tr, group=group_sizes_tr,       # counts per query, IN ORDER
           eval_set=[(X_va, y_va)], eval_group=[group_sizes_va],
           callbacks=[lgb.early_stopping(100)])
```

Three ways this goes wrong, all silent:

1. **`group` is a list of group *sizes* in row order**, not group ids. Rows must
   be sorted so each group is contiguous. Get this wrong and the model trains on
   nonsense groups without complaining.
2. **`label_gain` must cover your label range.** Default assumes `2^label - 1`
   gains; a label of 10 means a gain of 1023 and a wildly distorted objective.
3. Shuffling rows after building `group` destroys the alignment.

Features come from `feature_engineering.md`: BM25 score, dense cosine, lexical
overlap, plus **rank-based** features (rank of this doc under each retriever,
reciprocal rank, score minus the group's max score, score z-scored within
group). Within-group normalized features are the ones that matter most — they
make scores comparable across queries of different difficulty.

## Neural LTR: cross-encoder listwise losses

`sentence-transformers` cross-encoder losses `[VERIFIED]`, for
`(query, [doc1..docN])` with `[score1..scoreN]`:

`LambdaLoss` · `PListMLELoss` · `ListNetLoss` · `RankNetLoss` · `ListMLELoss` ·
`ADRMSELoss`

The library's own guidance `[VERIFIED]`: take `(anchor, positive)` pairs,
run `mine_hard_negatives(output_format="labeled-list")`, then **`LambdaLoss` is
the frequently-used choice for learning-to-rank**; with
`output_format="labeled-pair"`, `BinaryCrossEntropyLoss` "remains very
challenging to outperform".

That last clause is the important one: **a pointwise BCE cross-encoder is a very
strong reranker.** Do not jump to listwise losses assuming they are better —
measure.

## Ensembling rankings

Combine **ranks**, not scores, unless you have carefully normalized per query:

- **RRF** (`1/(k+rank)`, k=60) — robust, no normalization needed, the default.
- Rank averaging / Borda count — similar, sensitive to list length.
- Weighted score blending — needs per-query normalization (z-score or min-max
  *within* each group); more powerful, more brittle.

The `justin-aj/ml-retrieval` repo `[VERIFIED]` does apply
`scipy.stats.rankdata` + percentile conversion to make two incomparable scorers
(RF probabilities, cosine) comparable — that conversion instinct is right even
though the experiment around it isn't.

## Evaluating

- Compute the metric **per group, then average**. A global metric over pooled
  rows is a different (wrong) number.
- Report the **retrieval recall ceiling** alongside the ranking metric: the
  reranker cannot recover what stage 1 missed.
- Groups vary wildly in difficulty. Report the **distribution** of per-query
  nDCG, not just the mean — a mean improvement driven by 5 easy queries is not
  an improvement.
- Bootstrap over queries for a confidence interval before believing a small gap.
