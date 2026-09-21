# Playbook: Ranking / Learning-to-Rank / Reranking

## 0. Audit

- What is the **group** key? (query / user / session / document)
- Group size distribution — fixed-size candidate lists or variable?
- Labels: binary or graded? What scale?
- Metric: nDCG@k / MRR / MAP / Recall@k — and is it averaged per group?
- Where do the candidates come from — given, or must you retrieve them?

If candidates must be retrieved, this is a retrieval task first. Do
`retrieval.md`, then come back.

## 1. Validation

`GroupKFold` on the **group key**. Compute the metric **per group, then
average** — a pooled global metric is a different number.

Report the **distribution** of per-group scores, not only the mean. A mean lifted
by a handful of easy groups is not an improvement, and bootstrapping over groups
gives you the confidence interval you need to tell the difference.

## 2. Ladder

| Exp | Approach |
|---|---|
| L0 | rank by a single lexical score (BM25) — the floor |
| L1 | pointwise: GBDT classifier on pair features, rank by probability |
| L2 | **+ within-group normalized features** (z-score, rank, score−max) |
| L3 | `LGBMRanker(objective="lambdarank")` |
| L4 | cross-encoder pointwise (`BinaryCrossEntropyLoss`) |
| L5 | cross-encoder listwise (`LambdaLoss` + `labeled-list` negatives) |
| L6 | rank-level ensemble (RRF) of the best two |

**L2 is the cheapest real gain.** Raw scores aren't comparable across groups;
normalized ones are. Add it before moving to a more complex objective.

L4 before L5: the library's own guidance is that pointwise
`BinaryCrossEntropyLoss` is "very challenging to outperform" `[VERIFIED]`.

## 3. LightGBM ranker traps (all silent)

- `group` = list of group **sizes in row order**; rows must be contiguous per
  group. Wrong → trains on nonsense groups without erroring.
- `label_gain` must match your label scale.
- Never shuffle rows after building `group`.

## 4. Features

From `../techniques/feature_engineering.md`, retrieval section. The high-value
ones: BM25, dense cosine, IDF-weighted overlap, **rank under each retriever**,
**reciprocal rank**, **score z-scored within group**, `score − max(group)`,
group size, agreement between retrievers.

## 5. Traps

- Optimizing logloss when the metric is nDCG.
- Splitting by row instead of by group (leakage).
- Unjudged candidates treated as negatives.
- Reranking a shortlist whose recall you never measured.
- Comparing to a baseline that used a different candidate set — fix the
  candidate generator before comparing rerankers, or you're comparing two
  things at once.
