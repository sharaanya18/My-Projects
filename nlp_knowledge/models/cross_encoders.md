# Cross-Encoders

## Mechanism

`[CLS] query [SEP] document [SEP]` → transformer → single relevance score.
Full cross-attention between query and document tokens.

Versus a bi-encoder, which embeds each side independently: the bi-encoder must
compress the document into a fixed vector **before seeing the query**, so
information is lost at compression time. The cross-encoder never compresses.
That is the whole accuracy difference.

Cost: **one forward pass per pair**. No precomputed index is possible. Hence
shortlists only.

## When to use

- Reranking the top-k of a retriever (the canonical use).
- Pair classification / NLI / duplicate detection where n is bounded.
- Scoring a small candidate set per row (e.g. multiple-choice, entity linking
  against a short candidate list).

## When not to

- Retrieval over a large corpus (O(N) per query — use a bi-encoder).
- Anything needing a reusable document vector (clustering, dedup at scale, ANN).
- Tight latency budgets without distillation.
- **Before you know your stage-1 recall.** The reranker cannot exceed it.

## Training

```python
from sentence_transformers import CrossEncoder
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
```

Loss by data shape `[VERIFIED]`:

| Data | Loss |
|---|---|
| `(q, d)` + 0/1 or float | **`BinaryCrossEntropyLoss`** — start here |
| `(q, d)` + class | `CrossEntropyLoss` (`num_labels=num_classes`) |
| `(q, pos, neg...)` | `MultipleNegativesRankingLoss` |
| `(q, [docs], [labels])` | `LambdaLoss`, `PListMLELoss`, `ListNetLoss`, `RankNetLoss` |
| distillation from a bigger reranker | `MarginMSELoss`, `DistillKLDivLoss` |

**The rule that matters most:** train on negatives from **your own stage-1
retriever**, not random negatives. A reranker trained on easy negatives meets
the retriever's hard top-100 at test time and collapses. Use
`mine_hard_negatives` with the retriever you will actually deploy.

## Evaluation

`CrossEncoderRerankingEvaluator` / `CrossEncoderNanoBEIREvaluator`
`[VERIFIED, present in the tree]`. Report the **delta over stage-1 ordering**,
not the absolute number — that delta is the only thing the reranker is
responsible for.

## No-GPU alternative

A GBDT over pair features (BM25, TF-IDF cosine, IDF-weighted overlap, length
ratios, within-group rank/z-score features) is a legitimate reranker. Trains in
seconds on CPU, needs no model download, and captures a real share of the gain.
In this container it is the only reranker available. See
`../techniques/reranking.md`.

## Distillation

If a large reranker wins but is too slow for the submission budget: train a
small cross-encoder on the big one's scores with `MarginMSELoss` /
`DistillKLDivLoss`. Usually retains most of the quality at a fraction of the
cost — and it is the right answer to "the good model is too slow", rather than
abandoning reranking.
