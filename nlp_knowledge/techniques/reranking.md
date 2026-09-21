# Reranking

The single highest-value technique in this KB that the PDF barely mentions
(one line: "UKPLab/sentence-transformers — Cross-Encoder Re-ranking").

## Why it works

A **bi-encoder** embeds query and document *independently* — it must compress
each document into a fixed vector before ever seeing the query. Fast (embed the
corpus once), but information is lost at compression time.

A **cross-encoder** feeds `[query, document]` through the transformer *together*,
so every query token can attend to every document token. Far more accurate, and
O(n) forward passes per query — unusable for retrieval over a corpus, ideal over
a shortlist.

Hence the cascade (`sentence-transformers` `retrieve_rerank` docs `[VERIFIED]`):
retrieve ~100 candidates cheaply, rerank them expensively, return top 5–10.

## The rule that decides everything

> **The reranker's ceiling is the retriever's Recall@k.**

Always measure both. If Recall@100 = 0.85, no reranker gets you past 0.85.
When the pipeline underperforms, diagnose which stage is at fault before
touching either:

| Recall@k | Rerank metric | Diagnosis |
|---|---|---|
| low | low | fix the **retriever** (hybrid, chunking, prompts) |
| high | low | fix the **reranker** (fine-tune, bigger model, more k) |
| high | high | increase k for more headroom, or stop |

## Choosing k

k trades recall ceiling against latency (and against reranker precision — a
weak reranker over a longer list can *hurt*). Sweep k ∈ {10, 20, 50, 100} and
plot final metric vs k; it usually saturates. Pick the knee, not the max.

## Training a reranker

From the cross-encoder loss table `[VERIFIED]`:

- `(query, doc)` + binary/float label → **`BinaryCrossEntropyLoss`**. The
  library's own note: "a traditional option that remains very challenging to
  outperform." **Start here.**
- `(query, [docs], [labels])` → `LambdaLoss` for listwise LTR.
- Distillation: `MarginMSELoss` / `DistillKLDivLoss` to compress a big reranker
  into a fast one — the right move when inference budget is capped.

**Train the reranker on the retriever's actual output distribution.** A reranker
trained on random negatives sees easy negatives, then meets the retriever's hard
top-100 at test time and fails. Mine negatives *from your own stage-1*. This is
the single most common reranking mistake.

## Cheap rerankers when there is no GPU

Reranking is not only neural. When HF/GPU is unavailable (as in this container),
rerank the shortlist with:

- a **GBDT over pair features** (`feature_engineering.md`) — BM25, TF-IDF
  cosine, overlap, IDF-weighted overlap, length ratios, position/rank features;
- exact-match and entity-overlap boosts;
- per-query score normalization + a small learned blend.

This is a genuine reranker, trains in seconds on CPU, and often recovers a
solid fraction of the neural gain.

## Don't rerank when

- the corpus is small enough to score every pair directly (just cross-encode all);
- the metric only rewards top-1 and stage 1 already nails top-1;
- stage-1 recall is the bottleneck (fix that first);
- latency/compute budget makes it unusable at submission time — check the Shipd
  constraints *before* building the cascade.
