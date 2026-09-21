# Playbook: Retrieval

Document / passage / paragraph retrieval, and retrieval+X hybrids.

## 0. Audit

- Corpus size; document lengths; passage structure.
- Query count and length; are queries natural language or keywords?
- Metric: Recall@k / MRR / nDCG / MAP — and **at what k**?
- Are relevance labels graded or binary? Complete or sparse (pooled)?
- **Are unjudged documents negatives or unknown?** (They are unknown.)
- Language(s).
- Compute budget at inference — this caps the whole design.

## 1. Validation

Split by **query**. If queries cluster (topic, user, session), group on that
too. Never split by query-document pair.

## 2. Ladder

| Exp | Stage-1 | Stage-2 | Notes |
|---|---|---|---|
| R0 | random / overlap | — | the floor |
| R1 | **BM25** (tuned k1, b) | — | the real baseline; CPU-only |
| R2 | TF-IDF cosine (word + char) | — | second lexical view |
| R3 | dense bi-encoder | — | needs HF access; **check the prompts** |
| R4 | **BM25 + dense via RRF** | — | usually the biggest jump |
| R5 | R4 | **cross-encoder rerank top-k** | usually the second biggest |
| R6 | R4 | GBDT reranker on pair features | the CPU-only version of R5 |
| R7 | fine-tuned retriever (hard negatives) | R5 | last rung; expensive |

## 3. Instrument every stage, always

Report, for each configuration:

- **Recall@k of stage 1** — the ceiling. Nothing downstream can exceed it.
- The final metric after stage 2.
- Per-query results, not just the mean.

If recall is low → fix stage 1 (chunking, hybrid, prompts). If recall is high
and the final metric is low → fix stage 2. Without both numbers you are guessing
which half to work on, and that is the most common way retrieval work is wasted.

## 4. Chunking is a hyperparameter

Test at least: whole document · fixed windows (128/256/512 tokens) with 10–25%
overlap · natural units (paragraph/section). Chunking often moves the metric
more than swapping encoders. Decide it **before** embedding a large corpus —
re-embedding after a chunking change is where budgets die.

## 5. Efficiency

- Cache embeddings keyed by `(model_id, prompt, chunk_params, text_hash)` from
  the first run.
- Batch encoding; `normalize_embeddings=True` once, at index time.
- Exact search is fine up to ~10⁵–10⁶ vectors. Use FAISS/HNSW beyond that, and
  remember ANN is **approximate** — measure the recall you lose to it.
- BM25: `bm25s` over `rank_bm25` for anything non-trivial.

## 6. Shipd traps

- **Missing / incomplete reference documents** — the gold passage may simply not
  be in the corpus. Measure what fraction of queries have *any* retrievable gold
  document before blaming the model.
- **Distractor documents** deliberately similar to the gold.
- **Unanswerable queries** — does the metric reward abstaining?
- Duplicate documents with different ids — dedupe, or they crowd the top-k and
  suppress recall.
- Train/test query distribution differences (length, style, specificity).
