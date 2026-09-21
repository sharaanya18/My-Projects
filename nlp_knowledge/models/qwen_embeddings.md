# Qwen Embedding Models

> **Verification status.** `huggingface.co` **blocked** this session. All
> specifics below are `[UNVERIFIED]` model knowledge. Confirm on the model card.

## The idea

LLM-derived embeddings: a Qwen decoder LLM adapted into a text encoder
(the `Qwen3-Embedding` line, with matching `Qwen3-Reranker` models). This is the
same family of approach that `McGill-NLP/llm2vec` `[VERIFIED]` documents
explicitly — bidirectional attention + denoising + contrastive training over a
decoder base — and it is currently where the quality ceiling sits.

## Family (approximate, `[UNVERIFIED]`)

| Model | Params | Notes |
|---|---|---|
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | smallest; still large by encoder standards |
| `Qwen/Qwen3-Embedding-4B` | 4B | mid |
| `Qwen/Qwen3-Embedding-8B` | 8B | top quality, heavy |
| `Qwen/Qwen3-Reranker-{0.6B,4B,8B}` | — | matching cross-encoder rerankers |

Reported features `[UNVERIFIED]`: instruction-aware query prompts, **Matryoshka
(truncatable) dimensions**, long context, strong multilingual coverage.

## Usage notes

- **Instruction-style query prompts**, e.g.
  `"Instruct: <task description>\nQuery: <text>"` — documents unprefixed.
  `[UNVERIFIED]` — confirm the exact template; a wrong template is silent.
- **Last-token pooling** (typical for decoder-derived encoders), not mean.
  Loading via `sentence-transformers` handles this; hand-wiring does not.
- MRL: truncate the embedding to 256/512/1024 dims and re-measure. Often a
  4x index reduction for a small quality cost — verify on your own data.
- Normalize.

## When to choose Qwen embeddings

Late in the ladder, when **all** of these hold:

- cheaper rungs (BM25 → hybrid → reranker) have measurably plateaued,
- you have real GPU budget and HF access,
- the corpus is small enough that embedding it is affordable,
- the task is multilingual or semantically hard in a way lexical methods miss.

## When not to

- **Default case.** This is a heavy model. A `bge-reranker` over a BM25+dense
  hybrid usually beats a bigger retriever alone, for far less compute.
- CPU-only (unusable), tight latency, or very large corpora.
- Before instrumenting stage-1 recall — a better retriever cannot help if
  reranking is the bottleneck, and vice versa.

## Cost discipline

Before committing: embed 1,000 documents, time it, and extrapolate to the full
corpus **and** to any re-embedding you'll need after a chunking change. Chunking
changes are common; re-embedding a large corpus several times with an 8B model
is where experiment budgets die. Cache embeddings keyed by
`(model_id, prompt, chunk_params, text_hash)` from the very first run.
