# BGE Embeddings (BAAI General Embedding)

> **Verification status.** `huggingface.co` **blocked** this session. Everything
> below is `[UNVERIFIED]` model knowledge. Confirm on the model card before use.

## The idea

BAAI's general-purpose embedding family. Like E5, it uses an instruction
convention — but **asymmetrically**: an instruction on the **query** side only
for retrieval, and none on the document side.

## Family (approximate, `[UNVERIFIED]`)

| Model | Dim | Notes |
|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | fast |
| `BAAI/bge-base-en-v1.5` | 768 | good default |
| `BAAI/bge-large-en-v1.5` | 1024 | stronger |
| `BAAI/bge-m3` | 1024 | **multilingual, long context (~8k), and produces dense + sparse + multi-vector (ColBERT) representations from one model** |
| `BAAI/bge-reranker-{base,large,v2-m3}` | — | **cross-encoder rerankers** |

Two things in that table are worth more than the rest:

- **`bge-m3` is a hybrid retriever in a single model.** It emits dense, learned
  sparse, and multi-vector outputs together, so you get the BM25-style term
  matching and the dense semantics from one forward pass — a strong answer to
  the hybrid problem in `../techniques/information_retrieval.md`, with long
  context and multilingual coverage as well.
- **`bge-reranker-*` gives you the stage-2 model from the same family**, which
  is the cheapest route to a complete retrieve→rerank cascade.

## Usage notes

- Query instruction for **retrieval** (v1.5, English), documents unprefixed:
  `"Represent this sentence for searching relevant passages: "`.
  `[UNVERIFIED]` — confirm the exact string; it changed between versions.
- For **symmetric** tasks (STS, dedup), use **no** instruction on either side.
- BGE v1.5 uses **CLS pooling**, not mean pooling — a real difference from E5,
  and a source of quietly bad embeddings if you hand-wire the model instead of
  loading it through `sentence-transformers`.
- Normalize; cosine similarity.

## When to choose BGE

- You want dense + sparse + reranker from one family (`bge-m3` + `bge-reranker`).
- Long documents (`m3`'s extended context avoids aggressive chunking).
- Multilingual retrieval.
- You want a strong general default and don't want to run a model bake-off.

## When not to

- CPU-only at scale.
- When `bge-m3`'s size/latency exceeds the submission budget — measure inference
  cost against the Shipd constraints before committing.
- Before a BM25 baseline exists to compare against.

## Checklist

1. Confirm the instruction string and the pooling mode on the model card.
2. Instruction on queries only (asymmetric); none (symmetric).
3. Normalize.
4. If using `m3`, decide explicitly which of its three outputs you're using —
   and consider fusing them (that is the model's main selling point).
5. Pair with `bge-reranker-*` before reaching for a larger retriever.
