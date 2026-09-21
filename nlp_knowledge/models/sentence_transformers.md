# sentence-transformers (the library)

`[VERIFIED]` — cloned at **v6.2.0.dev0** this session. The PDF cites it as
`UKPLab/sentence-transformers`; the canonical repo now lives at
`huggingface/sentence-transformers` (the UKPLab path redirects).

The most valuable resource in the entire PDF. Treat it as the default toolkit
for anything involving matching, retrieval, reranking, or similarity.

## Four model classes in v6

| Class | What it does | Use for |
|---|---|---|
| `SentenceTransformer` (bi-encoder) | one vector per text | retrieval, clustering, STS, features |
| `CrossEncoder` | joint (A,B) → score | reranking, pair classification |
| `SparseEncoder` | learned sparse (SPLADE-style) | lexical-aware dense retrieval |
| `MultiVectorEncoder` | late interaction (ColBERT/XTR-style) | high-recall retrieval, higher index cost |

The sparse and multi-vector classes are worth knowing about: SPLADE-style sparse
retrieval gives you the term-matching strengths of BM25 with learned expansion,
which is an alternative to running BM25+dense hybrid.

## Core APIs

```python
from sentence_transformers import SentenceTransformer, CrossEncoder, util

model = SentenceTransformer("...")                # download: BLOCKED here
emb   = model.encode(texts, normalize_embeddings=True,
                     batch_size=64, show_progress_bar=True,
                     prompt_name="query")          # prompts matter — see below
hits  = util.semantic_search(q_emb, corpus_emb, top_k=100)

ce     = CrossEncoder("...")
scores = ce.predict([(q, d) for d in candidates])
```

## Evaluators — use one, always

`InformationRetrievalEvaluator` · `RerankingEvaluator` ·
`EmbeddingSimilarityEvaluator` · `BinaryClassificationEvaluator` ·
`ParaphraseMiningEvaluator` · `TripletEvaluator` · `NanoBEIREvaluator`
(cheap generalization probe) · `CrossEncoderNanoBEIREvaluator`.

Training-loss curves do not tell you whether retrieval improved. **Always attach
a task-shaped evaluator** to the training loop. This is the library's most
important convention and the easiest to skip.

## `mine_hard_negatives`

The highest-value utility in the package. Converts `(anchor, positive)` into
`triplet` / `n-tuple` / `labeled-pair` / `labeled-list` — i.e. it determines
which losses are legal for your data.

Guards against mining false negatives: `range_min` (skip top ranks),
`max_score`, `absolute_margin`, `relative_margin`.

NV-Retriever recipe, verbatim from the docstring as its strongest setting
`[VERIFIED]`: `relative_margin=0.05`, `num_negatives<=10`,
`sampling_strategy="top"`, `use_faiss=True`.

Supports `query_prompt`/`corpus_prompt` — mine with the **same** prompts you
will use at inference, or the mined negatives don't match the deployed geometry.

## Loss selection

Full tables in `../techniques/semantic_similarity.md` (bi-encoder) and
`../techniques/ranking.md` (cross-encoder LTR). The library's own headline
guidance `[VERIFIED]`:

- `(anchor, positive)` pairs → **`MultipleNegativesRankingLoss`** ★ (InfoNCE);
  `Cached...` variant for large effective batch without the VRAM.
- `(a, b)` + float score → **`CoSENTLoss` / `AnglELoss`** are superior drop-in
  replacements for the traditional `CosineSimilarityLoss`.
- Cross-encoder `(q, d)` + label → **`BinaryCrossEntropyLoss`**, "very
  challenging to outperform".
- Cross-encoder `(q, [docs], [labels])` → **`LambdaLoss`** for LTR.

## Loss modifiers

`MatryoshkaLoss` (truncatable embeddings), `AdaptiveLayerLoss` (drop layers at
inference), `Matryoshka2dLoss` (both). Real levers when index size or latency is
constrained.

## Gotchas

- **Downloads are blocked in this container.** Every path above needs
  huggingface.co. Verify access before planning around this library.
- Prompt conventions are model-specific and silent when wrong.
- `max_seq_length` truncates silently — check it against your text lengths.
- In-batch-negative losses assume no duplicates in a batch. **Deduplicate first.**
- Pooling must match the published model's training configuration; loading via
  `SentenceTransformer` handles it, hand-wiring `transformers` does not.
