# Information Retrieval

The PDF's weakest area: its entire IR content is one Wikipedia link to BM25
(`ElizaLo`), one toy cosine script (`justin-aj/ml-retrieval`), a dead framework
(`allennlp`), and one TF-IDF search notebook (`SimonLaub`). Everything below the
"what the PDF had" line is filled in from `sentence-transformers` `[VERIFIED]`
and standard practice.

## Frame the task correctly first

Retrieval is **not** classification over document IDs. `justin-aj/ml-retrieval`
`[VERIFIED]` demonstrates why: it trains a RandomForest with one document per
class and gets a top probability of 0.150 over 10 classes — barely above the
0.100 uniform prior. That framing cannot generalize to unseen documents, because
the label space *is* the corpus. If a Shipd task tempts you into per-document
classes, that is a red flag: score query-document **pairs** instead.

## The ladder

| # | Stage | Cost | Notes |
|---|---|---|---|
| 0 | Random / lexical-overlap ranking | ~0 | report it; it calibrates everything else |
| 1 | **BM25** | CPU, fast | the real baseline. Beats naive dense retrieval more often than people expect |
| 2 | TF-IDF cosine (word + char) | CPU, fast | cheap second lexical view |
| 3 | Dense bi-encoder retrieval | GPU for encode | needs HF access; **watch the prompt** |
| 4 | **Hybrid: BM25 + dense via RRF** | + | usually the biggest single jump |
| 5 | **Cross-encoder rerank of top-k** | GPU, O(k) per query | usually the second biggest. See `reranking.md` |
| 6 | Fine-tuned retriever with hard negatives | GPU, hours | late rung, needs labeled pairs |

**Rungs 4 and 5 are where the points are.** A cross-encoder reranking the top 50
of a *cheap* retriever generally beats a more expensive retriever with no
reranker, for less total compute. Spend there before upgrading the encoder.

## BM25

The PDF gives it one link. It deserves more: it is the strongest CPU-only
retrieval baseline, it needs no training, no GPU, and no model download — which
in a blocked-network environment makes it the **only** rung available above 2.

- `k1 ∈ [1.2, 2.0]` controls term-frequency saturation; `b ∈ [0.5, 0.9]`
  controls length normalization. **Tune them** — the defaults are not optimal
  for short documents, and this is a two-parameter sweep that costs nothing.
- Tokenization choices (lowercasing, stemming, stopwords) matter more for BM25
  than for dense retrieval. Ablate them.
- BM25 wins on: rare terms, exact identifiers, numbers, codes, names, jargon,
  and any domain the encoder never saw. It fails on synonymy and paraphrase.
- Implementations: `rank_bm25` (pure Python, fine up to ~10⁵ docs),
  `bm25s` (much faster), Lucene/Pyserini for real scale.

## Dense retrieval

- **Prompts/prefixes are mandatory** for E5/BGE/Qwen-style models and are the
  most common silent mistake. E5 needs `query: ` / `passage: `; BGE uses a query
  instruction for retrieval; Qwen3-Embedding uses instruct-style prompts.
  `[UNVERIFIED]` — huggingface.co was blocked this session; **read the model
  card before using one.** Asymmetric task → different prompt per side.
- Normalize embeddings, then inner product == cosine.
- **Chunking is a hyperparameter, not a detail.** Passage size and overlap
  change results more than swapping encoders. Sentence-level splitting (as in
  `SimonLaub/NLP_JobTrend`) fragments context; prefer overlapping windows of
  128–512 tokens, and **test at least two settings**.
- Watch the encoder's `max_seq_length` — silent truncation (llm2vec defaults to
  512) quietly discards the end of every long document.

## Hybrid: Reciprocal Rank Fusion

The best-value fusion, because it needs **no score normalization** — it uses
ranks, so BM25's unbounded scores and cosine's [-1,1] never have to be made
commensurable:

```
RRF(d) = Σ_over_systems  1 / (k + rank_system(d)),  k = 60 by convention
```

Implemented in `../code/retrieval.py`. Tune `k` and per-system weights on a
labeled query set. Score-level fusion (`α·bm25_norm + (1-α)·dense`) can beat RRF
but requires careful per-query normalization and is far more brittle —
try RRF first.

Hybrid works because the two arms fail on **different** queries: BM25 on
paraphrase, dense on rare literal tokens. Confirm that complementarity by
checking per-query win/loss, not just the mean metric — if one arm dominates on
every query, fusion will not help and you should spend the compute elsewhere.

## Metrics

Pick before you build: **Recall@k** (for a retriever feeding a reranker — only
recall matters at stage 1), **MRR@k** (one right answer), **nDCG@k** (graded
relevance), **MAP** (multiple relevant docs). Report the stage-1 recall ceiling
explicitly: *a reranker can never recover a document the retriever didn't
return.* If Recall@50 is 0.72, your pipeline's ceiling is 0.72, full stop.

## Validation

- Split by **query**, never by query-document pair — otherwise the same query's
  pairs land on both sides.
- If queries cluster by topic/entity/user, group on that too.
- **Unjudged documents are not negatives.** Treating "not in the qrels" as
  negative both trains on false negatives and under-reports your metric.
  `mine_hard_negatives(range_min=...)` exists to skip the top ranks for exactly
  this reason.

## Hard negative mining

The `sentence-transformers` NV-Retriever recipe `[VERIFIED]`, quoted from the
`mine_hard_negatives` docstring as its strongest setting (`TopK-PercPos (95%)`):

```python
dataset = mine_hard_negatives(
    dataset=dataset, model=model,
    relative_margin=0.05,     # negative must be >=5% below the positive's score
    num_negatives=10,         # "10 or less is recommended"
    sampling_strategy="top",
    range_min=10,             # skip the top ranks: they are probably positives
    use_faiss=True,
)
```

Negatives that are *too hard* are usually unlabeled positives, and training on
them actively damages the model. `relative_margin` and `range_min` are the two
guards against that. This is the most transferable single recipe in the entire
resource list.
