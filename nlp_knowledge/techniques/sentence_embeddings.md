# Sentence Embeddings

Sources: `sentence-transformers` v6 `[VERIFIED]`, `HKUNLP/instructor-embedding`
`[VERIFIED]`, `McGill-NLP/llm2vec` `[VERIFIED]`, `graykode/nlp-tutorial`
`[VERIFIED]`, `ElizaLo/Word Embeddings` `[VERIFIED]`.

## The generational map

| Gen | Examples | Status |
|---|---|---|
| Static word vectors | Word2Vec, GloVe, FastText | **legacy**, but not useless — see below |
| Averaged/pooled static | mean of GloVe vectors | weak; word-order blind |
| Early SBERT | `bert-base-nli-mean-tokens` | **deprecated by SBERT itself**; used in `SimonLaub` — do not copy |
| Modern SBERT | MiniLM, mpnet, `all-*` | strong, cheap default |
| Instruction-tuned | INSTRUCTOR, E5, BGE, GTE | current standard |
| LLM-derived | llm2vec, Qwen3-Embedding | highest quality, heaviest |

## Static vectors still have two honest uses

Do not dismiss GloVe/Word2Vec/FastText entirely:

1. **FastText for OOV and morphology.** Subword n-grams give a vector for words
   never seen in training — valuable for noisy, typo-heavy, agglutinative, or
   code-mixed text.
2. **They run anywhere.** No GPU, small memory, no model-hub access. In this
   container they are one of the few embedding options that could actually run
   (given a local vector file).

What they cannot do: word order, context-dependent sense, negation. "Dog bites
man" and "man bites dog" have identical mean vectors. If sense disambiguation is
part of the task, static vectors are disqualified.

## The prompt/instruction rule

Verified across three independent sources (INSTRUCTOR, llm2vec, and
`sentence-transformers`'s `query_prompt`/`corpus_prompt` parameters):

> The instruction/prefix is **part of the input** and changes the embedding.
> Asymmetric tasks (query↔document): **different** prompt per side.
> Symmetric tasks (STS, dedup): the **same** prompt on both sides.

llm2vec states it directly `[VERIFIED]`: *"While training, we provide
instructions for both sentences in symmetric tasks, and only for queries in
asymmetric tasks."*

Forgetting the prefix is a **silent** error — embeddings still come out,
cosines still rank, the numbers are just worse. Always check the model card's
exact prefix strings. `[UNVERIFIED]` for specific models: huggingface.co was
blocked this session.

## Pooling

- **Mean pooling** — the default, and the right default. (llm2vec defaults to
  mean.)
- **CLS pooling** — used by some models (BGE). Use whatever the model was
  trained with; mismatching pooling silently degrades quality.
- **Last-token pooling** — for decoder-derived models (Qwen, llm2vec variants).
- `sentence-transformers` handles this automatically when you load a published
  model. It only becomes your problem when you wire up a raw `transformers`
  model by hand — a common source of mysteriously bad embeddings.

## Normalization

Normalize to unit length → inner product equals cosine, and ANN indices behave.
`normalize_embeddings=True` in `encode()`. Be consistent between index time and
query time; mixing them is another silent failure.

## Matryoshka (MRL)

`MatryoshkaLoss` `[VERIFIED]` trains embeddings whose **prefixes** are valid
embeddings: truncate 1024-d → 256-d and keep most of the quality. Directly
useful under memory/latency constraints — index at 256-d, and if the metric
holds, you've cut the index 4x for free. Always measure the truncated quality;
the loss is small but not zero.

## Choosing a model (decision order)

1. **Can you download weights at all?** (Blocked here. Check first.)
2. **Language.** Non-English or mixed → multilingual model, full stop. A
   monolingual English encoder on Danish text still produces vectors that still
   rank — and mean much less (`SimonLaub` `[VERIFIED]`).
3. **Symmetric or asymmetric?** Picks the prompt convention.
4. **Sequence length.** Long documents need a long-context encoder or chunking.
   Check `max_seq_length` — silent truncation at 512 is the default in many
   setups.
5. **Budget.** Start small (MiniLM-class). Measure. Only go bigger if CV moves
   more than the noise band.

Bigger is not automatically better on a *specific* task: leaderboard averages
over many tasks say little about your domain. Measure on your CV.

## Using embeddings without fine-tuning

Often the best cost/benefit rung, and under-used:

- cosine as a **feature** in a GBDT (not as the final score) — see
  `semantic_similarity.md` rung 4;
- frozen embeddings + LogisticRegression for classification;
- embeddings from **two different model families** as separate features — they
  fail differently, which is what makes an ensemble work;
- kNN over training embeddings as a feature (distance to the k nearest training
  items per class).

## Fine-tuning

See the loss table in `semantic_similarity.md`. Practical notes:

- `MultipleNegativesRankingLoss` improves with batch size;
  `CachedMultipleNegativesRankingLoss` gets large effective batches without the
  VRAM `[VERIFIED]`.
- **Deduplicate before in-batch-negative training** — duplicates become false
  negatives.
- Use an in-loop evaluator shaped like the task (`InformationRetrievalEvaluator`,
  `EmbeddingSimilarityEvaluator`), not loss curves.
- Hard negatives: NV-Retriever settings in `information_retrieval.md`.
- Always compare against the **frozen** model on the same split. Fine-tuning on
  a few thousand pairs can easily make a strong pretrained encoder worse.
