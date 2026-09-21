# Access Log — Phase 2

Honest record of what was actually reachable. Session date: 2026-09-21.

## Source document

`2f883d69-...pdf` — "NLP Resources", 3 pages. Extracted with `pypdf` 6.19.0.

- 4,053 characters of text, one glyph per line (the PDF was produced by a
  renderer that emits per-character text runs).
- **0 link annotations** (`/Annots` → `/A` → `/URI`).
- **0 `http(s)://` strings** in the raw bytes, and 0 after decompressing every
  `stream` object with zlib.

**Conclusion: the PDF contains no URLs.** Every resource below was resolved from
its title. Repo identity was confirmed with `git ls-remote` before cloning —
all 14 named GitHub repositories exist at the paths the PDF implies.

## Network policy

| Host | Result | Consequence |
|---|---|---|
| `raw.githubusercontent.com` | 200 | File reads OK |
| `github.com` (git protocol) | OK | `git clone` works |
| `github.com` (HTML) | 403 | No web browsing of repos |
| `api.github.com` | 403 (session repo scope) | No API metadata (stars, dates via API) |
| `www.kaggle.com` | 403 `connect_rejected` | **All Kaggle resources inaccessible** |
| `huggingface.co` | no route | No model cards, no weight downloads |
| `arxiv.org` | no route | No paper verification |
| `pypi.org` | OK (in `noProxy`) | Package installs work |

403s from the egress proxy are organization policy denials. Per the proxy README
these were **not retried or routed around**.

## GitHub resources — all 14 fetched and read

| PDF entry | Resolved to | Method | Size |
|---|---|---|---|
| huggingface/transformers | `huggingface/transformers` | sparse clone (`examples/pytorch/text-classification`, `docs/source/en/tasks`) | 4.4 MB |
| Snigdho8869/Natural-Language-Processing-NLP-Projects | same | full shallow clone | 30 MB |
| ElizaLo/NLP-Natural-Language-Processing | same | full shallow clone | 163 MB |
| TechNBusiness/Natural-Language-Processing | same | full shallow clone | 249 MB |
| UKPLab/sentence-transformers | same (repo now lives under `huggingface/`) | full shallow clone | v6.2.0.dev0 |
| SimonLaub/NLP_JobTrend | same | full shallow clone | 17 MB |
| reddy-123/sentence_similarity_using_Python | same | full shallow clone | 2.2 MB |
| justin-aj/ml-retrieval | same | full shallow clone | 248 KB |
| allenai/allennlp | same | sparse clone | 580 KB |
| sebastianruder/NLP-progress | same | full shallow clone | 1.7 MB |
| HKUNLP/instructor-embedding | same | full shallow clone | 359 MB |
| McGill-NLP/llm2vec | same | full shallow clone | 3.2 MB |
| Denis2054/Transformers-for-NLP-2nd-Edition | same | full shallow clone | 165 MB |
| graykode/nlp-tutorial | same | full shallow clone | 708 KB |

## Kaggle resources — 0 of 9 accessible `[BLOCKED]`

The PDF lists nine Kaggle entries. Eight are phrased as sections of a single
notebook ("Approaching NLP — <section>"), one as the full title:

1. Approaching (Almost) Any NLP Problem on Kaggle — Text Classification
2. Approaching NLP — Grid Search & Pipelines
3. Approaching NLP — Deep Learning (LSTM/GRU)
4. Approaching NLP — Word Vectors & Cosine Similarity
5. Approaching NLP — Ensembling for Similarity Tasks
6. Approaching NLP — TF-IDF Feature Extraction for Retrieval
7. Approaching NLP — Ensembling / Ranking Predictions
8. Approaching NLP — Word Vectors (GloVe/Word2Vec/FastText)
9. Approaching NLP — LSTM with Embeddings / GRU + Attention

**None of these were read.** `www.kaggle.com` is denied by egress policy, and no
mirror is reachable (huggingface, arxiv, and github web are all blocked too).

What is stated about them in this KB is limited to what their **titles** assert,
and is labelled as such. The techniques the titles name (TF-IDF retrieval,
word-vector cosine similarity, GRU+attention, blending) are documented here from
sources I *did* read, not from the notebooks. **Re-run Phase 2 for these nine
entries from an environment with Kaggle access before treating any of it as
"what that notebook does".**

## Mis-citations found in the PDF

Worth knowing so you don't hunt for content that isn't there:

- **`sebastianruder/NLP-progress — Ranking tasks`**: the repo has **no ranking
  task file**. `english/` contains 38 task files; none covers ranking or
  learning-to-rank. Closest: `semantic_textual_similarity.md`,
  `question_answering.md`, `word_sense_disambiguation.md`. Repo last updated
  2024-06-23.
- **`ElizaLo/... — Models and Algorithms`** cited under *Ranking*: that file
  covers Levenshtein, CRFs, Formal Concept Analysis, NMF, KL divergence, LSA,
  PLSA, LDA, Gibbs sampling. **Nothing about ranking.**
- **`allenai/allennlp — Retrieval-based QA`**: AllenNLP has been in maintenance
  mode since 2022 (banner in its README; last commit 2022-11-21). Retrieval QA
  models lived in the separate `allennlp-models` repo, not this one.
- **`reddy-123/sentence_similarity_using_Python`** is a partial copy of
  Microsoft's `nlp-recipes` (`utils_nlp/` + the sentence-similarity examples
  README), last commit 2021-03-07. Attribute accordingly.
- **`UKPLab/sentence-transformers`**: the canonical repo moved to
  `huggingface/sentence-transformers`. The UKPLab path still clones (redirect).
