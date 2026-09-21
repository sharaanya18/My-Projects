# Resource Index — Phases 1 & 2

Every resource in the PDF, classified, accessed, and assessed.

Categories used: `TC` text classification · `SS` semantic similarity ·
`IR` information retrieval · `RANK` ranking/LTR · `EMB` sentence embeddings ·
`TRF` transformers · `QA` question answering · `RR` retrieval+rerank ·
`ENS` ensembling · `FE` feature engineering · `VAL` validation.

Verdict key: **MINE** = extract ideas, modernize · **USE** = usable as-is ·
**REF** = reference/lookup only · **SKIP** = no transferable value for Shipd.

---

## 1. `huggingface/sentence-transformers` (PDF: `UKPLab/sentence-transformers`)

`[VERIFIED]` — cloned at `v6.2.0.dev0`. Cited in the PDF under SS, RANK, EMB.

**Categories:** EMB, SS, IR, RANK, RR — the single most valuable resource in the list.

**Problem it solves.** Producing and fine-tuning text embeddings (bi-encoders),
cross-encoder rerankers, sparse encoders, and multi-vector (late-interaction)
encoders, with matching losses, evaluators, and mining utilities.

**Core approach.** Siamese/triplet fine-tuning of a pretrained encoder with a
loss chosen to match your *data format*. The library's central insight — and the
one thing to internalize — is the **loss ↔ data-format table**, reproduced in
`techniques/sentence_embeddings.md`.

**Algorithms/models present in the tree (verified by listing the source):**

- Bi-encoder losses: `MultipleNegativesRankingLoss` (InfoNCE / in-batch negatives),
  `CachedMultipleNegativesRankingLoss`, `CoSENTLoss`, `AnglELoss`,
  `CosineSimilarityLoss`, `ContrastiveLoss`, `OnlineContrastiveLoss`,
  `TripletLoss`, `Batch{All,Hard,SemiHard,HardSoftMargin}TripletLoss`,
  `GISTEmbedLoss`, `MarginMSELoss`, `DistillKLDivLoss`, `MatryoshkaLoss`,
  `AdaptiveLayerLoss`, `DenoisingAutoEncoderLoss` (TSDAE), `ContrastiveTensionLoss`.
- Cross-encoder losses: `BinaryCrossEntropyLoss`, `CrossEntropyLoss`,
  `MultipleNegativesRankingLoss`, and the **learning-to-rank family**:
  `LambdaLoss`, `ListNetLoss`, `ListMLELoss`, `PListMLELoss`, `RankNetLoss`,
  `ADRMSELoss`, `MarginMSELoss`.
- Evaluators: `InformationRetrievalEvaluator`, `RerankingEvaluator`,
  `EmbeddingSimilarityEvaluator`, `BinaryClassificationEvaluator`,
  `ParaphraseMiningEvaluator`, `TripletEvaluator`, `NanoBEIREvaluator`.
- `sentence_transformers.util.mine_hard_negatives` — full hard-negative miner.

**Feature engineering.** Not feature-based; the useful transferable artifacts are
(a) the retrieve→rerank pipeline and (b) hard-negative mining parameters.

**Validation strategy.** Evaluators run inside the training loop on a held-out
split; `NanoBEIR` gives a cheap generalization probe. *This is the right pattern:
a task-shaped evaluator, not loss curves.*

**Ensemble strategy.** Two-stage cascade (bi-encoder recall → cross-encoder
precision), which is an ensemble in everything but name and is almost always the
highest-value "ensemble" in a retrieval task.

**Implementation patterns worth stealing:**

1. **NV-Retriever hard-negative recipe**, verbatim from
   `sentence_transformers/util/hard_negatives.py` docstring — the settings the
   library calls its strongest (`TopK-PercPos (95%)`):
   `relative_margin=0.05`, `num_negatives<=10`, `sampling_strategy="top"`,
   `use_faiss=True`. `range_min` skips the top ranks to avoid mining unlabeled
   positives as negatives. This is the most reusable single fact in the repo.
2. `mine_hard_negatives(..., output_format=...)` converts `(anchor, positive)`
   into `triplet` / `n-tuple` / `labeled-pair` / `labeled-list` — i.e. it
   *changes which losses are legal for your data*. Use it to move between rungs.
3. `CachedMultipleNegativesRankingLoss` decouples effective batch size from GPU
   memory. In-batch-negative losses improve with batch size; this is how you get
   it without the VRAM.

**Strengths.** Actively maintained, correct, batteries-included, and the loss
table removes most of the guesswork in a matching task.

**Weaknesses.** Every path assumes you can download pretrained weights
(**blocked in this container**). Fine-tuning assumes a GPU. In-batch-negative
losses silently assume no duplicate/near-duplicate positives inside a batch —
with a deduplicated-poorly dataset they train against false negatives.

**Use when:** any pairing, matching, retrieval, reranking, STS, or duplicate
task, *and* you have GPU + model access.
**Do NOT use when:** ≤ a few thousand labeled pairs with a narrow vocabulary
(TF-IDF/BM25 often matches it), CPU-only, or when the metric is dominated by
exact lexical overlap.
**Modernization:** none needed — this *is* the modern baseline. **USE.**

---

## 2. `huggingface/transformers` — text-classification examples

`[VERIFIED]` — sparse clone of `examples/pytorch/text-classification`.

**Categories:** TC, TRF.

**Problem/approach.** Reference fine-tuning scripts: `run_glue.py` (9 GLUE
tasks), `run_classification.py` (arbitrary single-/multi-label CSV/JSON),
`run_glue_no_trainer.py` (bare Accelerate loop), `run_xnli.py` (multilingual).

**Patterns worth stealing:**

- `run_classification.py --text_column_name "review_title,review_body,product_category"
  --text_column_delimiter "\n"` — **multi-field text concatenation as a first-class
  feature**. The README reports 0.5958 acc from `review_body` alone vs **0.659**
  from title+body+category on `amazon_reviews_multi`, 1 epoch. That is a large,
  free gain from feature construction, not modeling — exactly the cheap win to
  test early on a Shipd task with metadata columns.
- `--ignore_mismatched_sizes` for reusing a checkpoint with a different head.
- Multi-label is the same script with `--metric_name f1` and a list-valued label
  column (`reuters21578` → micro-F1 ≈ 0.82).
- `--fp16` table: ~2x speedup at equal accuracy across all 9 GLUE tasks. Free.

**Validation strategy in the examples.** Single train/dev split, seeded. The
README itself warns its dev numbers differ materially from the GLUE test set —
a built-in reminder that a single split over-reports.

**Strengths.** Canonical, correct, maintained, trivially adapted to a CSV.
**Weaknesses.** No CV, no OOF, no ensembling, no threshold tuning, no class
weighting — everything that wins competitions is left to you. Defaults
(`lr=2e-5`, 3 epochs, `max_seq_length=128`) are a *starting* point; 128 tokens
silently truncates long documents.
**Use when:** you need a correct fine-tuning loop fast, with GPU access.
**Do NOT use when:** CPU-only, or before a TF-IDF baseline exists.
**Modernization:** current. **USE** (as a template, then add CV/OOF yourself).

---

## 3. `Snigdho8869/Natural-Language-Processing-NLP-Projects`

`[VERIFIED]` — 7 notebooks read, `Multiclass_Text_Classification.ipynb` read in full.

**Categories:** TC, ENS.

**Problem/approach.** Seven self-contained classification projects (BBC topics,
spam, IMDb sentiment, movie genre, language ID, suicidal-ideation detection,
summarization). Each sweeps ~13 models over one dataset.

**What it actually does** (verified from the BBC notebook):
`clean_text()` (lowercase → strip non-alpha → drop URLs → `RegexpTokenizer` →
curated stopword list that **keeps negations** → drop 1-char tokens →
`WordNetLemmatizer`) → `train_test_split(test_size=0.2, random_state=42)` →
`CountVectorizer(ngram_range=(1,2))` + `TfidfTransformer(norm='l2', sublinear_tf=True)`
fitted on train only → `GridSearchCV(cv=5)` per model → accuracy on the test split.

**The one genuinely good idea here:** the stopword list is *subtractive* — it
removes sklearn/NLTK stopwords **except** negations and modals (`not`, `no`,
`don't`, `won't`, `couldn't`, `shouldn't`, …). Blanket stopword removal destroys
negation, which matters for sentiment and entailment. Worth copying.
`sublinear_tf=True` is also the right default and is often omitted.

**Reported results (BBC, n=2225):** LR 95.96 · LinearSVC 96.40 · MNB 96.18 ·
RF 94.83 · GBC 94.38 · **soft-vote ensemble 96.40** · AdaBoost 94.16 ·
LSTM 92.58 · GRU 91.24 · CNN 95.06 · hybrid 91.46 · **BERT 98.20** · XLNet 97.75.

**Read those numbers correctly — this is the lesson, not the leaderboard:**

- **The ensemble did not beat its best member** (96.40 = LinearSVC exactly).
  Weights `[1,2,3,4]` were hand-picked, never tuned. Evidence for
  `techniques/ensembling.md`: soft-voting correlated TF-IDF linear models on a
  small clean dataset buys nothing.
- **Every number is one 80/20 split** of 2,225 rows (~445 test docs). One
  document ≈ 0.22 points. The 0.44-point gap between LinearSVC and MNB is ~2
  documents — **noise**. Choosing a model on this is model selection on the test set.
- 13 models were compared on the **same** test split, so the reported max is
  upward-biased.
- The BERT/XLNet numbers come from `learner.validate(val_data=test_data)` where
  `test_data` is the same split used for training feedback.
- Classical models sat at 94–96 while BERT reached 98 — on 2,225 clean,
  well-separated BBC articles. Do **not** generalize "BERT is +2 points" from this.

**Strengths.** Broad, readable, honest about its recipe; the negation-preserving
cleaner is reusable.
**Weaknesses.** Single-split evaluation throughout; no OOF; no calibration;
no per-class error analysis; `ktrain` (TF/Keras) is a dependency-heavy dead end.
**Use when:** you want a menu of classical classifiers to port.
**Do NOT use when:** you need trustworthy model comparison — the protocol
cannot support it.
**Modernization:** replace `CountVectorizer+TfidfTransformer` with
`TfidfVectorizer(sublinear_tf=True)`; replace single split with
`StratifiedKFold` + OOF; replace `ktrain` with `transformers` + `Trainer`;
drop AdaBoost/GBC on sparse TF-IDF (dominated by LinearSVC at a fraction of the
cost). **MINE.**

---

## 4. `ElizaLo/NLP-Natural-Language-Processing`

`[VERIFIED]` — cloned; read `Text Similarity/`, `Text Search and Information
Retrieval/`, `Models and Algorithms/`, `Transformers/`, `Transfer Learning/`,
`Question Answering System/`, and the root README.

**Categories:** TC, SS, IR, RANK(mis-cited), EMB, TRF, QA — cited 6× in the PDF,
more than any other resource.

**What it is.** A large curated *awesome-list* (39 top-level topic folders) with
prose, formulas, images, and short code snippets. **Not a codebase.** The PDF
cites it as if each section were an implementation; most sections are link tables.

**Section-by-section reality check:**

| PDF cites it for | What is actually there |
|---|---|
| Text Classification | README + one "Fine Tune BERT with TensorFlow" folder |
| **Text Similarity** | **15 KB README, genuinely good** — see below |
| Information Retrieval | 2.2 KB README: one BM25 Wikipedia link, Typesense, AWS Kendra/Comprehend. Thin. |
| Ranking → "Models and Algorithms" | Levenshtein, CRF, FCA, NMF, KL, LSA, PLSA, LDA, Gibbs. **No ranking content.** |
| Word Embeddings | 45 KB README + `Word_embeddings(Non-semantic).ipynb` + TF-IDF folder. Substantial. |
| Transformers | 708-byte README: 3 external links. Effectively empty. |
| Transfer Learning | ULMFiT paper + fast.ai tutorial link. |
| QA | Pointer to the author's separate BiDAF-on-SQuAD repo + a PDF write-up. |

**The Text Similarity README is the real asset.** It is a clear, correct
decision guide over Jaccard / Euclidean / cosine / Word Mover's Distance /
Tanimoto / STS, with runnable snippets, and it reaches the right conclusions:

- *"Euclidean distance doesn't work well with the sparse vectors of text
  embeddings. So cosine similarity is generally preferred over Euclidean when
  working with text data."* — correct, and the reason (length invariance) is
  explained with a worked example.
- Jaccard ignores repetition → right choice when repetition is noise (product
  descriptions), wrong when term frequency carries signal.
- Distance→similarity via `1/exp(d)` rather than min-max, because min-max is
  outlier-sensitive. A genuinely useful trick for turning a distance feature
  into a bounded one.
- The `percent_shared` helper (shared / diverging / union word counts, plus
  asymmetric coverage of each side) is a **ready-made lexical feature block** for
  sentence-pair tasks. Port it directly into `code/text_features.py`.

**Strengths.** Broad map of the field; the similarity guide is better than most
blog posts; useful for "what am I forgetting?" sweeps.
**Weaknesses.** Link rot risk; depth is wildly uneven; several sections cited by
the PDF are near-empty; spaCy `en_core_web_sm` similarity (used in its snippets)
uses **context-insensitive tagger vectors, not real embeddings** — never use
`nlp(...).similarity()` from the `sm` model as a semantic feature.
**Use when:** scoping a task type, or picking a similarity metric.
**Do NOT use when:** you need working training code.
**Modernization:** the similarity-metric reasoning is timeless; replace its
spaCy/Word2Vec vector examples with a sentence-transformer, keep the metric
guidance. **MINE (Text Similarity + Word Embeddings) / REF (rest).**

---

## 5. `TechNBusiness/Natural-Language-Processing`

`[VERIFIED]` — cloned; read `week8_text_classification.ipynb` and the file tree.

**Categories:** TC, EMB.

**What it is.** An 11-week university workshop: rule-based → pipelines →
text representation → BoW classifier → tokenization → word embeddings →
classification → LDA/NMF/BERTopic → generation/summarization. Ships datasets
(`bbc.zip`, `aclImdb.zip`, `spam_text.csv`, `webOfScience.xls`, tweets).

**Core approach (week 8).** TensorFlow Hub `nnlm-en-dim50/2` as a trainable
`KerasLayer` → Dense(16) → Dense(1) on IMDb, 40 epochs, batch 512, 10k/15k
train/val split.

**Verified defect — do not copy.** The model has a **single logit** output
(`Dense(1)`, `BinaryCrossentropy(from_logits=True)`), but prediction does
`classes_x = np.argmax(results_pred, axis=1)` over a shape-`(n,1)` array.
`argmax` over a length-1 axis is **always 0**. The prediction path is broken and
silently returns all-negative. A good reminder to always assert on the shape and
class balance of your prediction vector before submitting.

Also: 40 epochs with no `EarlyStopping` and no checkpoint restore — the final
weights are whatever epoch 40 gave, not the best epoch.

**Strengths.** Clean progression; bundled datasets are handy for smoke tests.
**Weaknesses.** TF/Keras + TF-Hub stack (TF Hub `tfhub.dev` URLs are themselves
now legacy); broken prediction path; no CV; 249 MB mostly of data and a video.
**Use when:** you want a teaching-order reference or a quick local dataset.
**Do NOT use when:** building anything for a leaderboard.
**Modernization:** the whole TF-Hub NNLM path is superseded by a sentence
embedding + linear head, which is both stronger and cheaper. **SKIP** for
modeling; **REF** for its datasets.

---

## 6. `SimonLaub/NLP_JobTrend`

`[VERIFIED]` — cloned; read `SentenceTransformerColabExampleEn.ipynb` and
`Prototype/PersonalCompetencies_TFIDF_Search.ipynb`.

**Categories:** SS, IR.

**Problem it solves.** An applied end-to-end case: mine Danish job ads, match
free-text competency descriptions against ad text, compare **TF-IDF search vs
transformer search** on the *same* corpus — the CSV artifacts of both runs are
committed side by side (`JobTrendAds_TFIDF_Search.csv` vs
`JobTrendAds_Transformer_Search.csv`, plus `..._ChangedJobText*` variants).

**The transferable idea — and it is a good one.** This is a **perturbation
test**: run the same search over original text *and* over deliberately modified
job texts, keep both outputs, and compare. That is exactly the robustness probe
Phase 13 asks for — does the retrieval method survive paraphrase, or is it
riding surface lexical overlap? Committing both result sets makes the comparison
auditable. Steal the *protocol*, not the code.

The `JobTrendPrototype4ROC.ipynb` file shows ROC analysis was applied to the
matching decision — i.e. threshold selection was treated as its own problem,
which is correct for a similarity-threshold task.

**Verified defects — do not copy:**

- Uses `SentenceTransformer('bert-base-nli-mean-tokens')`, which SBERT
  **officially deprecated** for producing low-quality embeddings. Any current
  MiniLM/mpnet/BGE/E5 model beats it substantially.
- Applies `jieba.lcut()` — a **Chinese** word segmenter — to Danish/English text
  in the TF-IDF pipeline. On non-Chinese input this does not do what the author
  thinks it does.
- The author's own note is nonetheless the most valuable line in the repo:
  *"In danish this is not as good as the model is not trained on danish texts."*
  **Monolingual-English encoders on non-English text is a silent failure mode**
  — the embeddings still come out, similarities still rank, the scores just mean
  much less. Check language before trusting any embedding score.
- Retrieval granularity is sentence-level via `nltk.sent_tokenize` with no
  passage windowing or overlap.

**Strengths.** Realistic messy-data pipeline; TF-IDF-vs-dense compared on one
corpus; perturbation artifacts committed; explicit multilingual failure note.
**Weaknesses.** Deprecated model; wrong tokenizer; Colab `files.upload()`
everywhere; no metric-driven evaluation of the two search methods against
labels.
**Use when:** you need a template for "compare lexical and dense retrieval on my
own corpus and probe robustness".
**Do NOT use when:** you need the model choice or the preprocessing.
**Modernization:** swap `bert-base-nli-mean-tokens` → a current multilingual
encoder (`multilingual-e5-*`, `bge-m3`, `paraphrase-multilingual-mpnet-base-v2`);
delete jieba; add BM25 as the lexical arm; score both against labels with
nDCG/Recall@k instead of eyeballing. **MINE (protocol only).**

---

## 7. `reddy-123/sentence_similarity_using_Python`

`[VERIFIED]` — cloned; README and `utils_nlp/` read. Last commit 2021-03-07.

**Categories:** SS, QA, VAL.

**What it actually is.** A partial copy of **Microsoft `nlp-recipes`** — the
`utils_nlp/` package plus the sentence-similarity examples README. Attribute it
to Microsoft, not to the repo name in the PDF.

**Core approach.** Its stated two-step framing is still the correct mental model:
*(1) obtain sentence embeddings, (2) take cosine similarity*; and its README
correctly notes these scores feed "search/retrieval, nearest-neighbor or
kernel-based classification, recommendations, and **ranking**" — i.e. one
similarity model can serve several Shipd problem families.

**The genuinely reusable assets (still valid in 2026):**

- `utils_nlp/eval/evaluate_squad.py` — the **canonical SQuAD `normalize_answer`
  / `f1_score` / `exact_match_score` / `metric_max_over_ground_truths`**
  implementation (lowercase → strip punctuation → drop articles → collapse
  whitespace; token-level F1; max over gold answers). If a Shipd task is
  extractive QA or short-answer matching, **use this exact normalization** —
  hand-rolled string comparison is the classic source of a wrongly-low score.
- `utils_nlp/eval/classification.py` — `eval_classification`,
  `compute_correlation_coefficients`, `plot_confusion_matrix`.
- `utils_nlp/eval/senteval.py` — SentEval harness config, useful as a
  "evaluate an embedding on many downstream probes" pattern.

**Weaknesses.** Everything model-side is 2019–2021 vintage: GenSen, BERTSum,
XLNet wrappers, `pytorch_modules/conditional_gru.py`, GloVe scripts, plus heavy
AzureML coupling. `nlp-recipes` itself is archived.
**Use when:** you need SQuAD-style metrics or classification eval utilities.
**Do NOT use when:** choosing a model — all of it is superseded.
**Modernization:** keep `eval/`, discard `models/`. **MINE (eval only).**

---

## 8. `justin-aj/ml-retrieval`

`[VERIFIED]` — 5 files, all read (248 KB).

**Categories:** IR, SS.

**Problem/approach.** A deliberately tiny comparative study: encode 10 documents
with `all-MiniLM-L6-v2` (384-d), then retrieve by (1) a `RandomForestClassifier`
trained to predict the document ID as a class, vs (2) plain cosine similarity.

**Critical assessment — this is a worked example of a bad idea, and it is useful
for exactly that reason.** The README argues both honestly ("Artificial
classification problem (documents aren't natural classes)"), but the experiment
cannot support any conclusion:

- **10 documents, 10 classes, one training example per class.** Nothing is held
  out. There is no validation set, and there cannot be one.
- The RF's top probability is **0.150** across 10 classes — barely above the
  0.100 uniform prior. The model learned essentially nothing; the ranking it
  produces is near-arbitrary.
- The README's headline ("both methods agree on the top result") is therefore
  evidence of **nothing**; one query, on documents whose lexical overlap with
  the query is obvious.
- Treating "which document is relevant" as a closed-set classification over
  document IDs **cannot generalize to unseen documents** — the label space is
  the corpus. The README concedes this ("Not truly scalable for dynamic
  document collections") but still presents it as a comparable method.

**Transferable lessons (the actual value):**

1. **Don't turn retrieval into document-ID classification.** Score
   query-document *pairs*; keep the label space task-shaped, not corpus-shaped.
   If a Shipd task tempts you into per-document classes, that is a red flag.
2. **A comparison on 10 items and 1 query is a demo, not a result.** Any
   "method A beats method B" claim needs a labeled query set and a retrieval
   metric (Recall@k / MRR / nDCG).
3. The `find_similar_documents(query, docs, model, top_k)` shape — encode
   corpus once, encode query, `cosine_similarity`, `argsort[::-1][:k]` — is the
   correct minimal dense-retrieval skeleton, and it is worth having as a
   two-line mental template.

**Strengths.** Small, readable, honest limitation section.
**Weaknesses.** No validation, no metrics, no scale; conclusions unsupported.
**Use when:** as a cautionary example, or for the 5-line retrieval skeleton.
**Do NOT use when:** deciding anything.
**Modernization:** replace RF-over-doc-IDs with BM25 + dense + a cross-encoder
reranker, and evaluate on labeled queries. **MINE (as an anti-pattern).**

---

## 9. `allenai/allennlp`

`[VERIFIED]` — sparse clone; README banner and last commit read.

**Categories:** IR, QA (as cited).

**Status: DEAD.** README banner: *"The AllenNLP library is now in maintenance
mode… we are no longer adding new features or upgrading dependencies… up until
December 16th, 2022."* Last commit **2022-11-21**.

The PDF cites it for "Retrieval-based QA", but retrieval/QA models lived in the
separate `allennlp-models` repo. This repo is the framework core.

**What remains transferable.** Only architecture-level ideas: the
`Model`/`DatasetReader`/`Predictor` separation, declarative JSONNET experiment
configs (a genuinely good idea — config-as-experiment-record, which
`EXPERIMENT_LOG.md` reproduces in spirit), and its `nn`/`modules/transformer`
building blocks as reading material.

**Do NOT install it.** Pinned to a 2022 PyTorch/transformers world; it will
fight every modern dependency.
**Use when:** reading about framework design. **Otherwise SKIP.**
**Modernization:** fully superseded by `transformers` + `datasets` +
`sentence-transformers`. **SKIP.**

---

## 10. `sebastianruder/NLP-progress`

`[VERIFIED]` — cloned; task list enumerated. Last commit 2024-06-23.

**Categories:** REF across all. Cited in the PDF under *Ranking* — **incorrectly**;
there is no ranking file among the 38 English task files.

**What it is.** SOTA leaderboards + dataset descriptions per task, in 16
languages plus a `structured/` section.

**Files relevant to likely Shipd families:** `text_classification.md`,
`semantic_textual_similarity.md`, `question_answering.md`,
`natural_language_inference.md`, `relationship_extraction.md`,
`relation_prediction.md`, `word_sense_disambiguation.md`,
`information_extraction.md`, `named_entity_recognition.md`,
`intent_detection_slot_filling.md`, `entity_linking.md`, `coreference_resolution.md`.

That list maps almost one-to-one onto the Phase 5 taxonomy — **use this repo to
name the task**, which is its real value: if you can find the academic task name,
you inherit its standard metric, its standard splits, and its known pitfalls.

**Strengths.** Excellent for identifying *which* well-studied problem a Shipd
challenge actually is, and what metric the literature uses for it.
**Weaknesses.** ~2 years stale (pre-dates most current LLM-era results); no code;
leaderboard numbers are not comparable to your CV.
**Use when:** Phase 6 task audit — naming the problem and picking the metric.
**Do NOT use when:** choosing a model (numbers are outdated).
**Modernization:** treat as a taxonomy, not a leaderboard. **REF.**

---

## 11. `HKUNLP/instructor-embedding`

`[VERIFIED]` — cloned; README, `requirements.txt`, `setup.py`, last commit
(2025-01-15) read.

**Categories:** EMB.

**Problem/approach.** Instruction-finetuned embeddings: prepend a task
instruction to each text ("Represent the Science title:") so **one** encoder
produces task-tailored vectors without fine-tuning. Trained on ~330 tasks
(MEDI); the paper claims SOTA on 70 embedding tasks.

**The idea that survives and matters.** *The instruction/prefix is part of the
input, and it changes the embedding.* This is now standard across the whole
modern embedding stack — E5's `query:`/`passage:`, BGE's query instruction,
Qwen3-Embedding's instruct prompts — and it is the single most common **silent**
mistake when using them: forget the prefix, or use the same prefix for queries
and documents in an asymmetric task, and you lose real points with no error
message. `sentence-transformers` exposes this as `prompt`/`prompt_name`, and
`mine_hard_negatives` takes `query_prompt`/`corpus_prompt` for the same reason.

**Also transferable:** asymmetric tasks (query vs document) want **different**
instructions per side; symmetric tasks (STS, dedup) want the **same** one.

**Strengths.** Clear idea, clean API (`model.encode([[instruction, text], ...])`),
`examples/faiss` shows the retrieval wiring.
**Weaknesses.** Practically superseded — the INSTRUCTOR checkpoints
(`hkunlp/instructor-{base,large,xl}`) are behind current BGE/E5/Qwen/GTE models
on MTEB `[UNVERIFIED]` — huggingface.co was blocked, so I could not check the
current leaderboard. Repo is quiet since Jan 2025; the README still instructs
`conda env create -n instructor python=3.7` while `requirements.txt` demands
`torch>=2.0`, `sentence-transformers>=3.0.1,<4.0` — **internally inconsistent**,
and the `<4.0` pin conflicts with the v6 sentence-transformers documented above.
**Use when:** you specifically want instruction-conditioned embeddings and can
pin an older sentence-transformers.
**Do NOT use when:** a current BGE/E5/Qwen model is available — same idea,
better weights, no dependency conflict.
**Modernization:** take the *instruction-prefix concept*, apply it with a
current model. **MINE (concept), SKIP (weights).**

---

## 12. `McGill-NLP/llm2vec`

`[VERIFIED]` — cloned; README read. Active (last commit 2026-04-04).

**Categories:** EMB, TRF.

**Problem/approach.** Convert a decoder-only LLM into a text encoder in three
steps: (1) enable **bidirectional attention**, (2) train with **masked next-token
prediction (MNTP)**, (3) **unsupervised contrastive learning (SimCSE)**;
optionally a supervised contrastive stage on E5 data. Ships LoRA adapters over
Llama-3 / Mistral / Gemma / Qwen-2 bases; mean pooling by default, `max_length`
512 by default.

**Why it is in this KB.** It is the clearest statement of *why* modern top-of-
leaderboard embedding models are LLM-derived, and it makes the recipe explicit
and legible — bidirectional attention + a denoising objective + contrastive
training. It also documents the instruction convention precisely: **instructions
on both sides for symmetric tasks, on queries only for asymmetric tasks** — the
same rule as INSTRUCTOR, stated more usefully.

The 2026 update adds LLM2Vec-Gen (embeddings that encode the LLM's *potential
answer* to a query rather than the query itself) — conceptually a learned
HyDE-like query expansion. `[UNVERIFIED]` — arxiv blocked; README only.

**Strengths.** Actively maintained; well-documented; strong quality ceiling;
LoRA keeps fine-tuning tractable.
**Weaknesses.** **Enormous** for competition work — a 7–8B encoder needs a
serious GPU, `flash-attn`, and bf16, and embedding a large corpus is slow and
expensive. 512-token default truncation is easy to miss. Completely unusable in
this container.
**Use when:** you have real GPU budget, a small-to-medium corpus, and the
metric gap justifies it — i.e. late in the ladder, after cheaper rungs plateau.
**Do NOT use when:** CPU-only, large corpus, tight inference budget, or before
a cross-encoder reranker has been tried (a reranker on top of a cheap retriever
usually beats a more expensive retriever, for less compute).
**Modernization:** current. **USE (late rung, with budget).**

---

## 13. `Denis2054/Transformers-for-NLP-2nd-Edition`

`[VERIFIED]` — cloned; chapter map read. Last commit 2024-01-04.

**Categories:** TRF, QA, TC.

**What it is.** Companion notebooks to Rothman's book, Ch. 2–17 + appendices:
Ch2 multi-head attention & positional encoding from scratch · Ch3 **BERT
fine-tuning for sentence classification** · Ch4 KantaiBERT (train a RoBERTa from
scratch) · Ch5 transformer tasks · Ch6 Trax translation · Ch7 GPT-3 fine-tuning ·
Ch8 T5 summarization · Ch9 GPT-2 training + tokenizer analysis · Ch10 semantic
role labeling · **Ch11 QA + a Haystack QA pipeline** · Ch12 sentiment · Ch13 fake
news · Ch14 **BertViz / Ecco interpretability** · Ch15–16 vision & multimodal ·
Ch17 all-in-one.

**Most relevant chapters for Shipd work:** Ch3 (classification fine-tuning
template), **Ch11 (`Haystack_QA_Pipeline.ipynb` — a retriever→reader pipeline,
the closest thing in the whole PDF to a real RAG/retrieval-QA implementation)**,
Ch9's tokenizer notebook (checking how your domain text tokenizes is a real,
under-used diagnostic), and Ch14 (attention visualization for error analysis).

**Strengths.** Broad, pedagogically ordered, working notebooks.
**Weaknesses.** Book-pinned dependencies; heavy OpenAI-API content that is now
several model generations stale; Trax is effectively abandoned; 165 MB.
**Use when:** you want a readable end-to-end reference for a transformer
pipeline, especially Ch11's retriever→reader structure.
**Do NOT use when:** you need current API/model choices.
**Modernization:** keep the pipeline *shapes*; replace model IDs and APIs.
**REF / MINE (Ch3, Ch11, Ch14).**

---

## 14. `graykode/nlp-tutorial`

`[VERIFIED]` — cloned; file tree and README read. Last commit **2021-07-25**.

**Categories:** TRF, EMB, TC.

**What it is.** Minimal from-scratch PyTorch implementations, each **under 100
lines**: NNLM · Word2Vec skip-gram · FastText · TextCNN · TextRNN · TextLSTM ·
Bi-LSTM · Seq2Seq · Seq2Seq+Attention · Bi-LSTM+Attention · **Transformer** ·
**BERT**. Old TF-v1 versions archived under `archive/tensorflow/v1/`.

**Value.** Purely educational — the best short read in the list for *understanding*
attention, positional encoding, and masked LM mechanics. The Transformer and BERT
files are excellent for building intuition you will later use in error analysis.

**Strengths.** Tiny, clear, PyTorch, no framework overhead.
**Weaknesses.** **Zero competition value.** Toy corpora (a handful of sentences),
no batching discipline, no validation, no pretrained weights. Training any of
these from scratch on a Shipd dataset would be strictly worse than a TF-IDF
baseline at vastly more cost. Unmaintained since 2021.
**Use when:** learning/debugging intuition about architecture internals.
**Do NOT use when:** building a solution. Never train these from scratch for a
leaderboard.
**Modernization:** not applicable — it is a teaching artifact, correctly so.
**REF.**

---

## 15–23. Kaggle entries (9) `[BLOCKED]`

`www.kaggle.com` → 403 `connect_rejected` (organization egress policy).
**Not accessed. Contents not reconstructed.** See `access_log.md` for the full
list and for what is and is not claimed about them.

Their *titles* map onto these categories — recorded so the resource list stays
complete, and so a future session with Kaggle access knows exactly what to fetch:

| # | Title (verbatim from PDF) | Category |
|---|---|---|
| 15 | Approaching (Almost) Any NLP Problem on Kaggle — Text Classification | TC |
| 16 | Approaching NLP — Grid Search & Pipelines | TC, VAL |
| 17 | Approaching NLP — Deep Learning (LSTM/GRU) | TC, TRF |
| 18 | Approaching NLP — Word Vectors & Cosine Similarity | SS, EMB |
| 19 | Approaching NLP — Ensembling for Similarity Tasks | SS, ENS |
| 20 | Approaching NLP — TF-IDF Feature Extraction for Retrieval | IR, FE |
| 21 | Approaching NLP — Ensembling / Ranking Predictions | RANK, ENS |
| 22 | Approaching NLP — Word Vectors (GloVe/Word2Vec/FastText) | EMB |
| 23 | Approaching NLP — LSTM with Embeddings / GRU + Attention | TRF, TC |

The techniques these titles name are covered in `techniques/` from sources I
**did** read. Nothing here is attributed to these notebooks.

**ACTION for a future session with Kaggle access:** re-run Phase 2 on these nine,
and specifically extract their *validation protocol* and *blend weights* — the
two things that transfer between competitions and that no GitHub repo in this
list documents well.

---

## Coverage gaps in the PDF

What the list does **not** cover, but a Shipd challenge plausibly needs. These
are filled in `techniques/` and `models/` from modern practice, flagged
`[UNVERIFIED]` where the source was blocked:

| Gap | Where it is covered here |
|---|---|
| **BM25 / sparse retrieval** (only a Wikipedia link in the whole PDF) | `techniques/information_retrieval.md` |
| **Hybrid sparse+dense fusion (RRF)** | `techniques/information_retrieval.md` |
| **Cross-encoder reranking** (cited once, no implementation) | `techniques/reranking.md`, `models/cross_encoders.md` |
| **Learning-to-rank / LambdaRank** | `techniques/ranking.md` |
| **BGE / E5 / Qwen / multilingual embeddings** (absent entirely) | `models/bge.md`, `models/e5.md`, `models/qwen_embeddings.md` |
| **LightGBM / XGBoost on text features** (absent) | `models/gbdt_on_text.md` |
| **OOF / stacking / blending discipline** | `validation/oof_predictions.md`, `techniques/ensembling.md` |
| **Leakage, distribution shift, noisy labels** | `validation/*` |
| **Threshold optimization & calibration** | `validation/threshold_and_calibration.md` |
| **Grouped/query-aware CV** | `validation/cross_validation.md` |
| **Error analysis as a first-class step** | `playbooks/*` |
