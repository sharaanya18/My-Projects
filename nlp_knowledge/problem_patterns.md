# Problem-Type Detector (Phase 5)

A reusable procedure for working out what an NLP challenge actually **is**.

**A task may belong to several categories. Do not force it into one.** The
detector below returns a *ranked set*, and hybrid tasks are the normal case,
not the exception.

Automated first pass: `code/task_detector.py`.

---

## Step 1 — Identify the prediction unit

The single most clarifying question: **what does one row of the submission
represent?**

| One row is… | Family |
|---|---|
| one text | classification / regression |
| one pair of texts | pair classification, similarity, matching |
| one (query, candidate) | ranking, reranking, retrieval scoring |
| one query, value = a list/ranking | retrieval, ranking |
| one span / token | structured prediction, extraction |
| one question | QA (check: extractive / MC / open-domain) |
| one (entity, entity) | relation extraction |
| one mention | entity linking, sense disambiguation |

## Step 2 — Identify the output type

| Output | Implies |
|---|---|
| one of k mutually exclusive labels | multiclass |
| 0/1 | binary |
| a set of labels | multilabel |
| a continuous score | regression / graded similarity |
| an ordered list | ranking |
| a span of input text | extractive |
| free text | generation (rare in Shipd; check scoring carefully) |
| an id from a fixed inventory | linking / disambiguation |

## Step 3 — Look for group structure

Ask: is there a column that **repeats** across rows and defines a natural unit?

Query id · document id · user id · session · entity · article · timestamp.

**If yes, that key drives validation** (`validation/cross_validation.md`) and
very often means the task is really ranking or retrieval even if it is presented
as per-row classification. This is the single most common mis-framing.

A strong tell: the metric is computed **per group and then averaged**. That is a
ranking metric no matter how the data is laid out.

## Step 4 — Read the metric backwards

The metric usually names the family:

| Metric | Family |
|---|---|
| accuracy, macro/micro-F1, MCC | classification |
| AUC | binary classification / ranking |
| log loss, Brier | calibrated classification |
| Spearman / Pearson | graded similarity (STS) |
| MRR, nDCG, MAP, Recall@k | retrieval / ranking |
| EM, token-F1 | extractive QA / span |
| span-F1 (exact/partial) | structured extraction |
| BLEU/ROUGE | generation |

If the metric and the apparent task disagree, **the metric is right.** Optimize
what is scored.

## Step 5 — The taxonomy (Phase 5 list, with tells)

| Category | Tells | Playbook |
|---|---|---|
| Binary classification | 2 labels, one text | `playbooks/classification.md` |
| Multiclass | k exclusive labels | same |
| Multilabel | label lists / multi-hot columns | same |
| Semantic similarity | 2 texts, graded score | `playbooks/similarity.md` |
| Sentence-pair classification | 2 texts, discrete label | same |
| Semantic matching | 2 texts, "do these correspond" | same |
| Information retrieval | query + corpus, no candidates given | `playbooks/retrieval.md` |
| Document retrieval | corpus unit = document | same |
| Paragraph retrieval | corpus unit = passage; chunking matters | same |
| Ranking | candidates given, order scored | `playbooks/ranking.md` |
| Learning-to-rank | graded labels + groups | same |
| Reranking | candidates + an existing score/order | same |
| Question answering | question + context or corpus | `techniques/question_answering.md` |
| Relation extraction | entity pairs → relation type | `playbooks/structured_nlp.md` |
| Structured prediction | output has internal structure | same |
| Entity extraction | spans + types | same |
| Sense disambiguation | mention + inventory of senses | same |
| Retrieval + classification | retrieve, then label | hybrid: `retrieval.md` → `classification.md` |
| Retrieval + ranking | retrieve, then order | hybrid: `retrieval.md` → `ranking.md` |
| Hybrid | several of the above | compose the playbooks |

## Step 6 — Hybrid detection

A task is a hybrid when **more than one** of these holds:

- the corpus is not given per row (you must retrieve) **and** the output is a
  label → retrieval + classification;
- candidates are given **and** the metric is per-group → retrieval + ranking;
- an answer must be located **and** verified → retrieval + QA + abstention;
- a mention must be matched to an inventory **and** typed → linking +
  classification.

**Solve hybrids stage by stage, and instrument each stage separately.** The most
common failure in a hybrid task is optimizing the end-to-end metric without
knowing which stage is the bottleneck.

## Step 7 — Shipd-specific probes (Phase 13)

Run these on every new challenge, before modeling:

| Probe | Looking for |
|---|---|
| Duplicate / near-duplicate scan | leakage, noisy labels, crowded top-k |
| Label consistency on duplicate texts | noise rate, or a hidden context variable |
| Adversarial validation (train vs test) | distribution shift, artifacts |
| Id/index vs target correlation | assembly artifacts |
| Per-source/per-field target rates | shortcut features |
| Length-only and metadata-only baselines | **if these score well, the benchmark has a shortcut** |
| Label vocabulary: train vs test | unseen classes |
| Coverage: does every test row have its evidence? | missing reference documents, unanswerable cases |
| Submission template vs train ids | duplicate ids, missing records |

The **metadata-only baseline** deserves special attention: train a model on
*everything except the text*. If it scores near your text model, the benchmark
is not measuring what it claims to, and your strategy should change accordingly
— that is a finding, not a trick.

## Step 8 — Write it down

Fill in `templates/SHIPD_TASK_ANALYSIS.md` before writing model code. If a field
is unknown, write "unknown" — an explicit unknown is a research task; a silently
assumed value is a bug.
