# Question Answering

Sources: `reddy-123/` (= MS nlp-recipes) `[VERIFIED]`,
`Denis2054` Ch11 `[VERIFIED]`, `ElizaLo/QA` `[VERIFIED]`,
`sebastianruder/NLP-progress` `question_answering.md` `[VERIFIED]`,
`allenai/allennlp` `[VERIFIED, dead]`.

## Identify the QA subtype first — they share almost no machinery

| Subtype | Output | Metric |
|---|---|---|
| **Extractive** | a span of the given context | EM / token-F1 |
| **Multiple choice** | index of an option | accuracy |
| **Retrieval QA (open-domain)** | answer + supporting passage | EM/F1, plus retrieval recall |
| **Boolean / yes-no** | a label | accuracy/F1 |
| **Abstractive** | generated text | ROUGE/BLEU/judge — hard to score, be careful |
| **Unanswerable-aware** | span **or** abstain | EM/F1 with a no-answer class |

**Check for the abstain case early.** SQuAD 2.0-style benchmarks include
unanswerable questions, and Phase 13 flags this explicitly for Shipd. A model
that never abstains can lose badly on a benchmark where a third of questions
have no answer — and the failure looks like a modeling problem when it is a
task-framing problem. Often the best design is an explicit no-answer score
compared against the best span score, with the **threshold tuned on OOF**.

## Use the canonical SQuAD metric — do not hand-roll it

`reddy-123/utils_nlp/eval/evaluate_squad.py` `[VERIFIED]` is the reference
implementation and is still correct:

```
normalize_answer(s):  lowercase → remove punctuation → remove articles (a/an/the)
                      → collapse whitespace
exact_match_score:    normalized strings equal
f1_score:             token-level precision/recall F1 over normalized tokens
metric_max_over_ground_truths:  max of the metric over all gold answers
```

Every one of those steps changes the score. Hand-rolled comparison (usually:
forgetting article removal or the max-over-golds) is a classic source of a
wrongly-low number that gets misdiagnosed as a model problem. It is also the
first thing to check if your local score and the leaderboard disagree.

`utils_nlp/eval/question_answering.py` in the same repo has the equivalent
`get_raw_scores` path.

## Retrieval QA

Structure it as retrieve → (rerank) → read. `Denis2054` Ch11's Haystack pipeline
`[VERIFIED]` is the closest thing in the whole resource list to a real
implementation of this.

Instrument **each stage separately**, otherwise you cannot debug it:

1. **Retrieval recall@k** — does the gold passage appear at all? This is the
   ceiling.
2. **Rerank precision@1/5** — is it ranked to the top?
3. **Reader EM/F1 given the gold passage** — an oracle-context upper bound.

End-to-end F1 alone tells you nothing about which stage to fix. The
oracle-context number is especially valuable: if the reader scores 0.88 with the
gold passage and the pipeline scores 0.41, the reader is fine and every point is
in retrieval.

See `information_retrieval.md` and `reranking.md` for stages 1–2.

## Extractive reading

- Predict start and end logits; decode with the standard constraints
  (`end >= start`, bounded span length, drop spans crossing the
  question/context boundary).
- **Long contexts:** use a sliding window with a document stride, and map
  offsets back to the original character positions carefully. Off-by-one
  offset-mapping bugs are common and produce answers that are subtly truncated —
  compare a sample of decoded spans against the raw text before trusting the
  pipeline.
- Ensemble by **averaging start/end logits** across models/seeds before decoding,
  not by voting on decoded strings.

## Legacy note

`ElizaLo`'s QA entry is a **BiDAF on SQuAD** implementation `[VERIFIED]` —
a 2017-era architecture (bidirectional attention flow over char+word embeddings),
and `allennlp` (cited by the PDF for retrieval QA) has been dead since 2022.
BiDAF's attention-flow idea is now subsumed by transformer self-attention. Read
it for intuition about query-aware context representation; do not build it.

## Validation

- Group folds by **document/context** — the same passage appearing in train and
  valid leaks the answer.
- Also group by question template/entity if the data is synthetically generated
  (very plausible for a Shipd benchmark).
- Report EM and F1 separately; they diverge exactly where partial answers live,
  and that divergence is diagnostic.
