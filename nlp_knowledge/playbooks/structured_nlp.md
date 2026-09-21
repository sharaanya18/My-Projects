# Playbook: Structured NLP

Entity extraction · relation extraction · structured prediction · sense
disambiguation · span tasks. Phase 5 lists all of these as plausible Shipd
families, and the PDF covers **none** of them — this playbook is built from
general practice plus `sebastianruder/NLP-progress` task definitions
`[VERIFIED]`.

## 0. Audit

- What is the prediction unit — token, span, entity pair, whole document?
- How is the output **scored**? Exact span match? Partial overlap? Type-aware?
- Is the label schema closed (fixed types) or open?
- Are spans nested or overlapping?
- How is the submission encoded — character offsets, token indices, strings?
  **Offsets are the #1 source of silent errors here.**

## 1. Name the task

Use `NLP-progress`'s task files to find the academic name. If it maps to NER,
relation extraction, entity linking, WSD, or slot filling, you inherit the
standard metric, the standard evaluation script, and the known pitfalls. That
lookup is worth 30 minutes.

## 2. Ladder

| Exp | Approach |
|---|---|
| T0 | gazetteer / regex / rule baseline — often shockingly strong, and it tells you the task's lexical ceiling |
| T1 | CRF over hand-crafted token features (word shape, casing, prefixes/suffixes, POS, context window) |
| T2 | token classification with a pretrained encoder (BIO tagging) |
| T3 | span-based model (enumerate candidate spans, classify) — handles nesting |
| T4 | pair classification over entity pairs (relation extraction) |
| T5 | retrieval + classification hybrid (entity linking, WSD against a sense inventory) |

T0 is not a joke. On a benchmark with a closed, well-defined entity vocabulary,
a gazetteer can be most of the way there — and its failures tell you exactly
what the learned model needs to add.

## 3. Sense disambiguation & context-dependent labels

The task family most likely to look like "noisy labels" when it isn't. If the
same surface string carries different labels in different rows, ask whether a
**context variable** determines the label before writing the rows off as noise
(see `../validation/noisy_labels.md`).

Approach: treat it as matching a mention **plus its context** against a sense
inventory — retrieval + reranking over sense glosses, rather than flat
classification over sense ids. This generalizes to senses unseen in training,
which flat classification cannot.

## 4. Evaluation

- Use the **official** scorer if one exists. Hand-rolled span matching is
  routinely wrong (boundary conventions, inclusive vs exclusive ends,
  type-aware vs type-agnostic matching).
- Report exact-match **and** partial-overlap F1 — the gap is diagnostic of
  boundary problems versus detection problems.
- Micro vs macro over entity types: know which one is scored.

## 5. Traps

- **Offset bugs.** Tokenizer offset mapping, whitespace normalization, and
  Unicode normalization all shift character offsets. Decode a sample of your
  predicted spans back out of the raw text and **read them** before submitting.
- Truncation dropping entities near the end of long documents.
- BIO encoding/decoding errors (invalid transitions like `I-` after `O`).
- Label imbalance: `O` dominates; per-class metrics are essential.
- Group folds by **document**, not by sentence or by entity.
