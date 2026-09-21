# SHIPD_TASK_ANALYSIS — <challenge name>

> Phase 6 template. **Fill this in before writing model code.**
> Where something is unknown, write `UNKNOWN` — an explicit unknown is a
> research task; a silently assumed value is a bug.

Date: · Analyst: · Data snapshot:

---

## 1. Objective

- **Exact task objective (one sentence):**
- **Prediction unit** (what is one submission row?):
- **Target / output** (type, range, vocabulary):
- **Evaluation metric** (exact definition, averaging, at what k):
- **Is the metric per-row or per-group?**

## 2. Data structure

| | Train | Test |
|---|---|---|
| rows | | |
| columns | | |
| text fields (+ length p50/p95/max) | | |
| metadata fields | | |
| id fields | | |
| label field | | |

- **Columns in train but not test:**
- **How was each split constructed (if stated)?**
- **Available metadata not in the main table:**
- **Missing information / what I wish I had:**

## 3. Label structure

- Label space (closed? complete? same in train and test?):
- Class balance / imbalance ratio:
- Multilabel? Hierarchical? Ordinal?
- **Abstain / unanswerable / "none of the above" case?** Is it scored?
- Graded or binary relevance (ranking/retrieval tasks):

## 4. Data hazards (Phase 13)

Fill in with **measurements**, not impressions. `../code/validation.py` runs most
of these.

| Check | Result | Action |
|---|---|---|
| Exact duplicates (within / across splits) | | |
| Near duplicates (char-3gram cosine) | | |
| Duplicate texts with **conflicting labels** (→ min noise rate) | | |
| Duplicate / missing ids vs the submission template | | |
| Missing data, per column | | |
| Adversarial validation AUC (train vs test) | | |
| Most train-like / test-like tokens | | |
| Length-only baseline score | | |
| **Metadata-only baseline score** (no text at all) | | |
| Target rate by each categorical column | | |
| Target vs row index correlation | | |
| Language distribution per text field | | |
| Vocabulary overlap train↔test / OOV rate | | |
| Evidence coverage (does every row have its evidence present?) | | |
| Distractors / deliberately similar negatives | | |

> If the **metadata-only** or **length-only** baseline scores near a real text
> model, the benchmark has a shortcut. That is a finding — record it and change
> the strategy, don't quietly exploit it without testing whether it survives a
> structural split.

## 5. Hidden structure

- Group keys present (query / user / document / entity / session / time):
- **Possible hidden variables** (context that determines the label):
- Evidence that labels are context-dependent (same text, different label):
- Signs of synthetic generation (templates, repeated phrasing, uniform lengths):
- Suspected train/test construction differences:

## 6. Problem family (Phase 5)

Run `python nlp_knowledge/code/task_detector.py train.csv test.csv`, then decide.

- **Detector output (ranked):**
- **My verdict — primary family:**
- **Secondary families (tasks are often hybrid):**
- **Why not the alternatives:**
- **Playbook(s) to follow:**

## 7. Validation design

- Split type and **why it matches how test was built**:
- Group key(s):
- n folds, seed, fold file path:
- What is deliberately held out:
- Known remaining leakage risk:
- **Noise band (fold-wise std of the baseline metric):**

> Every later claim of improvement must clear this noise band.

## 8. Constraints

- CPU / GPU available:
- RAM:
- Model downloads (huggingface.co) reachable? **yes / no**
- Wall-clock budget for training / inference:
- Submission-time compute limit:
- Estimated runtime of the intended top rung:

## 9. Relevant knowledge-base techniques

| KB file | Why it applies | Assumption it makes | How it could fail here | How I'll test it |
|---|---|---|---|---|
| | | | | |

## 10. Submission format

- File name / format:
- Exact columns, in order:
- Row count and id set (must equal ?):
- Ordering requirement:
- Value type and range:
- NaN policy:
- **Verified against the sample submission?** yes / no

## 11. Plan

| Rung | Experiment | Expected cost | What it would tell me |
|---|---|---|---|
| 0 | | | |
| 1 | | | |
| 2 | | | |

## 12. Open questions / risks

-
