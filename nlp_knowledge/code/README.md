# Code

Runnable helpers. **Everything here was executed and tested in this session**
on Python 3.11 with numpy 2.4.6 / pandas 3.0.6 / scikit-learn 1.9.1.

No GPU, no model downloads, no network. That is deliberate: in this container
huggingface.co is blocked, so these are the tools that actually run.

| File | What it gives you | Tested |
|---|---|---|
| `task_detector.py` | Phase 5 first pass: ranked candidate task families from a dataframe | 6 synthetic task shapes (classification, STS, LTR, QA, multilabel, imbalanced binary) |
| `baselines.py` | The classification baseline ladder + a **noise-band verdict** | easy + hard/imbalanced synthetic corpora |
| `text_features.py` | Lexical/pair/group-normalised features | reproduces ElizaLo's cat/mouse example |
| `validation.py` | Grouped folds, duplicate & near-duplicate scan, adversarial validation, OOF, thresholds, ensemble-worth-it | AUC 0.54 on matched data, 1.00 on shifted |
| `retrieval.py` | BM25 (from scratch), TF-IDF retrieval, RRF fusion, Recall@k/MRR/nDCG | toy corpus + metric hand-check |
| `experiment_log.py` | Append-only JSONL experiment log + artifacts + markdown table | 3-experiment round trip |

## Quick start on a new challenge

```bash
python nlp_knowledge/code/task_detector.py train.csv test.csv
```

Then, in order:

```python
import sys; sys.path.insert(0, "nlp_knowledge/code")
from validation import exact_duplicate_report, adversarial_validation, pair_group_ids
from baselines import run_ladder
from experiment_log import ExperimentLog

# 1. audit the data BEFORE modelling
print(exact_duplicate_report(train.text, train.label))   # noise + leakage
print(adversarial_validation(train.text, test.text))     # shift + artifacts

# 2. baseline ladder -- and record the NOISE BAND
df = run_ladder(train.text, train.label, scoring="f1_macro")

# 3. log every run
log = ExperimentLog("EXPERIMENT_LOG.jsonl")
```

## Two design decisions worth knowing

**BM25 is implemented from scratch** in `retrieval.py` rather than imported.
`rank_bm25` would be a fine dependency, but writing it out means the retrieval
rung runs with nothing but numpy — which is the difference between having a
retrieval baseline and not having one when the network is locked down.

**`run_ladder` prints a noise-band verdict.** It compares the gap between the
top two models against the larger of their fold-wise standard deviations, and
says `WITHIN NOISE -- do not claim a winner` when the gap doesn't clear it.
This exists specifically because the repositories the PDF recommends report
model rankings that their measurement cannot support (see `../antipatterns.md`).

## Dependencies

```
numpy  pandas  scikit-learn  scipy        # required; all used above
lightgbm                                   # optional: GBDT / LambdaRank rungs
sentence-transformers  torch               # optional: needs HF access (BLOCKED here)
```
