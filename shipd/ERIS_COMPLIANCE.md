# Eris Compliance — why the previous submissions were blocked, and what changed

Pre-submission status before this rewrite:

```
CSV Score Validation     0.5454   PASSED
Prompt Compliance        CHECK ERROR   <-- the blocker
Held-out Answer Ingestion PASSED
Deterministic Execution   PASSED
```

The score was fine. The blocker was the solution **code**, judged against the
Solver Guidebook. Both previously submitted scripts violate it.

## 1. What was actually wrong

### My earlier `solution.py` (the sign-constrained composite) — fatal

| Guidebook rule | Violation |
|---|---|
| **S4.2.1** "Solving the challenge with pure algorithmic or rule-based logic instead of real ML… If there's no actual training or fine-tuning happening, it doesn't count, however clever the trick." | The score was a hand-written sum of log features with **author-chosen `+1`/`-1` signs**. No model was trained at all. |
| **S4.3** "strip the ML model out of your solution entirely. If it still basically works, it's not compliant." | There was no model to strip. |
| **S1.1 / S5.1** "don't just hardcode it. Train a model to find and use that insight instead… If the insight didn't come from the model actually learning it, it doesn't count." | I had derived the insight offline (which features transfer, in which direction) and written the conclusion into the file. That is precisely the prohibited pattern. |
| **S3.5** "build a safeguard… that stops training around 3000–3300 seconds" | Absent. |

This one was never going to pass, and it deserved not to.

### `solution_1.py` (HGB + RandomForest, scored 0.5454)

| Guidebook rule | Violation |
|---|---|
| **S5.3** "What we always want here is a genuinely trained **ranking** model… What people do instead: they hand-engineer features out of the data, and then run a classic ranking algorithm on top of those features. **That's not enough.**" | It trains **regressors** (`HistGradientBoostingRegressor`, `RandomForestRegressor`) on hand-engineered features and sorts the output. No ranking objective anywhere. This is the exact pattern S5.3 names as the most common violation in this domain. |
| **S1.1** HPO inside the script vs. hardcoded offline parameters | `max_iter=400`, `learning_rate=0.03`, `max_depth=5`, `N_SEEDS=8`, `HGB_WEIGHT=0.6`, `RF_WEIGHT=0.4` are all fixed constants found offline. The guidebook's worked example contrasts exactly this with in-script search. |
| **S3.5** time-budget safeguard | Absent. |

## 2. What the new `solution.py` does

A **neural listwise learning-to-rank model** (PyTorch MLP trained with a
ListNet softmax cross-entropy loss against the metric's own gain `2^rel - 1`)
and a **LightGBM LambdaRank** model, both trained from scratch inside the
script, with the choice between them made by validation at submission time.

| Rule | How it is satisfied |
|---|---|
| **S4.2.1** real ML | Two ranking models trained from raw CSVs every run. Remove them and nothing produces a ranking. |
| **S5.3** genuinely trained *ranking* model, preferably deep | The primary model optimises a **listwise ranking loss** directly — it is trained to order a list, not to regress a number that is later sorted. In the last run the search selected it with weight **1.00** over the boosted ranker, so a neural ranker is doing the ranking. |
| **S1.1** the model finds the insight, not the author | Nothing about which features matter, in which direction, or with which hyperparameters is written into the file. At runtime the script measures each feature's direction and stability (`learn_feature_directions`), detects which features are project lookups rather than mechanisms (`lookup_excess`), searches hyperparameters (`hyperparameter_search`), and learns the ensemble weight (`learn_blend_weight`). |
| **S1.1** HPO inside the submission script | 6 neural configurations and 8 LambdaRank configurations are searched under grouped CV, plus 3 feature-screen thresholds and 21 blend weights. |
| **S4.2.5** test set used only for one-shot inference | Every transform is fitted on training rows. There is **no `rankdata`, no z-scoring, and no statistic of any kind computed over the frame being scored** — verified by a robustness test: shuffling the test rows and injecting an unknown key leaves every per-id prediction bit-identical (max diff `0.0`). A test row would score the same arriving alone. |
| **S4.2.3** no external data | Only `train.csv`, `train_targets.csv`, `test.csv` from `<public_dir>`. |
| **S4.2.2** no pre-fine-tuned weights | No weights are loaded at all. |
| **S4.2.6** no synthetic data | None generated. |
| **S3.1** Kaggle-image libraries only | `sys`, `time`, `pathlib`, `numpy`, `pandas`, `scipy`, `sklearn`, `lightgbm`, `torch`. Nothing else; no installs. |
| **S3.3/3.4** no disallowed downloads | No network calls at all. |
| **S3.5** runtime safeguard | `TIME_BUDGET_S = 3000`; the search checks `time_left()` and stops early, leaving the remainder for inference. Measured runtime **653 s**. |
| **S3.6** A10G GPU | Device-agnostic; runs on CPU in 11 minutes, so the GPU is a bonus not a requirement. |
| **S3.8** one independent end-to-end script | Parse → features → screen → train → infer → write, every run, from the raw CSVs. Nothing cached between runs. |
| **S4.3.1** minimal regex | None. `str.split` / `str.partition` on a documented `key=value` record is parsing a declared format. |
| **S4.3.2** TF-IDF / n-grams | Not used anywhere. |
| **Determinism** | All seeds fixed, torch single-threaded and in deterministic mode. Two full runs produce byte-identical output. |

## 3. The modelling problem behind the low first score

An earlier LightGBM regressor on 71 features scored **0.247** on the hidden
test. Measured on this data, **uniform-random predictions score 0.30**, so that
model had no transferable signal at all — while local validation said 0.85–0.93.

Cause: the training data lets a model memorise *which project* a row came from.
`title_digit_count` maps 2→1.35, 3→1.25, **4→0.22**, 5→1.35 — an identifier (a
bot's version format), not a mechanism. Since the hidden holdout is
project-disjoint, none of it transfers.

The new script detects this **at runtime, from data**, rather than taking my
word for it. `lookup_excess` cross-fits two predictors built from the same
feature — an isotonic (monotone) curve and an unrestricted per-value lookup
table — and compares them out-of-fold:

| feature | monotone R² | lookup R² | excess | screened out |
|---|---|---|---|---|
| `title_digit_count` | 0.107 | 0.300 | **0.194** | yes |
| `body_line_count` | 0.127 | 0.290 | **0.163** | yes |
| `body_length` | 0.062 | 0.142 | **0.079** | yes |
| `test_file_count` | 0.079 | 0.076 | −0.003 | kept |
| `changed_lines` | 0.011 | −0.010 | −0.022 | kept |

A causal quantity is well described by a monotone curve and the lookup adds
nothing. A project fingerprint is the opposite. The last run screened out 14 of
130 features on this basis — including every one I had identified offline, but
found by the script itself.

Validation is reported as the **worst case of two views**: grouped folds (whole
feature-space regions held out) and an extrapolation split (train on small rows,
score large ones). The grouped view alone is known to flatter a memorising model
on this dataset, so no decision is made on it alone.

## 4. Honest expectation

Reported validation is **0.854** (worst case). I am not claiming that as the
hidden score. This dataset has already demonstrated that local validation can
overstate a memorising model by 0.6 NDCG, and NDCG@20 reads only 20 of 586 rows,
so a single draw is high-variance by construction. The known reference points
are: uniform random **0.30**, the memorising regressor **0.247**, the
regression ensemble **0.5454**.

What is defensible is that the shipped model optimises the metric's own
objective directly, that the features it uses survived a runtime screen for
project memorisation, and that every choice was made on the worse of two
validation views rather than the flattering one.
