# LightGBM / XGBoost on Text

Absent from the PDF entirely. Essential in practice, and the main CPU-viable
rung above linear models.

## Where GBDT belongs

**Not** on raw sparse TF-IDF. A tree splits one feature at a time; over 100k
sparse columns it is slow and weak, and linear models dominate that regime.

GBDT wins when you have **dense, heterogeneous, hand-crafted features** —
exactly the situation in similarity, reranking, and metadata-rich
classification:

| Input | Use |
|---|---|
| TF-IDF sparse only | LinearSVC / LogisticRegression, not GBDT |
| SVD(TF-IDF) 100–300 dims + lengths + counts | **GBDT** |
| pair features (overlap, BM25, cosine, lengths) | **GBDT** — its home turf |
| retrieval features per (query, doc) | **LGBMRanker** |
| embeddings alone (384–1024 dense dims) | linear head usually ≥ GBDT |
| embeddings + lexical + metadata | **GBDT** |

The pattern that matters: GBDT is how you **combine a semantic score with
lexical and structural evidence**. A frozen encoder gives you one cosine number;
GBDT learns *when to trust it*. That is usually better than thresholding the
cosine directly.

## Settings

```python
import lightgbm as lgb
clf = lgb.LGBMClassifier(
    n_estimators=2000, learning_rate=0.03,
    num_leaves=31, min_child_samples=20,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, class_weight="balanced",
)
clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)])
```

- Early stopping on a **fold-internal** validation set, never on the fold you
  report. Otherwise the reported CV is chosen by the metric it reports.
- `n_estimators` high + early stopping beats tuning `n_estimators`.
- Ranking: `LGBMRanker(objective="lambdarank")` — see
  `../techniques/ranking.md` for the `group` / `label_gain` traps.

## XGBoost

Broadly interchangeable. Prefer LightGBM for speed on CPU and native
categorical support; XGBoost for `hist` on GPU and a somewhat more predictable
regularization story. Do not run both and pick the better — that's selection on
the validation set. Pick one, tune it, and only add the second as an ensemble
member if OOF correlation says it's complementary.

## Importance, honestly

Use **permutation importance on OOF predictions**. LightGBM's built-in split
importance is biased toward high-cardinality/continuous features and will
mislead you.

Any feature with outsized importance is a **leakage suspect** until you've
explained it. This is one of the most reliable leakage detectors available —
see `../validation/leakage_detection.md`.
