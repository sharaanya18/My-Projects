# TF-IDF + Linear Models

The baseline you never skip. CPU-only, seconds to train, fully interpretable,
and the reference every later rung must beat.

## Evidence

`Snigdho8869` on BBC (2,225 docs, 5 classes) `[VERIFIED]`:
LogisticRegression **95.96**, LinearSVC **96.40**, MultinomialNB **96.18**,
RandomForest 94.83, GradientBoosting 94.38, AdaBoost 94.16 — versus Keras
LSTM 92.58, GRU 91.24, CNN 95.06.

**Every from-scratch neural model lost to TF-IDF + a linear classifier**, at
orders of magnitude more compute. (Caveat: single 80/20 split, so treat the
ordering *within* the linear group as noise — the group-level gap is the
signal.)

## Configuration

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("feats", FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True,
                              min_df=2, max_df=0.9, strip_accents="unicode")),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5),
                              sublinear_tf=True, min_df=3)),
    ])),
    ("clf", LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")),
])
```

**Must be a `Pipeline`** so the vectorizer is fit inside each CV fold. Fitting
the vectorizer on the full dataset leaks IDF statistics and vocabulary from the
validation fold, and inflates CV in a way that does not transfer.

## Which linear model

| Model | Probabilities | Best at | Notes |
|---|---|---|---|
| LogisticRegression | yes | general default | needed if you must calibrate/threshold |
| LinearSVC | **no** | max accuracy on clean margins | wrap in `CalibratedClassifierCV` for probabilities |
| SGDClassifier | with `log_loss` | very large data | supports partial_fit |
| ComplementNB | yes | imbalanced, small data | better than MultinomialNB when skewed |
| Ridge | no | fast multiclass | strong and often overlooked |

**Trap:** `VotingClassifier(voting='soft')` needs `predict_proba`; `LinearSVC`
doesn't have it. `Snigdho8869` silently swaps in `SVC(probability=True)` for the
ensemble `[VERIFIED]` — which is a *different, much slower* model (O(n²) kernel
SVM), not the LinearSVC that produced the headline score. If you ensemble
LinearSVC, calibrate it explicitly and know you've changed the model.

## Hyperparameters worth tuning (in order)

1. `C` — log scale `[0.01, 0.1, 1, 10, 100]`. Biggest effect.
2. `ngram_range` — (1,1) vs (1,2) vs (1,3).
3. `min_df` / `max_df`.
4. `sublinear_tf` on/off.
5. `class_weight` balanced vs none.

Tune with `GridSearchCV` **inside** the outer CV, or accept that your CV is
mildly optimistic. `Snigdho8869` tunes on train, evaluates on test, then prints
`cross_val_score` on the same training data — the CV number there is
post-selection and therefore optimistic `[VERIFIED]`.

## Interpretability — use it

`coef_` gives per-class term weights. Read the top 30 positive and negative
terms per class. This routinely surfaces:

- **leakage** (an artifact token that perfectly predicts a class),
- **shortcuts** (formatting, boilerplate, source markers),
- **preprocessing bugs** (HTML entities, encoding mojibake, template text),
- label noise (a class whose top terms make no sense).

No transformer gives you this for free. It is the cheapest error-analysis tool
available, and it is why the baseline earns its place beyond just being a number.

## When it fails

Paraphrase without lexical overlap · word-order-dependent labels (negation
scope, NLI) · vocabulary mismatch between train and test · very short texts
(few tokens → sparse vectors) · cross-lingual matching.

Those failure modes are exactly what the next rungs address — which is why
running this one first tells you *which* rung to climb.
