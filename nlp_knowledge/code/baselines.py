"""Baseline ladder for text classification. CPU-only, no model downloads.

Run this BEFORE anything expensive. It gives you:
  - the metric floor (majority class)
  - the TF-IDF reference every later rung must beat
  - a fold-wise std, i.e. your NOISE BAND -- the number that decides whether
    any later "improvement" is real

See ../playbooks/classification.md for the full ladder.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC


def word_tfidf(**kw) -> TfidfVectorizer:
    params = dict(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_df=0.9,
                  strip_accents="unicode")
    params.update(kw)
    return TfidfVectorizer(**params)


def char_tfidf(**kw) -> TfidfVectorizer:
    params = dict(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=3)
    params.update(kw)
    return TfidfVectorizer(**params)


def build_ladder(balanced: bool = True) -> dict[str, Pipeline]:
    """The classification baseline ladder, as sklearn Pipelines.

    Everything is a Pipeline so the vectoriser is fit INSIDE each CV fold.
    Fitting it on the full dataset leaks IDF statistics from the validation
    fold and inflates CV.
    """
    cw = "balanced" if balanced else None
    union = FeatureUnion([("w", word_tfidf()), ("c", char_tfidf())])

    return {
        "B0_majority": Pipeline([
            ("vec", word_tfidf()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "B1_word_tfidf_logreg": Pipeline([
            ("vec", word_tfidf()),
            ("clf", LogisticRegression(max_iter=2000, class_weight=cw)),
        ]),
        "B2_word+char_logreg": Pipeline([
            ("vec", union),
            ("clf", LogisticRegression(max_iter=2000, class_weight=cw)),
        ]),
        "B3_word+char_linearsvc": Pipeline([
            ("vec", union),
            ("clf", LinearSVC(class_weight=cw)),
        ]),
        "B4_word_complementnb": Pipeline([
            ("vec", word_tfidf()),
            ("clf", ComplementNB()),
        ]),
        "B5_word+char_ridge": Pipeline([
            ("vec", union),
            ("clf", RidgeClassifier(class_weight=cw)),
        ]),
    }


def run_ladder(texts, y, scoring: str = "f1_macro", n_splits: int = 5,
               random_state: int = 42, models: dict | None = None,
               verbose: bool = True) -> pd.DataFrame:
    """Run the ladder under a fixed StratifiedKFold and report mean +/- std.

    The `std` column is your noise band. Any later experiment that improves
    the mean by less than this has not been shown to improve anything.
    """
    texts = list(texts)
    y = np.asarray(y)
    models = models or build_ladder()
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    rows = []
    for name, pipe in models.items():
        t0 = time.perf_counter()
        try:
            res = cross_validate(pipe, texts, y, cv=cv, scoring=scoring,
                                 n_jobs=-1, error_score="raise")
            scores = res["test_score"]
            rows.append({
                "experiment": name,
                "metric": scoring,
                "mean": float(scores.mean()),
                "std": float(scores.std()),
                "min": float(scores.min()),
                "max": float(scores.max()),
                "seconds": round(time.perf_counter() - t0, 2),
                "status": "ok",
            })
        except Exception as exc:                      # noqa: BLE001
            rows.append({"experiment": name, "metric": scoring, "mean": np.nan,
                         "std": np.nan, "min": np.nan, "max": np.nan,
                         "seconds": round(time.perf_counter() - t0, 2),
                         "status": f"FAILED: {type(exc).__name__}: {exc}"})
        if verbose:
            print(f"  {rows[-1]['experiment']:<28} "
                  f"{rows[-1]['mean']:.4f} +/- {rows[-1]['std']:.4f}  "
                  f"({rows[-1]['seconds']}s)  {rows[-1]['status']}")

    df = pd.DataFrame(rows).sort_values("mean", ascending=False, na_position="last")
    if verbose and len(df) > 1:
        ok = df[df.status == "ok"]
        if len(ok) >= 2:
            best, second = ok.iloc[0], ok.iloc[1]
            gap = best["mean"] - second["mean"]
            band = max(best["std"], second["std"])
            verdict = ("REAL" if gap > band else
                       "WITHIN NOISE -- do not claim a winner")
            print(f"\n  best={best['experiment']} gap over #2 = {gap:.4f}, "
                  f"noise band = {band:.4f} -> {verdict}")
    return df


def top_terms(pipe: Pipeline, class_index: int = 0, k: int = 25) -> dict:
    """Read a fitted linear model's top terms per class.

    The cheapest error-analysis and leakage detector available. Leakage
    tokens, template boilerplate, and preprocessing bugs show up here
    immediately -- no transformer gives you this for free.
    """
    vec = pipe.named_steps.get("vec")
    clf = pipe.named_steps.get("clf")
    names = np.asarray(vec.get_feature_names_out())
    coef = clf.coef_
    c = coef[class_index] if coef.ndim > 1 else coef
    order = np.argsort(c)
    return {"top_positive": names[order[::-1][:k]].tolist(),
            "top_negative": names[order[:k]].tolist()}
