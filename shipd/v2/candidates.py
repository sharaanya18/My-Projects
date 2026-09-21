"""Candidate rankers, ordered by capacity.

The evidence says capacity is the enemy here: a 71-feature fitted GBM scored
0.247 on the hidden test (below random, 0.30) while zero-parameter monotone
rules score 0.45-0.60 on train. So the candidate set spans from no parameters
to many, and each is scored on how much it ADDS over the unfitted rule --
the "memorisation premium", which is the part most at risk of not transferring.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb

# Features with a defensible causal link to "how much human review will this
# backport attract", and which are NOT project-style markers.
CAUSAL_POS = ['changed_lines', 'additions', 'deletions', 'file_count',
              'max_file_changes', 'test_file_count', 'code_file_count',
              'config_file_count', 'body_code_block_count', 'body_question_count']
CAUSAL_NEG = ['docs_file_count']

# Deliberately EXCLUDED as project-template style rather than mechanism:
#   title_length, title_digit_count, title_colon_count, title_word_count
#   body_length, body_line_count, body_unique_character_count, body_unique_word_count
#   body_mention_count (largest train->test drift in the schema)
#   every *_band categorical (banded duplicates of the numerics)
#   body_structure (template style)
EXCLUDED_STYLE = ['title_length', 'title_digit_count', 'title_colon_count',
                  'title_word_count', 'title_unique_word_count', 'body_length',
                  'body_line_count', 'body_unique_character_count',
                  'body_unique_word_count', 'body_mention_count',
                  'body_word_count', 'body_digit_count', 'body_colon_count',
                  'body_exclamation_count', 'title_exclamation_count',
                  'title_question_count', 'body_reference_count']

BINARY_TERMS = ['test_term', 'security_term', 'docs_term']


def _log(df, c):
    return np.log1p(df[c].astype(float).clip(lower=0).to_numpy())


def _z(v, mu=None, sd=None):
    v = np.asarray(v, dtype=float)
    mu = v.mean() if mu is None else mu
    sd = (v.std() + 1e-9) if sd is None else sd
    return (v - mu) / sd


class UnfittedComposite:
    """ZERO learned parameters beyond train-set standardisation constants.

    Ranks by a fixed, sign-constrained sum of log-scaled causal features. It
    cannot memorise a project because there is nothing to memorise: the weights
    are +1 / -1, set by causal reasoning, not fitted.
    """
    NAME = 'unfitted_composite'

    def __init__(self, pos=('changed_lines', 'test_file_count', 'code_file_count'),
                 neg=('docs_file_count',)):
        self.pos, self.neg = list(pos), list(neg)

    def fit(self, df, y=None):
        self.stats_ = {}
        for c in self.pos + self.neg:
            v = _log(df, c)
            self.stats_[c] = (v.mean(), v.std() + 1e-9)
        return self

    def predict(self, df):
        s = np.zeros(len(df))
        for c in self.pos:
            mu, sd = self.stats_[c]; s += _z(_log(df, c), mu, sd)
        for c in self.neg:
            mu, sd = self.stats_[c]; s -= _z(_log(df, c), mu, sd)
        return s


class MonotoneGBM:
    """LightGBM with hard monotone constraints on causal features only.

    Monotonicity is what makes extrapolation safe: a test row far outside the
    training range still gets a score in the causally correct direction, rather
    than whatever the last split happened to say. Capacity is held low on
    purpose (shallow, few leaves, high min_child_samples).
    """
    NAME = 'monotone_gbm'

    def __init__(self, n_estimators=300, num_leaves=4, min_child_samples=60,
                 learning_rate=0.05, seeds=5, use_terms=True):
        self.kw = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                       min_child_samples=min_child_samples,
                       learning_rate=learning_rate, verbose=-1)
        self.seeds, self.use_terms = seeds, use_terms

    def _X(self, df):
        cols, mono = [], []
        for c in CAUSAL_POS:
            cols.append(_log(df, c)); mono.append(1)
        for c in CAUSAL_NEG:
            cols.append(_log(df, c)); mono.append(-1)
        if self.use_terms:
            cols.append((df['test_term'] == 'present').astype(float).to_numpy()); mono.append(1)
            cols.append((df['security_term'] == 'present').astype(float).to_numpy()); mono.append(1)
            cols.append((df['docs_term'] == 'present').astype(float).to_numpy()); mono.append(-1)
        self.mono_ = mono
        return np.column_stack(cols)

    def fit(self, df, y):
        X = self._X(df)
        g = 2.0 ** np.asarray(y, dtype=float) - 1.0
        self.ms_ = [lgb.LGBMRegressor(random_state=s, monotone_constraints=self.mono_,
                                      **self.kw).fit(X, g) for s in range(self.seeds)]
        return self

    def predict(self, df):
        X = self._X(df)
        return np.mean([m.predict(X) for m in self.ms_], axis=0)


class PlainGBM:
    """Unconstrained GBM on an arbitrary feature set -- the previous approach."""
    NAME = 'plain_gbm'

    def __init__(self, cols=None, num_leaves=7, min_child_samples=40,
                 n_estimators=400, seeds=5, exclude_style=False):
        self.cols, self.exclude_style = cols, exclude_style
        self.kw = dict(n_estimators=n_estimators, num_leaves=num_leaves,
                       min_child_samples=min_child_samples, learning_rate=0.05,
                       colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
                       verbose=-1)
        self.seeds = seeds

    def _X(self, df):
        if self.cols is None:
            num = [c for c in df.columns
                   if c not in ('id', 'target') and df[c].dtype != object
                   and not str(df[c].dtype).startswith('str')]
            if self.exclude_style:
                num = [c for c in num if c not in EXCLUDED_STYLE]
            self.cols_ = num
        else:
            self.cols_ = self.cols
        return df[self.cols_].astype(float).to_numpy()

    def fit(self, df, y):
        X = self._X(df)
        g = 2.0 ** np.asarray(y, dtype=float) - 1.0
        self.ms_ = [lgb.LGBMRegressor(random_state=s, **self.kw).fit(X, g)
                    for s in range(self.seeds)]
        return self

    def predict(self, df):
        return np.mean([m.predict(df[self.cols_].astype(float).to_numpy())
                        for m in self.ms_], axis=0)
