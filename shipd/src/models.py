"""Candidate estimators. All expose .fit/.predict returning ONE ranking score."""
from __future__ import annotations
import numpy as np, lightgbm as lgb
from sklearn.base import BaseEstimator

BASE = dict(n_estimators=400, learning_rate=0.05, num_leaves=15, min_child_samples=20,
            colsample_bytree=0.8, verbose=-1, random_state=0)


class GainRegressor(BaseEstimator):
    """Regress the metric's own gain 2^r - 1."""
    def __init__(self, **kw):
        self.kw = kw
    def get_params(self, deep=True): return dict(self.kw)
    def set_params(self, **kw): self.kw.update(kw); return self
    def fit(self, X, y):
        p = dict(BASE, subsample=0.8, subsample_freq=1); p.update(self.kw)
        self.m_ = lgb.LGBMRegressor(**p).fit(X, 2.0 ** np.asarray(y, float) - 1.0)
        return self
    def predict(self, X): return self.m_.predict(X)


class GlobalRanker(BaseEstimator):
    """LambdaRank over one global list.

    CRITICAL: lambdarank_truncation_level defaults to 30. With a single group
    of ~1900 rows that means only 30 rows ever receive gradient and the model
    is starved. Setting it above the group size is worth ~+0.065 NDCG@20
    (measured). The metric's cutoff is 20, but the TRAINING truncation must be
    large -- they are not the same knob.
    """
    def __init__(self, truncation=4000, **kw):
        self.truncation = truncation; self.kw = kw
    def get_params(self, deep=True): return dict(self.kw, truncation=self.truncation)
    def set_params(self, **kw):
        self.truncation = kw.pop('truncation', self.truncation); self.kw.update(kw); return self
    def fit(self, X, y):
        y = np.asarray(y)
        p = dict(BASE); p.update(self.kw)
        self.m_ = lgb.LGBMRanker(objective='lambdarank', metric='ndcg',
                                 label_gain=[0, 1, 3],
                                 lambdarank_truncation_level=self.truncation, **p)
        self.m_.fit(X, y, group=[len(y)])
        return self
    def predict(self, X): return self.m_.predict(X)
