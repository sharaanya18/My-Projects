"""Proxy GroupKFold: cluster titles so that near-duplicate / same-institution-
style titles land in the same fold. The real split is by holding institution
(not present in the supplied columns), so this clustering is only a proxy
used to get a pessimistic, leakage-aware local validation signal -- never a
model input.
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

from normalize import normalize_title


def cluster_groups(titles: list[str], n_clusters: int = 24, svd_dim: int = 48, seed: int = 0) -> np.ndarray:
    norm = [normalize_title(t) for t in titles]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    X = vec.fit_transform(norm)
    n_comp = min(svd_dim, X.shape[1] - 1, X.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    Xr = svd.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    return km.fit_predict(Xr)


def group_kfold_splits(groups: np.ndarray, n_splits: int = 5, seed: int = 0):
    gkf = GroupKFold(n_splits=n_splits)
    dummy_X = np.zeros((len(groups), 1))
    return list(gkf.split(dummy_X, groups=groups))
