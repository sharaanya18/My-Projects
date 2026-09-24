"""Fixed sparse-text feature extractor: word n-grams + char n-grams -> SVD.

Fit on training titles only (never on test titles), per the challenge's
test-data discipline. Produces a fixed-size dense vector per title that
feeds a trainable projection + tag-embedding model.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize_title


class SparseTitleFeaturizer:
    def __init__(self, word_dim: int = 200, char_dim: int = 200, seed: int = 0):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000, sublinear_tf=True
        )
        self.word_svd = TruncatedSVD(n_components=word_dim, random_state=seed)
        self.char_svd = TruncatedSVD(n_components=char_dim, random_state=seed)
        self.dim = word_dim + char_dim

    def fit(self, titles: list[str]) -> "SparseTitleFeaturizer":
        norm = [normalize_title(t) for t in titles]
        Xw = self.word_vec.fit_transform(norm)
        Xc = self.char_vec.fit_transform(norm)
        self.word_svd.fit(Xw)
        self.char_svd.fit(Xc)
        return self

    def transform(self, titles: list[str]) -> np.ndarray:
        norm = [normalize_title(t) for t in titles]
        Xw = self.word_svd.transform(self.word_vec.transform(norm))
        Xc = self.char_svd.transform(self.char_vec.transform(norm))
        feat = np.concatenate([Xw, Xc], axis=1).astype(np.float32)
        # L2-normalize each block so neither dominates the learned projection.
        feat[:, : Xw.shape[1]] /= np.linalg.norm(feat[:, : Xw.shape[1]], axis=1, keepdims=True) + 1e-8
        feat[:, Xw.shape[1] :] /= np.linalg.norm(feat[:, Xw.shape[1] :], axis=1, keepdims=True) + 1e-8
        return feat
