"""Full (non-SVD-reduced) TF-IDF features: word 1-2gram + char 3-5gram,
fit on training titles only. Kept as scipy sparse matrices; the model's
input linear layer learns the dimensionality reduction end-to-end under the
task's own listwise loss, instead of an unsupervised SVD proxy."""
from __future__ import annotations

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from normalize import normalize_title


class RawTitleFeaturizer:
    def __init__(self, max_word_features: int = 40000, max_char_features: int = 40000):
        self.word_vec = TfidfVectorizer(
            analyzer="word", ngram_range=(1, 2), min_df=2, max_features=max_word_features, sublinear_tf=True
        )
        self.char_vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=max_char_features, sublinear_tf=True
        )

    def fit(self, titles: list[str]) -> "RawTitleFeaturizer":
        norm = [normalize_title(t) for t in titles]
        self.word_vec.fit(norm)
        self.char_vec.fit(norm)
        self.dim = len(self.word_vec.vocabulary_) + len(self.char_vec.vocabulary_)
        return self

    def transform(self, titles: list[str]):
        norm = [normalize_title(t) for t in titles]
        Xw = self.word_vec.transform(norm)
        Xc = self.char_vec.transform(norm)
        return sparse.hstack([Xw, Xc], format="csr")
