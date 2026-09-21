"""Lexical / pair / retrieval features for NLP competition tasks.

Pure numpy + sklearn. No GPU, no model downloads.

The pair-feature block generalises `percent_shared` from
ElizaLo/NLP-Natural-Language-Processing (Text Similarity), extended with
IDF weighting, n-gram views, and asymmetric coverage.

Every feature here is a *candidate*. Test feature groups with an ablation
before keeping them (see techniques/feature_engineering.md).
"""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

_WORD = re.compile(r"\w+", re.UNICODE)


def tokens(text: str, lower: bool = True) -> list[str]:
    t = text.lower() if lower else text
    return _WORD.findall(t)


def ngrams(seq, n: int) -> set:
    if n <= 1:
        return set(seq)
    return {tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)}


def char_ngrams(text: str, n: int = 4, lower: bool = True) -> set:
    t = text.lower() if lower else text
    return {t[i : i + n] for i in range(len(t) - n + 1)}


# --------------------------------------------------------------------------
# single-text features
# --------------------------------------------------------------------------

def single_text_features(text: str, prefix: str = "") -> dict:
    toks = tokens(text)
    n_chars = len(text)
    n_words = len(toks)
    uniq = len(set(toks))
    digits = sum(c.isdigit() for c in text)
    uppers = sum(c.isupper() for c in text)
    puncts = sum((not c.isalnum()) and (not c.isspace()) for c in text)
    return {
        f"{prefix}n_chars": n_chars,
        f"{prefix}n_words": n_words,
        f"{prefix}n_unique": uniq,
        f"{prefix}type_token_ratio": uniq / n_words if n_words else 0.0,
        f"{prefix}avg_word_len": (sum(len(t) for t in toks) / n_words) if n_words else 0.0,
        f"{prefix}digit_ratio": digits / n_chars if n_chars else 0.0,
        f"{prefix}upper_ratio": uppers / n_chars if n_chars else 0.0,
        f"{prefix}punct_ratio": puncts / n_chars if n_chars else 0.0,
        f"{prefix}n_sentences": text.count(".") + text.count("!") + text.count("?"),
    }


# --------------------------------------------------------------------------
# pair features
# --------------------------------------------------------------------------

def _overlap_stats(sa: set, sb: set, prefix: str) -> dict:
    inter = sa & sb
    union = sa | sb
    ni, nu, na, nb = len(inter), len(union), len(sa), len(sb)
    return {
        f"{prefix}_inter": float(ni),
        f"{prefix}_jaccard": ni / nu if nu else 0.0,
        f"{prefix}_cover_a": ni / na if na else 0.0,   # asymmetric: keep both
        f"{prefix}_cover_b": ni / nb if nb else 0.0,
        f"{prefix}_cover_diff": abs((ni / na if na else 0.0) - (ni / nb if nb else 0.0)),
        f"{prefix}_dice": (2 * ni / (na + nb)) if (na + nb) else 0.0,
    }


class IDFTable:
    """Document frequencies over a corpus, for IDF-weighted overlap."""

    def __init__(self, corpus, lower: bool = True):
        self.n_docs = 0
        df: dict[str, int] = {}
        for text in corpus:
            self.n_docs += 1
            for tok in set(tokens(text, lower=lower)):
                df[tok] = df.get(tok, 0) + 1
        self.df = df
        self._default = math.log((self.n_docs + 1) / 1.0) + 1.0

    def idf(self, token: str) -> float:
        d = self.df.get(token)
        if d is None:
            return self._default
        return math.log((self.n_docs + 1) / (d + 1)) + 1.0

    def weighted_overlap(self, sa: set, sb: set) -> tuple[float, float]:
        inter, union = sa & sb, sa | sb
        wi = sum(self.idf(t) for t in inter)
        wu = sum(self.idf(t) for t in union)
        return wi, (wi / wu if wu else 0.0)


def pair_features(a: str, b: str, idf: IDFTable | None = None) -> dict:
    ta, tb = tokens(a), tokens(b)
    sa, sb = set(ta), set(tb)

    feats: dict[str, float] = {}
    feats.update(_overlap_stats(sa, sb, "word"))
    feats.update(_overlap_stats(ngrams(ta, 2), ngrams(tb, 2), "bigram"))
    feats.update(_overlap_stats(char_ngrams(a, 4), char_ngrams(b, 4), "char4"))

    if idf is not None:
        wi, wr = idf.weighted_overlap(sa, sb)
        feats["idf_overlap_sum"] = wi
        feats["idf_overlap_ratio"] = wr

    la, lb = len(a), len(b)
    wa, wb = len(ta), len(tb)
    feats.update({
        "len_a": float(la), "len_b": float(lb),
        "len_absdiff": float(abs(la - lb)),
        "len_ratio": (min(la, lb) / max(la, lb)) if max(la, lb) else 0.0,
        "words_a": float(wa), "words_b": float(wb),
        "words_absdiff": float(abs(wa - wb)),
        "words_ratio": (min(wa, wb) / max(wa, wb)) if max(wa, wb) else 0.0,
    })

    # string similarity
    feats["seq_ratio"] = SequenceMatcher(None, a, b).ratio()

    # high-signal token classes: numbers and capitalised tokens
    num_a = {t for t in ta if t.isdigit()}
    num_b = {t for t in tb if t.isdigit()}
    feats["num_shared"] = float(len(num_a & num_b))
    feats["num_jaccard"] = len(num_a & num_b) / len(num_a | num_b) if (num_a | num_b) else 0.0

    cap_a = {t for t in _WORD.findall(a) if t[:1].isupper()}
    cap_b = {t for t in _WORD.findall(b) if t[:1].isupper()}
    feats["cap_shared"] = float(len(cap_a & cap_b))
    feats["cap_jaccard"] = len(cap_a & cap_b) / len(cap_a | cap_b) if (cap_a | cap_b) else 0.0

    return feats


def pair_feature_frame(pairs, idf: IDFTable | None = None) -> pd.DataFrame:
    """pairs: iterable of (a, b). Returns a DataFrame, one row per pair."""
    return pd.DataFrame([pair_features(a, b, idf) for a, b in pairs])


def distance_to_similarity(d: np.ndarray) -> np.ndarray:
    """1/exp(d) -- outlier-robust, unlike min-max (ElizaLo, Text Similarity)."""
    return 1.0 / np.exp(np.asarray(d, dtype=float))


# --------------------------------------------------------------------------
# group-normalised features (ranking / reranking)
# --------------------------------------------------------------------------

def group_normalize(df: pd.DataFrame, group_col: str, score_cols) -> pd.DataFrame:
    """Within-group rank / z-score / delta-to-max features.

    These are what make scores comparable ACROSS queries. Usually the single
    highest-value transformation for a ranking model.
    """
    out = df.copy()
    g = out.groupby(group_col, sort=False)
    for col in score_cols:
        s = out[col]
        out[f"{col}__rank"] = g[col].rank(ascending=False, method="average")
        out[f"{col}__rrank"] = 1.0 / (60.0 + out[f"{col}__rank"])
        mx = g[col].transform("max")
        mean = g[col].transform("mean")
        std = g[col].transform("std").fillna(0.0)
        out[f"{col}__minus_max"] = s - mx
        out[f"{col}__minus_mean"] = s - mean
        out[f"{col}__zscore"] = np.where(std > 0, (s - mean) / std.replace(0, np.nan), 0.0)
        out[f"{col}__zscore"] = out[f"{col}__zscore"].fillna(0.0)
    out["group_size"] = g[score_cols[0]].transform("size")
    return out
