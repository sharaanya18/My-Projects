"""Retrieval baselines and fusion. CPU-only, no model downloads.

BM25 is implemented here directly so the retrieval rung is available with no
extra dependency and no network -- which matters, because in a blocked
environment it is the strongest retriever you can actually run.

See ../playbooks/retrieval.md.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

_WORD = re.compile(r"\w+", re.UNICODE)


def simple_tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class BM25:
    """Okapi BM25.

    k1 controls term-frequency saturation (1.2-2.0).
    b  controls length normalisation (0.5-0.9).
    TUNE BOTH -- the defaults are not optimal for short documents, and this is
    a two-parameter sweep that costs nothing.
    """

    def __init__(self, corpus, k1: float = 1.5, b: float = 0.75, tokenizer=None):
        self.k1, self.b = k1, b
        self.tokenizer = tokenizer or simple_tokenize
        self.docs = [self.tokenizer(d) for d in corpus]
        self.n_docs = len(self.docs)
        self.doc_len = np.array([len(d) for d in self.docs], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0

        df: Counter = Counter()
        self.tf: list[Counter] = []
        for d in self.docs:
            c = Counter(d)
            self.tf.append(c)
            df.update(c.keys())

        # Robertson/Sparck-Jones idf with the +0.5 smoothing, floored at 0
        self.idf = {
            t: max(0.0, math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0))
            for t, n in df.items()
        }
        self.postings: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings[t].append(i)

    def get_scores(self, query: str) -> np.ndarray:
        q = self.tokenizer(query)
        scores = np.zeros(self.n_docs, dtype=float)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += idf * (f * (self.k1 + 1)) / denom
        return scores

    def top_k(self, query: str, k: int = 10):
        s = self.get_scores(query)
        k = min(k, self.n_docs)
        idx = np.argpartition(-s, k - 1)[:k] if k < self.n_docs else np.arange(self.n_docs)
        idx = idx[np.argsort(-s[idx])]
        return [(int(i), float(s[i])) for i in idx]


class TfidfRetriever:
    """TF-IDF cosine retrieval. A cheap second lexical view for fusion."""

    def __init__(self, corpus, analyzer: str = "word", ngram_range=(1, 2), **kw):
        params = dict(sublinear_tf=True, min_df=1)
        if analyzer == "char":
            params.update(analyzer="char_wb", ngram_range=(3, 5))
        else:
            params.update(analyzer="word", ngram_range=ngram_range)
        params.update(kw)
        self.vec = TfidfVectorizer(**params)
        self.X = self.vec.fit_transform(corpus)
        self.n_docs = self.X.shape[0]

    def get_scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        return np.asarray((self.X @ q.T).todense()).ravel()

    def top_k(self, query: str, k: int = 10):
        s = self.get_scores(query)
        idx = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in idx]


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def reciprocal_rank_fusion(rankings, k: float = 60.0, weights=None, top_k=None):
    """RRF over several ranked lists of doc ids.

    Needs NO score normalisation -- it uses ranks, so BM25's unbounded scores
    and cosine's [-1,1] never have to be made commensurable. That robustness
    is why it is the default fusion here.

    `rankings`: list of ranked id lists (best first), one per system.
    """
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for w, ranked in zip(weights, rankings):
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] += w / (k + rank)
    out = sorted(fused.items(), key=lambda kv: -kv[1])
    return out[:top_k] if top_k else out


def hybrid_search(query, retrievers, k_each: int = 100, rrf_k: float = 60.0,
                  weights=None, top_k: int = 10):
    """Run every retriever, fuse with RRF. The rung that usually jumps most."""
    rankings = [[i for i, _ in r.top_k(query, k_each)] for r in retrievers]
    return reciprocal_rank_fusion(rankings, k=rrf_k, weights=weights, top_k=top_k)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def recall_at_k(ranked_ids, relevant_ids, k: int) -> float:
    rel = set(relevant_ids)
    if not rel:
        return float("nan")
    return len(set(ranked_ids[:k]) & rel) / len(rel)


def reciprocal_rank(ranked_ids, relevant_ids) -> float:
    rel = set(relevant_ids)
    for i, d in enumerate(ranked_ids, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def dcg(gains) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids, relevance: dict, k: int) -> float:
    gains = [relevance.get(d, 0.0) for d in ranked_ids[:k]]
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    return (dcg(gains) / idcg) if idcg > 0 else 0.0


def evaluate_runs(runs: dict, qrels: dict, ks=(1, 5, 10, 50)) -> "pd.DataFrame":
    """Evaluate per query, then average -- never pool rows across queries.

    `runs`  : {query_id: [ranked doc ids]}
    `qrels` : {query_id: {doc_id: graded_relevance}}

    ALWAYS report Recall@k alongside the ranking metric: it is the ceiling a
    reranker can never exceed.
    """
    import pandas as pd

    rows = []
    for qid, ranked in runs.items():
        rel = qrels.get(qid, {})
        positives = [d for d, g in rel.items() if g > 0]
        row = {"query_id": qid, "mrr": reciprocal_rank(ranked, positives)}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, positives, k)
            row[f"ndcg@{k}"] = ndcg_at_k(ranked, rel, k)
        rows.append(row)
    per_query = pd.DataFrame(rows)
    summary = per_query.drop(columns=["query_id"]).agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]
    return per_query, summary
