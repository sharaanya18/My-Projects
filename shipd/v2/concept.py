"""Concept-averaged composite: no greedy selection, no correlated over-weighting.

C2 (equal weight over all 12 causal features) under-performs because seven of
them are size measures that are mutually correlated, so "size" silently gets 7x
the weight of "touches tests" -- which is the single strongest signal.

Grouping features into independent CONCEPTS and averaging within each concept
fixes that without fitting anything: the weights are 1/|concept|, set by the
structure of the feature list, not by the labels.
"""
import numpy as np

CONCEPTS = {
    'size':       (+1, ['changed_lines', 'additions', 'deletions', 'file_count',
                        'max_file_changes', 'max_file_additions', 'max_file_deletions']),
    'tests':      (+1, ['test_file_count']),
    'code':       (+1, ['code_file_count']),
    'discussion': (+1, ['body_code_block_count', 'body_question_count']),
    'docs':       (-1, ['docs_file_count']),
}


class ConceptComposite:
    """Unfitted apart from train-set log-mean/std used to standardise."""
    def __init__(self, concepts=CONCEPTS):
        self.concepts = concepts
        self.cols = [c for _, cs in concepts.values() for c in cs]

    def fit(self, df, y=None):
        self.st_ = {}
        for c in self.cols:
            v = np.log1p(df[c].astype(float).clip(lower=0).to_numpy())
            self.st_[c] = (float(v.mean()), float(v.std() + 1e-9))
        return self

    def _z(self, df, c):
        v = np.log1p(df[c].astype(float).clip(lower=0).to_numpy())
        mu, sd = self.st_[c]
        return (v - mu) / sd

    def predict(self, df):
        s = np.zeros(len(df))
        for sign, cs in self.concepts.values():
            s += sign * np.mean([self._z(df, c) for c in cs], axis=0)
        return s
