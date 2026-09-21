"""Official NDCG@20, copied verbatim from the challenge statement."""
import numpy as np
TOP_K = 20

def ndcg_at_k(truth, prediction, k=TOP_K):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size == 0:
        raise ValueError("The answer set cannot be empty.")
    cutoff = min(k, truth.size)
    gains = 2.0 ** truth - 1.0
    order = np.argsort(-prediction, kind="mergesort")[:cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    dcg = float(np.sum(gains[order] * discounts))
    ideal_gains = np.sort(gains)[::-1][:cutoff]
    ideal_dcg = float(np.sum(ideal_gains * discounts))
    return 0.0 if ideal_dcg <= 0.0 else dcg / ideal_dcg
