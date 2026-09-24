"""Exact reimplementation of the challenge grader's scoring formula.

Grader spec (from the problem description):
  * tokenize Unicode word runs and each non-whitespace punctuation character
  * Levenshtein distance D with unit insert/delete/substitute costs
  * similarity = 1 - D / max(len(P), len(T), 1)
  * case score  = 0.5 * similarity + 0.5 * exact
  * final score = arithmetic mean over cases
"""
import re

# A "word run" is a maximal run of Unicode word characters; every other
# non-whitespace character becomes a token of its own.
_TOK_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def score_tokenize(s: str):
    """Tokenize exactly the way the grader does."""
    return _TOK_RE.findall(s or "")


def levenshtein(a, b) -> int:
    """Unit-cost edit distance over two token sequences."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    cur = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[lb]


def case_score(pred: str, truth: str) -> float:
    p = score_tokenize(pred)
    t = score_tokenize(truth)
    d = levenshtein(p, t)
    sim = 1.0 - d / max(len(p), len(t), 1)
    exact = 1.0 if p == t else 0.0
    return 0.5 * sim + 0.5 * exact


def corpus_score(preds, truths) -> float:
    if not truths:
        return 0.0
    return sum(case_score(p, t) for p, t in zip(preds, truths)) / len(truths)


def score_breakdown(preds, truths):
    """Return (final, mean_similarity, exact_match_rate) for diagnostics."""
    sims, exacts = [], []
    for p, t in zip(preds, truths):
        tp, tt = score_tokenize(p), score_tokenize(t)
        d = levenshtein(tp, tt)
        sims.append(1.0 - d / max(len(tp), len(tt), 1))
        exacts.append(1.0 if tp == tt else 0.0)
    n = max(len(sims), 1)
    ms, me = sum(sims) / n, sum(exacts) / n
    return 0.5 * ms + 0.5 * me, ms, me
