"""Compositional-holdout fold construction (development instrument only).

The real test set is 100% compositional shift: every evaluation command contains
an ordered head->head adjacency that appears in *no* training command.  A random
90/10 split therefore massively over-reports.  These folds reproduce the real
construction on the public training data so model selection is done against a
number that actually predicts leaderboard behaviour.

Nothing in this module runs inside the graded submission script.
"""
import random
from collections import Counter, defaultdict

from shlexutil import split_top_level_pipes, stage_head


def row_pairs(output: str):
    """Ordered adjacent head pairs of one command."""
    heads = [stage_head(s) for s in split_top_level_pipes(output)]
    return list(zip(heads, heads[1:]))


def pair_index(rows):
    """pair -> set of row indices containing it, and the global pair counter."""
    idx = defaultdict(set)
    cnt = Counter()
    for i, r in enumerate(rows):
        for p in row_pairs(r["output"]):
            idx[p].add(i)
            cnt[p] += 1
    return idx, cnt


def canonical_group(output: str):
    """Group key used to keep paraphrases of the same command on one side.

    The challenge states groups are formed by canonically tokenized complete
    commands and never cross the split; we mirror that locally.
    """
    return " ".join(split_top_level_pipes(output))


def make_compositional_folds(rows, n_folds=6, target_val=180, seed=0):
    """Hold out whole ordered head-pairs, mimicking the real split.

    For each fold we greedily accumulate pairs (mixing singletons with slightly
    more frequent ones, as the real held-out set does) until the union of rows
    containing them reaches roughly `target_val` rows.  Every row containing a
    held-out pair leaves training for that fold and becomes validation.
    """
    idx, cnt = pair_index(rows)
    # Only pairs from multi-stage commands are eligible (a 1-stage command has
    # no adjacency); restrict to pairs that are not overwhelmingly frequent so
    # a fold does not delete a large slice of training.
    eligible = [p for p, c in cnt.items() if c <= 17]
    rng = random.Random(seed)
    folds = []
    for f in range(n_folds):
        r = random.Random(seed * 1000 + f)
        pool = list(eligible)
        r.shuffle(pool)
        # bias toward the frequency mix of the real held-out set: a few pairs
        # with real support plus a tail of rare ones
        pool.sort(key=lambda p: -cnt[p] if r.random() < 0.35 else 0)
        held, val = [], set()
        for p in pool:
            if len(val) >= target_val:
                break
            add = idx[p]
            if not add:
                continue
            held.append(p)
            val |= add
        # every row sharing a canonical command with a validation row also moves
        groups = {canonical_group(rows[i]["output"]) for i in val}
        val |= {i for i, row in enumerate(rows) if canonical_group(row["output"]) in groups}
        train = [i for i in range(len(rows)) if i not in val]
        folds.append({"held_pairs": held, "train": train, "val": sorted(val)})
    return folds


def random_fold(rows, frac=0.1, seed=0):
    """Plain random split -- secondary underfitting check only."""
    r = random.Random(seed)
    order = list(range(len(rows)))
    r.shuffle(order)
    k = int(len(rows) * frac)
    return {"held_pairs": [], "val": sorted(order[:k]), "train": sorted(order[k:])}


def make_frequent_pair_folds(rows, n_folds=5, pairs_per_fold=6, min_head_rows=20,
                             max_per_pair=50, seed=0):
    """Folds that mirror the documented test construction more faithfully.

    The problem statement says the evaluation set covers a handful of ordered
    pairs (six represented, <=50 cases each, 160 cases in total -- so these are
    *frequent* pairs) and that every evaluated head keeps >=20 training
    examples.  So: choose among the more frequent pairs whose two heads would
    each still have >=min_head_rows training rows after the removal, hold out
    `pairs_per_fold` of them, and remove every row containing any of them.
    """
    idx, cnt = pair_index(rows)
    head_rows = defaultdict(set)
    for i, r in enumerate(rows):
        for h in {stage_head(s) for s in split_top_level_pipes(r["output"])}:
            head_rows[h].add(i)
    # frequent pairs, excluding self-loops like (grep, grep)
    cand = [p for p, c in cnt.most_common() if c >= 4 and p[0] != p[1]]
    folds = []
    for f in range(n_folds):
        r = random.Random(seed * 7919 + f)
        pool = list(cand)
        r.shuffle(pool)
        held, val = [], set()
        for p in pool:
            if len(held) >= pairs_per_fold:
                break
            new_val = val | idx[p]
            ok = all(len(head_rows[h] - new_val) >= min_head_rows for h in p)
            ok = ok and all(len(head_rows[h] - new_val) >= min_head_rows
                            for q in held for h in q)
            if ok:
                held.append(p)
                val = new_val
        groups = {canonical_group(rows[i]["output"]) for i in val}
        val |= {i for i, row in enumerate(rows) if canonical_group(row["output"]) in groups}
        train = [i for i in range(len(rows)) if i not in val]
        folds.append({"held_pairs": held, "train": train, "val": sorted(val)})
    return folds
