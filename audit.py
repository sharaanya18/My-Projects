import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import solution as sol
import validate as V

LABELS = np.array([1, 2, 3])


def group_keys(frame):
    sorted_tokens = frame["report_tokens"].map(lambda s: " ".join(sorted(str(s).split())))
    all_fields = frame[sol.TOKEN_FIELDS].agg("|".join, axis=1)
    return {
        "report_exact": frame["report_tokens"].to_numpy(),
        "report_multiset": sorted_tokens.to_numpy(),
        "all_fields": all_fields.to_numpy(),
    }


def jaccard_clusters(frame, threshold=0.8):
    token_sets = [set(str(s).split()) for s in frame["report_tokens"]]
    postings = {}
    for i, tokens in enumerate(token_sets):
        for t in tokens:
            postings.setdefault(t, []).append(i)
    parent = list(range(len(token_sets)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, tokens in enumerate(token_sets):
        candidates = set()
        for t in tokens:
            if len(postings[t]) <= 200:
                candidates.update(postings[t])
        for j in candidates:
            if j <= i:
                continue
            a, b = token_sets[i], token_sets[j]
            union_size = len(a | b)
            if union_size and len(a & b) / union_size >= threshold:
                union(i, j)
    return np.array([find(i) for i in range(len(token_sets))])


def fold_eval(train, evaluate, y_train, variant, key_train, key_eval):
    probabilities = sol.fold_probabilities(train, evaluate, y_train, variant) if hasattr(sol, "fold_probabilities") \
        else V.fold_probabilities(train, evaluate, y_train, variant)
    priors = np.array([(y_train == c).mean() for c in LABELS])
    predictions = sol.decide(probabilities, priors)
    matched = np.isin(key_eval, key_train)
    return predictions, matched


def run_scheme(frame, variant, splitter, groups, key, seeds, n_jobs):
    y = frame["target"].to_numpy(int)
    keys = group_keys(frame)[key]
    rows = []
    for seed in seeds:
        splits = list(splitter(seed, frame, y, groups))
        results = Parallel(n_jobs=n_jobs)(
            delayed(fold_eval)(
                frame.iloc[fit], frame.iloc[val], y[fit], variant, keys[fit], keys[val]
            )
            for fit, val in splits
        )
        predictions = np.zeros(len(y), int)
        matched = np.zeros(len(y), bool)
        for (fit, val), (pred, match) in zip(splits, results):
            predictions[val] = pred
            matched[val] = match
        total, recall, _, _ = V.score_predictions(frame["id"].to_numpy(), y, predictions)
        sub = {}
        for label, mask in (("matched", matched), ("unmatched", ~matched)):
            if mask.sum() > 30:
                sub[label] = (
                    V.score_predictions(frame["id"].to_numpy()[mask], y[mask], predictions[mask])[0],
                    int(mask.sum()),
                )
        rows.append((total, recall, matched.mean(), sub))
    return rows


def plain_splitter(seed, frame, y, groups):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(frame, y)


def grouped_splitter(seed, frame, y, groups):
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(frame, y, groups)


def match_calibrated_estimate(frame, test, variant, seeds, n_jobs, key="report_exact"):
    y = frame["target"].to_numpy(int)
    keys_train = group_keys(frame)[key]
    keys_test = group_keys(test)[key]

    loo_matched = np.zeros(len(frame), bool)
    counts = pd.Series(keys_train).value_counts()
    for i, k in enumerate(keys_train):
        loo_matched[i] = counts[k] > 1
    p_match_train = np.array([loo_matched[y == c].mean() for c in LABELS])
    test_match_rate = np.isin(keys_test, keys_train).mean()
    scale = test_match_rate / loo_matched.mean()

    recalls = {"matched": [], "unmatched": []}
    cv_match = []
    for seed in seeds:
        splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(frame, y))
        results = Parallel(n_jobs=n_jobs)(
            delayed(fold_eval)(frame.iloc[fit], frame.iloc[val], y[fit], variant,
                               keys_train[fit], keys_train[val])
            for fit, val in splits
        )
        predictions = np.zeros(len(y), int)
        matched = np.zeros(len(y), bool)
        for (fit, val), (pred, match) in zip(splits, results):
            predictions[val] = pred
            matched[val] = match
        cv_match.append(np.array([matched[y == c].mean() for c in LABELS]))
        for label, mask in (("matched", matched), ("unmatched", ~matched)):
            recalls[label].append(
                [float((predictions[(y == c) & mask] == c).mean()) if ((y == c) & mask).sum() else np.nan
                 for c in LABELS]
            )

    r_matched = np.nanmean(recalls["matched"], axis=0)
    r_unmatched = np.nanmean(recalls["unmatched"], axis=0)
    cv_match = np.mean(cv_match, axis=0)
    p_test = np.clip(p_match_train * scale, 0, 1)

    cv_recall = cv_match * r_matched + (1 - cv_match) * r_unmatched
    test_recall = p_test * r_matched + (1 - p_test) * r_unmatched
    print(f"\n=== match-rate calibrated estimate (key={key}) ===")
    print(f"class-conditional recall | matched   = {np.round(r_matched, 3)}")
    print(f"class-conditional recall | unmatched = {np.round(r_unmatched, 3)}")
    print(f"P(matched|class) in plain CV         = {np.round(cv_match, 3)}")
    print(f"P(matched|class) expected for test   = {np.round(p_test, 3)}")
    print(f"implied balanced coverage: CV={cv_recall.mean():.4f}  test={test_recall.mean():.4f}"
          f"  (difference {test_recall.mean() - cv_recall.mean():+.4f})")
    print(f"implied score shift from overlap alone: {0.75 * (test_recall.mean() - cv_recall.mean()):+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("public_dir", type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    train, test = sol.load_data(args.public_dir)
    train = V.add_extra_crossings(train).reset_index(drop=True)
    test = V.add_extra_crossings(test).reset_index(drop=True)
    y = train["target"].to_numpy(int)
    seeds = tuple(range(args.seeds))
    variant = V.VARIANTS["oof_te"]

    print("=== overlap of evaluation rows with their fitting rows ===")
    train_keys, test_keys = group_keys(train), group_keys(test)
    for key in train_keys:
        test_rate = np.isin(test_keys[key], train_keys[key]).mean()
        fold_rates = []
        for seed in seeds:
            for fit, val in StratifiedKFold(5, shuffle=True, random_state=seed).split(train, y):
                fold_rates.append(np.isin(train_keys[key][val], train_keys[key][fit]).mean())
        print(
            f"{key:16s} test-vs-full-train={test_rate:.3f}  plain-CV val-vs-fit={np.mean(fold_rates):.3f}"
            f"  (gap {np.mean(fold_rates) - test_rate:+.3f})"
        )

    print("\n=== validation schemes, same model (oof_te) ===")
    schemes = [("plain StratifiedKFold", plain_splitter, None, "report_exact")]
    for key in ("report_exact", "report_multiset", "all_fields"):
        schemes.append((f"grouped by {key}", grouped_splitter, group_keys(train)[key], key))
    clusters = jaccard_clusters(train, 0.8)
    print(f"jaccard>=0.8 clusters: {len(np.unique(clusters))} for {len(train)} rows")
    schemes.append(("grouped by jaccard>=0.8", grouped_splitter, clusters, "report_exact"))

    for name, splitter, groups, key in schemes:
        rows = run_scheme(train, variant, splitter, groups, key, seeds, args.n_jobs)
        scores = np.array([r[0] for r in rows])
        recall = np.mean([r[1] for r in rows], axis=0)
        match_rate = np.mean([r[2] for r in rows])
        line = f"{name:26s} score={scores.mean():.4f} +/-{scores.std():.4f}  recall={np.round(recall,3)}  match_rate={match_rate:.3f}"
        print(line)
        for label in ("matched", "unmatched"):
            vals = [r[3][label] for r in rows if label in r[3]]
            if vals:
                print(f"    {label:10s} n~{int(np.mean([v[1] for v in vals])):5d} score={np.mean([v[0] for v in vals]):.4f}")

    match_calibrated_estimate(train, test, variant, seeds, args.n_jobs)


if __name__ == "__main__":
    main()
