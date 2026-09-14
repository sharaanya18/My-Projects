import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

import argparse
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

import solution as sol
import validate as V

LABELS = np.array([1, 2, 3])


def utilities(probabilities, priors):
    return (
        0.75 / 3.0 * (probabilities / np.maximum(priors, 1e-9)[None, :])
        + 0.25 * (probabilities @ sol.ORDINAL.T)
    )


def quota_assign(probabilities, priors, lower, upper):
    n = len(probabilities)
    cost = -utilities(probabilities, priors).reshape(-1)
    rows = sparse.kron(sparse.eye(n, format="csr"), np.ones((1, 3))).tocsr()
    per_class = sparse.kron(np.ones((1, n)), sparse.eye(3, format="csr")).tocsr()
    result = linprog(
        cost,
        A_eq=rows,
        b_eq=np.ones(n),
        A_ub=sparse.vstack([per_class, -per_class]).tocsr(),
        b_ub=np.concatenate([upper, -lower]),
        bounds=(0, 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return LABELS[result.x.reshape(n, 3).argmax(axis=1)]


def bounds_for(n, priors, tolerance):
    target = priors * n
    if tolerance is None:
        low = np.floor(target).astype(float)
        high = np.ceil(target).astype(float)
        return low, high
    low = np.maximum(target * (1.0 - tolerance), 0.0)
    high = np.minimum(target * (1.0 + tolerance), float(n))
    return low, high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("public_dir", type=Path)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    dev, _ = V.load_dev_holdout(args.public_dir)
    y = dev["target"].to_numpy(int)
    ids = dev["id"].to_numpy()
    n = len(y)

    # quotas come from training labels only; the published split is stratified
    priors = np.array([(y == c).mean() for c in LABELS])
    print(f"dev={n} rows  train priors={np.round(priors, 4)}  implied counts={np.round(priors * n, 1)}")

    per_seed = V.cross_validate(dev, V.VARIANTS["oof_te"], tuple(range(args.seeds)), args.n_jobs)
    oof_by_seed = [r[1] for r in per_seed]

    rules = [("baseline decide()", None)]
    rules.append(("quota exact", None))
    for tolerance in (0.10, 0.25, 0.50, 1.00):
        rules.append((f"quota +/-{int(tolerance * 100)}%", tolerance))

    results = {}
    for name, tolerance in rules:
        scores, recalls, confusions, distributions = [], [], [], []
        for probabilities in oof_by_seed:
            if name == "baseline decide()":
                predictions = sol.decide(probabilities, priors)
            else:
                low, high = bounds_for(n, priors, tolerance)
                predictions = quota_assign(probabilities, priors, low, high)
            total, recall, confusion, distribution = V.score_predictions(ids, y, predictions)
            scores.append(total)
            recalls.append(recall)
            confusions.append(confusion)
            distributions.append(distribution)
        results[name] = np.array(scores)
        line = f"{name:20s} cv={np.mean(scores):.4f} +/-{np.std(scores):.4f}"
        if name != "baseline decide()":
            gain = np.array(scores) - results["baseline decide()"]
            line += f"  gain={gain.mean():+.4f} +/-{gain.std():.4f}"
        print("\n" + line)
        print(f"    per-seed  ={np.round(scores, 4)}")
        if name != "baseline decide()":
            print(f"    per-seed gain={np.round(np.array(scores) - results['baseline decide()'], 4)}")
        print(f"    per-class recall={np.round(np.mean(recalls, axis=0), 3)}")
        print(f"    predicted (seed 0)={distributions[0]}")
        print(f"    confusion summed (rows=true):\n{np.sum(confusions, axis=0)}")


if __name__ == "__main__":
    main()
