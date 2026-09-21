#!/usr/bin/env python3
"""Standalone inference: score an arbitrary profile file with a fitted model.

`final_solution.py` fits and predicts in one pass (it takes ~2 s, so there is
no reason to persist a model for the competition itself). This script exists
for the case where the model must be fitted once and applied later, or applied
to a different file with the same schema.

    python inference.py --fit                      # fit and save working/model.pkl
    python inference.py --predict raw/test.csv --out working/submission.csv
"""
from __future__ import annotations
import argparse, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))
from data import load, parse_profile                 # noqa: E402
from features import build, cat_dtypes               # noqa: E402
from final_solution import DROP, PARAMS, N_SEEDS, validate_submission   # noqa: E402

MODEL = HERE / "working" / "model.pkl"


def fit(raw: Path, n_seeds: int = N_SEEDS):
    TR, TE = load(raw)
    X, XT = build(TR, TE, drop=DROP)
    gain = 2.0 ** TR.target.to_numpy(dtype=float) - 1.0
    models = [lgb.LGBMRegressor(random_state=s, **PARAMS).fit(X, gain)
              for s in range(n_seeds)]
    bundle = dict(models=models, columns=list(X.columns),
                  dtypes=cat_dtypes(TR, TE), drop=DROP)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with MODEL.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"fitted {n_seeds} seeds on {len(TR)} rows -> {MODEL}")
    return bundle


def predict(bundle, profile_csv: Path, raw: Path) -> pd.DataFrame:
    TR, _ = load(raw)                       # only for consistent category levels
    df = pd.read_csv(profile_csv)
    P = parse_profile(df)
    X, XN = build(TR, P, drop=bundle["drop"], dtypes=bundle["dtypes"])
    XN = XN[bundle["columns"]]              # enforce identical column order
    assert list(XN.columns) == bundle["columns"], "feature columns drifted"
    pred = np.mean([m.predict(XN) for m in bundle["models"]], axis=0)
    return pd.DataFrame({"id": P.id.values, "prediction": pred})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(HERE / "raw"))
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--predict", default=None)
    ap.add_argument("--out", default=str(HERE / "working" / "submission.csv"))
    a = ap.parse_args()
    raw = Path(a.raw)

    bundle = fit(raw) if a.fit or not MODEL.exists() else pickle.load(MODEL.open("rb"))
    if a.predict:
        sub = predict(bundle, Path(a.predict), raw)
        ids = pd.read_csv(a.predict).id
        validate_submission(sub, ids)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(a.out, index=False)
        print(f"wrote {a.out} rows={len(sub)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
