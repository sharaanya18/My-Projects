import json
import random

import pandas as pd

from response_critique_pipeline.splitting import add_split_column, split_dataframe


def _make_relevance(n_true, seed):
    rng = random.Random(seed)
    idx = rng.sample(range(16), n_true)
    rel = [0] * 16
    for i in idx:
        rel[i] = rng.choice([-1, 1])
    return json.dumps(rel)


def _synthetic_df(n=400):
    domains = ["general", "stem", "code", "multilingual"]
    rows = []
    for i in range(n):
        domain = domains[i % 4]
        n_true = 2 + (i % 6)  # 2..7
        rows.append(
            {
                "id": f"item_{i}",
                "domain": domain,
                "context": "[]",
                "response_a": "a",
                "response_b": "b",
                "candidates": json.dumps([f"c{j}" for j in range(16)]),
                "relevance": _make_relevance(n_true, seed=i),
            }
        )
    return pd.DataFrame(rows)


def test_every_row_assigned_exactly_once():
    df = _synthetic_df()
    out = add_split_column(df, seed=1)
    assert out["split"].isna().sum() == 0
    assert set(out["split"].unique()) <= {"dev", "calibration", "final_check"}
    assert len(out) == len(df)


def test_split_proportions_roughly_match_ratios():
    df = _synthetic_df(n=800)
    parts = split_dataframe(df, ratios=(0.7, 0.15, 0.15), seed=1)
    total = sum(len(p) for p in parts.values())
    assert total == len(df)
    dev_frac = len(parts["dev"]) / total
    assert 0.6 < dev_frac < 0.8


def test_deterministic_with_same_seed():
    df = _synthetic_df()
    a = add_split_column(df, seed=7)
    b = add_split_column(df, seed=7)
    assert (a["split"] == b["split"]).all()


def test_no_id_appears_in_two_splits():
    df = _synthetic_df()
    parts = split_dataframe(df, seed=3)
    ids = [set(p["id"]) for p in parts.values()]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
