import json

import pandas as pd
import pytest

from response_critique_pipeline import calibration


def _synthetic_pairs(n=200):
    import random

    rng = random.Random(0)
    scores, labels = [], []
    for _ in range(n):
        y = rng.choice([-1, 0, 0, 0, 1])  # skew toward 0, like real pools
        noise = rng.uniform(-0.3, 0.3)
        scores.append(max(-1.0, min(1.0, y * 0.6 + noise)))
        labels.append(y)
    return scores, labels


def test_fit_isotonic_predictions_in_range():
    scores, labels = _synthetic_pairs()
    cal = calibration.fit_isotonic(scores, labels)
    preds = cal.predict(scores)
    assert all(-1.0 <= p <= 1.0 for p in preds)


def test_fit_isotonic_is_monotonic_nondecreasing():
    scores, labels = _synthetic_pairs()
    cal = calibration.fit_isotonic(scores, labels)
    xs = sorted(set(round(s, 3) for s in scores))
    preds = cal.predict(xs)
    assert all(preds[i] <= preds[i + 1] + 1e-9 for i in range(len(preds) - 1))


def test_fit_platt_predictions_in_range():
    scores, labels = _synthetic_pairs()
    cal = calibration.fit_platt(scores, labels)
    preds = cal.predict(scores)
    assert all(-1.0 <= p <= 1.0 for p in preds)


def test_build_pairs_by_domain_and_fit_per_domain():
    rel_a = [1, -1] + [0] * 14
    rel_b = [0, 1] + [0] * 14
    rel_c = [-1] + [0] * 15
    df = pd.DataFrame(
        [
            {"domain": "general", "score": [0.5, -0.5] + [0.0] * 14, "relevance": json.dumps(rel_a)},
            {"domain": "general", "score": [0.0, 0.9] + [0.0] * 14, "relevance": json.dumps(rel_b)},
            {"domain": "code", "score": [-0.9] + [0.0] * 15, "relevance": json.dumps(rel_c)},
        ]
    )
    pairs = calibration.build_pairs_by_domain(df)
    assert set(pairs.keys()) == {"general", "code"}
    assert len(pairs["general"][0]) == 32
    assert len(pairs["code"][0]) == 16

    calibrators = calibration.fit_per_domain_calibrators(pairs, method="isotonic")
    out = calibration.apply_calibration([0.5, -0.5], "general", calibrators)
    assert len(out) == 2


def test_save_and_load_calibrators_roundtrip(tmp_path):
    scores, labels = _synthetic_pairs()
    calibrators = {"general": calibration.fit_isotonic(scores, labels)}
    calibration.save_calibrators(calibrators, str(tmp_path))
    loaded = calibration.load_calibrators(str(tmp_path), ["general"])
    assert loaded["general"].predict([0.1]) == calibrators["general"].predict([0.1])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        calibration.fit_per_domain_calibrators({"general": ([0.1], [1])}, method="bogus")
