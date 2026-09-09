"""Per-domain calibration fitters.

Each fitter operates only on flat (score, label) number pairs -- score is
the raw p_b - p_a value in [-1, 1], label is the ground-truth relevance
value in {-1, 0, 1} -- decoupled from row/JSON structure so it's trivial to
unit test and to swap. A separate helper builds those pairs from a
calibration-split dataframe.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .utils import load_relevance


class Calibrator:
    def predict(self, scores: Sequence[float]) -> List[float]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: str) -> "Calibrator":
        return pickle.loads(Path(path).read_bytes())


@dataclass
class IsotonicCalibrator(Calibrator):
    model: object  # sklearn.isotonic.IsotonicRegression

    def predict(self, scores: Sequence[float]) -> List[float]:
        out = self.model.predict(np.asarray(scores, dtype=float))
        return [float(np.clip(v, -1.0, 1.0)) for v in out]


@dataclass
class PlattCalibrator(Calibrator):
    """calibrated = tanh(a * score + b), fit by least squares against label."""

    a: float
    b: float

    def predict(self, scores: Sequence[float]) -> List[float]:
        arr = np.asarray(scores, dtype=float)
        out = np.tanh(self.a * arr + self.b)
        return [float(np.clip(v, -1.0, 1.0)) for v in out]


def fit_isotonic(scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=-1.0, y_max=1.0, out_of_bounds="clip")
    model.fit(np.asarray(scores, dtype=float), np.asarray(labels, dtype=float))
    return IsotonicCalibrator(model=model)


def fit_platt(scores: Sequence[float], labels: Sequence[int]) -> PlattCalibrator:
    from scipy.optimize import minimize

    x = np.asarray(scores, dtype=float)
    y = np.asarray(labels, dtype=float)

    def loss(params):
        a, b = params
        pred = np.tanh(a * x + b)
        return float(np.mean((pred - y) ** 2))

    res = minimize(loss, x0=np.array([1.0, 0.0]), method="Nelder-Mead")
    a, b = res.x
    return PlattCalibrator(a=float(a), b=float(b))


FITTERS = {"isotonic": fit_isotonic, "platt": fit_platt}


def build_pairs_by_domain(
    df,
    score_col: str = "score",
    domain_col: str = "domain",
    relevance_col: str = "relevance",
) -> Dict[str, Tuple[List[float], List[int]]]:
    """Flattens a calibration-split dataframe (one row per example, a
    16-length `score` list and JSON `relevance` column) into per-domain
    parallel (scores, labels) arrays of individual (candidate, label) pairs.
    """
    pairs: Dict[str, Tuple[List[float], List[int]]] = {}
    for _, row in df.iterrows():
        domain = row[domain_col]
        scores = row[score_col]
        labels = load_relevance(row[relevance_col]) if isinstance(row[relevance_col], str) else row[relevance_col]
        if len(scores) != len(labels):
            raise ValueError(f"row has {len(scores)} scores but {len(labels)} labels")
        s_list, l_list = pairs.setdefault(domain, ([], []))
        s_list.extend(float(s) for s in scores)
        l_list.extend(int(y) for y in labels)
    return pairs


def fit_per_domain_calibrators(
    pairs_by_domain: Dict[str, Tuple[Sequence[float], Sequence[int]]],
    method: str = "isotonic",
) -> Dict[str, Calibrator]:
    if method not in FITTERS:
        raise ValueError(f"unknown calibration method {method!r}, choose from {list(FITTERS)}")
    fitter = FITTERS[method]
    return {domain: fitter(scores, labels) for domain, (scores, labels) in pairs_by_domain.items()}


def apply_calibration(
    scores: Sequence[float], domain: str, calibrators: Dict[str, Calibrator]
) -> List[float]:
    if domain not in calibrators:
        raise KeyError(f"no calibrator fit for domain {domain!r}")
    return calibrators[domain].predict(scores)


def save_calibrators(calibrators: Dict[str, Calibrator], out_dir: str) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    for domain, cal in calibrators.items():
        cal.save(str(out_path / f"{domain}.pkl"))


def load_calibrators(out_dir: str, domains: Sequence[str]) -> Dict[str, Calibrator]:
    out_path = Path(out_dir)
    return {d: Calibrator.load(str(out_path / f"{d}.pkl")) for d in domains}
