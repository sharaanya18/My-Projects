"""Experiment logging (Phase 9).

One append-only JSONL record per experiment, plus a markdown table. The point
is that six weeks later you can answer "why did we do it this way?" -- and
that you never re-run an experiment you already ran.

Saves OOF/test prediction arrays alongside each record so blending, threshold
tuning and calibration can be redone later without refitting anything.

Usage:
    log = ExperimentLog("EXPERIMENT_LOG.jsonl")
    log.record(exp_id="E007", model="tfidf+logreg", features="word 1-2 + char 3-5",
               cv="StratifiedKFold(5, seed=42)", metric="f1_macro",
               score=0.6428, score_std=0.0282, runtime_s=0.11,
               diff_from="E006: added char n-grams",
               decision="keep -- gain 0.024 > noise band 0.031? NO -> within noise",
               next_step="tune threshold on OOF",
               oof=oof_array, test_pred=test_array)
    print(log.to_markdown())
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FIELDS = ["exp_id", "timestamp", "model", "features", "preprocessing",
          "cv", "metric", "score", "score_std", "runtime_s", "peak_mem_mb",
          "device", "diff_from", "delta", "error_notes", "decision",
          "next_step", "git_sha", "artifacts"]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:                                  # noqa: BLE001
        return None


class ExperimentLog:
    def __init__(self, path="EXPERIMENT_LOG.jsonl", artifact_dir="artifacts"):
        self.path = Path(path)
        self.artifact_dir = Path(artifact_dir)

    # -- write ------------------------------------------------------------
    def record(self, exp_id: str, model: str, metric: str, score: float,
               oof=None, test_pred=None, fold_ids=None, **kw) -> dict:
        prev = self.last_score(metric)
        rec = {f: None for f in FIELDS}
        rec.update({
            "exp_id": exp_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "metric": metric,
            "score": float(score),
            "device": kw.pop("device", platform.platform()),
            "git_sha": _git_sha(),
        })
        rec.update({k: v for k, v in kw.items() if k in FIELDS})

        if rec.get("delta") is None and prev is not None:
            rec["delta"] = round(float(score) - prev, 6)

        arts = {}
        if oof is not None or test_pred is not None or fold_ids is not None:
            d = self.artifact_dir / exp_id
            d.mkdir(parents=True, exist_ok=True)
            for name, arr in (("oof", oof), ("test", test_pred), ("folds", fold_ids)):
                if arr is not None:
                    p = d / f"{name}.npy"
                    np.save(p, np.asarray(arr))
                    arts[name] = str(p)
        rec["artifacts"] = arts or None

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        return rec

    # -- read -------------------------------------------------------------
    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def last_score(self, metric: str) -> float | None:
        recs = [r for r in self.load() if r.get("metric") == metric]
        return recs[-1]["score"] if recs else None

    def best(self, metric: str) -> dict | None:
        recs = [r for r in self.load() if r.get("metric") == metric]
        return max(recs, key=lambda r: r["score"]) if recs else None

    def load_artifact(self, exp_id: str, name: str = "oof"):
        for r in self.load():
            if r["exp_id"] == exp_id and (r.get("artifacts") or {}).get(name):
                return np.load(r["artifacts"][name])
        raise KeyError(f"no artifact {name!r} for {exp_id!r}")

    # -- report -----------------------------------------------------------
    def to_markdown(self, cols=("exp_id", "model", "features", "cv", "metric",
                                "score", "score_std", "delta", "runtime_s",
                                "decision")) -> str:
        recs = self.load()
        if not recs:
            return "_(no experiments logged yet)_"
        head = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join("---" for _ in cols) + "|"
        lines = [head, sep]
        for r in recs:
            vals = []
            for c in cols:
                v = r.get(c)
                if isinstance(v, float):
                    v = f"{v:.4f}"
                vals.append("" if v is None else str(v).replace("|", "\\|"))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    def write_markdown(self, path="EXPERIMENT_LOG.md") -> Path:
        p = Path(path)
        p.write_text(
            "# Experiment Log\n\n"
            "Append-only. One row per experiment. Never delete a row -- a failed\n"
            "experiment is a result, and re-running it is pure waste.\n\n"
            + self.to_markdown() + "\n", encoding="utf-8")
        return p


class timed:
    """`with timed() as t: ...` -> t.seconds"""

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.perf_counter() - self._t0, 3)
        return False
