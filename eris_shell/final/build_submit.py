"""Assemble final/submit.py (and solution.ipynb) from the src/ modules.

The graded run must be ONE self-contained script, while development happens in
separate modules.  Generating the script by inlining the exact module sources
keeps the two from drifting apart.  Only the modules the submission needs are
included -- no dev harness, probe, or sweep code.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULES = ["tokenizer", "metric", "shlexutil", "splits", "detok",
           "dataset", "model", "engine", "decode"]
LOCAL = set(MODULES)

HEADER = '''#!/usr/bin/env python3
# =============================================================================
# Unseen-Pair Shell Pipeline Synthesis -- end-to-end solution
#
# Standing constraint checklist (build spec sec. 0 / Solver Guidebook):
#  [x] From scratch only: no pretrained models, embeddings, tokenizers or weights.
#      Every vocabulary and every weight is fit here, on train.csv, every run.
#  [x] No synthetic data: no (input, output) pair is invented; training outputs
#      are only re-cut into stage-level targets (Guidebook 4.2.6).
#  [x] No source lookup, and `id` is never used as a feature.
#  [x] CPU only; watchdog stops training at ~52 min, inference follows
#      (Guidebook 3.5).
#  [x] Test discipline: vocabularies never see test.csv; each test row is
#      decoded on its own; no pseudo-labels, no test-set statistics (4.2.5).
#  [x] One independent script: reads ./dataset/public/*.csv, writes only
#      ./working/submission.csv, loads no cached artifacts (3.8).
#  [x] The model does the solving: regex/shlex are used only to cut training
#      labels into stages and to check quote balance of generated text (4.3.1).
#
# Approach: shared Transformer encoder over the description, trained jointly
# with (1) a PLAN decoder that emits the ordered command heads, (2) a REALIZE
# decoder that writes one pipeline stage at a time conditioned on its head,
# with a pointer-generator copy channel for literal file names/patterns, and
# (3) an auxiliary unordered head-set classifier.  A compositional holdout
# (whole head->head pairs removed from training) is built inside this script
# for early stopping and decoding choices; then full-data seeds are trained
# and ensembled.
# =============================================================================
import os
'''


def strip_local_imports(src: str) -> str:
    out = []
    for line in src.splitlines():
        m = re.match(r"\s*from (\w+) import ", line) or re.match(r"\s*import (\w+)\s*$", line)
        if m and m.group(1) in LOCAL:
            continue
        out.append(line)
    return "\n".join(out)


def main():
    parts = [HEADER]
    for mod in MODULES:
        src = (ROOT / "src" / f"{mod}.py").read_text()
        parts.append(f"\n# {'=' * 77}\n# ---- module: {mod}.py\n# {'=' * 77}\n")
        parts.append(strip_local_imports(src))
    cfg = (ROOT / "final" / "final_cfg.py").read_text()
    parts.append("\n# " + "=" * 77 + "\n# ---- final configuration\n# " + "=" * 77 + "\n")
    parts.append(cfg)
    parts.append((ROOT / "final" / "main_block.py").read_text())
    script = "\n".join(parts)
    (ROOT / "final" / "submit.py").write_text(script)

    # notebook version: identical code, one cell per section
    cells = []
    def md(t):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
    def code(t):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": t})
    md("# Unseen-Pair Shell Pipeline Synthesis\n\n"
       "Plan + realize sequence model with a pointer-generator copy channel, trained from "
       "scratch on `train.csv` only (CPU).  Reads `./dataset/public/`, writes "
       "`./working/submission.csv`.  The same code ships as `final/submit.py`.")
    code(HEADER)
    for mod in MODULES:
        src = strip_local_imports((ROOT / "src" / f"{mod}.py").read_text())
        md(f"## `{mod}`")
        code(src)
    md("## Final configuration")
    code(cfg)
    md("## End-to-end run")
    code((ROOT / "final" / "main_block.py").read_text().replace(
        'if __name__ == "__main__":\n    main()', "main()"))
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
          "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (ROOT / "final" / "solution.ipynb").write_text(json.dumps(nb, indent=1))
    print("wrote final/submit.py and final/solution.ipynb")


if __name__ == "__main__":
    main()
