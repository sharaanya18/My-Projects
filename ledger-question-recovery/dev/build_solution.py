"""Assemble the standalone solution.py from ad.py + ledger.py + a main()."""
import re, sys, json, pathlib

HERE = pathlib.Path(__file__).parent

HEADER = '''"""
Mixed scientific ledger: witness-cluster recovery and question restoration.

A transformer encoder-decoder is trained from scratch on the supplied training
ledgers only. The same trained network produces both submitted fields:

  * witnesses -- contextual record states from the encoder are scored pairwise
    by a trained head; the subset of size record_count-2 with the highest mean
    pairwise score is submitted.
  * question  -- the trained decoder generates the question autoregressively
    with deterministic diverse beam search; the candidate with the highest
    expected character n-gram agreement over the model's own beam is submitted.

Nothing is retrieved, looked up, copied from a table, or produced by a
hand-written template, and no external data or pretrained weights are used.
Every row is encoded and decoded in isolation: predictions for a row never
depend on any other row, so no cross-row or same-paper information is shared.

Usage: python3 solution.py <public_dir> <submission_out>
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import itertools
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260920

'''

SHIM = '''

ad = sys.modules[__name__]

'''


def strip_module(src, drop_prefixes):
    out = []
    for line in src.splitlines():
        if any(line.startswith(p) for p in drop_prefixes):
            continue
        out.append(line)
    return "\n".join(out)


def main():
    cfg_path = HERE / "final_config.json"
    if not cfg_path.exists():
        print("final_config.json missing; write the chosen hyperparameters first", file=sys.stderr)
        return 1
    cfg = json.loads(cfg_path.read_text())
    adsrc = strip_module((HERE / "ad.py").read_text(),
                         ('"""', "import numpy"))
    ledsrc = strip_module((HERE / "ledger.py").read_text(),
                          ('"""', "import ad", "from ad import", "import numpy", "import itertools",
                           "import json", "import math", "import re", "import unicodedata",
                           "from collections import"))
    tail = (HERE / "solution_main.py").read_text()
    body = (HEADER
            + "# " + "-" * 68 + "\n# reverse-mode autodiff\n# " + "-" * 68 + "\n"
            + adsrc + SHIM
            + "# " + "-" * 68 + "\n# data, model, training, inference\n# " + "-" * 68 + "\n"
            + ledsrc + "\n\n"
            + "# " + "-" * 68 + "\n# configuration chosen by group-held-out validation on train.csv\n"
            + "# " + "-" * 68 + "\nCFG = " + json.dumps(cfg, indent=4).replace("true", "True").replace("false", "False") + "\n\n"
            + tail)
    out = HERE.parent / "solution.py"
    out.write_text(body)
    import ast
    ast.parse(body)
    print(f"wrote {out} ({len(body.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
