"""Builds a single-file Kaggle GPU kernel that runs validate.py (held-out-cluster CV) with solution.py.
Usage: python make_kernel.py <out_dir> <kernel_slug> <dataset_slug> '<cfg json>' [cluster_seeds] [only_folds]
Only code is pushed. The dataset must already exist on the user's Kaggle account."""
import json, sys
from pathlib import Path

out, slug, ds, cfg = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
seeds = sys.argv[5] if len(sys.argv) > 5 else "0"
only = sys.argv[6] if len(sys.argv) > 6 else ""
here = Path(__file__).resolve().parent.parent
sol, val = (here / "solution.py").read_text(), (here / "validate.py").read_text()
user, name = slug.split("/")
out.mkdir(parents=True, exist_ok=True)
run = f'''import os, subprocess, sys
from pathlib import Path
SOL = {sol!r}
VAL = {val!r}
Path("/kaggle/working/solution.py").write_text(SOL)
Path("/kaggle/working/validate.py").write_text(VAL)
data = next(Path(r) for r, _, fs in os.walk("/kaggle/input") if "paragraphs.csv" in fs)
print("data dir:", data, flush=True)
import torch; print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
cmd = [sys.executable, "-u", "validate.py", str(data), "/kaggle/working/cv_result.json", "--cfg", {cfg!r}, "--cluster-seeds", {seeds!r}]
if {only!r}: cmd += ["--only-folds", {only!r}]
subprocess.run(cmd, cwd="/kaggle/working", check=True)
'''
(out / "run_kernel.py").write_text(run)
(out / "kernel-metadata.json").write_text(json.dumps(dict(
    id=slug, title=name.replace("-", " "), code_file="run_kernel.py", language="python", kernel_type="script",
    is_private=True, enable_gpu=True, enable_internet=True, dataset_sources=[ds], competition_sources=[], kernel_sources=[]), indent=1))
print("built", out)
