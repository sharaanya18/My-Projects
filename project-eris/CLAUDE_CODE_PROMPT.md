# Paste into Claude Code (in a folder with solution.py, APPROACH.md and dataset/public/)

Read APPROACH.md and solution.py. The problem description's rules come first. Two platform rules are
absolute:
(a) No behaviour may depend on run time. That means no time checks, no "if time remains", no
    time-based early stopping.
(b) Every explicit requirement must be met exactly: output format, no case_id or row/pool order as
    a signal, no external data, and so on.
Do not add either kind of problem.

1. Environment check: `python3 -c "import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"`.
   A GPU is expected. torch >= 2.6 is needed, because the pinned model commits ship pytorch_model.bin weights.
2. Run: `python3 solution.py ./dataset/public ./working/submission.csv 2>&1 | tee run1.log`
3. Report from run1.log:
   - the per-epoch `holdout MAP` and selected epoch for each backbone
   - total runtime (must be under about 55 min)
   - that both permutation tests passed
4. Determinism on the GPU: run it a second time to ./working/submission2.csv, then
   `md5sum working/submission*.csv`. The two files must be identical. If they differ, find the
   non-deterministic op (warnings from torch.use_deterministic_algorithms appear in the log) and fix it
   deterministically. Never fix it with a time-based workaround.
5. Target: best holdout MAP >= 0.37 for the ensemble's backbones, which is about 0.40 on unseen
   institutions. If it falls short, you may ONLY change the fixed constants at the top of solution.py
   (max_epochs bound, lr, D, LAYER_DECAY, TAG_L2), one at a time. Judge only on the holdout MAP printed
   by the script. Never use a random split, test.csv, or the public LB for tuning.
6. Final checks:
   - 4,792 rows, columns exactly `case_id,ranked_tags`, no index column
   - every row is exactly its pool's 80 codes
   - identical md5 across two runs
