#!/usr/bin/env python3
"""Offline experiment harness: same folds/model/decoder as solution.py, fixed number of seeds per fold.
Usage: python3 experiments.py PUBLIC_DIR NAME ROUNDS '{"cfg_key": value, ...}'
Reports the out-of-fold score (GroupKFold(5) by validation_group) after each round. Never touches test.csv."""
import json, os, sys, time
import pandas as pd, torch
import solution as S

pub, name, rounds, over = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.loads(sys.argv[4] if len(sys.argv) > 4 else '{}')
torch.set_num_threads(1)
tr = pd.read_csv(os.path.join(pub, 'train.csv')); tg = pd.read_csv(os.path.join(pub, 'train_targets.csv'))
cases = S.build_cases(tr, dict(zip(tg.case_id, tg.answer_json)))
folds = S.prepare_folds(cases, tr.validation_group.values)
cfg = dict(S.CFG, **over)
S.log(f'EXP {name} cfg={cfg}')
t = time.time()
res = S.run_cv(folds, cfg, rounds=rounds)
h = res['history'][-1]
S.log(f'RESULT {name}: rounds={h[0]} OOF={h[1]:.4f} raw={h[2]:.4f} time={time.time() - t:.0f}s')
