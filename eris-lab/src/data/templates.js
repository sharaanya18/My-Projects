/* Reusable, copyable templates. The code templates follow the platform
 * contract: argv paths, constants instead of machine checks, seeds everywhere,
 * no fallbacks, output validation before writing. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.TEMPLATES = [
    {
      id: "skeleton-tabular",
      name: "Compliant solution skeleton (tabular, GBDT)",
      lang: "python",
      note: "A starting point. Replace TARGET, ID_COL, the metric and the fold strategy with what your challenge defines.",
      code: `"""<Challenge name> - solution.

Run: python3 solution.py <public_dir> <submission_out>
"""
import os

# Pin thread counts before numpy/sklearn load so float reductions don't vary
# with the machine's core count.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import random
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5
N_TRIALS = 20          # fixed trial count, never a time budget
TARGET = "target"
ID_COL = "id"
GROUP_COL = "group_id"  # the entity the hidden test is built on

random.seed(SEED)
np.random.seed(SEED)

PUBLIC_DIR = Path(sys.argv[1])
SUBMISSION_OUT = Path(sys.argv[2])


def metric(y_true, y_pred):
    # Re-implement the challenge metric exactly, then test it on the sample submission.
    return roc_auc_score(y_true, y_pred)


train = pd.read_csv(PUBLIC_DIR / "train.csv")
test = pd.read_csv(PUBLIC_DIR / "test.csv")
sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")

features = [c for c in train.columns if c not in (TARGET, ID_COL, GROUP_COL)]
X, y, groups = train[features], train[TARGET].to_numpy(), train[GROUP_COL].to_numpy()

# Folds are built once and reused by every model, so comparisons are paired.
folds = list(StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X, y, groups))


def cv_score(params):
    oof = np.zeros(len(train))
    best_iters = []
    for tr, va in folds:
        model = lgb.LGBMClassifier(**params, n_estimators=5000, random_state=SEED, verbose=-1)
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
        best_iters.append(model.best_iteration_)
    fold_scores = [metric(y[va], oof[va]) for _, va in folds]
    return oof, fold_scores, int(np.median(best_iters))


def objective(trial):
    params = dict(
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 200),
        subsample=trial.suggest_float("subsample", 0.6, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )
    _, scores, _ = cv_score(params)
    return float(np.mean(scores))


# In-script HPO: seeded sampler + fixed n_trials, so the run is deterministic.
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=N_TRIALS)
best = study.best_params | {"subsample_freq": 1}
oof, fold_scores, n_iter = cv_score(best)
print(f"CV {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} folds={np.round(fold_scores, 5).tolist()}")
print(f"baseline constant: {metric(y, np.full(len(y), y.mean())):.5f}")

# Final fit on all training data with a fixed iteration count taken from CV.
final = lgb.LGBMClassifier(**best, n_estimators=n_iter, random_state=SEED, verbose=-1).fit(X, y)
pred = final.predict_proba(test[features])[:, 1]

submission = pd.DataFrame({ID_COL: test[ID_COL], sample.columns[1]: pred})[sample.columns]

# Validate before writing: a malformed file ranks below a score of zero.
assert list(submission.columns) == list(sample.columns), "column mismatch"
assert len(submission) == len(test), "row count mismatch"
assert submission[ID_COL].is_unique and set(submission[ID_COL]) == set(test[ID_COL]), "id mismatch"
assert np.isfinite(submission.iloc[:, 1]).all(), "non-finite predictions"
assert submission.iloc[:, 1].nunique() > 1, "constant prediction"

SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(SUBMISSION_OUT, index=False)
print(f"wrote {SUBMISSION_OUT} rows={len(submission)}")
`,
    },
    {
      id: "skeleton-torch",
      name: "Seed and device header (PyTorch)",
      lang: "python",
      note: "Follows the guidebook's §3.4 seed template, with constants instead of machine checks.",
      code: `import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import sys
from pathlib import Path

import numpy as np
import torch

SEED = 42
NUM_WORKERS = 4                 # constant, never os.cpu_count()
BATCH_SIZE = 32                 # constant, never derived from free memory
EPOCHS = 4                      # measured to fit the runtime budget
DEVICE = torch.device("cuda")   # constant, never "cuda if available"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PUBLIC_DIR = Path(sys.argv[1])
SUBMISSION_OUT = Path(sys.argv[2])


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


loader_generator = torch.Generator()
loader_generator.manual_seed(SEED)

# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
#     worker_init_fn=seed_worker, generator=loader_generator,
# )
`,
    },
    {
      id: "local-check",
      name: "Local pre-submission check (bash)",
      lang: "bash",
      note: "Runs the exact platform command twice. It reports wall time and peak memory, compares the output hashes, and validates the CSV against the sample.",
      code: `#!/usr/bin/env bash
# usage: ./check_submission.sh solution.py ./dataset/public
set -euo pipefail
SOL="\${1:-solution.py}"; PUB="\${2:-./dataset/public}"
mkdir -p working/run1 working/run2

for run in run1 run2; do
  echo "== $run"
  /usr/bin/time -v python3 "$SOL" "$PUB" "working/$run/submission.csv" 2> "working/$run/time.log" | tail -n 5
  grep -E "Elapsed|Maximum resident" "working/$run/time.log"
done

h1=$(sha256sum working/run1/submission.csv | cut -d' ' -f1)
h2=$(sha256sum working/run2/submission.csv | cut -d' ' -f1)
[ "$h1" = "$h2" ] && echo "DETERMINISTIC: identical output hashes" || echo "WARNING: outputs differ between runs"

python3 - "$PUB" working/run1/submission.csv <<'PY'
import sys, numpy as np, pandas as pd
pub, sub_path = sys.argv[1], sys.argv[2]
sample = pd.read_csv(f"{pub}/sample_submission.csv")
sub = pd.read_csv(sub_path)
test = pd.read_csv(f"{pub}/test.csv")
id_col = sample.columns[0]
problems = []
if list(sub.columns) != list(sample.columns): problems.append(f"columns {list(sub.columns)} != {list(sample.columns)}")
if len(sub) != len(sample): problems.append(f"rows {len(sub)} != {len(sample)}")
if sub[id_col].duplicated().any(): problems.append("duplicate ids")
if id_col in test and set(sub[id_col]) != set(test[id_col]): problems.append("id set differs from test.csv")
num = sub.drop(columns=[id_col]).select_dtypes("number")
if not np.isfinite(num.to_numpy()).all(): problems.append("NaN/inf values")
print("CSV OK" if not problems else "CSV PROBLEMS:\\n- " + "\\n- ".join(problems))
PY
`,
    },
    {
      id: "claude-md",
      name: "CLAUDE.md for a challenge folder",
      lang: "markdown",
      note: "Drop this into the challenge working directory so every Claude Code session starts with the rules loaded.",
      code: `# Eris challenge: <name>

## Contract (never violate)
- Entry: \`python3 solution.py <public_dir> <submission_out>\`. Read Path(sys.argv[1]) and Path(sys.argv[2]) with no fallbacks.
- One end-to-end script. Train inside it. No cached artifacts, no pip installs, no GitHub models.
- Kaggle Docker image libraries only. Pretrained general-purpose weights from Hugging Face or timm are OK unless the category forbids them.
- Determinism: seed everything; DEVICE = torch.device("cuda"); constants for workers and batch size; fixed epochs; HPO with a seeded sampler and fixed n_trials.
- No clock- or machine-dependent logic, no try/except fallbacks, no env-var or debug switches.
- Runtime under 75 min on an A10G (limit 1.5 h). Challenge-specific limits override.
- Test data: one-sample or one-query-list inference only. No pseudo-labels, test-fitted transforms or test-distribution calibration.
- No external data, no synthetic training data, no LLM API calls, no hardcoded labels or rules.

## Category: <general | NLP | CV | RAG | Fine-tuning | From-scratch | bio-chem>
<paste the category rules from Eris Solver Lab>

## Validation
- Split: <GroupKFold on X | time split on Y | leave-one-group-out on Z>. Fold seed 42. Folds saved once and reused.
- Stress split: <leave-one-group-out on ...>
- Metric: <exact definition>. Direction: <higher | lower> is better.

## Evidence rules
- Inspect files before assuming. Report only numbers you computed, with the log line.
- Tag claims VERIFIED or HYPOTHESIS. Never claim private-LB performance.
- One change per experiment. Gains within about 2× the paired fold SE are noise.

## Experiment log
| id | parent | change | CV mean ± std | stress | runtime | verdict |
|----|--------|--------|---------------|--------|---------|---------|
`,
    },
    {
      id: "exp-log",
      name: "Experiment log entry",
      lang: "markdown",
      note: "One block per run. The tracker imports the same fields.",
      code: `### exp-<n>: <short name>
- Parent: exp-<m>
- Change (exactly one): <what changed>
- Hypothesis: <why it should help on unseen data>
- Folds / seed: <scheme>, seed 42 (same as parent)
- Fold scores: [.., .., .., .., ..]  -> mean ± std
- Stress split (leave-one-group-out): <score>
- Train score: <score>  (gap to CV: <delta>)
- Public LB (if submitted): <score>
- Runtime: <min>  Peak memory: <GB>
- Verdict: KEEP / REVERT / INCONCLUSIVE, because <evidence>
- Status: VERIFIED (log: <path or line>) / UNVERIFIED
`,
    },
    {
      id: "reviewer-q",
      name: "Reviewer question (Discord)",
      lang: "markdown",
      note: "§5.4: when unsure whether a line is a real restriction, ask and wait for the answer before submitting.",
      code: `Hi <reviewer>, a question about challenge "<name>" before I submit:

The description says: "<exact quote>".
I plan to: <one-sentence description of the technique>.
It is trained inside the script / uses no test data beyond per-sample inference / uses only <libraries>.

Is this compliant for this challenge's category (<category>)? If not, is <alternative> acceptable?
Thanks!`,
    },
  ];
})();
