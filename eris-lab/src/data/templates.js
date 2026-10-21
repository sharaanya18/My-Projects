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
      id: "skeleton-tabular-reg",
      name: "Rubric-ready tabular regression (log-space metric)",
      lang: "python",
      note: "Built from the platform's worked example: a requirements map up top, every feature type encoded on purpose, an L1 objective on log1p(y), out-of-fold entity statistics for a random split, the train-fit score printed next to CV, and the grader's own checks before writing. Swap the column lists for your challenge and drop the blocks it doesn't have.",
      code: `"""<Challenge name> - solution.

Run: python3 solution.py <public_dir> <submission_out>

Requirements map (rubric criterion / constraint -> where it is met):
- Drop target-leaking columns (defensive, only if present) -> LEAKY, load section
- Drop entirely-null columns -> ALL_NULL, load section
- Log-transform the skewed target to match the metric -> fit_target(), y_log
- Split pipe-separated lists and multi-hot encode them -> encode_multi()
- Dates parsed with the documented format -> encode_dates(), DATE_FORMAT
- Text column handled with TF-IDF fitted on train only -> encode_text()
- Gradient boosting model -> LGBMRegressor in run_cv()
- Train vs validation gap reported -> "train-fit" line in the CV log
- Predictions clipped to the valid range -> np.clip before writing
- Output checked against the grader's own rules -> asserts before to_csv
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
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
N_FOLDS = 5
N_THREADS = 4
ID_COL, TARGET = "id", "target"
LEAKY = ["fork_count", "views", "comments_count"]   # columns the description says leak the target
ALL_NULL = ["unused_a", "unused_b"]                 # columns the description says are entirely null
NUMERIC = ["num_a", "num_b"]
CATEGORICAL = ["cat_a"]
DATES = ["created_date"]
DATE_FORMAT = "%d/%m/%Y"                            # the format the description documents
MULTI = ["tags"]                                    # pipe-separated lists
TEXT = "title"
ENTITY = "author"                                   # key that recurs in train and test (random split)
PARAMS = dict(objective="l1", learning_rate=0.03, num_leaves=63, min_child_samples=40,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
              n_estimators=4000, random_state=SEED, n_jobs=N_THREADS, verbose=-1)

random.seed(SEED)
np.random.seed(SEED)

PUBLIC_DIR = Path(sys.argv[1])
SUBMISSION_OUT = Path(sys.argv[2])


def metric(y_true, y_pred):
    # MAE in log space, as the challenge defines it. Lower is better.
    return float(np.mean(np.abs(np.log1p(y_true) - np.log1p(y_pred))))


train = pd.read_csv(PUBLIC_DIR / "train.csv")
test = pd.read_csv(PUBLIC_DIR / "test.csv")
sample = pd.read_csv(PUBLIC_DIR / "sample_submission.csv")
# Leaking and empty columns go first, and only where present (the rubric asks for it even as a no-op).
train = train.drop(columns=[c for c in LEAKY + ALL_NULL if c in train.columns])
test = test.drop(columns=[c for c in LEAKY + ALL_NULL if c in test.columns])
y = train[TARGET].to_numpy(dtype=float)
# The metric is an L1 error on log1p(y), minimised by the median of log1p(y): train there with an L1 objective.
y_log = np.log1p(y)


def encode_dates(df):
    out = pd.DataFrame(index=df.index)
    for c in DATES:
        d = pd.to_datetime(df[c], format=DATE_FORMAT)
        out[c + "_days"] = (d - pd.Timestamp("1970-01-01")).dt.days
    return out


def encode_multi(tr, te):
    # Fitted on training rows only; values unseen in train are ignored for test rows.
    blocks_tr, blocks_te = [], []
    for c in MULTI:
        mlb = MultiLabelBinarizer(sparse_output=True)
        split = lambda s: s.fillna("").str.split("|").apply(lambda v: [x for x in v if x])
        blocks_tr.append(mlb.fit_transform(split(tr[c])).astype(np.float32))
        known = set(mlb.classes_)
        blocks_te.append(mlb.transform(split(te[c]).apply(lambda v: [x for x in v if x in known])).astype(np.float32))
    return sp.hstack(blocks_tr).tocsr(), sp.hstack(blocks_te).tocsr()


def encode_text(tr, te):
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    return tfidf.fit_transform(tr[TEXT].fillna("")).astype(np.float32), tfidf.transform(te[TEXT].fillna("")).astype(np.float32)


def dense_block(df):
    out = df[NUMERIC].astype(float).copy()
    out[NUMERIC] = np.log1p(out[NUMERIC].clip(lower=0))   # skewed counts
    out = out.join(encode_dates(df))
    for c in CATEGORICAL:
        out[c] = df[c].astype("category").cat.codes
    return out


def entity_stats(keys_fit, y_fit, keys_apply):
    # Mean log-target and row count per entity, fitted on the given rows only (inside each fold).
    g = pd.DataFrame({"k": keys_fit.to_numpy(), "y": y_fit}).groupby("k")["y"].agg(["mean", "count"])
    prior, k = float(np.mean(y_fit)), 10.0
    smooth = (g["mean"] * g["count"] + prior * k) / (g["count"] + k)
    m = keys_apply.map(smooth).fillna(prior).to_numpy()
    n = keys_apply.map(g["count"]).fillna(0).to_numpy()
    return np.column_stack([m, n]).astype(np.float32)


D_tr, D_te = dense_block(train), dense_block(test)
M_tr, M_te = encode_multi(train, test)
T_tr, T_te = encode_text(train, test)
X_base_tr = sp.hstack([sp.csr_matrix(D_tr.to_numpy(np.float32)), M_tr, T_tr]).tocsr()
X_base_te = sp.hstack([sp.csr_matrix(D_te.to_numpy(np.float32)), M_te, T_te]).tocsr()

# The prepare script split rows at random, so shuffled KFold mirrors the hidden test.
folds = list(KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED).split(X_base_tr))


def with_entity(X, stats):
    return sp.hstack([X, sp.csr_matrix(stats)]).tocsr()


oof_log = np.zeros(len(train))
best_iters, fold_scores, fit_scores = [], [], []
for f, (tr, va) in enumerate(folds):
    s_tr = np.zeros((len(tr), 2), dtype=np.float32)
    inner = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + f).split(tr)
    for a, b in inner:   # out-of-fold entity stats for the training rows too
        s_tr[b] = entity_stats(train[ENTITY].iloc[tr[a]], y_log[tr[a]], train[ENTITY].iloc[tr[b]])
    s_va = entity_stats(train[ENTITY].iloc[tr], y_log[tr], train[ENTITY].iloc[va])
    X_tr, X_va = with_entity(X_base_tr[tr], s_tr), with_entity(X_base_tr[va], s_va)
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr, y_log[tr], eval_set=[(X_va, y_log[va])], callbacks=[lgb.early_stopping(200, verbose=False)])
    oof_log[va] = model.predict(X_va)
    best_iters.append(model.best_iteration_)
    fold_scores.append(metric(y[va], np.clip(np.expm1(oof_log[va]), 0, None)))
    fit_scores.append(metric(y[tr], np.clip(np.expm1(model.predict(X_tr)), 0, None)))
    print(f"fold {f}: valid {fold_scores[-1]:.5f} train-fit {fit_scores[-1]:.5f} iters {best_iters[-1]}")

print(f"CV {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f} train-fit {np.mean(fit_scores):.5f} folds={np.round(fold_scores, 5).tolist()}")
print(f"baseline constant median: {metric(y, np.full(len(y), np.median(y))):.5f}")

# Final fit on all training rows with a fixed iteration count from CV.
s_all = np.zeros((len(train), 2), dtype=np.float32)
for a, b in folds:
    s_all[b] = entity_stats(train[ENTITY].iloc[a], y_log[a], train[ENTITY].iloc[b])
s_test = entity_stats(train[ENTITY], y_log, test[ENTITY])
final = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": int(np.median(best_iters))})
final.fit(with_entity(X_base_tr, s_all), y_log)
pred = np.clip(np.expm1(final.predict(with_entity(X_base_te, s_test))), 0, None)

submission = pd.DataFrame({ID_COL: test[ID_COL], TARGET: pred})[list(sample.columns)]

# The grader's own checks: exact columns, one row per test id, values >= 0. Plus finite values.
assert set(submission.columns) == set(sample.columns), "column mismatch"
assert len(submission) == len(test) and submission[ID_COL].is_unique, "row/id mismatch"
assert np.isfinite(submission[TARGET]).all() and (submission[TARGET] >= 0).all(), "invalid predictions"

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
- Fixed work plan ≤ 50 min on the target hardware with ≥ 30% headroom (limit 1.5 h). CPU-only or shorter limits in the description override.
- Source: plain readable UTF-8 under 512 KB. Write submission.csv exactly once, after validation. A requirements-map docstring quotes each rule.
- Test data: one-sample or one-query-list inference only. No pseudo-labels, test-fitted transforms or test-distribution calibration.
- No external data, no synthetic training data, no LLM API calls, no hardcoded labels or rules.

## Category: <general | NLP | CV | RAG | Fine-tuning | From-scratch | bio-chem>
<paste the category rules from Eris Solver Lab>

## Rubric (copy it from the challenge page)
- REQUIRED: <criterion> -> <where the code meets it>
- RECOMMENDED: <criterion> -> <done, or why skipped>
Every REQUIRED item is met visibly and listed in the requirements-map docstring. Print the train-fit score next to CV.

## Grader contract (from the grading script)
- Columns: <exact set>. Rows: one per test id. Values: <range, e.g. >= 0>. Assert all of it, plus finite values, before writing.

## Validation
- Split: mirror the prepare script (<random KFold | GroupKFold on X | time split on Y>). Fold seed 42. Folds saved once and reused.
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
