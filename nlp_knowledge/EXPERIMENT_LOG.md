# Experiment Log

Append-only. One row per experiment, including the failures — **a failed
experiment is a result**, and re-running one you already ran is pure waste.

Generated/updated by `code/experiment_log.py`
(`ExperimentLog.write_markdown()`).

## Rules

1. **Fixed folds.** Same fold assignment for every row in this table, saved to
   disk. Comparing models across different fold assignments compares noise.
2. **Record the noise band** (fold-wise std). An improvement smaller than it has
   not been demonstrated.
3. **One change per row.** If you changed two things, you learned nothing about
   either.
4. **Record rejections.** The `decision` column exists so you don't retry a dead
   end in three weeks.
5. **Save OOF + test arrays** per experiment. Blending, threshold tuning and
   calibration can then be redone without refitting anything.

## Columns

| Column | Meaning |
|---|---|
| `exp_id` | E001, E002, … |
| `model` | estimator / architecture |
| `features` | feature set used |
| `preprocessing` | cleaning / tokenisation applied |
| `cv` | scheme, n folds, seed |
| `metric` | the **competition** metric |
| `score` | CV mean |
| `score_std` | fold-wise std — the **noise band** |
| `delta` | change vs the previous experiment |
| `runtime_s` / `peak_mem_mb` / `device` | cost |
| `diff_from` | what changed vs which experiment |
| `error_notes` | what the errors looked like |
| `decision` | keep / reject / inconclusive — **and why** |
| `next_step` | the experiment this one motivates |
| `git_sha` / `artifacts` | reproducibility |

## Ablation ladder (fill downward)

```
baseline → lexical features → embeddings → stronger embeddings
        → hybrid retrieval → reranker → ranking model → ensemble
```

Never assume the most complex rung is best. Each rung must clear the noise band
of the one below it.

---

| exp_id | model | features | cv | metric | score | score_std | delta | runtime_s | decision |
|---|---|---|---|---|---|---|---|---|---|
| _(no experiments yet — no Shipd challenge has been provided)_ | | | | | | | | | |
