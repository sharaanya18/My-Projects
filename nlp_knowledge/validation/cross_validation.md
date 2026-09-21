# Cross-Validation

**Build the validation scheme before the first model.** A wrong split makes
every subsequent experiment uninterpretable, and you usually find out on the
leaderboard.

## Choosing the scheme

Answer these in order. The first "yes" decides it.

| Question | If yes → |
|---|---|
| Do rows share a **query / group / document / user / entity**? | `GroupKFold` on that key |
| Is there a **time** dimension, or does test come after train? | time-based split (`TimeSeriesSplit` or a fixed cutoff) |
| Are there **near-duplicate** texts across rows? | group by duplicate-cluster id |
| Is the target **imbalanced**? | `StratifiedKFold` |
| Both groups and imbalance? | `StratifiedGroupKFold` |
| Multi-label? | iterative stratification (`skmultilearn`), or stratify on the rarest label |
| None of the above | `KFold(shuffle=True, random_state=...)` |

Random K-fold is the **last resort**, not the default. Most NLP competition data
has structure, and a random split over structured data reports a number you
cannot reproduce on test.

## Group leakage — the dominant failure in pair tasks

If sentence X appears in pairs (X, A) and (X, B), a random split puts one in
train and the other in validation. The model memorizes X and your CV is
inflated, sometimes enormously.

**Fix:** build a graph over all texts, connect any two that co-occur in a pair,
take **connected components**, and use the component id as the group key.
Implemented in `../code/validation.py::pair_group_ids`.

**Check the component count.** If the pair graph is densely connected,
everything collapses into one or two components and grouped CV becomes
impossible. That is a real finding about the dataset, not a tooling failure:
it means *every* text is reachable from every other, so no split can prevent
shared-text leakage. Pick a different split axis (time, source, a held-out set
of entities) and state the residual leakage risk explicitly.
`pair_group_ids` warns when this happens.

The same logic applies to retrieval (group by query), QA (group by context
document), and any task with repeated entities.

## Matching the competition structure

Your CV should mimic how test was constructed. Concretely:

- If test contains **unseen queries** → group by query.
- If test contains **unseen documents** → group by document.
- If test contains **both unseen** → group by both (or use the stricter one).
- If test is a **future time slice** → time split, and never shuffle.
- If test has a **different class balance** → report on both the natural balance
  and the test-like balance, and know which one the metric uses.

Ask explicitly: *what is held out in the real test set?* Then hold out the same
thing.

## Number of folds

5 is the default. 10 when data is small and you can afford it (lower bias,
higher variance, 2x cost). Repeated K-fold (e.g. 5×3 with different seeds) when
fold-to-fold variance is large relative to the effects you're chasing — which is
common on small datasets.

## Discipline

- **Fix the fold assignment once**, save it to disk, and reuse it for every
  experiment. Comparing two models across different fold assignments compares
  noise.
- Fit **all** preprocessing inside the fold: vectorizers, scalers, target
  encoding, SVD, imputation. Use a `Pipeline` so this is structural, not a thing
  you have to remember.
- Report **mean and per-fold std**. A +0.003 gain against ±0.02 fold std is not
  a gain. The std is your noise band, and it is the number that decides whether
  an experiment succeeded.
- Nest hyperparameter search inside the outer CV, or accept (and state) that
  your CV is optimistic. `Snigdho8869` tunes with `GridSearchCV` on train, then
  prints `cross_val_score` on the same training data `[VERIFIED]` — that CV
  number is post-selection.
- Keep a **final holdout** you never touch until submission, if data volume
  allows. It is the only honest check on how much you've overfitted CV.

## Trusting CV vs the leaderboard

Track the pair `(CV, LB)` for every submission and **plot them**. What the
relationship looks like tells you what's wrong:

| Pattern | Meaning |
|---|---|
| CV and LB move together | trust CV; iterate locally |
| CV improves, LB flat/worse | overfitting CV, or leakage in the split |
| LB consistently below CV by a constant | distribution shift or a harder test set — the *offset* is fine, the *correlation* is what matters |
| Noisy, uncorrelated | your split doesn't match test construction. **Fix it before anything else.** |

A large constant offset is survivable. A broken correlation is not — it means
you are flying blind, and no amount of modeling will fix it.
