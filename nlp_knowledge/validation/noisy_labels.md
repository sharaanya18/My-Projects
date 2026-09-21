# Noisy Labels

Phase 13 lists noisy labels as an expected Shipd hazard. On a noisy benchmark,
the last few points of CV are often **unreachable by construction** — and
chasing them wastes the experiment budget.

## Detecting label noise

1. **Duplicate texts with different labels.** The cleanest evidence. Count them:
   the rate is a direct lower bound on the noise rate, and it also caps the
   achievable score.
2. **High-confidence errors.** Train a model with CV, collect OOF predictions,
   and sort by `P(predicted) - P(true label)`. Read the top 50 by hand. If the
   model is confidently "wrong" and you agree with the model, the label is wrong.
3. **Cross-model agreement.** Rows that *every* diverse model gets wrong are
   noise candidates, not hard examples.
4. **Per-annotator / per-source error rates** if that metadata exists. A single
   bad source often explains most of the noise.
5. **Confident learning** (cleanlab-style): estimate the joint distribution of
   noisy and true labels from OOF probabilities.

## Responses, in order

1. **Quantify first.** Estimate the noise rate before acting. If it's ~1%,
   ignore it and spend the time elsewhere. If it's 15%, it dominates your
   strategy.
2. **Robust losses** — label smoothing (0.05–0.1), symmetric cross-entropy,
   Huber for regression, bounded losses generally.
3. **Don't over-train.** Networks fit clean patterns first and memorize noise
   later; earlier stopping is a genuine noise defense.
4. **Sample weighting** — down-weight suspected-noisy rows rather than deleting
   them. Less destructive, and reversible.
5. **Removal** — only for clearly-wrong rows, and **only from training, never
   from validation.** Removing hard rows from validation is self-deception: it
   makes the number go up and tells you nothing.
6. **Ensembling** averages away some noise sensitivity.

## Critical rule

> **Never clean the validation set with a model's own predictions.**

You will remove exactly the rows your model gets wrong, your CV will improve,
and nothing real will have changed. If you must produce a cleaned evaluation,
clean it by a rule independent of the model (duplicates, schema violations,
manual review) and **report both the cleaned and uncleaned numbers**.

## Noise vs. a hidden variable

Before concluding "noise", check whether the label depends on context you
haven't used:

- another column you dropped,
- document position / surrounding rows,
- annotation guidelines that changed over time,
- a **sense-dependent** label (the same string means different things in
  different contexts — Phase 5's sense-disambiguation category).

Phase 13 calls these hidden definitions and context-dependent labels. A
"contradictory" pair of rows with identical text and different labels is
sometimes the strongest available clue that a context variable exists and you
have not found it yet. That is a modeling opportunity, not noise — and it is
worth real investigation time before you write the rows off.
