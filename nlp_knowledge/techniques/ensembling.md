# Ensembling & Blending

The PDF cites ensembling three times. The one ensemble I could actually verify
in the cited code **did not beat its best single member** — start there.

## Evidence first

`Snigdho8869/Multiclass Text Classification` `[VERIFIED]`:

```python
ec = VotingClassifier(
    estimators=[('MNB', mnb), ('RF', rfc), ('LR', lr), ('SVM', svc)],
    voting='soft', weights=[1, 2, 3, 4])
```

Result: **96.40%** — *exactly* LinearSVC's standalone score. Four models, hand-
picked untuned weights, zero gain. Why:

1. All four sit on the **same** TF-IDF feature matrix → highly correlated errors.
2. Weights `[1,2,3,4]` were chosen by hand, never tuned on OOF.
3. On 445 test documents the measurement can't resolve differences smaller than
   ~0.22 points anyway.

**Ensembling is not free and is not automatic.** It is a technique with
preconditions.

## The precondition: complementary errors

Before ensembling, compute on **OOF predictions**:

- Pearson/Spearman correlation between model prediction vectors. `> 0.95` →
  almost certainly no gain.
- **Disagreement rate** and, more importantly, the accuracy **on the rows where
  they disagree**. If model A is right on 80% of disagreements, don't blend —
  just use A.
- Per-segment performance (by class, length, domain, language). An ensemble pays
  when models win on **different segments**.

Genuinely complementary pairings — worth testing because they fail differently:

| Pair | Why they differ |
|---|---|
| TF-IDF linear + transformer | lexical vs contextual |
| BM25 + dense retrieval | literal terms vs paraphrase |
| word n-grams + char n-grams | semantics vs morphology/typos |
| GBDT on features + neural score | tabular interactions vs text |
| different pretrained families | different pretraining data |
| different seeds/folds of one model | variance reduction only — real but small |

## Methods, in order of how often they're worth it

1. **Rank averaging** — for ranking/AUC metrics. Immune to scale differences.
   Usually the safest first blend.
2. **Weighted average of probabilities** — weights fit on OOF (scipy
   `minimize`, or a simple grid). **Fit weights on OOF, never on the test set.**
3. **RRF** — for combining retrieval systems (`1/(k+rank)`, k=60). No
   normalization needed.
4. **Stacking** — a meta-model (usually LogisticRegression, kept deliberately
   simple) over OOF predictions. Strongest, most leak-prone. See
   `../validation/oof_predictions.md` — the meta-model must be trained on
   *out-of-fold* predictions only, and needs its own nested validation.
5. **Snapshot / seed ensembling** — average several seeds of the same
   architecture. Modest, reliable, cheap if you're training anyway.

## Rules

- **Never blend on the test set.** Weights, thresholds, and meta-models come
  from OOF.
- **Always compare the blend to its best single member** on the same CV. If the
  blend doesn't clear it by more than the noise band, ship the single model —
  it's simpler, faster, and less likely to break at submission.
- Estimate the noise band: bootstrap the CV metric, or report fold-wise std. A
  +0.002 blend gain on ±0.01 fold std is nothing.
- Keep the blend **reproducible**: fixed seeds, fixed fold assignment, logged
  weights. A blend you can't regenerate is a liability at submission time.
- Prefer few, strong, diverse members over many weak ones.
- A **cascade** (retriever → reranker) is usually a better use of compute than a
  parallel ensemble in retrieval tasks. Try it first.

## Pseudo-labeling

Justified only when: test data is available and plentiful, your CV is
trustworthy, and there is a genuine distribution shift you're adapting to.

Rules: use **confident** predictions only (tuned threshold), keep pseudo-labeled
data at a modest fraction of the real training set, **re-generate labels
per-fold** to avoid leaking valid-fold information, and always compare against
the non-pseudo-labeled baseline on the same folds. It is easy to build a
self-confirming loop that improves CV and hurts the leaderboard — treat any
pseudo-labeling gain with suspicion until it replicates on a held-out split you
didn't use to build the pseudo-labels.
