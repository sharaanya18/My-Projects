# Eris — Catalogue-Text Tag Ranking

Solution for the Shipd/Eris challenge *"Catalogue-Text Tag Ranking"*: given a
museum catalogue title and a pool of 80 opaque tag codes, rank the 3 codes
that independent describers actually agreed on above the 77 decoys. Scored
by mean average precision (MAP) on evaluation cases drawn from institutions
that supply **no** training case (whole-institution holdout).

## Layout

```
data/                 # place train.csv, train_labels.csv, test.csv, sample_submission.csv here (gitignored)
src/
  metric.py            # MAP@pool reimplementation, sanity-checked against the challenge's stated baselines
  normalize.py          # NFKC + lowercase + strip digits/punctuation
  grouping.py            # title-clustering GroupKFold — a proxy for the real (invisible) institution split
  data.py                 # CSV loading / tag-vocab building
  features_sparse.py       # word + char n-gram TF-IDF -> SVD, fit on train only
  model.py                  # shared trainable head: projection -> unit-norm dot product with trainable tag embeddings
  model_encoder.py           # fine-tuned multilingual transformer branch (ungated HF backbones only)
  train_sparse_cv.py          # group-CV driver for the sparse branch
solution.ipynb                 # self-contained notebook for the Eris runtime (reads ./dataset/public/, writes ./working/submission.csv)
```

## Approach

Two branches feed one shared scoring head (unit-norm title embedding · unit-norm
tag embedding, temperature-scaled), trained with a multi-positive softmax loss
over each case's own 80-candidate pool (not the full ~5.7k tag vocabulary):

1. **Sparse branch** — word 1-2gram + char 3-5gram TF-IDF, reduced by SVD
   (fit on training titles only), then a small trainable MLP into the shared
   space.
2. **Encoder branch** — an ungated multilingual transformer
   (`paraphrase-multilingual-MiniLM-L12-v2` by default, with fallbacks),
   fine-tuned end-to-end, mean-pooled, projected into the same shared space.

Per-tag embeddings are trainable parameters (5,667 test-pool codes all occur
in the training pools, so this is transductive over the tag vocabulary even
though evaluation institutions are unseen).

Both branches are trained separately and blended by averaging their
per-case z-scored logits (no searched blend weights — see rationale below).

## What the rules ruled out, and why it mattered

The pools are **frequency-band matched**: within a pool, decoys are drawn
from the commonness band immediately above/below the correct answer. A
quick local check confirms why this rule exists — ranking candidates by
*"how often this code is a correct answer across all of train"* looks
predictive in-sample (because a case's own label leaks into its own tag's
global count) but the challenge statement reports it scores *at or below
chance* on the real held-out test (0.072 descending vs. ~0.084 chance). No
pool-frequency, rank, or position feature is used anywhere in this model,
and title clustering for the group-CV split is used only to build folds,
never as a model input.

## Validation

`train_sparse_cv.py` clusters normalized titles (char n-gram TF-IDF -> SVD ->
KMeans) into pseudo-institutions and runs GroupKFold on top, since the real
holding-institution field isn't in the supplied columns. This is a
deliberately pessimistic proxy for the true unseen-institution split, not
the real thing.

## Honest score expectations

The challenge statement's own measured baselines: order-supplied 0.084,
best non-learned route (title-word/tag association) 0.185, ceiling 1.0.
See `RESULTS.md` (generated after the CV run) for this submission's measured
group-CV MAP. Getting to a MAP of 0.5 on real unseen institutions is not
supported by anything measured here or in the reference non-learned
baselines the challenge itself reports — treat any pre-submission number in
this repo as a floor/ceiling estimate from a proxy split, not a guarantee.
