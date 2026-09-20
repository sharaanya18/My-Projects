# Ledger witness recovery and question restoration

Recovers, for each mixed scientific ledger, the coherent cluster of answer-and-evidence
witnesses and the missing natural-language question shared by exactly that cluster.

```bash
python3 solution.py <public_dir> <submission_out>
```

Reads `train.csv`, `train_labels.csv` and `test.csv` from `<public_dir>`, trains from
scratch on every invocation, and writes `id,witnesses,question` for all 190 evaluation
rows. Requires only `numpy` and `pandas`. Runtime is about 35 minutes on one CPU core.

## Headline result — read this first

Group-held-out validation over all 424 training ledgers (4 paper-level folds):

| fold | n | witness accuracy | Q | final `mean(W x Q)` |
|---|---|---|---|---|
| 0 | 106 | 0.2642 | 0.0734 | 0.0244 |
| 1 | 104 | 0.3269 | 0.0439 | 0.0181 |
| 2 | 103 | 0.3301 | 0.0422 | 0.0103 |
| 3 | 111 | 0.1982 | 0.0608 | 0.0168 |
| **all** | **424** | **0.2783** | **0.0553** | **0.0174** |

**0.0174 is far below the 0.10 bar, so this should not replace a solution that already
scores higher.** It is shipped as a fully rule-compliant, reproducible reference
implementation, and because the measurements below explain *why* the score is low.

Both components are beaten by trivial baselines that the rules forbid as solvers:

| component | this model | trivial baseline | baseline method |
|---|---|---|---|
| witness accuracy | 0.278 | **0.50** | pick the subset with the highest answer word overlap |
| question Q | 0.055 | **0.137** | one fixed content-free word bag fitted on training questions |
| question Q | 0.055 | 0.098 | best single training question, chosen on train |

The gap is a data-size problem, not a bug. The witness decision is mostly lexical
relatedness, which word overlap encodes for free and which a from-scratch encoder
cannot learn from 318 training ledgers; the decoder has only ~320 questions to learn
conditional generation from, so its specific guesses cost precision without gaining
recall. Using the lexical statistic directly would score far better, but
"nearest-neighbour lexical matching" is explicitly barred as the component that
determines predictions.

## Approach

Two transformer encoder-decoders trained from scratch on the supplied training ledgers.
No pretrained weights, no downloads, no external data. Reverse-mode autodiff, the
transformer, Adam, masked-language-model pretraining, diverse beam search and MBR
selection are all implemented in the file; every autodiff op is gradient-checked
against finite differences.

**Representation.** Each record becomes `[CLS] answer [SEP] evidence` under balanced
per-record truncation, so every witness stays visible at any record count.

**Encoder.** A shared transformer encodes each record independently. Three pooled views
are taken — the `[CLS]` state, a mean over the *answer* span, a mean over the *evidence*
span — and combined into one record vector. Pooling the answer separately matters: on
the training data a word-overlap probe scores 0.50 on answers alone but 0.37 once
evidence is mixed in, so a single averaged vector dilutes the signal. A
permutation-equivariant transformer over the record vectors then contextualises each
record against the others in its ledger.

**Witnesses.** A trained pairwise head scores every record pair. Training uses a
*listwise* objective: each of the `C(R, R-2)` candidate subsets is scored by its mean
internal pair score and trained with cross-entropy against the true subset, which
optimises exactly what the metric gates on. This was worth +0.085 accuracy over
per-pair binary cross-entropy. Inference submits the best subset of size
`record_count - 2`, so the output is always a valid two-to-four-token cluster.

**Question.** The decoder generates autoregressively, cross-attending to the witness
records' token states and record summaries with a learned flag marking the selected
records. Decoding is deterministic diverse beam search; the submitted candidate
maximises expected character n-gram agreement over the model's own beam
(minimum-Bayes-risk selection against the evaluation metric).

The two models are sized separately because the tasks pull in opposite directions: the
witness head needs a small, heavily regularised network (d=64, dropout 0.35) or it
memorises, while the decoder needs a larger one (d=128) or it degenerates into
repetition. A single shared-size model was measurably worse at one end or the other.

Because the metric is recall-weighted chrF (beta=2), the tuned decoding settings favour
long, hedged questions (~20 words). That is what scored best on held-out folds; setting
`"gen_temp": 1.0` in `dev/final_config.json` produces much more natural-looking
questions at a small measured cost (-0.012 Q on one fold, within noise).

## Validation method

Validation is **group-held-out on `group`** (paper level), never row level. The training
set holds 424 ledgers from only 213 papers, so a random row split would put same-paper
siblings on both sides and inflate every number. All hyperparameters were selected on
these folds; `test.csv` was never used to fit anything.

## Generalisation and row independence

Tested, not assumed — run `python3 test_isolation.py <public_dir>`:

* **Unseen data.** Vocabularies and every fitted parameter come from `train.csv` only.
  Edge cases — empty answer, empty evidence, both empty, punctuation only, wholly unseen
  tokens, CJK/unicode, 4000-word evidence, numeric only — all produce valid,
  correctly-sized rows.
* **No sibling behaviour.** Every row is encoded and decoded in isolation: records are
  encoded as independent sequences, attention is confined within a row, and each row's
  question is decoded from its own memory only. Verified numerically — a row's predicted
  witnesses and question are byte-identical whichever rows share its batch, and its pair
  logits move by <1e-7 when batch padding changes. Nothing links rows by paper.
* **Determinism.** Fixed seeds, single-threaded BLAS, fixed epoch counts, no sampling at
  inference, no cross-run cache. Repeated runs are identical.

`python3 check_submission.py submission.csv <public_dir>/test.csv <public_dir>/sample_submission.csv`
checks the submission contract: 190 rows, exact column order, ids, witness tokens sorted
and sized `record_count - 2`, and question length/character constraints.

## Rule compliance

* Trains a real sequence model on the supplied training data on every run; both
  submitted fields come from that trained system.
* No TF-IDF, bag-of-n-gram retrieval, Markov model or nearest-neighbour lexical matcher
  anywhere in the prediction path; no logistic-regression or Ridge stage; no
  hand-written question templates and no source spans copied into the question.
* No external data, no upstream-record lookup, no synthetic training rows, no
  hardcoded outputs, no sibling or paper-level linkage.
* No library installs and no pretrained weights are downloaded.
* Nothing is fitted on `test.csv`; per-row preprocessing and batched inference only.

## Files

| file | purpose |
|---|---|
| `solution.py` | standalone graded entry point, no local imports |
| `test_isolation.py` | row-independence, robustness and determinism checks |
| `check_submission.py` | submission-contract validator |
| `submission.csv` | output of `solution.py` on the supplied evaluation set |
| `dev/` | modules `solution.py` is generated from, plus the validation harnesses |
