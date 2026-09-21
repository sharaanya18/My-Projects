# Transformer Fine-Tuning

Practical recipe. Source for the verified numbers:
`huggingface/transformers` examples `[VERIFIED]`.

## Before you start

1. TF-IDF baseline exists and is recorded. (If not, stop and do that.)
2. Token-length distribution measured → `max_seq_length` chosen from data, not
   from the default.
3. CV scheme fixed, with the same folds the baseline used.
4. GPU + model download confirmed available. (**Blocked in this container.**)
5. Runtime estimated: `steps × batch × seq_len` → hours. Know the cost before
   spending it.

## Defaults

```
model          deberta-v3-base (strong default) / roberta-base / a domain model
lr             2e-5   (base) | 1e-5 (large)
epochs         3      (2-5; small data overfits fast)
batch          16-32  (+ grad accumulation if VRAM-bound)
max_seq_len    from the data; 128 is a default, not a decision
warmup         6-10% of total steps, linear decay
weight_decay   0.01
precision      fp16/bf16 — ~2x speedup at equal accuracy [VERIFIED GLUE table]
early stopping on the eval metric, restore best checkpoint
```

## Things that go wrong silently

- **Truncation.** `max_seq_length=128` on 600-token documents throws away most
  of the input and the loss curve looks fine.
- **Keeping the last epoch** instead of the best. (`TechNBusiness` trains 40
  epochs with no early stopping `[VERIFIED]`.)
- **Seed variance.** On small datasets, seed-to-seed spread can exceed the gap
  you're claiming. Run ≥3 seeds; report mean ± std.
- **Evaluating on the split used for early stopping.** That number is optimistic
  by construction. (Both the Snigdho BERT/XLNet numbers do this `[VERIFIED]`.)
- **Argmax over a single logit.** `np.argmax` over shape `(n,1)` is always 0 —
  a verified bug in `TechNBusiness` `[VERIFIED]`. Assert on the shape and the
  class balance of your predictions before submitting.
- **Class imbalance ignored.** Use weighted loss, or tune the threshold on OOF.

## Getting more from the same model

In rough order of value per hour spent:

1. **Better input construction** — concatenating metadata fields took
   `amazon_reviews_multi` from 0.5958 → **0.659** in one epoch `[VERIFIED]`.
   Cheaper and larger than most modeling changes.
2. **Threshold tuning on OOF** (binary/multilabel).
3. **CV + OOF ensembling** across folds (you're training k models anyway).
4. **Multi-seed averaging.**
5. Layer-wise LR decay; longer training with a lower LR.
6. A bigger model. **Last**, not first.

## Multi-stage training

- **Domain-adaptive pretraining (MLM on task text)** — helps when your domain is
  far from the pretraining distribution. Check the tokenizer diagnostics in
  `../techniques/transformers.md` first: a high token-per-word ratio is the
  signal that this is worth doing.
- **Intermediate task fine-tuning** (e.g. NLI → your pair task).
- **Pseudo-labeling** — only under the rules in
  `../techniques/ensembling.md`.

## Budget reality

With no GPU (as measured in this container: 4 CPU cores, 15 GB RAM), transformer
fine-tuning is **not viable**. Plan the solution so the lexical/GBDT rungs carry
real weight, and treat this document as conditional on confirming GPU access.
