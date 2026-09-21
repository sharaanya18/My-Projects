# Transformers

Sources: `huggingface/transformers` `[VERIFIED]`, `graykode/nlp-tutorial`
`[VERIFIED]`, `Denis2054/Transformers-for-NLP-2nd-Edition` `[VERIFIED]`,
`allenai/allennlp` `[VERIFIED, dead]`.

## When a transformer is the right answer

**Yes when:** word order / composition / negation / long-range context decides
the label; enough labeled data (rule of thumb ≥ ~5k, less with a strong
pretrained model); vocabulary in test differs from train (pretraining
generalizes where TF-IDF cannot); the task is inherently pairwise-contextual
(NLI, reranking, extractive QA).

**No when:** the signal is lexical (spam keywords, topic words, IDs); tiny data;
CPU-only or tight latency; the metric is dominated by a threshold you haven't
tuned yet; **you haven't run a TF-IDF baseline.** Without the baseline you
cannot tell whether the transformer is earning its cost.

## Never train from scratch

`graykode/nlp-tutorial` `[VERIFIED]` contains beautiful <100-line Transformer
and BERT implementations. They are **teaching artifacts**. Training them from
scratch on a Shipd dataset would be strictly worse than TF-IDF at enormously
more cost. Read them to understand attention and masking; then use pretrained
weights. Same verdict for TF/Keras LSTM/GRU/CNN stacks built on randomly
initialized `Embedding` layers (`Snigdho8869` `[VERIFIED]`): on BBC they scored
**91–95%** against TF-IDF+LinearSVC's **96.4%** — worse, and far slower.

That is the honest lesson about the PDF's "Deep Learning (LSTM/GRU)" and
"GRU + Attention" entries: as *architectures* they were superseded; as
*pretrained* models they were replaced. Don't rebuild them.

## Fine-tuning defaults that work

From the HF examples `[VERIFIED]`:

```
lr 2e-5 (2e-5–5e-5 for base models; lower, ~1e-5, for large)
epochs 3 (2–5; small datasets overfit fast)
batch 16–32, max_seq_length 128–512
warmup 6–10% of steps, linear decay
weight_decay 0.01
fp16/bf16 on  (~2x speedup at equal accuracy — verified GLUE table)
```

- **`max_seq_length` is a silent data-destroyer.** Measure your token length
  distribution before accepting 128.
- **Seed variance is large** on small datasets. Report mean ± std over ≥3 seeds
  or you are reporting a seed, not a method.
- Save the **best** checkpoint by the eval metric, not the last epoch
  (`TechNBusiness` trains 40 epochs with no early stopping and keeps epoch 40
  `[VERIFIED]`).
- `--ignore_mismatched_sizes` to reuse a checkpoint with a different head.

## Model-size discipline

A `deberta-v3-base` fine-tuned well, with proper CV, a tuned threshold, and a
good validation split, beats a large model trained once on a random split.
Spend on validation before parameters.

## Tokenizer diagnostics (under-used)

Ch9 of `Denis2054` `[VERIFIED]` inspects tokenizer behavior — worth copying as a
routine check. Run your domain text through the tokenizer and look at:

- the fraction of `[UNK]` / byte-fallback tokens;
- how identifiers, numbers, URLs, and code fragment;
- token-per-word ratio vs. normal English (a high ratio means your domain is
  out-of-distribution for that model and your effective context is shorter than
  you think).

If your key entities shatter into 8 subword pieces each, that tells you
something concrete — and often argues for keeping char n-gram lexical features
in the ensemble.

## Interpretability for error analysis

Ch14 of `Denis2054` `[VERIFIED]` (BertViz, Ecco). Use attention/attribution
visualization on **your actual errors**, not on cherry-picked examples. The
useful question is narrow: *is the model attending to the evidence span, or to a
spurious artifact?* If it's the latter, you likely have a shortcut feature — see
`../validation/leakage_detection.md`.

## Do not use AllenNLP

`[VERIFIED]` — maintenance mode since 2022, last commit 2022-11-21. Pinned to a
2022 dependency world. Its one transferable idea is declarative
config-as-experiment-record, which `../EXPERIMENT_LOG.md` reproduces.
