# Text Classification

Covers binary, multiclass, and multilabel. Sources: `Snigdho8869/*` `[VERIFIED]`,
`huggingface/transformers` examples `[VERIFIED]`, `TechNBusiness/*` `[VERIFIED]`,
modern practice `[UNVERIFIED]` where marked.

## The ladder

Climb in order. Each rung must beat the previous **on CV**, not on a hunch.

| # | Model | Cost | Beats the previous when… |
|---|---|---|---|
| 0 | Majority class / prior | ~0 | never — it is the floor you must report |
| 1 | TF-IDF word(1,2) + LogisticRegression | seconds | always. **Never skip.** |
| 2 | + char_wb(3,5) n-grams, union | seconds | typos, morphology, code/IDs, non-English, short text |
| 3 | LinearSVC / SGDClassifier | seconds | high-dim sparse, clean margins; no probabilities |
| 4 | ComplementNB / MultinomialNB | instant | very small data, strong class imbalance |
| 5 | LightGBM on SVD(TF-IDF) + handcrafted features | minutes | metadata/length/structure carries real signal |
| 6 | Frozen sentence embeddings + linear head | minutes (GPU) | vocabulary mismatch train↔test, paraphrase, semantics |
| 7 | Fine-tuned transformer | hours (GPU) | enough labels (≥ ~5k) and word order/context matter |
| 8 | Ensemble of 6+7 + rung 1 | — | **only if OOF shows low correlation.** See `ensembling.md` |

Rung 1 is not a formality. On BBC (2,225 docs) the Snigdho notebook measured
TF-IDF+LinearSVC at **96.40%** vs BERT at 98.20% — and that BERT number came
from an evaluation protocol that reuses the test split (see
`../antipatterns.md`). The real gap on a clean, well-separated topic task is
usually smaller than the ladder's cost ratio implies.

## Vectorizer settings that actually matter

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

word = TfidfVectorizer(
    ngram_range=(1, 2),
    sublinear_tf=True,      # 1+log(tf). Almost always helps. Frequently omitted.
    min_df=2,               # kills hapax noise; raise to 3-5 on large corpora
    max_df=0.9,             # drops corpus-wide boilerplate
    strip_accents="unicode",
    lowercase=True,
)
char = TfidfVectorizer(
    analyzer="char_wb",     # char_wb, NOT char: respects word boundaries
    ngram_range=(3, 5),
    sublinear_tf=True,
    min_df=3,
)
feats = FeatureUnion([("w", word), ("c", char)])
```

- `sublinear_tf=True` — verified as used in the Snigdho pipeline
  (`TfidfTransformer(norm='l2', sublinear_tf=True)`). Dampens the effect of a
  term repeated 50 times. Test it; it is close to free.
- `char_wb(3,5)` is the highest-value single addition for noisy, short,
  morphologically rich, or multilingual text. It also makes the model robust to
  the adversarial typos Phase 13 warns about.
- **Fit the vectorizer inside the CV fold**, never on train+test. Use a
  `Pipeline` so `cross_val_predict` cannot leak vocabulary/IDF statistics.
  (The Snigdho notebook does get this right — it fits on `x_train` only.)

## Preprocessing: do less than you think

Verified good idea from `Snigdho8869`: a **subtractive** stopword list that
removes standard stopwords *except* negations and modals —
`not, no, don't, doesn't, didn't, won't, can, couldn't, shouldn't, mustn't,
wasn't, wouldn't, hadn't, mightn't, should, should've, why, some, do, does,
did, will` and the pronouns. Blanket `stop_words='english'` deletes the word
that flips the label in sentiment, NLI, and claim-verification tasks.

Otherwise: **test every preprocessing step as an ablation.** With `sublinear_tf`
and `min_df` doing the heavy lifting, stemming/lemmatization often changes CV by
less than its noise band, and lowercasing can destroy signal (acronyms, ticker
symbols, gene names, code identifiers). Never stack five cleaning steps and
attribute the result to the model.

## Class imbalance

In priority order:

1. **Fix the metric first.** Accuracy on a 95/5 split is meaningless. Know
   whether you are scored on macro-F1, micro-F1, balanced accuracy, AUC, or MCC
   — each implies a different optimum.
2. `class_weight="balanced"` on LogisticRegression / LinearSVC / LightGBM.
   Cheap, usually most of the benefit.
3. **Threshold tuning on OOF predictions** — for binary/multilabel F1 this is
   typically a larger gain than any model change. See
   `../validation/threshold_and_calibration.md`.
4. Resampling (SMOTE etc.) last, and rarely: on sparse text features it tends to
   interpolate nonsense. Prefer weights.

Do **not** tune the threshold on the same fold you fit on.

## Multilabel specifics

- One-vs-rest linear models are a genuinely strong baseline and are trivially
  parallel.
- **Tune one threshold per label** on OOF, not one global threshold — label
  frequencies differ by orders of magnitude.
- Micro-F1 rewards getting frequent labels right; macro-F1 rewards rare ones.
  Check which one is scored before optimizing.
- Watch for label correlation/hierarchy; a classifier chain or a post-hoc
  consistency fix can help when labels are nested.

## Transformer fine-tuning notes

From `huggingface/transformers` examples `[VERIFIED]`:

- Defaults `lr=2e-5`, 3 epochs, `max_seq_length=128`, batch 32 are a reasonable
  start. **128 tokens silently truncates** — measure your token-length
  distribution first; a long-document task loses badly here.
- `--fp16` gives ~2x speedup at equal accuracy across all 9 GLUE tasks
  (verified table in the README). Always on, with a compatible GPU.
- **Concatenating text fields is a real feature-engineering lever**, not an
  afterthought: 0.5958 → **0.659** accuracy on `amazon_reviews_multi` just by
  using `review_title,review_body,product_category` instead of the body alone.
  Test field combinations before testing bigger models.
- Seed variance on small datasets is large. Report mean ± std over ≥3 seeds, or
  your "improvement" is a seed.

## Error analysis — do this before the next rung

Cheap, and it decides what to build next:

1. Confusion matrix, then read **20 actual errors** per confused pair.
2. Split errors by: text length, class, source/domain field, language, presence
   of negation, duplicates.
3. Check whether errors are **label noise** (see `../validation/noisy_labels.md`)
   — on a noisy benchmark, chasing the last 2% is chasing mislabeled rows.
4. Ask the deciding question: *are the errors lexical (the model never saw the
   word → go to rung 2/6) or semantic (the words are there but composed
   differently → go to rung 7)?* That answer, not fashion, picks the next rung.
