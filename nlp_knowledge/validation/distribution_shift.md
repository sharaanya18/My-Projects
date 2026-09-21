# Train/Test Distribution Shift

Phase 13 flags **deliberate train/test mismatch** as a plausible Shipd design.
Assume it until measured otherwise.

## Kinds of shift

| Kind | What moves | Consequence |
|---|---|---|
| Covariate shift | P(x) | features differ; the mapping x→y still holds |
| Label shift | P(y) | class balance differs; thresholds and priors are wrong |
| Concept drift | P(y\|x) | the *rule* changed — the hardest case |
| Domain shift | source/genre/language | vocabulary and style differ |

## Detection

1. **Adversarial validation** (`leakage_detection.md`) — the primary tool. AUC
   well above 0.5 means measurable shift; the top features name it.
2. **Compare marginals** between train and test: text length distribution,
   vocabulary overlap, OOV rate, language mix, punctuation/casing statistics,
   per-field null rates.
3. **Vocabulary coverage**: what fraction of test tokens appear in train? A low
   number predicts that TF-IDF will underperform relative to CV and that
   pretrained embeddings will do relatively better.
4. **Per-field null/format differences** — often the giveaway that test was
   assembled by a different process.
5. If test labels are unavailable, compare the **distribution of your
   predictions** on train-CV vs test. A large divergence is a warning.

## Responses

| Situation | Response |
|---|---|
| Covariate shift | build validation that mimics test (weight or subsample the most test-like train rows) |
| Label shift | recalibrate priors; re-tune thresholds for the test balance |
| Concept drift | you cannot fix this with modeling — reduce reliance on the features that moved; prefer robust/simple models |
| Vocabulary shift | favor pretrained embeddings and char n-grams over word n-grams |
| Domain shift | domain-adaptive pretraining, or train on the most test-like subset |

General hedges that help under shift:

- Prefer **robust** features to sharp ones: rank/percentile features over raw
  magnitudes, char n-grams over rare word n-grams, binned over continuous.
- **Regularize harder** than CV alone suggests. Under shift, the CV-optimal
  complexity is past the test-optimal complexity.
- Blend a simple model in. It degrades more gracefully.
- Be very cautious with features whose *scale* differs between train and test —
  normalize within group/query where possible.

## The honest framing

Under real concept drift, a CV improvement can be **anti-correlated** with test
performance, because you're fitting a relationship that no longer holds. If your
CV/LB plot shows that pattern (see `cross_validation.md`), stop tuning and go
back to the task audit. More modeling will make it worse, not better.
