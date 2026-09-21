# Feature Engineering for NLP

The rule: **no feature enters the model without an ablation showing it helps
CV.** Features are cheap to add and expensive to trust.

## Lexical (single text)

| Feature | Notes |
|---|---|
| word n-grams (1,2) TF-IDF | the baseline representation |
| char_wb n-grams (3,5) TF-IDF | typos, morphology, code, multilingual |
| length: chars, words, sentences | surprisingly strong; also a shortcut risk |
| avg word length, type-token ratio | style/complexity |
| punctuation, digit, uppercase ratios | formatting artifacts, shouting, IDs |
| counts of URLs, emails, numbers, hashtags | structural signal |
| OOV rate vs a reference vocabulary | domain-shift detector |
| language ID | **always compute it** on multilingual data |

## Pair features

The core block for similarity/matching/reranking. Implemented and tested in
`../code/text_features.py`. Ported and extended from `ElizaLo`'s
`percent_shared` `[VERIFIED]`:

**Overlap**
- Jaccard `|A∩B|/|A∪B|`; raw `|A∩B|`
- **asymmetric coverage `|A∩B|/|A|` and `|A∩B|/|B|` — keep both**; their
  difference distinguishes containment from equality
- the same three over **bigrams** and **char 4-grams**
- **IDF-weighted overlap**: `Σ idf(shared) / Σ idf(union)` — a shared rare term
  means far more than a shared stopword. Usually the strongest single overlap
  feature.

**String**
- normalized edit distance; longest-common-subsequence ratio
- shared numbers, shared capitalized tokens, shared OOV tokens — numbers and
  proper nouns are high-signal and embeddings represent them poorly

**Length**
- `len(A)`, `len(B)`, `|Δlen|`, `len(A)/len(B)`

**Semantic**
- TF-IDF cosine (word and char views separately)
- embedding cosine, from **two different model families** if available
- embedding L2 distance, converted with `1/exp(d)` (`ElizaLo` `[VERIFIED]`)
  rather than min-max — outlier-robust

## Retrieval features (for reranking / LTR)

Per (query, candidate):

- BM25 score; TF-IDF cosine; dense cosine
- **rank** under each retriever; **reciprocal rank** `1/(k+rank)`
- `score - max(score in group)` and `score - mean(score in group)`
- **z-score of the score within the group** — makes queries of different
  difficulty comparable; usually the highest-value transformation here
- number of candidates in the group; candidate's position in the source document
- agreement: does this candidate appear in the top-k of *both* retrievers?

**Within-group normalization is the point.** Raw scores are not comparable
across queries; normalized ones are.

## Metadata

Do not ignore non-text columns: category, source, author, timestamp, document
length, section, IDs. The HF example `[VERIFIED]` shows **concatenating metadata
fields into the text** took `amazon_reviews_multi` from 0.5958 → **0.659**
accuracy in one epoch. Test field combinations before testing bigger models —
it is the cheapest large gain available.

But: check every metadata feature for **leakage**. An ID that correlates with
the label, a timestamp that encodes collection order, a source field that
perfectly separates classes — these boost CV and die on test. See
`../validation/leakage_detection.md`.

## Testing features honestly

1. **Baseline first**, recorded.
2. Add one feature **group** at a time (not one feature) — single features are
   below the noise floor.
3. Same folds, same seed, every time.
4. Keep only groups whose gain exceeds the fold-wise std.
5. Re-test interactions at the end: groups that individually did nothing can
   help together (and vice versa).
6. For GBDT, check **permutation importance on OOF**, not the built-in split
   importance — split importance is biased toward high-cardinality features.
7. Any feature with suspiciously high importance is a **leakage suspect** until
   proven otherwise. Investigate before celebrating.

## Dimensionality

- Linear models: feed sparse TF-IDF directly. Do not densify.
- GBDT: sparse TF-IDF is a poor fit (trees split one feature at a time over
  100k sparse columns). Use `TruncatedSVD` to 100–300 dims, or hand-crafted
  features, or both.
- Never run PCA on sparse text — it densifies and destroys memory. `TruncatedSVD`
  (LSA) works on sparse input directly.
