# Playbook: Semantic Similarity / Pair Tasks

STS · duplicate detection · paraphrase · sentence-pair classification · matching.

## 0. Audit

- Symmetric or **asymmetric**? (Decides prompts and everything after.)
- Output: graded score, binary label, or a ranking over candidates?
- Metric: Spearman / AUC / F1 / accuracy?
- **Pair graph structure**: how often does one text appear in many pairs?
- Are negatives sampled or exhaustive? How were they generated?
- Class balance of the positive class.

## 1. Validation first — this is where pair tasks die

Build the pair graph, take connected components, use them as `GroupKFold` keys
(`../code/validation.py::pair_group_ids`). A random split over pairs leaks
through shared texts and can inflate CV dramatically.

Then check: do the same texts appear in test? Are test pairs generated the same
way as train pairs? A difference in negative-sampling procedure between train
and test is a classic benchmark artifact — and adversarial validation will
usually catch it.

## 2. Ladder

| Exp | Approach |
|---|---|
| S0 | length difference + token overlap only — the floor |
| S1 | TF-IDF cosine (word), thresholded |
| S2 | + char n-gram cosine |
| S3 | **full lexical feature block + LightGBM** (`../code/text_features.py`) |
| S4 | frozen embedding cosine alone |
| S5 | **S3 + embedding cosine as a feature** ← usually the best cost/benefit |
| S6 | fine-tuned bi-encoder (`CoSENTLoss` / `MNRL`) |
| S7 | cross-encoder on the pair |
| S8 | blend S5 + S7 if OOF correlation is low |

S5 over S4 is the key move: a cosine is one number; a GBDT learns *when to trust
it* alongside lexical evidence.

## 3. Threshold

For binary outputs, tune on OOF. Report AUC (ranking quality) **and** F1 at the
tuned threshold — they answer different questions and a model can win one while
losing the other.

## 4. Robustness probe

The `SimonLaub/NLP_JobTrend` protocol `[VERIFIED]`: re-run scoring on perturbed
text (paraphrase, word-order shuffle, synonym swap, typo injection) and diff the
rankings against the original. If they collapse, you're riding lexical overlap
— which matters a great deal if the Shipd test set is paraphrased or
adversarial. Keep both result sets so the comparison is auditable.

## 5. Traps

- **In-batch negatives + duplicates = false negatives.** Deduplicate before
  training with `MultipleNegativesRankingLoss`.
- Symmetric prompts used on an asymmetric task (silent quality loss).
- Monolingual encoder on multilingual text (silent quality loss) — check the
  language of every field.
- Positive-pair generation artifacts: if positives were built by a rule
  (e.g. same-article pairs), a model can learn the rule rather than similarity.
  Test for it: can a model separate positives from negatives using **only**
  length and formatting features? If yes, the sampling is leaking.
