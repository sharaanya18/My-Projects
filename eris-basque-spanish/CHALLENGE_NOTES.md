# Challenge notes: Basque/Spanish speech-continuity retrieval (Eris)

## Status
```
STATUS: baseline recipe measured on 5 held-out-cluster folds (Kaggle T4): frozen 0.2530 -> tuned 0.3979.
        Below the 0.5 target on held-out data. CFG constants are still PROVISIONAL, not tuned.
PUBLIC_SCORE: none submitted
```

## Contract (quoted from the challenge text)
- Metric: "The metric is the mean reciprocal rank." A query whose ranking omits the answer scores 0.
- Runtime command: `python3 solution.py <public_dir> <submission_out>`.
- The challenge states no runtime limit -> guidebook default: A10G, 1.5 h expected max.
- Banned signals: id strings; row order; order inside candidate_paragraph_ids; candidate frequency
  across galleries; "each candidate answers at most one query" / reallocating candidates between
  queries; outside copies of the proceedings; external translation; private/API models.
- Allowed: fine-tuning a public multilingual encoder using the training queries and answers.
- Published reference points on the evaluation queries: random 0.0913, char TF-IDF 0.2067,
  frozen public encoders 0.2167 / 0.2701 ("the strongest route measured that involves no learned model").

## Measured data facts (this repo's data, train side unless noted)
- 52 train galleries / 4,287 queries; 45 test galleries / 1,791 queries; candidates per gallery
  min/median/max: train 26/47/60, test 27/52/71.
- Galleries are fully disjoint on both sides: no paragraph, query or candidate is shared between
  galleries, and none between train and test. Every row of a gallery lists the same candidate set.
- Train differs structurally from test: every train candidate is an answer and has ~2 different
  Spanish queries (median 2, max 2); test has at most 1 query per candidate (~80% coverage).
  So train has ~2.2k distinct target speeches, not 4.3k independent examples.
- Reproduced diagnostics on train galleries: random MRR 0.100 (test 0.0913, smaller galleries),
  char 3-5gram TF-IDF (fit on train, diagnostic only) 0.214 (published test 0.2067), frozen
  multilingual-e5-base mean-pool with "query: " prefix 0.253. Raw difficulty is comparable.
- Per-gallery frozen MRR ranges 0.175-0.38, so a fold of ~10 galleries is noisy: use repeated splits.

## Validation design
- Gallery-key grouping only stops literal leakage, and galleries are already disjoint.
- The private set is a different legislative term, so folds hold out whole clusters of galleries
  (competitor_patterns.md B12; B7: match the magnitude of the gap). Gallery vector = mean frozen
  Spanish-query embedding + mean frozen Basque-candidate embedding. KMeans (k=12) clusters are packed
  greedily into 5 folds by size. A gallery's nearest neighbour lies in the same fold 58-71% of the
  time with cluster folds vs 18% with random folds.
- Limitation: no dates are supplied, so time order is not recoverable; clusters are a topical proxy
  for "new agenda", and cannot reproduce the new-speaker share (36 of 84 speakers are new/returning).
- Ensembling: 52 train galleries (fewer effective clusters) is too few to trust a learned blend
  (LEARNINGS: OOF-optimised blend over ~54 groups failed to transfer). Single fine-tuned model first;
  if ensembling later, a fixed equal-weight average only.

## Decisions taken from the workspace docs
- No wall-clock-dependent control flow (LEARNINGS [DETERMINISM]); fixed epochs; timing logged only.
  This overrides CLAUDE.md's time-guard template. compliance_scan.py will warn "NO TIME GUARD":
  expected, intentional.
- Positional sys.argv, no CPU fallback, no import fallback, pinned HF revision, eager attention,
  TF32 off, deterministic algorithms on. On the T4 the deterministic flags raised no errors; bit-exact reproducibility across two runs is NOT yet verified (run twice, diff).
- Nothing is fitted on test text (Q2): no TF-IDF, no test-side normalisation.

## Experiment log (fill from GPU runs; one variable per experiment)
| id | change | cluster-CV tuned MRR (mean, sd) | frozen MRR same folds | note |
|----|--------|-------------------------------|-----------------------|------|
| e1 | provisional CFG: e5-base, 4 ep, lr 2e-5, max_len 192, 16 q/step, fp16 + grad-checkpoint, Kaggle T4, 5 held-out-cluster folds (seed 0) | **0.3979 (sd 0.0116)**; folds 0.412/0.395/0.391/0.410/0.381 | 0.2530 (sd 0.0092) | +0.145 on every fold. Below the 0.5 target. NOT the private score. ~13 min/fold on T4 |
