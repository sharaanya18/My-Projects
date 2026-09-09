# Response Critique Point Grounding — pipeline

A pipeline for the Eris "Response Critique Point Grounding" challenge:
given a prompt, two candidate responses, and a masked pool of 16 candidate
critique statements, predict a signed grounding score in `[-1, 1]` per
candidate (magnitude ranks true critique points above distractors, sign
routes a true critique point to response A or B).

## Modules

- **`scoring.py`** — exact local reproduction of the grading pseudocode:
  `sign`, `average_precision` (with tie-block credit — see its docstring for
  why a naive tie-break doesn't reproduce the spec's "constant vector scores
  exactly the pool prevalence" property), `row_score`, `grade`, plus
  `grade_submission`/`is_malformed` for the "malformed rows score 0" rule
  and `to_score`/`to_scores` for `score = p_b - p_a`.
- **`splitting.py`** — stratified `dev` / `calibration` / `final_check`
  split of `train.csv`, stratified jointly by `domain` and the count of
  non-zero relevance values per row (2–7), via seeded largest-remainder
  apportionment per stratum (robust to tiny domain×n_true cells).
- **`api_client.py`** — batched Anthropic caller: one request per row
  (context + both responses + all 16 candidates), expects a JSON array of
  16 `{"reasoning", "p_a", "p_b"}` objects, retries malformed JSON up to 2
  times, falls back to an all-zero vector after that. Every row's full
  result — including every candidate's full reasoning text, not just the
  scores — is persisted to `<cache_dir>/<row_id>.json`, which doubles as
  the audit log and the resumability cache (re-running `run_batch` skips
  rows that already have a cache file).
- **`calibration.py`** — per-domain isotonic or Platt calibration, fit only
  on flat `(score, label)` number pairs from the calibration split.
- **`diagnostics.py`** — per-domain grounding/routing/calibration/row_score
  breakdown on the final-check split, and flags any domain that lags the
  best domain by more than a threshold on a chosen metric.
- **`submission.py`** — validates (every test id exactly once, each a
  16-element finite-float list in `[-1, 1]`) before writing the submission
  CSV; raises with every problem found rather than writing a partial file.
- **`pipeline.py`** — CLI wiring the stages together (`split`, `call-api`,
  `build-scores`, `calibrate`, `diagnose`, `submit`).

## Usage

```bash
export ANTHROPIC_API_KEY=...

python -m response_critique_pipeline.pipeline split \
    --train train.csv --out-dir splits/

python -m response_critique_pipeline.pipeline call-api \
    --csv splits/calibration.csv --cache-dir cache/calibration --max-workers 4
python -m response_critique_pipeline.pipeline build-scores \
    --csv splits/calibration.csv --cache-dir cache/calibration --out calibration_scores.csv

python -m response_critique_pipeline.pipeline calibrate \
    --scores-csv calibration_scores.csv --out-dir calibrators/ --method isotonic

python -m response_critique_pipeline.pipeline call-api \
    --csv splits/final_check.csv --cache-dir cache/final_check
python -m response_critique_pipeline.pipeline build-scores \
    --csv splits/final_check.csv --cache-dir cache/final_check --out final_check_scores.csv
python -m response_critique_pipeline.pipeline diagnose \
    --scores-csv final_check_scores.csv --calibrators-dir calibrators/ --threshold 0.05

python -m response_critique_pipeline.pipeline call-api \
    --csv test.csv --cache-dir cache/test
python -m response_critique_pipeline.pipeline build-scores \
    --csv test.csv --cache-dir cache/test --out test_scores.csv
python -m response_critique_pipeline.pipeline submit \
    --test-csv test.csv --scores-csv test_scores.csv \
    --calibrators-dir calibrators/ --out submission.csv
```

Re-running `call-api` on the same `--cache-dir` after an interruption only
pays for rows that don't already have a cache file; pass `--force` to
recompute everything.

## Tests

```bash
pip install -r requirements.txt
pytest response_critique_pipeline/tests -q
```

The test suite is fully offline — the API caller tests use an injected fake
`caller` callable instead of hitting the Anthropic API.
