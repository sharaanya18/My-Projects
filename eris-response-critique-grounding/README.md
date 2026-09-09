# Response Critique Point Grounding — Eris submission

`solution.ipynb` is a self-contained, classical-ML (no LLM calls, no
pretrained embeddings) solution to Eris's "Response Critique Point
Grounding" challenge. See the notebook's own intro and "Rubric alignment"
cells for the full design rationale; this file is just the run instructions.

## Design summary

Two calibrated LightGBM classifiers per candidate instance:

- **Model G (grounding)** — is this candidate a genuine critique point?
- **Model R (routing)** — trained on genuine instances only, does it
  concern response B?
- **Combined score** — `clip(m × (2r − 1), -1, 1)` where `m`, `r` are the
  isotonic-calibrated outputs of G and R.

Every feature is hand-engineered from the raw text (TF-IDF cosine
similarity, n-gram Jaccard overlap, LCS ratio, length ratios, a
data-mined absence/negation cue-phrase list, all computed against both
responses plus their difference) — no generative model, no pretrained
encoder. Candidate pool position and candidate frequency are never used as
features. All cross-validation is `StratifiedGroupKFold` grouped by row id
(so a row's 16 candidates never split across train/val) and stratified by
domain; every reported metric is out-of-fold.

## Running

```bash
mkdir -p dataset/public
# copy train.csv, test.csv, sample_submission.csv into dataset/public/
jupyter nbconvert --to notebook --execute solution.ipynb --output solution.ipynb
# -> ./working/submission.csv
```

Or open `solution.ipynb` directly in Jupyter/Kaggle and run all cells.
Only standard Docker-image libraries are required: `pandas`, `numpy`,
`scikit-learn`, `lightgbm`. No internet access is needed at runtime.

`dataset/public/` and `working/` are placeholders — the actual challenge
data and generated submission are not committed to this repository.
