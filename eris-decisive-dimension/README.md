# Eris "Decisive Dimension" Challenge

Predicts which of three grounds (`correctness`, `instruction_compliance`,
`completeness`) a human review panel's discussion was actually about, given
a three-way `[REQUEST]/[RESPONSE A]/[RESPONSE B]` comparison record.

- `solution.ipynb` — the full, self-contained pipeline (EDA, locale-aware
  grouped CV, feature engineering, model stacking, inference).
- `dataset/public/` — `train.csv`, `test.csv`, `sample_submission.csv`.
- `working/submission.csv` — produced by running the notebook end-to-end.
