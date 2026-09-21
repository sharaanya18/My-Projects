# Experiment Log

Append-only. One row per experiment. Never delete a row -- a failed
experiment is a result, and re-running it is pure waste.

| exp_id | model | features | cv | metric | score | score_std | delta | runtime_s | decision |
|---|---|---|---|---|---|---|---|---|---|
| R00 | random | - | resample586 | ndcg@20 | 0.3635 | 0.0787 |  |  | floor |
| R01 | constant | - | resample586 | ndcg@20 | 0.3194 | 0.1010 | -0.0441 |  | floor |
| R00 | random | - | resample586 | ndcg@20 | 0.3635 | 0.0787 | 0.0441 |  | floor |
| R01 | constant | - | resample586 | ndcg@20 | 0.3194 | 0.1010 | -0.0441 |  | floor |
| T_tfidf word 1g | tfidf+logreg | tfidf word 1g | GroupKFold(100clust) | ndcg@20 | 0.8200 | 0.0723 | 0.5006 |  | see report |
| T_tfidf word 1-2g | tfidf+logreg | tfidf word 1-2g | GroupKFold(100clust) | ndcg@20 | 0.8671 | 0.0606 | 0.0472 |  | see report |
| E03_leaky | LGBMRegressor | raw+derived+cat | StratifiedKFold(5) LEAKY | ndcg@20 | 0.9867 | 0.0200 | 0.1196 |  | REJECT as evidence -- project leakage |
| E04_honest | LGBMRegressor | raw+derived+cat | GroupKFold(100clust)+shift | ndcg@20 | 0.9028 | 0.0478 | -0.0839 |  | HONEST REFERENCE |
