# Eris: Unseen-Pair Shell Pipeline Synthesis

From-scratch, CPU-only solution for the Shipd/Eris challenge.

- `src/`: development modules (tokenizer, metric, splits, model, training, decoding, dev harnesses)
- `final/build_submit.py`: inlines the needed `src/` modules into one standalone `final/submit.py` and `final/solution.ipynb`
- `final/submit.py`: the graded script. It reads `./dataset/public/*.csv` and writes `./working/submission.csv`

Model: a shared Transformer encoder with three parts. A plan decoder predicts the command heads. A realize decoder writes each stage and can copy literals (file names, patterns) from the description. An auxiliary head-set classifier helps the encoder spot which commands are needed. The script validates on a compositional holdout of whole head-to-head pairs removed from train.csv, and uses it for early stopping and decoding choices. It then trains full-data seeds and ensembles them, with a watchdog at about 52 minutes.

Put the challenge CSVs in `data/` for development. They are not committed.

Work in progress: the final config and the submission are still being produced.
