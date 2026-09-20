# Development modules

`solution.py` in the parent directory is generated from these files and is the only
file the grader runs. Nothing here is imported by it.

| file | purpose |
|---|---|
| `ad.py` | reverse-mode autodiff over numpy arrays (all ops gradient-checked) |
| `ledger.py` | data pipeline, transformer, MLM pretraining, training, beam search, MBR |
| `common.py` | data loading, vocabulary construction, group-fold assignment |
| `final_config.json` | the shipped hyperparameters (chosen on group-held-out folds) |
| `solution_main.py` | the `solve()` entry point appended to the generated file |
| `build_solution.py` | assembles `../solution.py` from `ad.py` + `ledger.py` + `solution_main.py` |
| `full_val.py` | group-held-out validation of the shipped two-model configuration |
| `validate.py`, `wit_val.py`, `q_val.py` | harnesses used to tune the witness and question models separately |

Regenerate the entry point after editing `ledger.py`, `ad.py` or `final_config.json`:

```bash
cd dev && python3 build_solution.py
```
