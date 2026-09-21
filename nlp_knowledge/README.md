# NLP Competition Knowledge Base

A reusable problem-solving system for Shipd/Eris NLP challenges.

Built from the resource list in the supplied PDF (`NLP Resources`, 3 pages, 6 categories),
extended with modern methods where the cited material is outdated.

## What this is — and is not

This is **not** a summary of the PDF. The PDF is a bare list of titles with no URLs
(verified: zero link annotations, zero `http` strings in the raw file, including inside
compressed streams). Every resource here was resolved by name, fetched where the network
allowed, and **read**. Claims are tagged:

| Tag | Meaning |
|---|---|
| `[VERIFIED]` | I read the actual file/code in this session. Path in `resources_index.md`. |
| `[UNVERIFIED]` | From model knowledge. The source host was blocked. Check before relying on it. |
| `[BLOCKED]` | Could not be accessed at all. Contents are **not** reconstructed. |

Nothing in `[BLOCKED]` sections is invented. See `access_log.md`.

## Layout

```
resources_index.md      Every PDF resource: category, what it solves, strengths,
                        weaknesses, when to use / not use, modernization verdict.
access_log.md           Network reality: what was reachable, what was not, and why.
problem_patterns.md     Phase 5 problem-type detector (taxonomy + decision procedure).
antipatterns.md         Concrete broken patterns found in the cited repos. Do not copy.
techniques/             Transferable methodology per task family.
models/                 Per-model-family notes: cost, when it wins, failure modes.
validation/             Validation, leakage, shift, noisy labels, OOF, thresholds.
playbooks/              Ordered, runnable attack plans per problem family.
code/                   Runnable helpers. Tested in this session (see code/README.md).
```

## How to use it on a new challenge

1. Run `code/task_detector.py` on the raw data → candidate problem families.
2. Read the matching `playbooks/*.md`. Each is an ordered ladder, cheapest first.
3. Write `SHIPD_TASK_ANALYSIS.md` (template: `templates/SHIPD_TASK_ANALYSIS.md`).
4. Build the validation split **before** the first model (`validation/cross_validation.md`).
5. Run the baseline ladder. Log every run in `EXPERIMENT_LOG.md`.
6. Only climb the ladder when the current rung's CV says the next rung is worth it.

## Environment constraints observed in this session

Measured, not assumed:

- **4 CPU cores, 15 GB RAM, no GPU.**
- **PyPI reachable** → sklearn / LightGBM / rank_bm25 baselines run here.
- **huggingface.co BLOCKED** → `SentenceTransformer("...")` and any `from_pretrained`
  download **will fail** in this container. Transformer-based rungs need an environment
  with HF access, or pre-downloaded weights.
- **kaggle.com, arxiv.org BLOCKED.**
- **github.com API/HTML blocked; `raw.githubusercontent.com` and `git clone` work.**

This is the single biggest practical constraint: **the top of every ladder in these
playbooks is currently unreachable from this container.** Plan experiments so the
sparse/lexical rungs carry real weight, and confirm GPU/HF availability before
committing to a dense-retrieval or fine-tuning plan.
