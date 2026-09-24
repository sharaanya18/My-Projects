# Project Eris: Crowd-Tag Ranking. Solution Approach for Claude Code

Goal: rank the 80 candidate tag codes per title so the 3 crowd-agreed tags come first.
Target: MAP ≥ 0.40 on unseen institutions. No shortcuts, fully deterministic, fully compliant.

---

## 0. Rule hierarchy (read first)

1. **Problem Description**: hard rules. Violating one is a compliance failure.
2. **Guidebook**: applies wherever the description is silent (environment, runtime, "real ML" rules).
3. **Platform execution rule**: the run must be deterministic. **No behaviour may depend on wall-clock
   time or on which packages happen to import.** That means no time-based early stopping, no
   "if time remains, do X", and no silent fallback model.

### Hard constraints, with where each one comes from

| # | Constraint | Source |
|---|---|---|
| 1 | Inputs: only `record_title`, that case's candidate pool (its *contents*), and what is learned from `train.csv` + `train_labels.csv` | Description, Rules |
| 2 | Never use `case_id` as a signal | Description |
| 3 | Never use row order of any file, or candidate order inside a pool | Description |
| 4 | No looking up the item, collection, archive, or picture anywhere | Description |
| 5 | Public pretrained text models may be fine-tuned. No gated, private, or API-key models. No external inference API | Description, Pretrained policy |
| 6 | Output: exactly `case_id,ranked_tags`, one row per test case, all 80 codes, space-separated, no index, no duplicate `case_id` | Description, Submission |
| 7 | Real training happens inside the script, every run, from raw CSVs. No cached embeddings or self-hosted fine-tuned weights | Guidebook 1.1, 3.8, 4.2 |
| 8 | No external data, no synthetic data, no pseudo-labelling, no test-set-wide statistics or calibration. Each test case is scored independently | Guidebook 4.2 |
| 9 | The ranking itself must be a trained deep model, not hand-made features fed to a classical ranker | Guidebook 5.3 |
| 10 | TF-IDF, n-grams, and regex are grey-area. They may only be auxiliary inputs to the deep model, never the core | Guidebook 4.3 |
| 11 | Kaggle Docker libraries only (torch, transformers, numpy, pandas, sklearn). Weights from Hugging Face only | Guidebook 3.1, 3.3, 3.4 |
| 12 | A10G GPU. Design total runtime ≤ ~55 min | Guidebook 3.5, 3.6, 4.4 |
| 13 | Deterministic: fixed seeds, fixed epoch counts, no time-based or import-based branching | Platform rule |
| 14 | CLI: `python3 solution.py <public_dir> <submission_out>` | Submission instructions |

### Explicitly NOT used (these would be score gaming)
- The frequency-band construction of the pools (e.g. finding the tag at the "centre" of a frequency
  neighbourhood). The description states frequency is information-free by design. Exploiting how the
  decoys were sampled is a data-generation exploit, not modelling.
- Tag gold-rate priors. The description measured these at or below chance.
- Anything order-based or `case_id`-based.

---

## 1. Verified facts about the data (measured on the real files)

- 15,003 train cases, 4,792 test cases, 80 unique candidates per case, exactly 3 gold, gold ⊂ pool.
- 5,668 distinct tag codes. **Every test-pool code also appears in a train pool**, so a closed
  tag-embedding table is valid.
- 4,909 codes are gold at least once in training. Gold counts are long-tailed: median 4, and 998 codes
  are gold only once. Rare tags need a good embedding **initialisation**, not just random init.
- The language mix shifts between splits. Train is about 61% German, 13% Dutch, 7.5% English,
  6% French. Test is about 72% German, <1% Dutch, 8.5% French, 8% English. This is the
  held-out-institution effect.
- The metric implementation reproduces the description's anchors: order baselines 0.085–0.089, title
  co-occurrence 0.171 on the Dutch holdout vs 0.185 on the real test.
- **Validation calibration (important):**

  | Same baseline | MAP |
  |---|---|
  | random split | 0.265 |
  | real test | 0.185 |
  | Dutch holdout | 0.171 |

  Random splits overstate the test score by about 43%. The Dutch holdout understates it by about 8%.
  **Always judge models on the Dutch holdout. The gate for 0.40 on test is ≥ 0.37 on the Dutch holdout.**
- Pool coherence is a real signal. The 3 gold tags describe the same picture, so they relate to each
  other, while decoys are random. Dense co-gold tag embeddings alone give 0.124 (chance is 0.086) and
  add about +0.01 on top of the title signal. It is modest but complementary, so it goes **inside**
  the model as a learned component.
- A fully trained model **without** pretrained semantics (hashed char and word n-grams) plateaus at
  about 0.22 on the Dutch holdout. The training mechanism works. The remaining gap is representation
  quality, which is what the pretrained multilingual encoder supplies.

---

## 2. Model

### 2.1 Title tower
- Backbone: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (public, XLM-R base, covers
  every title language). Secondary / ensemble member: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Mean pooling over tokens with the attention mask, `max_length=64`. Titles are four words at the
  median and 54 at most.
- Projection head: `Linear(768→256) → LayerNorm → Dropout(0.1)`.
- **Full fine-tuning** with layer-wise LR decay (0.9 per layer, top layer 3e-5). Freeze only the
  embedding layer.

### 2.2 Tag tower: a trainable table with data-driven initialisation
- `nn.Embedding(5668, 256)`, vocabulary = union of candidate codes in train.csv and test.csv. These
  are code strings only, not labels.
- **Initialisation, computed inside the script from training data only:**
  1. Encode every training title with the *frozen, not yet fine-tuned* backbone. This is one fast
     no-grad pass.
  2. For each tag, average the embeddings of the titles where it is gold. That is its "prototype".
  3. Project the prototype through the freshly initialised projection head. Tags never gold in
     training get the global mean.
  4. Add a small co-gold component: an SVD of the tag-tag PPMI matrix built from training gold
     triples (64-dim), concatenated or projected in. This gives rare tags a meaningful starting point.

  Everything here is learned from the supplied training cases, which satisfies the description's
  requirement that "the association between phrasing and code has to be learned here".

### 2.3 Pool-context scorer (uses pool *contents*, provably not pool *order*)
- Input sequence: `[title_vec] + [tag_emb(c_1..c_80)]`. Add a learned type embedding
  (title vs tag). **No positional encoding.**
- 2-layer Transformer encoder (d=256, 4 heads, dropout 0.1). With no positional information it is
  permutation-equivariant over candidates, so pool order cannot leak.
- Per-candidate logit = `cos(title_vec, tag_vec_i) * scale + MLP(contextual_tag_i)`. The first term
  is the direct title→tag match. The second learns coherence among candidates and title-conditioned
  interactions.
- During training, **shuffle each pool** before every forward pass.
- Required unit test: permute a pool at inference and assert the resulting ranking is identical.

### 2.4 Optional auxiliary input (grey area, so keep it small)
- Hashed char (2–4)-gram features of the title → `Linear → 32 dims`, concatenated into `title_vec`
  before the projection. It is only an auxiliary input to the deep model (Guidebook 5.2 / 5.4
  pattern). Keep it only if it adds ≥ +0.005 on the Dutch holdout. Otherwise drop it.

### 2.5 Loss
- Listwise multi-positive softmax over the case's own 80 candidates:
  `L = −Σ_j y_j log softmax(s)_j`, where `y_j = 1/3` for gold and 0 otherwise. This matches exactly
  what MAP ranks over.
- Label smoothing 0.05 over the pool.
- L2 pull of the tag embeddings toward their initialisation (λ≈1e-4). This stops rare tags drifting
  on 1–3 examples.

### 2.6 Regularisation against overfitting
- Title word dropout 10% (train only). Dropout 0.1 everywhere. AdamW weight_decay 0.01. Gradient clip 1.0.
- Few epochs (see section 4), cosine schedule with 6% warmup.
- Monitor the train-vs-Dutch-holdout MAP gap. A gap > 0.25 means overfitting (lower LR or epochs,
  raise the tag-L2). Both low means underfitting (raise the LR on the head, add an epoch, check the init).

---

## 3. Validation protocol (offline development)

1. Hold out Dutch-language training titles, about 2,000 cases. Use `langdetect` with seed 0 during
   **offline** development only.
2. Train on the rest. Evaluate with the exact AP definition from the description.
3. Sanity gates before any model work:
   - supplied, shuffled, and code-sorted order ≈ 0.085–0.09
   - title co-occurrence ≈ 0.17
4. Milestone gates on the Dutch holdout:

   | Milestone | What | Gate |
   |---|---|---|
   | M1 | Frozen backbone + tag prototypes, cosine ranking, no gradient training | > 0.22 (must beat the n-gram model) |
   | M2 | + fine-tuned two-tower, listwise loss | ≥ 0.30 |
   | M3 | + pool-context transformer | ≥ M2 + 0.02 |
   | M4 | + second backbone / seed ensemble (average per-case z-scored logits) | **≥ 0.37 → submit** |

5. If you are stuck below a gate, apply these levers in this order: fix the tag init (it matters most
   for rare tags), full fine-tune instead of partial, LR sweep {1e-5, 2e-5, 3e-5} (run offline and fix
   the winner as a constant), increase `d` to 384, add the MiniLM ensemble member.
6. Never judge on a random split (it is inflated by about 43%). Never look at test.csv answers or
   tune on the public LB.

---

## 4. Final `solution.py` (what actually runs on the grader)

```
python3 solution.py <public_dir> <submission_out>
```

1. **Determinism setup, first lines:**
   `os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"`; seed `random`, `numpy`, and `torch` (+cuda) = 42;
   `torch.backends.cudnn.deterministic=True`; `benchmark=False`;
   `torch.use_deterministic_algorithms(True, warn_only=True)`; DataLoader `num_workers=0` with a seeded
   `generator`.
2. Load the CSVs and merge the labels. Build the tag vocabulary. Build pool index tensors.
3. Compute tag initialisations from the training data (section 2.2).
4. **Epoch selection in-script (deterministic):** for each backbone, train `max_epochs` (a fixed bound)
   on the non-Dutch rows, record the Dutch-holdout MAP after every epoch, and pick the best epoch
   (earliest wins ties). This depends only on data and seeds, never on time. Guidebook §1.1 prefers
   selection inside the script over hand-tuning offline.
5. Retrain on **all 15,003 cases** for exactly the selected number of epochs, with the same LR
   schedule length. Training rows are put in a content-based canonical order, so file row order and
   `case_id` cannot affect the model. Model downloads are pinned to exact commit hashes, and
   attention runs on the deterministic math kernel.
6. Inference: score each test case independently, ensemble by averaging per-case z-scored logits,
   `argsort` descending, and emit **all 80 codes**.
7. Validate before writing: 4,792 rows, unique `case_id`, each row is exactly its pool's 80 codes.
   Raise on failure. Then `to_csv(index=False)` after `mkdir(parents=True, exist_ok=True)`.
8. **No `try/except` fallback to a different model.** If torch or transformers fail, the run should
   fail loudly. A silent downgrade is exactly the environment-dependent behaviour the platform
   penalises. Remove the fallback path from the earlier draft.

### Runtime plan (A10G, measure once offline, then hard-code)
- mpnet-base, 15k titles, batch 64, max_len 64: about 40–60 s/epoch. MiniLM: about 15–20 s/epoch.
- Prototype pass plus SVD: under 2 minutes.
- Full-data training of both backbones: about 10–15 minutes. Optional diagnostic holdout run: about
  the same again. Inference: under 1 minute.
- **Total ≈ 25–35 minutes**, well inside the limit. Time is not a constraint here, so there is no
  reason for any time-based logic.

---

## 5. Compliance self-check (put this in the script's docstring)

- [ ] Only title text, pool contents, and training labels are used
- [ ] No `case_id`, no row order, no pool order (the permutation test passes)
- [ ] No external data, archive lookup, image retrieval, or API calls. The backbone is public on HF
- [ ] All training happens in-script from raw CSVs. Nothing is cached or loaded from earlier runs
- [ ] No pseudo-labels or test-wide statistics. Each test case is scored alone
- [ ] Fixed seeds and fixed epochs. No time-based or import-based branching. No fallback model
- [ ] Output has 4,792 rows × all 80 codes, correct header, no index

---

## 6. What was measured vs what still needs your run

**Measured in development on the real data:**
- the metric
- all baselines
- the validation calibration (random vs holdout vs real test)
- the pool-coherence signal (+0.01)
- the no-pretraining ceiling (~0.22)

**Not measured:** the transformer milestones M1–M4. They need a GPU and Hugging Face access, which the
development sandbox did not have. They are **gates, not guarantees**. Run M1–M4 on the Dutch holdout
first. Only submit once M4 is ≥ 0.37 there, which corresponds to about 0.40 on the unseen test.
