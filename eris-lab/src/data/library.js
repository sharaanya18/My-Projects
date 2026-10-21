/* Reusable strategy library, organised by task type.
 * kind: lesson (observed in your files) | strategy | pitfall
 * prov: verified | rule | hypothesis | verify  (see platform.js header)
 * Lessons from high_score.py describe what the code does. Its score effect is
 * unverified because no scores or logs were supplied with the file. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.TASK_TYPES = [
    { id: "general", name: "All tasks" },
    { id: "tabular", name: "Tabular ML" },
    { id: "nlp", name: "NLP classification / regression" },
    { id: "retrieval", name: "Retrieval & ranking" },
    { id: "cv", name: "Computer vision" },
    { id: "seqgen", name: "Sequence generation" },
    { id: "recsys", name: "Recommendation" },
    { id: "structured", name: "Structured prediction" },
    { id: "finetune", name: "Fine-tuning challenges" },
    { id: "scratch", name: "From-scratch challenges" },
    { id: "biochem", name: "Biology / chemistry" },
  ];

  const L = (task, kind, prov, title, body, validate, cite) => ({ task, kind, prov, title, body, validate, cite });

  E.LIBRARY = [
    // ---- Lessons from your previous solution (high_score.py) ----
    L("retrieval", "lesson", "verified", "Fold by query, never by row",
      "high_score.py groups every candidate row into a per-query slate and assigns folds to whole slates. Candidates of one query never sit on both sides of a fold, so out-of-fold scores reflect unseen queries.",
      "Compare row-level KFold with slate-level folds. If the row-level CV is noticeably higher, it was leaking.", "S5"),
    L("retrieval", "lesson", "verified", "Stress-test with leave-one-direction-out",
      "Besides 5-fold grouped CV, the script holds out each language direction in turn (LODO). That simulates a test set dominated by a pair the model saw less of. Blend weights maximise 0.5·CV + 0.5·LODO instead of CV alone.",
      "Log CV and LODO side by side for every candidate. A model that gains CV but loses LODO is memorising group-specific quirks.", "S5"),
    L("retrieval", "lesson", "verified", "Keep the trained neural ranker the major component",
      "The blend search only considers weights where the neural ranker counts for at least as much as the linear and GBDT members combined. That keeps the solution aligned with §6.3 (a genuinely trained ranker) even if the classical members score similarly.",
      "Report each member's standalone CV. If a simple heuristic or the GBDT alone matches the blend, a reviewer may see the neural model as decoration.", "S5, S2 §6.3"),
    L("general", "lesson", "verified", "Size vocabularies and encoders from training data only",
      "The bucket vocabulary is computed from train.csv. Unseen ids at inference are clipped and carry no signal, so inference never depends on the shape of the test set.",
      "Grep for any statistic (max, unique, mean, vocab) computed on test data before predicting.", "S5, S2 §5.2.5"),
    L("general", "lesson", "verified", "Pin BLAS/OpenMP threads before importing numpy",
      "Thread-count environment variables are set to 1 before numpy and torch load, so float reductions don't vary with the number of cores the machine exposes.",
      "Run the script twice on two machines (or with different core counts) and diff the CSV hashes.", "S5"),
    L("general", "lesson", "verified", "Validate the submission before writing it",
      "Before writing, the script checks row count, duplicate ids, the id set against test.csv, finite values and a non-constant prediction, and raises on failure. A malformed output ranks below zero (§4.6).",
      "Feed the validator a deliberately broken frame in a quick local test. It should refuse to write.", "S5, S2 §4.6"),
    L("general", "lesson", "verified", "Log trivial baselines next to the model",
      "Random ordering and a one-line similarity heuristic are scored with the same metric. This proves the model adds value and gives the strip-the-model test (§5.3) a concrete number.",
      "The gap between the heuristic and the model should be clearly larger than fold-to-fold noise.", "S5, S2 §5.3"),
    L("retrieval", "lesson", "verified", "Ensemble different inductive biases, then seed-bag",
      "A small attention-based listwise ranker, a low-capacity softmax regression and shallow gradient-boosted trees are blended. The neural member is bagged over 8 seeds in the final fit.",
      "Check prediction correlation between members on OOF data. Members that are nearly identical add runtime but no diversity.", "S5"),
    L("retrieval", "lesson", "verify", "Capacity was chosen offline",
      "A comment says the small hyperparameter grid was 'established offline'. §1.1 prefers searching inside the script. Widening the in-script search (seeded, fixed trial count) is safer for review.",
      "Move the offline comparison into the script with a fixed trial budget and confirm the runtime still fits.", "S5, S2 §1.1"),

    // ---- General strategies ----
    L("general", "strategy", "hypothesis", "Make the validation split mimic how the test set was built",
      "Read the challenge text for hints about the hidden split: new users, later dates, unseen domains. Rebuild the same split inside the training data (GroupKFold, time-based or leave-one-group-out). A CV that matches the split construction tracks the private LB far better than a random split.",
      "Every experiment logs both the matched split and a random split. If they rank models differently, trust the matched one.", ""),
    L("general", "strategy", "hypothesis", "Generate out-of-fold predictions for every model",
      "OOF predictions on the same folds let you compare models fairly, tune blends and thresholds without touching test data, and run error analysis on every training row.",
      "Every model in the script exposes oof[] and test_pred[] produced with identical fold indices.", ""),
    L("general", "strategy", "hypothesis", "Change one thing per experiment",
      "Hold folds, seeds and the rest of the pipeline fixed. Report the delta against the parent run together with fold-level noise. Several changes at once make wins impossible to attribute.",
      "The tracker marks a gain smaller than about 2× the paired fold standard error as within noise.", ""),
    L("general", "pitfall", "rule", "Don't chase the public LB",
      "The public LB is a small, noisy slice and the private LB decides prizes. Choosing the final submission by public score alone is a known way to drop at the reveal.",
      "Pick finals by matched-split CV. Use the public LB only to catch pipeline bugs (a large unexplained gap).", "S2 §1.2"),
    L("general", "pitfall", "rule", "Preprocessing fitted on train+test",
      "Fitting scalers, vocabularies, PCA or target encoders on concatenated train and test lets test statistics shape the model. That breaks the one-sample inference rule.",
      "Every .fit / .fit_transform call receives training-fold data only.", "S2 §5.2.5"),

    // ---- Tabular ----
    L("tabular", "strategy", "hypothesis", "Gradient-boosted trees as the first strong baseline",
      "LightGBM, XGBoost and CatBoost with seeded, early-stopped training on OOF folds are usually the strongest single tabular baseline. Tune learning rate, depth or leaves, regularisation and subsampling inside the script with Optuna (seeded sampler, fixed n_trials).",
      "Compare against logistic or linear regression. If boosted trees barely win, the signal is mostly linear or the features are weak.", ""),
    L("tabular", "strategy", "hypothesis", "Fit target encoding inside the folds",
      "Compute category statistics on the training fold only, with smoothing. Statistics computed on the full train set leak the label into validation.",
      "The OOF score falls noticeably if you switch to full-data encoding. A rise means the leak was inflating it.", ""),
    L("tabular", "strategy", "hypothesis", "Handle imbalance with weights, not synthesis",
      "Use class weights, focal-style losses or threshold tuning on OOF. Oversamplers that create synthetic rows (SMOTE) conflict with the no-synthetic-data rule.",
      "Report per-class recall on OOF before and after.", "S2 §5.2.6"),
    L("tabular", "pitfall", "hypothesis", "Using the id or row order as a feature",
      "If the prepare script sorted the data, ids and row order can correlate with the label in train but not in the hidden split.",
      "Train on the id alone. Any score above chance is a red flag; drop it.", ""),

    // ---- NLP ----
    L("nlp", "strategy", "hypothesis", "Fine-tune a pretrained transformer as the core model",
      "A DeBERTa, RoBERTa or multilingual encoder fine-tuned on the provided data, with fixed epochs, a linear-warmup schedule and AMP. Log the measured epoch time and set the epoch count so the full run fits under about 75 min on an A10G.",
      "Show the gain over a TF-IDF + linear baseline on the same folds. That baseline is for reference only, because TF-IDF is a grey area (§5.3.2).", "S2 §4.4"),
    L("nlp", "pitfall", "rule", "Exploiting synthetic-data templates with regex",
      "Synthetic NLP datasets often carry generator artefacts. Noticing them is fine. Hardcoding a regex or template around them gets rejected; the model has to learn them.",
      "Strip-the-model test: remove the model. If the regex alone still scores, you are over-relying on it.", "S2 §5.3.1, §6.1"),
    L("nlp", "strategy", "hypothesis", "Choose max length from the training length distribution",
      "Choose the token budget from train-set percentiles (for example p95). Truncate head+tail for long documents. Never size it from test lengths.",
      "Compare OOF at two max lengths and record the runtime cost of each.", ""),

    // ---- Retrieval ----
    L("retrieval", "strategy", "rule", "Frozen retriever + trained reranker",
      "An off-the-shelf embedding model for candidate generation is fine. The reranker on top must be genuinely trained, ideally a fine-tuned cross-encoder. Fine-tuning the retriever too is recommended.",
      "Measure retriever recall@K first. The reranker can never recover a positive that isn't in the candidate set.", "S2 §6.3"),
    L("retrieval", "strategy", "hypothesis", "Mine hard negatives from the training retriever",
      "Train the reranker on negatives the retriever ranks highly, not random ones. Mine them from training queries only, inside the folds.",
      "Compare MRR@10 on OOF for random negatives against mined ones at an equal number of training steps.", ""),
    L("retrieval", "strategy", "hypothesis", "Use a listwise or pairwise loss for list metrics",
      "For MRR, NDCG or MAP, a softmax listwise loss or pairwise logistic loss over each query's candidates usually aligns better than pointwise BCE.",
      "Same folds, same features: compare the loss functions on the challenge metric itself.", ""),
    L("retrieval", "pitfall", "rule", "Hand features + an off-the-shelf ranker",
      "Feature engineering followed by an off-the-shelf LambdaMART alone is cited as insufficient for retrieval challenges.",
      "Make sure a deep model you trained produces most of the ranking signal, and show it in the logs.", "S2 §6.3"),

    // ---- CV ----
    L("cv", "strategy", "hypothesis", "Fine-tune a pretrained CNN or ViT with strong augmentation",
      "Fine-tune a timm or torchvision backbone with standard augmentations (flips, crops, colour jitter), a cosine schedule, AMP and a fixed epoch count. Average a few TTA views at inference, one image at a time.",
      "Report OOF with and without TTA. TTA is allowed because it touches one sample at a time.", "S2 §5.1.3"),
    L("cv", "pitfall", "rule", "Tabular models on raw pixels",
      "Flattening images into a GBDT or random forest is an immediate rejection.",
      "Any tree model's input must not be a pixel array.", "S2 §6.2"),
    L("cv", "pitfall", "hypothesis", "Near-duplicate images across folds",
      "Augmented copies or burst frames of one scene in both the train and validation folds inflate CV.",
      "Group folds by a perceptual hash (ImageHash is in the allowed list) or by any source or patient id.", "S4"),

    // ---- Seq generation ----
    L("seqgen", "strategy", "hypothesis", "Fine-tune a seq2seq model and tune decoding on validation",
      "Fine-tune T5, mT5 or BART-style models. Choose beam size, length penalty and max length on a validation fold from training data, and keep them fixed in the script.",
      "Report the metric (BLEU, chrF or ROUGE) for greedy versus beam decoding on OOF.", ""),
    L("seqgen", "pitfall", "rule", "\"Decoding tricks are fine\" lines",
      "Lines like this in a challenge description are treated as boilerplate. Decoding settings are fine, but the solution still has to train.",
      "Confirm with a reviewer if a description seems to permit rule-based output.", "S2 §5.4"),

    // ---- Recsys ----
    L("recsys", "strategy", "hypothesis", "Split by time or by user, matching the test",
      "Recommendation tests usually hold out later interactions or unseen users. Reproduce that split in training. A random split leaks future behaviour.",
      "Compare the metric under random and temporal splits. A big gap means the random split leaks.", ""),
    L("recsys", "strategy", "hypothesis", "Two-stage: candidate generation, then a trained ranker",
      "Recall-oriented candidates (co-visitation, embeddings, popularity), then a ranker trained on the candidates with hard negatives.",
      "Track candidate recall separately from final ranking quality.", ""),

    // ---- Structured ----
    L("structured", "strategy", "hypothesis", "Model the output structure explicitly",
      "For token tagging, spans or graphs, use a token-classification head with BIO tags, a CRF layer or a constrained decoder rather than independent per-token predictions.",
      "Measure invalid-structure rate on OOF (broken spans, inconsistent tags).", ""),

    // ---- Fine-tuning challenges ----
    L("finetune", "pitfall", "rule", "Embeddings + XGBoost is not fine-tuning",
      "In a Fine-tuning category challenge, extracting embeddings and fitting a tabular model is grey and likely rejected. Bio/chem Fine-tuning challenges require fine-tuning a domain model.",
      "Make sure the backbone's weights receive gradient updates (requires_grad on the encoder layers).", "S2 §6.4"),
    L("finetune", "strategy", "hypothesis", "Parameter-efficient fine-tuning when VRAM is tight",
      "LoRA or adapter fine-tuning keeps a larger backbone inside A10G memory and the runtime limit. Check that the library you need (for example peft) is in the image first.",
      "Compare full fine-tuning of a small model with LoRA on a larger one at equal runtime.", "S2 §4.5"),

    // ---- From scratch ----
    L("scratch", "pitfall", "rule", "Anything pretrained",
      "No pretrained weights, embeddings or distillation teachers. Standard tokenizers are fine by default unless the challenge says otherwise.",
      "Search the code for from_pretrained, timm.create_model(pretrained=True) and weights= arguments.", "S2 §6.5"),
    L("scratch", "strategy", "hypothesis", "Budget architecture to the 1.5 h limit",
      "Measure step time on a small run, then fix the steps or epochs so training plus inference fits with a margin. Small, well-regularised models trained to convergence usually beat large undertrained ones.",
      "Plot validation loss against steps from the log. If it is still falling at the end, the model is undertrained for its size.", "S2 §4.4"),

    // ---- Bio / chem ----
    L("biochem", "strategy", "rule", "Check the challenge category first",
      "An NLP-categorized bio/chem challenge allows tabular models on frozen embeddings and structure-aware features. A Fine-tuning categorized one requires fine-tuning a domain model (for example ChemBERTa).",
      "Write the category and its consequences at the top of the script's comments.", "S2 §6.4, §6.6"),
  ];
})();
