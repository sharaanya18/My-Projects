/* Beginner learning track: lessons with quizzes, the "Allowed or banned?"
 * rules game, a glossary, and the XP / level / badge definitions. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.LESSONS = [
    {
      id: "metric", title: "Metrics: how your score is computed", mins: 4, demo: "threshold", tool: "leaderboard",
      body: [
        "Every challenge names one metric, and the grader computes only that. Before anything else, find out which metric it is and whether higher or lower is better.",
        "Some metrics read probabilities (AUC, log loss). Others need hard labels (accuracy, F1). For label metrics, the threshold that turns a probability into a label is a lever you can tune, but only on your own validation predictions, never on the test set.",
        "Move the slider below. The model's scores never change, yet accuracy and F1 do. That's why the threshold belongs to the solution, not the model.",
      ],
      quiz: [
        { q: "The metric is AUC. Does changing the 0.5 threshold change your score?", options: ["Yes", "No, AUC only uses the order of the predictions", "Only for imbalanced data"], a: 1, why: "AUC measures ranking quality across all thresholds, so any monotonic change to the scores leaves it unchanged." },
        { q: "Where should you tune an F1 threshold?", options: ["On out-of-fold training predictions", "On the test-set score distribution", "On the public leaderboard"], a: 0, why: "Tuning on test statistics is banned (§5.2.5), and LB probing wastes credits and overfits. OOF predictions are training data." },
        { q: "RMSE is the metric. Is a smaller number better?", options: ["Yes", "No"], a: 0, why: "Error metrics (RMSE, MAE, log loss) are lower-is-better. Record the direction in the tracker settings." },
      ],
    },
    {
      id: "cv", title: "Validation: your private leaderboard", mins: 5, demo: "split", tool: "generalization",
      body: [
        "You can't see the private leaderboard, so you build your own: split the training data into folds, train on some, score on the rest, and repeat.",
        "The split has to copy how the hidden test was built. If the test has unseen users or plants, the same user must never sit in both your training and validation folds. Otherwise the model memorises the user, and your score looks better than it will be on the hidden test.",
        "Switch between Random and Grouped below and watch the leak counter. Grouped folds keep each colour in one fold.",
      ],
      quiz: [
        { q: "The test set contains customers who never appear in train. Which split mimics that?", options: ["Random KFold", "GroupKFold on customer id", "A split sorted by row number"], a: 1, why: "GroupKFold keeps every customer in a single fold, so validation customers are unseen, just as in the test." },
        { q: "Your random-split CV is 0.91 but the grouped-split CV is 0.84. Which number should you trust for the private LB?", options: ["0.91", "0.84", "Their average"], a: 1, why: "The grouped split matches the test construction. The random split leaks group identity and inflates the score." },
        { q: "Why fix the fold indices once and reuse them for every model?", options: ["It's faster", "So comparisons between models are paired and fair", "The platform requires it"], a: 1, why: "Identical folds remove split luck from model comparisons. The tracker then compares fold-by-fold." },
      ],
    },
    {
      id: "oof", title: "Out-of-fold predictions", mins: 4, demo: "oofstep", tool: "oof",
      body: [
        "During cross-validation, every training row is predicted exactly once, by the model that did not see it. Collect those predictions and you get out-of-fold (OOF) predictions for the whole training set.",
        "OOF predictions are the most useful thing a solution produces besides the submission. You can blend models on them, tune thresholds, check calibration and run error analysis, all without touching the test set.",
        "Step through the folds below to see the OOF column fill up.",
      ],
      quiz: [
        { q: "How many times is each training row predicted in 5-fold OOF?", options: ["Once", "Five times", "Four times"], a: 0, why: "Each row lands in exactly one validation fold, so it gets exactly one out-of-fold prediction." },
        { q: "You want blend weights for three models. What data do you fit them on?", options: ["Test predictions", "OOF predictions from identical folds", "Training-fit predictions"], a: 1, why: "OOF predictions are honest (unseen-by-model) and come from training data only." },
        { q: "Which of these is NOT a good use of OOF predictions?", options: ["Error analysis by subgroup", "Pseudo-labelling the test set", "Choosing an F1 threshold"], a: 1, why: "Pseudo-labelling is banned on Eris (§5.2.5), whatever predictions you use for it." },
      ],
    },
    {
      id: "fit", title: "Overfitting and underfitting", mins: 4, demo: "capacity", tool: "generalization",
      body: [
        "A tiny model can't capture the pattern (underfitting): train and validation scores are both poor. A huge model memorises noise (overfitting): train looks great, validation gets worse.",
        "The sweet spot is where the validation score peaks. Regularisation, more data, fewer epochs or a smaller model move you back from overfitting.",
        "Drag the capacity slider and watch the gap between the two curves.",
      ],
      quiz: [
        { q: "Train AUC 0.99, CV AUC 0.78. What's most likely?", options: ["Underfitting", "Overfitting", "A perfect model"], a: 1, why: "A large train-validation gap is the classic overfitting signal." },
        { q: "Train 0.71, CV 0.70, and both are low. What should you try?", options: ["More capacity or better features", "Stronger regularisation", "Fewer epochs"], a: 0, why: "Train ≈ CV, both low, means the model is too simple for the signal. Add capacity within the runtime budget." },
        { q: "Is early stopping on your validation score allowed on Eris?", options: ["Yes, explicitly allowed (§3.3)", "No, it's a time-based decision"], a: 0, why: "Early stopping depends on data and seed, not the clock, so it passes the determinism check." },
      ],
    },
    {
      id: "leak", title: "Leakage: scores too good to be true", mins: 4, demo: null, tool: "auditor",
      body: [
        "Leakage is when information that won't exist at prediction time sneaks into training or validation. It makes CV look great, then collapses on the hidden test.",
        "Common sources: fitting scalers or vocabularies on train+test, target encoding computed on the full train set, ids or row order that correlate with the label, and the same entity in both train and validation folds.",
        "On Eris, some leakage is also a rule violation. Any statistic computed over the test set breaks the one-sample-inference rule (§5.2.5). The auditor flags these patterns.",
      ],
      quiz: [
        { q: "You fit a StandardScaler on pd.concat([train, test]). Is that OK?", options: ["Yes, it doesn't use labels", "No: test statistics shape the model, which is banned (§5.2.5)"], a: 1, why: "The rule is about realism, not labels. Inference must work one sample at a time." },
        { q: "Target-mean encoding computed on the whole training set, then CV. What's wrong?", options: ["Nothing", "Validation rows' labels leak into their own features"], a: 1, why: "Compute encodings inside each training fold only." },
        { q: "A model trained only on the 'id' column scores well above chance. What does that tell you?", options: ["Ids are a great feature", "Row order or id correlates with the label. Drop it, it won't transfer"], a: 1, why: "The hidden split may not keep that correlation." },
      ],
    },
    {
      id: "lb", title: "Public vs private leaderboard", mins: 4, demo: "lbnoise", tool: "leaderboard",
      body: [
        "The public leaderboard scores only a slice of the test set. The private leaderboard decides prizes and is revealed at the end (§1.2).",
        "With a few thousand rows, two models a point apart can swap places on the public LB just by chance. Chasing the public LB means fitting that noise.",
        "Run the simulator. With a small public slice, the worse model wins surprisingly often.",
      ],
      quiz: [
        { q: "Your CV says model A is better, but the public LB prefers B by a tiny margin. What should you do?", options: ["Pick B", "Trust the well-built CV, and check whether the LB gap is within noise", "Submit both forever"], a: 1, why: "A matched-split CV over all training data is far less noisy than a small public slice." },
        { q: "A competitor shows an impossibly high public score. What should you do?", options: ["Try grey-area tricks to keep up", "Keep improving your own solution. It may be a violation that hasn't been caught yet."], a: 1, why: "The guidebook says exactly this (§2). Solutions are reviewed after close." },
        { q: "Is uploading your CSV for a self-check free?", options: ["Yes, it uses no credits", "No, it costs a credit"], a: 0, why: "Only the scripted run consumes credits (§2, §4.6)." },
      ],
    },
    {
      id: "rules", title: "Eris rules in a nutshell", mins: 5, demo: "checks", tool: "rules",
      body: [
        "The model has to do the learning. Real training happens inside one end-to-end script, and nothing is hardcoded or loaded from a previous run (§1.1, §4.7).",
        "The script must be deterministic: seed everything, use constants instead of machine checks, no clock-based decisions, no try/except fallbacks (§3.3).",
        "Data rules: no external data, no synthetic training data, no test-set adaptation, no weights you fine-tuned elsewhere (§5.2). Domain rules add more: no tabular models on pixels, and retrieval needs a trained ranker (§6).",
        "Every submission passes four checks. Only the final script run costs a credit.",
      ],
      quiz: [
        { q: "Which line fails the determinism check?", options: ["DEVICE = torch.device(\"cuda\")", "DEVICE = \"cuda\" if torch.cuda.is_available() else \"cpu\"", "torch.manual_seed(42)"], a: 1, why: "Choosing behaviour by inspecting the machine is named as a rejection (§3.3)." },
        { q: "Can you load embeddings you computed in an earlier submission?", options: ["Yes, it saves time", "No: every run must start from raw data (§4.7)"], a: 1, why: "There is no previous run to load from. Each script is independent." },
        { q: "Which check costs a credit if it fails?", options: ["CSV grading", "Prompt compliance", "Determinism", "The script run"], a: 3, why: "Failures in checks 1–3 are refunded after a 5-minute cooldown (§3.2)." },
      ],
    },
    {
      id: "claude", title: "Working with Claude Code", mins: 4, demo: null, tool: "prompts",
      body: [
        "Claude Code can read your files, run experiments and fix bugs. Your prompts decide whether it works from evidence or guesses.",
        "Good prompts ask Claude to inspect the actual files first, change one thing per experiment, report numbers only from commands it ran, and tag every claim VERIFIED or HYPOTHESIS. The prompt generator builds these for you, with the platform rules included.",
        "Put a CLAUDE.md with the rules in each challenge folder (see Templates), so every new session starts compliant.",
      ],
      quiz: [
        { q: "Claude reports 'CV improved to 0.83' but shows no log line. What should you do?", options: ["Accept it", "Ask it to show the command and output, and treat the claim as HYPOTHESIS until it does"], a: 1, why: "The evidence standard: only numbers you can trace to a run count." },
        { q: "Why one change per experiment?", options: ["So you know which change caused the gain", "Claude can't do more than one"], a: 0, why: "Several changes at once make wins impossible to attribute or revert cleanly." },
        { q: "Claude suggests pseudo-labelling to boost the score. What do you do?", options: ["Allow it", "Refuse: banned on Eris (§5.2.5)"], a: 1, why: "The generated prompts list it under prohibited shortcuts." },
      ],
    },
  ];

  /* answer: allowed | grey | banned */
  E.SCENARIOS = [
    { s: "You average predictions over horizontally flipped copies of each test image (TTA).", a: "allowed", why: "TTA touches one sample at a time, so it works in production.", cite: "§5.1.3" },
    { s: "You fit a TF-IDF vectorizer on train + test text before training.", a: "banned", why: "The vocabulary and IDF weights come from the test set, which is test-set adaptation.", cite: "§5.2.5" },
    { s: "You download general-purpose DeBERTa weights from Hugging Face inside the script and fine-tune them.", a: "allowed", why: "General-purpose backbones may be downloaded and fine-tuned at run time.", cite: "§4.2, §5.1" },
    { s: "You fine-tune a model on your laptop, upload it to Hugging Face, and load it in the submission.", a: "banned", why: "All training must happen inside the submission script.", cite: "§5.2.2" },
    { s: "You run Optuna with n_trials=30 and TPESampler(seed=42) inside the script.", a: "allowed", why: "In-script HPO with a fixed seed and trial count is exactly what's encouraged.", cite: "§1.1, §3.3" },
    { s: "You run Optuna with timeout=1800 so it stops in time.", a: "banned", why: "A time budget makes the result depend on machine speed, which fails the determinism check.", cite: "§3.3" },
    { s: "You predict the test set, then train again on test rows with confident predictions as labels.", a: "banned", why: "Pseudo-labelling is explicitly prohibited.", cite: "§5.2.5" },
    { s: "You oversample the minority class with SMOTE.", a: "banned", why: "SMOTE generates synthetic rows, and training on self-generated synthetic data isn't allowed. Use class weights.", cite: "§5.2.6" },
    { s: "You use class weights in the loss to handle a 4% positive rate.", a: "allowed", why: "Reweighting training rows by their own labels is ordinary training.", cite: "§5.1" },
    { s: "You write DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'.", a: "banned", why: "Machine-conditioned code is named as a rejection. Write the device as a constant.", cite: "§3.3" },
    { s: "You stop training after 50 minutes to stay inside the limit.", a: "banned", why: "Clock-based stopping fails the determinism check. Fix epochs or steps in advance.", cite: "§3.3, §4.4" },
    { s: "You use early stopping on your validation fold's score.", a: "allowed", why: "Explicitly listed as fine: it depends on data and seed only.", cite: "§3.3" },
    { s: "You train XGBoost on flattened raw image pixels.", a: "banned", why: "Tabular models on raw pixels are an immediate rejection.", cite: "§6.2" },
    { s: "You extract embeddings from a frozen pretrained CNN and train LightGBM on them (CV challenge).", a: "grey", why: "Grey area: it may be rejected if it's not robust. Fine-tuning the CNN is safer.", cite: "§6.2" },
    { s: "You add colour histograms as extra inputs to a fine-tuned CNN's head.", a: "allowed", why: "Hand features feeding a real deep model are just extra signal.", cite: "§6.2" },
    { s: "You use a regex to strip HTML tags before tokenising.", a: "allowed", why: "Regex for cleaning or deterministic extraction is fine.", cite: "§5.3.1" },
    { s: "You noticed the synthetic generator puts the label word in brackets, so you regex for it.", a: "banned", why: "Exploiting how the data was generated with hand rules gets rejected. The model must learn it.", cite: "§5.3.1, §6.1" },
    { s: "You use TF-IDF + logistic regression as a minor ensemble member next to a fine-tuned transformer.", a: "grey", why: "TF-IDF is grey, judged by the reviewer on how much you rely on it.", cite: "§5.3.2" },
    { s: "Retrieval challenge: you use a frozen sentence-transformer for candidates and train a cross-encoder reranker.", a: "allowed", why: "A frozen retriever is fine when the ranking stage is genuinely trained.", cite: "§6.3" },
    { s: "Retrieval challenge: you hand-craft overlap features and rank with an off-the-shelf LambdaMART only.", a: "banned", why: "Feature engineering plus an off-the-shelf ranker is called out as not enough. Train a ranking model.", cite: "§6.3" },
    { s: "You pip install a package that isn't in the Kaggle image.", a: "banned", why: "Only preinstalled libraries are allowed, and there's no approval route.", cite: "§4.1" },
    { s: "You load a model with torch.hub.load from a GitHub repository.", a: "banned", why: "Models from GitHub or other channels are rejected immediately.", cite: "§4.3" },
    { s: "You calibrate final probabilities so their mean matches the test-set average prediction.", a: "banned", why: "Calibrating to the test distribution is explicitly prohibited.", cite: "§5.2.5" },
    { s: "You fit temperature scaling on your out-of-fold predictions.", a: "allowed", why: "Calibration fitted on training data (OOF) is ordinary practice.", cite: "§5.2.5" },
    { s: "You blend LightGBM, CatBoost and an MLP inside one script.", a: "allowed", why: "Training and ensembling several models in one end-to-end script is fine.", cite: "§4.7" },
    { s: "You cache test embeddings from submission 2 and load them in submission 3.", a: "banned", why: "Every run starts from raw data. Caching across runs isn't possible or allowed.", cite: "§4.7" },
    { s: "You use an external public dataset of similar reviews for pretraining.", a: "banned", why: "No external datasets, only the challenge data plus backbone weights.", cite: "§5.2.3" },
    { s: "From-scratch challenge: you use a standard BPE tokenizer the description doesn't mention.", a: "grey", why: "Tokenizers are fine by default, but reviewers may reject an unfair edge. Confirm with a reviewer if unsure.", cite: "§6.5" },
    { s: "You share your notebook privately with a friend competing on the same challenge.", a: "banned", why: "Private sharing between solvers is banned. Public sharing in #general is fine.", cite: "§5.2.4" },
    { s: "You post your approach publicly in the #general Discord channel.", a: "allowed", why: "Public sharing is allowed and encouraged.", cite: "§5.1.4" },
    { s: "The description says 'no internet access', so you skip downloading pretrained weights.", a: "grey", why: "That line is usually leftover boilerplate (§5.4). Downloading weights is allowed, so skipping them is legal but may cost score. Ask a reviewer if unsure.", cite: "§5.4" },
    { s: "You seed a DataLoader with num_workers=4, worker_init_fn and a seeded generator.", a: "allowed", why: "A multi-worker DataLoader with a fixed worker count and seeding is fine.", cite: "§3.3, §3.4" },
  ];

  E.GLOSSARY = [
    ["AUC", "Area under the ROC curve. The probability that a random positive is ranked above a random negative. 0.5 = random, 1 = perfect."],
    ["CV (cross-validation)", "Splitting training data into folds, training on some and scoring on the rest, then averaging. Your stand-in for the hidden test."],
    ["Fold", "One slice of the training data used for validation in one CV round."],
    ["OOF (out-of-fold)", "Predictions for each training row from the model that didn't train on it."],
    ["GroupKFold", "A CV split that keeps all rows of one group (user, patient, query) in the same fold."],
    ["Stress split", "An extra validation split that copies the hardest plausible shift, e.g. leave one language pair out."],
    ["LODO", "Leave-one-direction-out: hold out each language direction in turn. Used in your high_score.py."],
    ["Leakage", "Information in training or validation that won't be available at prediction time. It inflates scores."],
    ["SE (standard error)", "How much a mean score would wobble if you repeated the experiment. Gains under about 2×SE are noise."],
    ["Paired comparison", "Comparing two models fold by fold on identical folds, which removes split luck."],
    ["Blend", "A weighted average of several models' predictions."],
    ["Calibration", "Whether a predicted 0.8 really means 80% of such cases are positive."],
    ["Threshold", "The cut-off that turns a probability into a class label."],
    ["Public / private LB", "Leaderboards on two slices of the test set. Only the private one decides prizes."],
    ["Determinism check", "The platform's LLM code review for seeds and clock- or machine-dependent logic."],
    ["TTA", "Test-time augmentation: averaging predictions over transformed copies of one input."],
    ["Pseudo-labelling", "Training on your own predictions for unlabeled test data. Banned on Eris."],
    ["HPO", "Hyperparameter optimisation, e.g. Optuna with a fixed number of seeded trials."],
  ];

  E.LEVELS = [
    { xp: 0, name: "Seed" }, { xp: 120, name: "Sprout" }, { xp: 300, name: "Sapling" },
    { xp: 550, name: "Blossom" }, { xp: 850, name: "Apple" }, { xp: 1200, name: "Golden Apple" },
  ];
  E.BADGES = {
    "first-lesson": { name: "First steps", desc: "Finished your first lesson" },
    "all-lessons": { name: "Scholar", desc: "Finished every lesson" },
    "rule-keeper": { name: "Rule keeper", desc: "Scored 80%+ in Allowed or banned?" },
    "perfect-game": { name: "Flawless", desc: "Perfect score in Allowed or banned?" },
    "first-run": { name: "Baseline", desc: "Logged your first real run" },
    "clean-audit": { name: "Spotless", desc: "Audited a script with no blocker or high findings" },
    "oof-scientist": { name: "OOF scientist", desc: "Analysed out-of-fold predictions" },
    "log-reader": { name: "Log reader", desc: "Filled a run from a training log" },
    "organised": { name: "Organised", desc: "Created a second challenge workspace" },
  };
})();
