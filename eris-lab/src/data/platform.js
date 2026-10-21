/* Platform facts and rules extracted from the documents in the user's "Claude"
 * Drive folder. Everything here is paraphrased and cites its source so each
 * claim can be traced. Provenance tags used across the app:
 *   rule        - stated by a platform document (cite = where)
 *   verified    - observed directly in the user's own files
 *   hypothesis  - general practice; not validated on the user's challenges
 *   verify      - ambiguous or conflicting; confirm before relying on it
 */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.SOURCES = [
    {
      id: "S1",
      file: "shipd_docs.docx",
      title: "Project Eris / Shipd documentation",
      role: "Creator-side docs: datasets, challenge files, grading, rubrics, solution requirements, credits, prize pool.",
      used: "Solution requirements, credit refill, prize split, rubric categories, Kaggle-image core packages.",
    },
    {
      id: "S2",
      file: "Project Eris - Solver Guidebook (1).pdf",
      title: "Project Eris Solver Guidebook",
      role: "The solver rulebook. It says challenge-specific instructions override it.",
      used: "The four submission checks, determinism rules, allowed and banned techniques, domain rules, review process.",
    },
    {
      id: "S3",
      file: "solution_sub_instructions.txt",
      title: "Solution submission instructions",
      role: "Current runtime contract: python3 solution.py <public_dir> <submission_out>.",
      used: "Command-line contract, old vs new path handling, local test command.",
    },
    {
      id: "S4",
      file: "libs_allowed.txt",
      title: "Allowed libraries list",
      role: "About 150 package names from the Kaggle image. Core packages such as numpy, pandas, lightgbm and xgboost are not listed.",
      used: "Import checks in the auditor and runtime checker.",
    },
    {
      id: "S5",
      file: "high_score.py",
      title: "Previous solution: cross-lingual legal-provision ranking",
      role: "Your one previous solution. No scores, logs or audit notes came with it.",
      used: "Lessons in the strategy library and the auditor case study. Code patterns are verified; their effect on score is not.",
    },
  ];

  E.MISSING_INPUTS = [
    "Audit reports",
    "Benchmark / leaderboard scores",
    "Experiment results or training logs",
    "Challenge problem descriptions",
  ];

  E.KEY_FACTS = [
    { label: "Runtime command", value: "python3 solution.py <public_dir> <submission_out>", cite: "S3" },
    { label: "Expected max runtime", value: "1.5 h (+ up to 30 min grace, usually only 5–10 min)", cite: "S2 §4.4" },
    { label: "Hardware", value: "NVIDIA A10G GPU", cite: "S2 §4.5" },
    { label: "Credits per challenge", value: "6, +1 every 4 h", cite: "S2 §4.6, S1" },
    { label: "Global credits", value: "~15–25 / day, each returns 24 h after use", cite: "S2 §4.6" },
    { label: "Final ranking", value: "Private LB, revealed at close", cite: "S2 §1.2" },
    { label: "Payout condition", value: "Beat the AI baseline", cite: "S1" },
    { label: "Closing trigger", value: "10 distinct graded solvers, then a 24 h countdown", cite: "S1" },
  ];

  E.CHECKS = [
    { n: 1, name: "CSV grading check", what: "Your uploaded submission.csv is run through grade.py. Wrong columns, missing ids or bad values fail here.", credit: "Refunded, 5-min cooldown" },
    { n: 2, name: "Prompt compliance check", what: "An LLM reads your script against the challenge overview and its rules.", credit: "Refunded, 5-min cooldown" },
    { n: 3, name: "Determinism check", what: "An LLM reviews the code (no rerun) for seeds, clock-dependent and machine-dependent logic.", credit: "Refunded, 5-min cooldown" },
    { n: 4, name: "Script run", what: "Full end-to-end run on the test environment. Any error (OOM, ValueError, missing package) fails it.", credit: "Credit consumed. A malformed output scores as an error and ranks below a 0." },
  ];

  E.CONFLICTS = [
    {
      topic: "Solution file and paths",
      a: "S1: notebook solution.ipynb that reads ./dataset/public/ and writes ./working/submission.csv.",
      b: "S3: one Python script or notebook. Paths come from sys.argv[1] and sys.argv[2], and the hardcoded paths are marked 'Old'.",
      resolution: "Use the argv contract from S3. It is the newer instruction. Test locally with python3 solution.py ./dataset/public ./working/submission.csv.",
    },
    {
      topic: "Runtime budget",
      a: "S1: completes in approximately 1 hour.",
      b: "S2 §4.4: 1.5 h expected max, grace of up to 30 min (usually 5–10). A lower limit in a challenge's own rules is binding (§5.4).",
      resolution: "Plan for 75 min or less on an A10G, leaving headroom for inference and writing the file. If the challenge states a lower limit, use it.",
    },
    {
      topic: "Library list completeness",
      a: "S4 lists about 150 packages but omits numpy, pandas, lightgbm, xgboost, timm and sentence-transformers.",
      b: "S1 says the Kaggle image includes pandas, numpy, scikit-learn, xgboost, lightgbm, TensorFlow and PyTorch. S2 §4.2 mentions timm as a weight source.",
      resolution: "Treat S1's core packages plus S4 as allowed. Verify anything else (timm, sentence-transformers, opencv, peft) against the actual Kaggle image before you rely on it.",
    },
    {
      topic: "\"No LLM outputs\"",
      a: "S1 (challenge-creation section): you may not use any LLM outputs as part of your submission.",
      b: "S2 (solver rules) does not prohibit AI-assisted coding. It bans external data, synthetic data, self-hosted fine-tuned weights and hardcoded insights.",
      resolution: "Never call LLM APIs from the solution, and never train on LLM-generated labels or text. If you are unsure whether AI-assisted authorship is acceptable for a challenge, ask a reviewer.",
    },
    {
      topic: "Permissive lines in a challenge description",
      a: "A challenge may say 'no internet access', 'decoding tricks are fine' or 'fully rule-based solutions are okay'.",
      b: "S2 §5.4: these are leftover boilerplate. Ignore them; real ML is still required.",
      resolution: "Follow genuine restrictions (model limits, lower runtimes). Ignore the boilerplate permissions. Ask a reviewer when unsure.",
    },
  ];

  /* severity: how a violation is treated. blocker = rejection or failed check;
   * high = likely rejection; grey = reviewer discretion; info = context. */
  E.RULES = [
    // Core ML requirement
    { id: "ML-1", group: "What counts as a solution", sev: "blocker", cite: "S2 §1.1, §5.2.1", title: "Real training must happen inside the script", text: "Inference-only solutions and purely rule-based or algorithmic solutions are rejected, however well they score." },
    { id: "ML-2", group: "What counts as a solution", sev: "high", cite: "S2 §1.1, §6.1", title: "Let the model learn the insight", text: "If you find a pattern in the data, train a model to discover and use it. Do not hardcode values, templates or regexes around it." },
    { id: "ML-3", group: "What counts as a solution", sev: "grey", cite: "S2 §1.1", title: "Prefer in-script HPO to hardcoded offline parameters", text: "Searching hyperparameters inside the script (seeded, fixed trial count) teaches a reusable procedure. Pasting parameters found offline is weaker and reviewers may weigh it against you." },
    { id: "ML-4", group: "What counts as a solution", sev: "info", cite: "S2 §5.3", title: "Strip-the-model test", text: "Remove the ML model from your solution. If it still basically works, it is not compliant." },
    // Environment
    { id: "ENV-1", group: "Environment", sev: "blocker", cite: "S2 §4.1", title: "Kaggle Docker image libraries only", text: "No installs from inside the script and no approval route for extra packages." },
    { id: "ENV-2", group: "Environment", sev: "info", cite: "S2 §4.2", title: "Internet is fine for pretrained weights", text: "Allowed packages may download general-purpose weights from Hugging Face, timm and similar hubs." },
    { id: "ENV-3", group: "Environment", sev: "blocker", cite: "S2 §4.3", title: "No models from GitHub or other channels", text: "GitHub-hosted models usually need their own codebase installed, which the library rule forbids." },
    { id: "ENV-4", group: "Environment", sev: "high", cite: "S2 §4.4, §5.4", title: "1.5 h expected max runtime", text: "Up to 30 min grace for genuinely good solutions, usually only 5–10 min in practice. A shorter limit in a challenge's own rules is binding." },
    { id: "ENV-5", group: "Environment", sev: "blocker", cite: "S2 §4.4, §3.3", title: "Fix training length in advance", text: "Choose epochs or steps you have measured to fit. Never stop training on elapsed time." },
    { id: "ENV-6", group: "Environment", sev: "info", cite: "S2 §4.5", title: "Target hardware is an NVIDIA A10G", text: "No general model-size cap. Challenge-specific size limits override this." },
    { id: "ENV-7", group: "Environment", sev: "blocker", cite: "S2 §4.7", title: "One independent end-to-end script", text: "Preprocessing, features, training and inference run fresh every time. Never load cached outputs such as embeddings from a previous run. Several models in one script are fine." },
    { id: "ENV-8", group: "Environment", sev: "blocker", cite: "S3", title: "Command-line contract", text: "The platform runs python3 solution.py <public_dir> <submission_out>. Read sys.argv[1] and sys.argv[2], and create the output's parent directory." },
    // Determinism
    { id: "DET-1", group: "Determinism check", sev: "blocker", cite: "S2 §3.3, §3.4", title: "Seed every source of randomness", text: "Seed Python random, NumPy, torch (CPU and CUDA), and the random_state or seed argument of every model, split and sampler. Seed DataLoader workers and use a seeded generator." },
    { id: "DET-2", group: "Determinism check", sev: "blocker", cite: "S2 §3.3", title: "No clock-conditioned decisions", text: "No time budgets, no stopping after N minutes, and no epoch counts derived from how fast an epoch ran." },
    { id: "DET-3", group: "Determinism check", sev: "blocker", cite: "S2 §3.3", title: "No machine-conditioned code", text: "No os.cpu_count(), no torch.cuda.device_count(), no 'cuda if available else cpu', and no batch size chosen from free memory. Write these values as constants." },
    { id: "DET-4", group: "Determinism check", sev: "blocker", cite: "S2 §3.3", title: "No try/except fallbacks", text: "Never quietly switch to a different method or model when something fails." },
    { id: "DET-5", group: "Determinism check", sev: "blocker", cite: "S2 §3.3", title: "One fixed path through the script", text: "No environment-variable switches, debug flags, or if-branches that choose between methods, even when only one branch ever runs." },
    { id: "DET-6", group: "Determinism check", sev: "info", cite: "S2 §3.3", title: "Explicitly allowed", text: "In-script HPO with a fixed seed and fixed trial count, early stopping on validation score, AMP/bf16, and a seeded multi-worker DataLoader with a constant worker count." },
    // Test-set use
    { id: "TEST-1", group: "Using the test set", sev: "blocker", cite: "S2 §5.2.5", title: "Test data is for one-sample or one-batch inference only", text: "Not allowed: pseudo-labelling, reweighting training data with test data, test-time adaptation, and calibrating outputs to the test set's overall distribution. This applies even though you never see test labels." },
    { id: "TEST-2", group: "Using the test set", sev: "info", cite: "S2 §5.1.3", title: "Production-realistic inference is fine", text: "TTA on one sample at a time and batch-level batch-norm statistics behave the same in production and are allowed." },
    // Data
    { id: "DATA-1", group: "Data and weights", sev: "blocker", cite: "S2 §5.2.3", title: "No external datasets", text: "Use only the challenge data plus general-purpose pretrained backbone weights." },
    { id: "DATA-2", group: "Data and weights", sev: "blocker", cite: "S2 §5.2.6", title: "No self-generated synthetic training data", text: "Synthetic data is a legitimate technique elsewhere, but not here. The challenge measures what you can earn from the data you were given." },
    { id: "DATA-3", group: "Data and weights", sev: "blocker", cite: "S2 §5.2.2", title: "No weights you fine-tuned elsewhere", text: "Do not fine-tune offline, host the weights and load them back. Only general-purpose backbones may be loaded." },
    // Conduct
    { id: "CON-1", group: "Conduct", sev: "blocker", cite: "S2 §5.2.4, §5.7", title: "No private sharing; one account; no teams", text: "Sharing your approach publicly in the #general Discord channel is allowed and encouraged." },
    { id: "CON-2", group: "Conduct", sev: "info", cite: "S2 §2", title: "Ignore suspicious public-LB scores", text: "Solutions are reviewed after close. An unbelievable score may be a violation nobody has caught yet. Don't drift into grey areas to chase it." },
    // Grey areas
    { id: "GREY-1", group: "Grey areas", sev: "grey", cite: "S2 §5.3.1", title: "Regex: keep it minimal", text: "Fine for cleaning and for perfectly deterministic extraction such as numbers. Leaning on regex to exploit how the data was generated gets rejected." },
    { id: "GREY-2", group: "Grey areas", sev: "grey", cite: "S2 §5.3.2", title: "TF-IDF, n-grams, Markov chains", text: "Reviewer's discretion, possibly differently per participant. It depends on your solution's quality and novelty and how much it relies on the technique." },
    { id: "GREY-3", group: "Grey areas", sev: "grey", cite: "S2 §5.3", title: "Explicit permission doesn't remove the grey area", text: "Even when a challenge allows a technique, reviewers can reject over-reliance on the dataset's distribution." },
    // Domain
    { id: "CV-1", group: "Computer vision", sev: "blocker", cite: "S2 §6.2", title: "No tabular models on raw pixels", text: "Immediate rejection. Train or fine-tune a real CV model (CNN, ViT)." },
    { id: "CV-2", group: "Computer vision", sev: "grey", cite: "S2 §6.2", title: "Hand-made image features", text: "Fine as extra inputs to a deep model alongside the image. Fragile and grey when they only feed a tabular model." },
    { id: "CV-3", group: "Computer vision", sev: "grey", cite: "S2 §6.2", title: "Tabular model on pretrained embeddings", text: "Grey area. It may be rejected if it is not robust." },
    { id: "RAG-1", group: "Retrieval and ranking", sev: "high", cite: "S2 §6.3", title: "Train a genuine ranking model", text: "Usually a deep model. Hand-engineered features plus an off-the-shelf ranking algorithm is not enough. This is the second-highest-violation domain." },
    { id: "RAG-2", group: "Retrieval and ranking", sev: "high", cite: "S2 §6.3", title: "A frozen retriever is fine; a fully frozen pipeline is not", text: "A major portion of the solution must be trained. Fine-tuning the retriever is recommended." },
    { id: "FT-1", group: "Fine-tuning", sev: "high", cite: "S2 §6.4", title: "Actually fine-tune the backbone", text: "Pulling embeddings and fitting a tabular model on them is grey and may be rejected." },
    { id: "FT-2", group: "Fine-tuning", sev: "high", cite: "S2 §6.4, §6.6", title: "Bio/chem depends on the challenge category", text: "Categorized as NLP: tabular models on frozen embeddings are fine. Categorized as Fine-tuning: you must fine-tune a domain model, and XGBoost or LightGBM alone is not tolerated. Ensembling is grey." },
    { id: "SCR-1", group: "From scratch", sev: "blocker", cite: "S2 §6.5", title: "Nothing pretrained", text: "No pretrained models for embeddings, distillation or anything else. Tokenizers and BPE are fine unless the challenge says otherwise, but reviewers may reject an unfair edge." },
    // Challenge-specific
    { id: "CH-1", group: "Challenge-specific rules", sev: "high", cite: "S2 §5.4", title: "Real challenge restrictions override the guidebook", text: "Model restrictions and shorter runtimes in the description are binding. Boilerplate permissions ('no internet', 'decoding tricks fine', 'rule-based OK') are not." },
    { id: "CH-2", group: "Challenge-specific rules", sev: "info", cite: "S2 §5.4", title: "When unsure, ask a reviewer first", text: "Wait for the answer before you submit. Appeals happen after close and never refund credits." },
    // Submission & review
    { id: "SUB-1", group: "Submission and review", sev: "info", cite: "S2 §3.1–3.2", title: "Four checks; only the run costs a credit", text: "Failing the CSV, compliance or determinism check refunds the credit after a 5-min cooldown. Fix the script before resubmitting." },
    { id: "SUB-2", group: "Submission and review", sev: "blocker", cite: "S2 §4.6", title: "A malformed output ranks below zero", text: "Missing rows or a broken CSV from the scripted run shows as an error and places you last. The credit is still consumed." },
    { id: "SUB-3", group: "Submission and review", sev: "info", cite: "S2 §2", title: "Your uploaded CSV is a self-check", text: "It costs no credits and doesn't count for the leaderboard. Only the scripted run's output counts." },
    { id: "SUB-4", group: "Submission and review", sev: "info", cite: "S2 §5.5–5.6", title: "Relative review after close", text: "Neighbouring ranks are compared pairwise on appropriateness. A rank with a much worse appropriateness rating than the one below it is removed. Credits are never refunded." },
    { id: "SUB-5", group: "Submission and review", sev: "info", cite: "S1", title: "Show progression", text: "3–5 complete submissions, from a solid baseline to your best model. Comment your reasoning at each step." },
    { id: "LB-1", group: "Submission and review", sev: "info", cite: "S2 §1.2", title: "The private LB decides prizes", text: "Treat the public LB as a soft signal. Over-optimizing for it can drop you sharply at the reveal." },
  ];

  /* Packages: S4 verbatim names plus the core packages S1 names explicitly. */
  E.ALLOWED_PACKAGES = [
    "jax","jax-cuda12-pjrt","jax-cuda12-plugin","jaxlib","keras","keras-hub","keras-nlp","tensorflow","tensorflow-datasets","tensorflow-hub","tensorflow-metadata","tensorflow-probability","tensorflow-text","tensorflow_decision_forests","tf_keras","torch","torchao","torchaudio","torchdata","torchsummary","torchtune","torchvision","Boruta","Cartopy","ImageHash","Janome","PyArabic","PyUpSet","Pympler","Rtree","shapely","SimpleITK","TPOT","Wand","bayesian-optimization","boto3","catboost","category-encoders","cesium","comm","cytoolz","datasets","deap","dipy","docker","easyocr","emoji","fasttext","featuretools","fiona","fury","fuzzywuzzy","geojson","gensim","google-adk","google-cloud-aiplatform","google-cloud-videointelligence","google-cloud-vision","google-genai","gpxpy","h2o","haversine","hep-ml","igraph","ipympl","ipywidgets","isoweek","jedi","jiter","jupyter-lsp","jupyter_server","jupyter_server_proxy","jupyterlab","jupyterlab-lsp","kaggle","kaggle-environments","keras-cv","keras-tuner","kornia","langid","libpysal","lime","line_profiler","mamba","matplotlib","mlcrate","mne","mpld3","nbconvert","nbdev","nilearn","olefile","onnx","openslide-bin","openslide-python","optuna","pandas-profiling","pandasql","papermill","path","path.py","pdf2image","plotly-express","preprocessing","pudb","pyLDAvis","pycryptodome","pydicom","pyemd","pyexcel-ods","pymongo","pypdf","pytesseract","python-lsp-server","pytorch-ignite","pytorch-lightning","qgrid","qtconsole","ray","rgf-python","s3fs","scikit-learn","scikit-learn-intelex","scikit-multilearn","scikit-optimize","scikit-plot","scikit-surprise","segment-anything","squarify","tensorflow-cloud","tensorflow-io","torchinfo","torchmetrics","transformers","vtk","wavio","xvfbwrapper","ydata-profiling",
  ];
  E.CORE_PACKAGES = ["pandas", "numpy", "scikit-learn", "xgboost", "lightgbm", "tensorflow", "torch"];

  /* import name -> distribution name, for the import checker */
  E.IMPORT_ALIASES = {
    sklearn: "scikit-learn", skopt: "scikit-optimize", PIL: "pillow", cv2: "opencv-python",
    yaml: "pyyaml", bs4: "beautifulsoup4", lightgbm: "lightgbm", xgboost: "xgboost",
    category_encoders: "category-encoders", tensorflow_decision_forests: "tensorflow_decision_forests",
    pytorch_lightning: "pytorch-lightning", lightning: "pytorch-lightning", ignite: "pytorch-ignite",
    segment_anything: "segment-anything", surprise: "scikit-surprise", skmultilearn: "scikit-multilearn",
    scikitplot: "scikit-plot", keras_nlp: "keras-nlp", keras_cv: "keras-cv", keras_hub: "keras-hub",
    keras_tuner: "keras-tuner", google: "google-genai", sentence_transformers: "sentence-transformers",
    timm: "timm", peft: "peft", accelerate: "accelerate", scipy: "scipy", statsmodels: "statsmodels",
    tqdm: "tqdm", seaborn: "seaborn", nltk: "nltk", spacy: "spacy", tokenizers: "tokenizers",
    safetensors: "safetensors", albumentations: "albumentations", imblearn: "imbalanced-learn",
    huggingface_hub: "huggingface_hub", openai: "openai", anthropic: "anthropic",
    faiss: "faiss", rank_bm25: "rank_bm25", catboost: "catboost", optuna: "optuna",
    torchvision: "torchvision", torchaudio: "torchaudio", transformers: "transformers", datasets: "datasets",
  };
  /* Packages that are very likely in the Kaggle image but are not on either
   * supplied list. The checker labels these "verify" instead of "unknown". */
  E.LIKELY_IN_IMAGE = ["scipy", "pillow", "opencv-python", "pyyaml", "seaborn", "statsmodels", "tqdm", "nltk", "spacy", "tokenizers", "safetensors", "albumentations", "timm", "sentence-transformers", "peft", "accelerate", "huggingface_hub", "beautifulsoup4", "imbalanced-learn", "faiss", "rank_bm25"];
  E.STDLIB = "abc argparse array ast asyncio base64 bisect builtins bz2 calendar collections concurrent contextlib copy csv ctypes dataclasses datetime decimal difflib enum errno fnmatch fractions functools gc getpass glob gzip hashlib heapq hmac html http importlib inspect io ipaddress itertools json keyword locale logging lzma math mimetypes multiprocessing numbers operator os pathlib pickle platform pprint queue random re resource sched secrets select shlex shutil signal socket sqlite3 ssl stat statistics string struct subprocess sys tarfile tempfile textwrap threading time timeit tokenize traceback types typing unicodedata unittest urllib uuid warnings weakref xml zipfile zlib __future__".split(" ");
})();
