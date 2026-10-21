/* Static solution auditor. Heuristic pattern checks against the platform rules.
 * It can't prove compliance. It points at lines a reviewer or the platform's
 * LLM checks are likely to flag, and explains the rule and the fix. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const SEV_ORDER = { blocker: 0, high: 1, medium: 2, low: 3, info: 4 };

  /* ---------- source preparation ---------- */

  function fromNotebook(text) {
    try {
      const nb = JSON.parse(text);
      if (!nb || !Array.isArray(nb.cells)) return null;
      const out = [];
      nb.cells.forEach((c, i) => {
        if (c.cell_type !== "code") return;
        out.push(`# ---- cell ${i + 1} ----`);
        const src = Array.isArray(c.source) ? c.source.join("") : String(c.source || "");
        out.push(...src.split("\n"));
      });
      return out.join("\n");
    } catch (_) {
      return null;
    }
  }

  /* Returns per-line {raw, code, comment, inDoc}. code has comments and string
   * contents of docstrings removed so patterns match real code only. */
  function prepare(text) {
    const lines = text.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let inDoc = null; // the quote style of an open triple-quoted string
    for (const raw of lines) {
      let code = "";
      let comment = "";
      let i = 0;
      let quote = null;
      let docLine = inDoc !== null;
      while (i < raw.length) {
        if (inDoc) {
          const end = raw.indexOf(inDoc, i);
          if (end === -1) { i = raw.length; break; }
          i = end + 3;
          inDoc = null;
          continue;
        }
        const ch = raw[i];
        if (quote) {
          code += ch;
          if (ch === "\\") { code += raw[i + 1] || ""; i += 2; continue; }
          if (ch === quote) quote = null;
          i++;
          continue;
        }
        const three = raw.substr(i, 3);
        if (three === '"""' || three === "'''") {
          // Treat any triple-quoted string as non-code (docstrings, long text).
          const end = raw.indexOf(three, i + 3);
          code += '""';
          if (end === -1) { inDoc = three; docLine = true; i = raw.length; break; }
          i = end + 3;
          continue;
        }
        if (ch === "#") { comment = raw.slice(i + 1); break; }
        if (ch === '"' || ch === "'") quote = ch;
        code += ch;
        i++;
      }
      out.push({ raw, code, comment, inDoc: docLine && !code.trim() });
    }
    return out;
  }

  /* Join a call statement that spans several lines (until brackets balance). */
  function statementAt(L, idx, maxLines = 25) {
    let depth = 0;
    let text = "";
    for (let j = idx; j < Math.min(L.length, idx + maxLines); j++) {
      const c = L[j].code;
      text += " " + c;
      for (const ch of c) {
        if (ch === "(" || ch === "[" || ch === "{") depth++;
        else if (ch === ")" || ch === "]" || ch === "}") depth--;
      }
      if (depth <= 0 && j >= idx) break;
    }
    return text;
  }

  function indentOf(s) {
    const m = s.match(/^\s*/);
    return m ? m[0].length : 0;
  }

  /* ---------- rules ---------- */

  const SEEDED_CTORS = [
    "RandomForestClassifier", "RandomForestRegressor", "ExtraTreesClassifier", "ExtraTreesRegressor",
    "GradientBoostingClassifier", "GradientBoostingRegressor", "HistGradientBoostingClassifier", "HistGradientBoostingRegressor",
    "train_test_split", "ShuffleSplit", "StratifiedShuffleSplit", "GroupShuffleSplit", "RepeatedKFold", "RepeatedStratifiedKFold",
    "KMeans", "MiniBatchKMeans", "TSNE", "SGDClassifier", "SGDRegressor", "MLPClassifier", "MLPRegressor", "IsolationForest",
    "LGBMClassifier", "LGBMRegressor", "LGBMRanker", "XGBClassifier", "XGBRegressor", "XGBRanker",
    "CatBoostClassifier", "CatBoostRegressor", "CatBoostRanker", "TPESampler", "RandomSampler", "CmaEsSampler",
    "TruncatedSVD", "NMF", "LatentDirichletAllocation", "BaggingClassifier", "BaggingRegressor",
  ];
  const SHUFFLE_SPLITTERS = ["KFold", "StratifiedKFold", "StratifiedGroupKFold", "GroupKFold"];
  const TABULAR_MODELS = /\b(LGBM\w+|XGB\w+|CatBoost\w+|RandomForest\w+|ExtraTrees\w+|HistGradientBoosting\w+|GradientBoosting\w+|LogisticRegression|SVC|SVR|KNeighbors\w+|lgb\.train|xgb\.train)\b/;
  const DEEP_MODEL = /\b(nn\.Module|torchvision\.models|timm\.create_model|Conv2d|from_pretrained|keras\.applications|tf\.keras\.Model|keras\.Sequential|AutoModel\w*|Trainer\()/;
  const TRAINING = /\.fit\(|\.train\(|\.backward\(|optimizer\.step\(|\bopt\.step\(|\.step\(\)|\bTrainer\(|lgb\.train\(|xgb\.train\(|\.partial_fit\(|\.optimize\(/;
  const TEST_NAME = /^(test|x_test|xtest|test_df|df_test|te|te_df|test_x|x_te|test_data|test_features|test_X)\b/i;

  function f(id, sev, cat, title, why, fix, cite, line, snippet) {
    return { id, sev, cat, title, why, fix, cite, line: line || null, snippet: snippet ? snippet.trim().slice(0, 160) : "" };
  }

  function lineRules(L, ctx) {
    const out = [];
    const push = (x) => out.push(x);
    const outVar = ctx.outVar;

    L.forEach((ln, i) => {
      const c = ln.code;
      const n = i + 1;
      if (!c.trim()) {
        if (/tuned offline|found offline|established offline|best params from|from (a )?previous (run|experiment)|copied from (optuna|local)|hand[- ]?tuned/i.test(ln.comment)) {
          push(f("ml-offline-hparams", "medium", "Compliance", "Hyperparameters chosen outside the script",
            "A comment says parameters were tuned offline. §1.1 prefers in-script search so the procedure is visible and learnable. Reviewers may weigh hardcoded offline results against you.",
            "Move the search into the script with a seeded sampler and a fixed trial count, or widen the in-script grid, and log the result.", "S2 §1.1", n, ln.raw));
        }
        return;
      }

      // Determinism: machine-conditioned
      if (/torch\.cuda\.is_available\(\)|tf\.test\.is_gpu_available|tf\.config\.list_physical_devices/.test(c))
        push(f("det-cuda-available", "blocker", "Determinism", "Device chosen by checking hardware",
          "'cuda if available, otherwise cpu' is named as a rejection in the determinism check.",
          'Write the device as a constant: DEVICE = torch.device("cuda").', "S2 §3.3", n, ln.raw));
      if (/os\.cpu_count\(\)|multiprocessing\.cpu_count\(\)|torch\.cuda\.device_count\(\)|psutil\.|mem_get_info\(|get_device_properties\(|torch\.cuda\.memory_reserved|nvidia-smi|torch\.get_num_threads\(\)/.test(c))
        push(f("det-machine", "blocker", "Determinism", "Logic depends on the machine",
          "Core counts, GPU counts or free memory make the run depend on the hardware it lands on.",
          "Replace with constants (NUM_WORKERS = 4, BATCH_SIZE = 32) measured to fit the A10G.", "S2 §3.3", n, ln.raw));

      // Determinism: clock-conditioned
      const clock = /time\.time\(\)|time\.perf_counter\(\)|time\.monotonic\(\)|datetime\.(datetime\.)?now\(\)|time\.process_time\(\)/.test(c);
      const decision = /^\s*(if|elif|while)\b|\bbreak\b|\b(budget|deadline|remaining|time_left|max_time|time_limit)\b/i.test(c);
      if (clock && decision)
        push(f("det-clock", "blocker", "Determinism", "Decision based on elapsed time",
          "Time budgets or stopping after N minutes make results depend on machine speed.",
          "Fix the epoch, step or trial counts in advance from measured timings.", "S2 §3.3, §4.4", n, ln.raw));
      else if (/^\s*(if|elif|while)\b.*\b(elapsed|time_left|remaining|deadline|time_budget|TIME_LIMIT|TIME_BUDGET)\b/.test(c))
        push(f("det-clock", "blocker", "Determinism", "Branch on an elapsed-time variable",
          "Branching on elapsed time is a runtime-conditioned decision.",
          "Remove the branch and fix the amount of work in advance.", "S2 §3.3", n, ln.raw));
      else if (clock)
        push(f("det-clock-log", "info", "Determinism", "Clock used (logging only?)",
          "Reading the clock is fine for logging. Make sure no decision uses it.",
          "Keep timing values out of every if/while condition and loop bound.", "S2 §3.3", n, ln.raw));

      if (/\.optimize\(.*timeout\s*=|\btimeout\s*=\s*\d/.test(statementAt(L, i)) && /optimi[sz]e|study/.test(c))
        push(f("det-hpo-timeout", "blocker", "Determinism", "HPO stopped by a time budget",
          "A timeout makes the number of trials, and so the result, depend on machine speed.",
          "Use a fixed n_trials with a seeded sampler.", "S2 §3.3", n, ln.raw));

      // try/except fallbacks
      if (/^\s*try\s*:/.test(c)) {
        const ind = indentOf(c);
        let exceptIdx = -1;
        for (let j = i + 1; j < Math.min(L.length, i + 80); j++) {
          if (L[j].code.trim() && indentOf(L[j].code) === ind && /^\s*except\b/.test(L[j].code)) { exceptIdx = j; break; }
          if (L[j].code.trim() && indentOf(L[j].code) < ind) break;
        }
        let body = "";
        if (exceptIdx >= 0) {
          for (let j = exceptIdx + 1; j < Math.min(L.length, exceptIdx + 12); j++) {
            if (L[j].code.trim() && indentOf(L[j].code) <= ind) break;
            body += L[j].code + "\n";
          }
        }
        const exceptLine = exceptIdx >= 0 ? L[exceptIdx].code : "";
        const switches = /\bimport\b|from_pretrained|Model|model\s*=|=\s*\w+\(|load/.test(body) || /ImportError|ModuleNotFoundError/.test(exceptLine);
        const silent = /^\s*(pass|continue)\s*$/m.test(body);
        push(f("det-try", switches ? "high" : "medium", "Determinism",
          switches ? "try/except switches to another method" : "try/except block",
          switches ? "The except branch loads or builds something different, which is a fallback path the determinism check rejects."
            : silent ? "A silent except hides failures and creates a second code path." : "Reviewers check try/except blocks for fallbacks that switch methods.",
          "Remove the try/except and call the one intended method directly. Let real errors fail loudly.", "S2 §3.3", n, ln.raw + (exceptLine ? " … " + exceptLine.trim() : "")));
      }

      // env switches / flags / branches
      if (/os\.environ\.get\(|os\.getenv\(/.test(c) || /os\.environ\[[^\]]+\](?!\s*=)/.test(c))
        push(f("det-env-switch", "high", "Determinism", "Behaviour read from an environment variable",
          "Environment-variable switches are rejected even when only one path runs.",
          "Replace with a constant at the top of the script.", "S2 §3.3", n, ln.raw));
      else if (/os\.environ(\.setdefault\(|\[[^\]]+\]\s*=)/.test(c))
        push(f("det-env-set", "info", "Determinism", "Environment variable set",
          "Setting thread counts or CUBLAS config to constants is fine and helps determinism.",
          "Keep the values constant.", "S2 §3.3", n, ln.raw));
      if (/^\s*(DEBUG|FAST_DEV_RUN|FAST|QUICK|SMOKE(_TEST)?|DRY_RUN|LOCAL|IS_LOCAL|KAGGLE|ON_KAGGLE)\s*=/.test(c) || /^\s*(if|elif)\s+(not\s+)?(DEBUG|FAST|QUICK|SMOKE|DRY_RUN|LOCAL|IS_LOCAL|ON_KAGGLE|args\.\w+)\b/.test(c))
        push(f("det-flag", "high", "Determinism", "Debug or mode flag",
          "Debug flags and switches are rejected even if they're off in the submitted run.",
          "Delete the flag and keep one fixed path.", "S2 §3.3", n, ln.raw));
      if (/^\s*(if|elif)\s+.*\b\w*(MODEL|BACKBONE|METHOD|MODE|ARCH|STRATEGY)\w*\s*(==|in)\s*/i.test(c))
        push(f("det-method-branch", "high", "Determinism", "Branch that chooses between methods",
          "if-branches that select a model or method count as unused switches.",
          "Keep only the method you actually use.", "S2 §3.3", n, ln.raw));
      if (/^\s*(if|elif)\b.*(os\.path\.exists|os\.path\.isfile|\.exists\(\)|\.is_file\(\))/.test(c))
        push(f("det-file-branch", "medium", "Determinism", "Behaviour depends on which files exist",
          "Probing for alternative file names creates paths that depend on the environment.",
          "Read the exact file names given in the challenge description and fail loudly if one is missing.", "S2 §3.3", n, ln.raw));
      if (/\bargparse\b|\bclick\.|fire\.Fire/.test(c))
        push(f("io-argparse", "medium", "Contract", "Command-line option parser",
          "The platform passes exactly two positional arguments. Extra options become switches.",
          "Use Path(sys.argv[1]) and Path(sys.argv[2]) directly.", "S3, S2 §3.3", n, ln.raw));
      if (/len\(\s*sys\.argv\s*\)/.test(c))
        push(f("io-argv-default", "low", "Contract", "Fallback when arguments are missing",
          "A default path for missing arguments is a second code path. It never runs on the platform but reads as a switch.",
          "Use PUBLIC_DIR = Path(sys.argv[1]) and SUBMISSION_OUT = Path(sys.argv[2]) unconditionally.", "S3, S2 §3.3", n, ln.raw));

      // seeds
      if (/cudnn\.benchmark\s*=\s*True/.test(c))
        push(f("det-cudnn-bench", "high", "Determinism", "cudnn.benchmark = True",
          "Autotuning picks kernels per machine and run.",
          "Set cudnn.benchmark = False and cudnn.deterministic = True.", "S2 §3.4", n, ln.raw));
      if (/np\.random\.default_rng\(\s*\)|random\.Random\(\s*\)/.test(c) || (/torch\.Generator\(\s*\)/.test(c) && !/manual_seed/.test(c + " " + (L[i + 1] ? L[i + 1].code : ""))))
        push(f("det-unseeded-gen", "high", "Determinism", "Random generator created without a seed", "An unseeded generator draws from OS entropy on every run.",
          "Pass SEED: np.random.default_rng(SEED), torch.Generator().manual_seed(SEED).", "S2 §3.3", n, ln.raw));
      const ctor = SEEDED_CTORS.find((name) => new RegExp("\\b" + name + "\\(").test(c));
      if (ctor) {
        const st = statementAt(L, i);
        const seeded = /random_state\s*=|\bseed\s*=|random_seed\s*=|\*\*\w+/.test(st);
        if (!seeded)
          push(f("det-no-seed", "high", "Determinism", `${ctor} without a seed`,
            "Every model, split and sampler needs its random_state or seed set.",
            `Add random_state=SEED (or seed=SEED for samplers and CatBoost's random_seed).`, "S2 §3.3, §3.4", n, ln.raw));
      }
      const splitter = SHUFFLE_SPLITTERS.find((name) => new RegExp("\\b" + name + "\\(").test(c));
      if (splitter) {
        const st = statementAt(L, i);
        if (/shuffle\s*=\s*True/.test(st) && !/random_state\s*=/.test(st))
          push(f("det-no-seed", "high", "Determinism", `${splitter}(shuffle=True) without random_state`,
            "Shuffled folds change on every run, so every CV number changes too.", "Add random_state=SEED.", "S2 §3.3", n, ln.raw));
      }
      if (/\bDataLoader\(/.test(c)) {
        const st = statementAt(L, i);
        if (/shuffle\s*=\s*True/.test(st) && !/generator\s*=/.test(st))
          push(f("det-loader-gen", "high", "Determinism", "Shuffled DataLoader without a seeded generator",
            "Batch order is unseeded.", "Pass generator=torch.Generator().manual_seed(SEED).", "S2 §3.4", n, ln.raw));
        const nw = st.match(/num_workers\s*=\s*([\w.]+)/);
        if (nw && nw[1] !== "0" && !/worker_init_fn\s*=/.test(st))
          push(f("det-loader-workers", "medium", "Determinism", "Worker processes without worker_init_fn",
            "Each worker's random state should be derived from the seed.", "Add worker_init_fn=seed_worker (see the §3.4 template).", "S2 §3.4", n, ln.raw));
      }
      if (/(for\s+\w+\s+in|list\(|enumerate\()\s*set\(/.test(c))
        push(f("det-set-order", "low", "Determinism", "Iterating over a set",
          "Set iteration order for strings changes between Python runs (hash randomisation), which can reorder features or classes.",
          "Use sorted(set(...)).", "", n, ln.raw));

      // I/O contract
      if (/['"](\.\/)?dataset\/public|\/kaggle\/input|\/kaggle\/working|['"](\.\/)?working\/submission\.csv/.test(ln.raw.replace(/#.*$/, "")) && !/len\(\s*sys\.argv/.test(c))
        push(f("io-hardcoded-path", "high", "Contract", "Hardcoded data or output path",
          "The current runtime passes paths as arguments. Hardcoded paths are the old format.",
          "Use Path(sys.argv[1]) / 'train.csv' and Path(sys.argv[2]).", "S3", n, ln.raw));
      if (/answers\.csv|['"\/]private\//.test(ln.raw.replace(/#.*$/, "")))
        push(f("io-private", "blocker", "Compliance", "Reads private answers",
          "Private ground truth is never available to solutions, and trying to read it is an exploit.",
          "Remove it.", "S2 §5.5", n, ln.raw));
      if (/\.to_csv\(/.test(c)) {
        const st = statementAt(L, i);
        if (outVar && st.includes(outVar) && !/index\s*=\s*False/.test(st))
          push(f("io-index", "medium", "Contract", "Submission written with the DataFrame index",
            "to_csv writes an unnamed index column unless index=False, which breaks the column check.",
            "Use .to_csv(SUBMISSION_OUT, index=False).", "S2 §3.1", n, ln.raw));
      }

      // environment
      if (/\bpip\s+install\b|subprocess\.\w+\(.*pip|os\.system\(|^\s*[!%]\s*(pip|conda)\b|conda\s+install/.test(ln.raw))
        push(f("env-install", "blocker", "Environment", "Installs or shells out",
          "No installs from inside the script, and no approval route for extra packages.",
          "Use only packages already in the Kaggle image.", "S2 §4.1", n, ln.raw));
      else if (/^\s*[!%]/.test(ln.raw) && !ln.inDoc)
        push(f("env-magic", ctx.isNotebook ? "medium" : "high", "Environment", "Notebook magic or shell line",
          "python3 solution.py treats ! and % lines as syntax errors.", "Remove it or replace it with Python.", "S3", n, ln.raw));
      if (/github\.com|raw\.githubusercontent|git\s+clone|torch\.hub\.load\(/.test(ln.raw.replace(/#.*$/, "")))
        push(f("env-github", "blocker", "Environment", "Model or code from GitHub",
          "Installing or loading models through GitHub or other channels is rejected immediately.",
          "Use Hugging Face or timm weights through installed packages.", "S2 §4.3", n, ln.raw));
      if (/requests\.(get|post)\(|urllib\.request|urlretrieve\(|\bwget\b|\bcurl\b|kaggle\s+(datasets|competitions)\s+download/.test(c) ||
          (/load_dataset\(/.test(c) && !/load_dataset\(\s*['"](csv|json|parquet|text|imagefolder)['"]/.test(c)))
        push(f("env-external-data", "high", "Compliance", "Downloads data from outside the challenge",
          "External datasets are banned. Only pretrained backbone weights may come from the internet.",
          "Use only the files under the public directory.", "S2 §5.2.3", n, ln.raw));
      if (/torch\.load\(|joblib\.load\(|pickle\.load\(|\bnp\.load\(|pd\.read_pickle\(|load_state_dict\(/.test(c)) {
        const savedInScript = ctx.text.match(/torch\.save\(|joblib\.dump\(|pickle\.dump\(|np\.save\(|to_pickle\(/);
        push(f("env-artifact-load", savedInScript ? "info" : "high", "Environment",
          savedInScript ? "Loads an artifact (saved earlier in this script?)" : "Loads a pre-built artifact",
          "Every run starts from the raw data. Loading embeddings or weights from a previous run is not allowed.",
          "Make sure the file is produced earlier in the same run, or recompute it.", "S2 §4.7, §5.2.2", n, ln.raw));
      }
      if (/\bopenai\b|\banthropic\b|google\.generativeai|genai\.(Client|GenerativeModel)|ChatCompletion|messages\.create\(/.test(c))
        push(f("env-llm", "blocker", "Compliance", "Calls an LLM API",
          "LLM outputs can't be part of the solution, and using one is not training a model.",
          "Remove the call.", "S1, S2 §1.1", n, ln.raw));
      const fp = c.match(/from_pretrained\(\s*["']([^"']+)["']/) || ln.raw.match(/from_pretrained\(\s*["']([^"']+)["']/);
      if (/from_pretrained\(|timm\.create_model\(.*pretrained\s*=\s*True|weights\s*=\s*["']?\w*(DEFAULT|IMAGENET)/i.test(c)) {
        if (ctx.category === "scratch")
          push(f("scr-pretrained", "blocker", "Compliance", "Pretrained weights in a from-scratch challenge",
            "Nothing pretrained is allowed in from-scratch challenges.", "Initialise from scratch.", "S2 §6.5", n, ln.raw));
        else
          push(f("env-pretrained", "info", "Compliance", "Loads pretrained weights" + (fp ? `: ${fp[1]}` : ""),
            "Fine when it's a general-purpose backbone downloaded at run time. It must not be weights you fine-tuned elsewhere.",
            "Use public base checkpoints and fine-tune them inside the script.", "S2 §4.2, §5.2.2", n, ln.raw));
      }
      if (/torch\.compile\(/.test(c))
        push(f("rt-compile", "low", "Runtime", "torch.compile",
          "Adds compile time and can introduce run-to-run variation.", "Measure the end-to-end time with and without it.", "", n, ln.raw));

      // test-set usage
      const fit = c.match(/\.(fit|fit_transform|partial_fit|adapt)\(\s*([\w.\[\]'"]+)/);
      if (fit && TEST_NAME.test(fit[2].replace(/^.*\./, "").replace(/\[.*$/, "")) || (fit && /\btest/i.test(fit[2]) && !/train/i.test(fit[2])))
        push(f("test-fit", "blocker", "Leakage", "Something is fitted on test data",
          "Fitting transforms, vocabularies or models on test inputs adapts the solution to this test set.",
          "Fit on training folds only, and only transform test.", "S2 §5.2.5", n, ln.raw));
      if (/(concat|vstack|hstack|append|cat)\(\s*\[?\s*\(?\s*[\w.]*(train|tr)\w*[\w.\[\]"']*\s*,\s*[\w.]*(test|te)\w*/i.test(c) ||
          /(concat|vstack|hstack|cat)\(\s*\[?\s*\(?\s*[\w.]*(test|te)\w*[\w.\[\]"']*\s*,\s*[\w.]*(train|tr)\w*/i.test(c))
        push(f("test-concat", "high", "Leakage", "Train and test concatenated",
          "This is fine only for identical per-row transforms. Any statistic, vocabulary or fit computed on the combined frame uses test data.",
          "Process train and test separately with transforms fitted on train.", "S2 §5.2.5", n, ln.raw));
      if (/pseudo[_\s-]?label|self[_-]?training|\bsoft[_-]?labels?\b|test[_-]?time[_-]?(adapt|training)|\bTENT\b/i.test(c))
        push(f("test-pseudo", "blocker", "Leakage", "Pseudo-labelling or test-time adaptation",
          "Explicitly prohibited, even without test labels.", "Remove it.", "S2 §5.2.5", n, ln.raw));
      if (/\b(test|x_test|test_df|df_test|te_df|test_pred\w*|pred_test\w*|test_preds?)\b(\[[^\]]+\])*\.(mean|std|median|quantile|describe|value_counts|var|rank)\(/i.test(c))
        push(f("test-stats", "high", "Leakage", "Statistic computed over the test set",
          "Test-wide statistics used for features, scaling or post-processing make predictions depend on the rest of the test set.",
          "Use statistics from training data or OOF predictions.", "S2 §5.2.5", n, ln.raw));
      if (/np\.(quantile|percentile)\(\s*\w*(pred|test)/i.test(c) || /len\(\s*(test|x_test|test_df)\w*\s*\)\s*\*\s*0?\.\d+/i.test(c))
        push(f("test-quantile", "high", "Leakage", "Threshold or count derived from the test distribution",
          "Choosing thresholds from test-score quantiles, or assuming a prevalence in test, calibrates to the test set.",
          "Choose thresholds on OOF predictions.", "S2 §5.2.5", n, ln.raw));

      // data
      if (/\bSMOTE\b|\bADASYN\b|\bimblearn\b|make_classification\(|make_regression\(|back[_-]?translat|\bparaphras/i.test(c) || /synthetic/i.test(c))
        push(f("data-synthetic", "high", "Compliance", "Synthetic training data",
          "Generating your own synthetic data and training on it is not allowed.",
          "Use class weights or loss reweighting instead of oversampling with generated rows.", "S2 §5.2.6", n, ln.raw));
      const nums = (c.match(/-?\b\d+\.?\d*(e-?\d+)?\b/g) || []).length;
      if (nums >= 30 && /[\[{(]/.test(c))
        push(f("data-hardcoded", "high", "Compliance", "Large literal table of numbers",
          "Hardcoded values or labels are removed outright in review.", "Have the model learn the values from data.", "S2 §5.5", n, ln.raw));

      // grey areas
      if (/TfidfVectorizer|CountVectorizer|HashingVectorizer|ngram_range|markov/i.test(c))
        push(f("grey-tfidf", "medium", "Grey area", "TF-IDF / n-gram / Markov features",
          "These are grey-area techniques, allowed or rejected at the reviewer's discretion depending on how much the solution relies on them.",
          "Keep them as a minor component next to a trained model, and show the trained model carries most of the signal.", "S2 §5.3.2", n, ln.raw));

      // runtime / version risks
      if (/np\.fromstring\(/.test(c))
        push(f("rt-fromstring", "low", "Runtime", "np.fromstring",
          "Deprecated API. It still parses text with sep= in current NumPy, but it's fragile across versions.",
          "Use np.array(s.split(), dtype=int).", "", n, ln.raw));
      if (/\bnp\.(float|int|bool|object)\(|\.ix\[|load_boston\(/.test(c))
        push(f("rt-removed-api", "high", "Runtime", "API removed in current library versions",
          "np.float/np.int, .ix and load_boston no longer exist and will crash.", "Use float/int, .loc, or another dataset.", "", n, ln.raw));
      if (/(early_stopping_rounds|verbose_eval)\s*=/.test(c) && /lgb|lightgbm|LGBM|\.fit\(/.test(statementAt(L, Math.max(0, i - 3), 6)))
        push(f("rt-lgb-api", "medium", "Runtime", "Old LightGBM/XGBoost early-stopping argument",
          "LightGBM 4 removed early_stopping_rounds and verbose_eval from train/fit, and XGBoost 2 moved early_stopping_rounds to the constructor.",
          "LightGBM: callbacks=[lgb.early_stopping(N)]. XGBoost: XGBClassifier(early_stopping_rounds=N).", "", n, ln.raw));
      if (/warnings\.(filterwarnings|simplefilter)\(\s*["']ignore/.test(c))
        push(f("rt-warnings", "info", "Runtime", "All warnings silenced",
          "This hides deprecation warnings that predict crashes on a newer image.", "Silence specific categories only.", "", n, ln.raw));
    });
    return out;
  }

  function fileRules(L, ctx) {
    const out = [];
    const t = ctx.codeText;
    const lineOf = (re) => { const i = L.findIndex((l) => re.test(l.code)); return i >= 0 ? i + 1 : null; };

    if (!/sys\.argv\s*\[\s*1\s*\]|sys\.argv\s*\[\s*1\s*:|sys\.argv\s*\[\s*1\s*:\s*3\s*\]/.test(t))
      out.push(f("io-argv-missing", "blocker", "Contract", "Doesn't read the platform's path arguments",
        "The platform runs python3 solution.py <public_dir> <submission_out> and provides the paths only as arguments.",
        "PUBLIC_DIR = Path(sys.argv[1]); SUBMISSION_OUT = Path(sys.argv[2]); SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)", "S3"));
    else if (!/sys\.argv\s*\[\s*2\s*\]|sys\.argv\s*\[\s*1\s*:/.test(t))
      out.push(f("io-argv-out", "blocker", "Contract", "Output path argument not used",
        "The submission must be written to sys.argv[2].", "SUBMISSION_OUT = Path(sys.argv[2])", "S3"));
    if (ctx.outVar && !/mkdir\(|makedirs\(/.test(t))
      out.push(f("io-mkdir", "medium", "Contract", "Output directory not created",
        "If the parent folder of the output path doesn't exist, to_csv crashes and the credit is consumed.",
        "SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)", "S3"));

    const usesTorch = /\bimport torch\b|\bfrom torch\b/.test(t);
    const usesNpRandom = /np\.random\.(rand|randn|randint|choice|permutation|shuffle|normal|uniform|seed)\(/.test(t);
    if (usesTorch && !/torch\.manual_seed\(/.test(t))
      out.push(f("det-torch-seed", "blocker", "Determinism", "torch is used but never seeded",
        "Weight initialisation and dropout differ on every run.", "torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)", "S2 §3.4", lineOf(/import torch/)));
    if (usesTorch && /\bcuda\b|torch\.device/.test(t) && !/cuda\.manual_seed_all\(|cuda\.manual_seed\(|torch\.manual_seed\(/.test(t))
      out.push(f("det-cuda-seed", "low", "Determinism", "CUDA RNG not seeded explicitly", "torch.manual_seed covers CUDA, but the guidebook template also calls cuda.manual_seed_all.", "torch.cuda.manual_seed_all(SEED)", "S2 §3.4"));
    if (usesTorch && /nn\.Conv|torchvision|timm/.test(t) && !/cudnn\.deterministic\s*=\s*True/.test(t))
      out.push(f("det-cudnn", "medium", "Determinism", "cudnn.deterministic not set", "Convolution kernels can be nondeterministic on GPU.", "torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False", "S2 §3.4"));
    if (usesNpRandom && !/np\.random\.seed\(/.test(t) && !/default_rng\(\s*\w/.test(t))
      out.push(f("det-np-seed", "high", "Determinism", "NumPy random functions used without np.random.seed", "Global NumPy draws differ between runs.", "np.random.seed(SEED) at the top.", "S2 §3.4", lineOf(/np\.random\./)));
    if (/\brandom\.(random|randint|choice|shuffle|sample|uniform)\(/.test(t) && !/\brandom\.seed\(/.test(t))
      out.push(f("det-py-seed", "high", "Determinism", "Python random used without random.seed", "Python's global RNG is unseeded.", "random.seed(SEED)", "S2 §3.4", lineOf(/\brandom\.(random|randint|choice|shuffle|sample|uniform)\(/)));

    const trains = TRAINING.test(t);
    if (!trains)
      out.push(f("ml-no-training", "blocker", "Compliance", "No training step found",
        "Real training or fine-tuning must happen inside the script. Inference-only and rule-based solutions are rejected.",
        "Train or fine-tune a model in the script.", "S2 §1.1, §5.2.1"));
    const hasPretrained = /from_pretrained\(|pretrained\s*=\s*True/.test(t);
    const hasBackward = /\.backward\(|Trainer\(|\.fit\(\s*\w*(loader|dataset|ds)\b|model\.fit\(/i.test(t);
    if (hasPretrained && !hasBackward && TABULAR_MODELS.test(t)) {
      const sev = ctx.category === "finetune" ? "high" : ctx.category === "cv" ? "medium" : "medium";
      out.push(f("ft-embed-tabular", sev, "Grey area", "Pretrained embeddings feeding a tabular model, backbone not fine-tuned",
        "Grey area in CV and NLP. In Fine-tuning challenges it's likely rejected because the backbone must be fine-tuned.",
        "Fine-tune the backbone (full or parameter-efficient) and use the tabular model, if at all, as a minor member.", "S2 §6.2, §6.4"));
    }
    const images = /cv2\.imread|Image\.open|imread\(|read_image\(|decode_image|load_img\(/.test(t);
    if (images && TABULAR_MODELS.test(t) && !DEEP_MODEL.test(t)) {
      const pixels = /\.flatten\(\)|\.ravel\(\)|reshape\(\s*-1\s*\)|reshape\(\s*\w+\s*,\s*-1\s*\)|reshape\(len\(/.test(t);
      out.push(f(pixels ? "cv-pixels-tabular" : "cv-tabular-only", pixels ? "blocker" : "high", "Compliance",
        pixels ? "Tabular model on raw pixels" : "Images handled only by a tabular model",
        pixels ? "Tabular models on raw pixels are rejected immediately." : "Hand-made image features used only by a tabular model are grey and fragile.",
        "Train or fine-tune a CNN or ViT, and feed any hand features into it as extra inputs.", "S2 §6.2"));
    }
    const ranker = /LGBMRanker|XGBRanker|lambdarank|rank:pairwise|rank:ndcg|BM25|rank_bm25/.test(t);
    if ((ctx.category === "retrieval" || ranker) && ranker && !/nn\.Module|Trainer\(|\.backward\(/.test(t))
      out.push(f("rag-offshelf", "high", "Compliance", "Off-the-shelf ranker without a trained deep ranker",
        "Retrieval challenges want a genuinely trained ranking model, usually deep. Hand features plus an off-the-shelf ranking algorithm is called out as not enough.",
        "Make the main ranking signal a trained (cross-)encoder or neural ranker.", "S2 §6.3"));

    const reCalls = (t.match(/\bre\.(sub|findall|search|match|compile|split|fullmatch|finditer)\(/g) || []).length;
    if (reCalls > 8)
      out.push(f("grey-regex", "medium", "Grey area", `Heavy regex use (${reCalls} calls)`,
        "Regex beyond cleaning or deterministic extraction reads as over-reliance on how the data was generated.",
        "Keep regex for cleaning only. Let the model learn the patterns, then run the strip-the-model test.", "S2 §5.3.1"));

    if (ctx.outVar && !/assert|raise\b|problems|validate/i.test(t))
      out.push(f("io-validate", "low", "Contract", "No output validation before writing",
        "A malformed output ranks below a score of 0 and the credit is consumed.",
        "Assert columns, row count, unique ids matching test, and finite values before to_csv.", "S2 §4.6"));

    // imports
    const imports = new Set();
    L.forEach((l) => {
      const m = l.code.match(/^\s*import\s+([\w.]+(\s*,\s*[\w.]+)*)/);
      const m2 = l.code.match(/^\s*from\s+([\w.]+)\s+import\b/);
      if (m) m[1].split(",").forEach((s) => imports.add(s.trim().split(".")[0].split(" ")[0]));
      if (m2 && !m2[1].startsWith(".")) imports.add(m2[1].split(".")[0]);
    });
    const allowed = new Set([...E.ALLOWED_PACKAGES, ...E.CORE_PACKAGES].map((s) => s.toLowerCase()));
    const unknown = [];
    const verifyList = [];
    [...imports].sort().forEach((mod) => {
      if (E.STDLIB.includes(mod)) return;
      const dist = (E.IMPORT_ALIASES[mod] || mod).toLowerCase();
      if (allowed.has(dist) || allowed.has(mod.toLowerCase()) || allowed.has(dist.replace(/_/g, "-"))) return;
      if (E.LIKELY_IN_IMAGE.map((s) => s.toLowerCase()).includes(dist)) verifyList.push(mod);
      else unknown.push(mod);
    });
    if (unknown.length)
      out.push(f("env-unknown-import", "high", "Environment", "Imports not on the supplied package lists: " + unknown.join(", "),
        "A missing package fails the script run and consumes the credit.",
        "Confirm each one exists in the Kaggle Docker image or replace it.", "S2 §4.1, S4"));
    if (verifyList.length)
      out.push(f("env-verify-import", "low", "Environment", "Imports to confirm in the image: " + verifyList.join(", "),
        "Usually present in the Kaggle image, but not on the lists you supplied.",
        "Check the image's pip list once and record the result.", "S4"));

    if (L.length > 1500)
      out.push(f("cx-long", "info", "Complexity", `${L.length} lines`,
        "Very long scripts are harder for reviewers and for the compliance LLM to read.",
        "Remove dead code and unused experiments. Keep one fixed path.", "S2 §3.3"));
    const families = new Set((t.match(TABULAR_MODELS) ? t.match(new RegExp(TABULAR_MODELS.source, "g")) : []).map((s) => s.replace(/(Classifier|Regressor|Ranker)$/, "")));
    if (families.size > 4)
      out.push(f("cx-families", "info", "Complexity", `${families.size} model families`,
        "Each ensemble member should earn its place with an OOF gain larger than noise.",
        "Drop members whose removal doesn't hurt OOF by more than fold noise.", ""));
    return { findings: out, imports: [...imports].sort() };
  }

  function audit(text, opts = {}) {
    let isNotebook = false;
    let src = String(text || "");
    if (/^\s*\{/.test(src)) {
      const nb = fromNotebook(src);
      if (nb !== null) { src = nb; isNotebook = true; }
    }
    const L = prepare(src);
    const codeText = L.map((l) => l.code).join("\n");
    const outM = codeText.match(/(\w+)\s*=\s*(Path\(\s*)?sys\.argv\s*\[\s*2\s*\]/) || codeText.match(/(\w+)\s*=\s*sys\.argv\s*\[\s*2\s*\]\s+if/);
    const ctx = { category: opts.category || "general", isNotebook, text: src, codeText, outVar: outM ? outM[1] : null };
    const a = lineRules(L, ctx);
    const b = fileRules(L, ctx);
    // collapse repeated identical findings (same id + title) into one with a count
    const merged = [];
    const key = new Map();
    for (const x of [...a, ...b.findings]) {
      const k = x.id + "|" + x.title;
      if (key.has(k)) { const m = key.get(k); m.count++; m.lines.push(x.line); continue; }
      const m = { ...x, count: 1, lines: [x.line] };
      key.set(k, m);
      merged.push(m);
    }
    merged.sort((p, q) => SEV_ORDER[p.sev] - SEV_ORDER[q.sev] || (p.line || 0) - (q.line || 0));
    const counts = { blocker: 0, high: 0, medium: 0, low: 0, info: 0 };
    merged.forEach((m) => counts[m.sev]++);
    const verdict = counts.blocker ? "blocked" : counts.high ? "at-risk" : counts.medium ? "review" : "clean";
    return { findings: merged, counts, verdict, imports: b.imports, lines: L.length, isNotebook, category: ctx.category };
  }

  E.auditor = { audit, prepare, SEV_ORDER };
})();
