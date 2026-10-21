/* Static solution auditor. Heuristic pattern checks against the platform rules.
 * It can't prove compliance. It points at lines a reviewer or the platform's
 * LLM checks are likely to flag, and explains the rule and the fix.
 * Calibrated on real Eris solutions (top-5 files and the owner's own scripts):
 * see tests/engines.test.cjs for the false-positive cases it must not raise. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const SEV_ORDER = { blocker: 0, high: 1, medium: 2, low: 3, info: 4 };
  const SOURCE_LIMIT = 512000; // bytes; larger sources fail "source cannot be inspected"

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

  /* Per line: raw text, code (comments and docstrings removed, string contents
   * kept), bare (string contents removed too), comment, and the bracket depth
   * before the line (so continuation lines can be told apart). */
  function prepare(text) {
    const lines = text.replace(/\r\n?/g, "\n").split("\n");
    const out = [];
    let inDoc = null;
    let depth = 0;
    for (const raw of lines) {
      let code = "", bare = "", comment = "";
      let i = 0, quote = null;
      const docLine = inDoc !== null;
      const depthBefore = depth;
      while (i < raw.length) {
        if (inDoc) {
          const end = raw.indexOf(inDoc, i);
          if (end === -1) { i = raw.length; break; }
          i = end + 3; inDoc = null; continue;
        }
        const ch = raw[i];
        if (quote) {
          code += ch;
          if (ch === "\\") { code += raw[i + 1] || ""; i += 2; continue; }
          if (ch === quote) { quote = null; bare += ch; }
          i++; continue;
        }
        const three = raw.substr(i, 3);
        if (three === '"""' || three === "'''") {
          const end = raw.indexOf(three, i + 3);
          code += '""'; bare += '""';
          if (end === -1) { inDoc = three; i = raw.length; break; }
          i = end + 3; continue;
        }
        if (ch === "#") { comment = raw.slice(i + 1); break; }
        if (ch === '"' || ch === "'") { quote = ch; bare += ch; code += ch; i++; continue; }
        if ("([{".includes(ch)) depth++;
        else if (")]}".includes(ch)) depth = Math.max(0, depth - 1);
        code += ch; bare += ch; i++;
      }
      out.push({ raw, code, bare, comment, depthBefore, inDoc: docLine && !code.trim() });
    }
    return out;
  }

  /* Join a call statement that spans several lines (until brackets balance). */
  function statementAt(L, idx, maxLines = 25) {
    let depth = 0, text = "";
    for (let j = idx; j < Math.min(L.length, idx + maxLines); j++) {
      const c = L[j].code;
      text += " " + c;
      for (const ch of c) {
        if ("([{".includes(ch)) depth++;
        else if (")]}".includes(ch)) depth--;
      }
      if (depth <= 0) break;
    }
    return text;
  }
  const indentOf = (s) => (s.match(/^\s*/) || [""])[0].length;
  const nextCode = (L, i) => { for (let j = i + 1; j < Math.min(L.length, i + 4); j++) if (L[j].code.trim()) return L[j].code; return ""; };
  const isLogLine = (c) => /^\s*(log|print|logger\.\w+|logging\.\w+|tqdm\.write|sys\.stdout\.write|warnings\.warn)\s*\(/.test(c);
  const raisesNext = (L, i) => /\braise\b|sys\.exit|\bassert\b/.test(L[i].code + " " + nextCode(L, i));

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
  const DEEP_MODEL = /\b(nn\.Module|torchvision\.models|timm\.create_model|Conv[12]d|from_pretrained|keras\.applications|tf\.keras\.Model|keras\.Sequential|AutoModel\w*|Trainer\(|nn\.GRU|nn\.LSTM|nn\.Transformer\w*)/;
  const TRAINING = /\.fit\(|\.train\(|\.backward\(|optimizer\.step\(|\bopt\.step\(|\.step\(\)|\bTrainer\(|lgb\.train\(|xgb\.train\(|\.partial_fit\(|\.optimize\(|\.fit_resample\(/;
  const TEST_NAME = /^(test|x_test|xtest|test_df|df_test|te|te_df|test_x|x_te|test_data|test_features|test_X)\b/i;
  const PATHY_ENV = /DIR|PATH|ROOT|INPUT|OUTPUT|DATA|HOME|CACHE|OUT\b/i;

  function f(id, sev, cat, title, why, fix, cite, line, snippet) {
    return { id, sev, cat, title, why, fix, cite, line: line || null, snippet: snippet ? snippet.trim().slice(0, 160) : "" };
  }

  function lineRules(L, ctx) {
    const out = [];
    const push = (x) => out.push(x);
    const R = ctx.restrict || {};

    L.forEach((ln, i) => {
      const c = ln.code, b = ln.bare;
      const n = i + 1;
      if (!c.trim()) {
        if (/tuned offline|found offline|established offline|best params from|from (a )?previous (run|experiment)|copied from (optuna|local)|hand[- ]?tuned|selected on (train )?folds? (earlier|offline|in a previous)/i.test(ln.comment))
          push(f("ml-offline-hparams", "medium", "Compliance", "Constants chosen outside the script",
            "A comment says values were tuned offline. Your guidebook prefers in-script search, and reviewers treat offline-tuned constants (params, thresholds, calibration floats) as hardcoding.",
            "Search them inside the script on OOF with a fixed grid or seeded trials, and log the choice.", "S2 §1.1", n, ln.raw));
        return;
      }
      const logLine = isLogLine(c);

      // ---- determinism: machine-conditioned
      if (/torch\.cuda\.is_available\(\)|tf\.test\.is_gpu_available|tf\.config\.list_physical_devices/.test(c)) {
        if (/^\s*assert\b/.test(c) || (/^\s*if\b/.test(c) && /\braise\b|sys\.exit/.test(c + " " + nextCode(L, i))))
          push(f("det-cuda-guard", "low", "Determinism", "GPU presence checked as a hard guard",
            "A fail-fast guard is not a fallback, so it is usually fine. Make sure no branch switches to a CPU path.",
            "Keep it as assert/raise only.", "S2 §3.3", n, ln.raw));
        else
          push(f("det-cuda-available", "blocker", "Determinism", "Device chosen by checking hardware",
            "'cuda if available, otherwise cpu' is named as a rejection in the determinism check. Four of the top-5 files in your corpus still have it; ranks are not acceptance.",
            'Write the device as a constant: DEVICE = torch.device("cuda") (or "cpu" for CPU-only challenges).', "S2 §3.3", n, ln.raw));
      }
      if (!logLine && /os\.cpu_count\(\)|multiprocessing\.cpu_count\(\)|torch\.cuda\.device_count\(\)|psutil\.|mem_get_info\(|get_device_properties\(|torch\.cuda\.memory_reserved|nvidia-smi|torch\.get_num_threads\(\)/.test(c))
        push(f("det-machine", "blocker", "Determinism", "Logic depends on the machine",
          "Core counts, GPU counts or free memory make the run depend on the hardware it lands on.",
          "Replace with constants (NUM_WORKERS = 4, BATCH_SIZE = 32) measured to fit the target hardware.", "S2 §3.3", n, ln.raw));
      if (/\b(n_jobs|thread_count|num_threads|nthread)\s*=\s*-1\b/.test(c))
        push(f("det-all-cores", "low", "Determinism", "Uses all available cores (-1)",
          "'All cores' depends on the machine. sklearn forests give identical results at any n_jobs, but GBDT and BLAS reductions can differ in the last digits. The platform's own exemplary solution uses n_jobs=-1, so this alone is unlikely to fail review.",
          "Set a fixed number (e.g. n_jobs=4, thread_count=4).", "S2 §3.3", n, ln.raw));

      // ---- determinism: clock-conditioned
      const clock = /time\.time\(\)|time\.perf_counter\(\)|time\.monotonic\(\)|datetime\.(datetime\.)?now\(\)|time\.process_time\(\)/.test(c);
      const decision = /^\s*(if|elif|while)\b|\bbreak\b|\b(budget|deadline|remaining|time_left|max_time|time_limit)\b/i.test(c);
      if (clock && decision && !logLine)
        push(f("det-clock", "blocker", "Determinism", "Decision based on elapsed time",
          "Time budgets and 'stop after N minutes' were the exact pattern behind your real Deterministic Execution rejections.",
          "Fix the epoch, step or trial counts in advance from measured timings.", "S2 §3.3, §4.4", n, ln.raw));
      else if (/^\s*(if|elif|while)\b.*\b(elapsed|time_left|remaining|deadline|time_budget|TIME_LIMIT|TIME_BUDGET|CUTOFF|DEADLINE)\b/.test(c) || /\b(budget_ok|time_ok|within_budget)\s*=\s*lambda/.test(c))
        push(f("det-clock", "blocker", "Determinism", "Branch on an elapsed-time variable",
          "Branching on elapsed time is a runtime-conditioned decision, even when the deadline variable defaults to None.",
          "Remove the branch and fix the amount of work in advance.", "S2 §3.3", n, ln.raw));
      else if (clock)
        push(f("det-clock-log", "info", "Determinism", "Clock used (logging only?)",
          "Reading the clock is fine for logging. Make sure no decision uses it.",
          "Keep timing values out of every if/while condition, loop bound and argument.", "S2 §3.3", n, ln.raw));
      if (/signal\.alarm\(|signal\.setitimer\(|threading\.Timer\(|\b(time_limit|timelimit|max_time|time_budget)\s*=\s*[\d.]/.test(c))
        push(f("det-alarm", "blocker", "Determinism", "Wall-clock alarm or time-limit argument",
          "Alarms and library time limits (Optuna timeout, tuner time_limit) make the work done depend on machine speed.",
          "Remove it. Use fixed trial, round and epoch counts.", "S2 §3.3", n, ln.raw));
      if (/\.optimize\(.*timeout\s*=|\btimeout\s*=\s*\d/.test(statementAt(L, i)) && /optimi[sz]e|study/.test(c))
        push(f("det-hpo-timeout", "blocker", "Determinism", "HPO stopped by a time budget",
          "A timeout makes the number of trials, and so the result, depend on machine speed.",
          "Use a fixed n_trials with a seeded sampler.", "S2 §3.3", n, ln.raw));

      // ---- try/except fallbacks
      if (/^\s*try\s*:/.test(c)) {
        const ind = indentOf(c);
        let exceptIdx = -1;
        for (let j = i + 1; j < Math.min(L.length, i + 80); j++) {
          if (L[j].code.trim() && indentOf(L[j].code) === ind && /^\s*except\b/.test(L[j].code)) { exceptIdx = j; break; }
          if (L[j].code.trim() && indentOf(L[j].code) < ind) break;
        }
        let body = "";
        if (exceptIdx >= 0) for (let j = exceptIdx + 1; j < Math.min(L.length, exceptIdx + 12); j++) {
          if (L[j].code.trim() && indentOf(L[j].code) <= ind) break;
          body += L[j].code + "\n";
        }
        const exceptLine = exceptIdx >= 0 ? L[exceptIdx].code : "";
        const writes = /to_csv\(|open\(|\.write\(/.test(body);
        const switches = writes || /\bimport\b|from_pretrained|Model|model\s*=|=\s*\w+\(|load/.test(body) || /ImportError|ModuleNotFoundError|OSError/.test(exceptLine);
        const silent = /^\s*(pass|continue)\s*$/m.test(body);
        push(f("det-try", switches ? "high" : "medium", "Determinism",
          writes ? "try/except around an output write" : switches ? "try/except switches to another method" : "try/except block",
          writes ? "Writing the submission through a fallback path is the placeholder/fallback pattern your notes flag in several top files."
            : switches ? "The except branch loads, imports or builds something different: a fallback path the determinism check rejects."
              : silent ? "A silent except hides failures and creates a second code path." : "Reviewers check try/except blocks for fallbacks that switch methods.",
          "Remove the try/except and call the one intended method directly. Let real errors fail loudly.", "S2 §3.3", n, ln.raw + (exceptLine ? " … " + exceptLine.trim() : "")));
      }

      // ---- env switches / path fallback chains / flags
      const envRead = /os\.environ\.get\(|os\.getenv\(/.test(c) || /os\.environ\[[^\]]+\](?!\s*=)/.test(c);
      if (envRead) {
        const name = (c.match(/(?:environ\.get|getenv)\(\s*["']([^"']+)/) || c.match(/environ\[\s*["']([^"']+)/) || [])[1] || "";
        if (PATHY_ENV.test(name) || /environ\.get\(["'][^"']*(DIR|PATH)/i.test(c))
          push(f("det-path-chain", "medium", "Contract", "Paths resolved through an environment variable",
            "A chain like argv → env var → candidate folders is a second code path. It doesn't change what is trained, but a reviewer may ask about it.",
            "Read Path(sys.argv[1]) and Path(sys.argv[2]) directly.", "S3, S2 §3.3", n, ln.raw));
        else
          push(f("det-env-switch", "high", "Determinism", "Behaviour read from an environment variable",
            "Environment-variable switches (EPOCHS, MODE, DEBUG…) are rejected even when only one path runs.",
            "Replace with a constant at the top of the script.", "S2 §3.3", n, ln.raw));
      } else if (/os\.environ(\.setdefault\(|\[[^\]]+\]\s*=)/.test(c))
        push(f("det-env-set", "info", "Determinism", "Environment variable set",
          "Setting thread counts, PYTHONHASHSEED, CUBLAS config or HF_HOME to constants is fine.",
          "Keep the values constant.", "S2 §3.3", n, ln.raw));
      if (/^\s*(DEBUG|FAST_DEV_RUN|FAST|QUICK|SMOKE(_TEST)?|DRY_RUN|LOCAL|IS_LOCAL|KAGGLE|ON_KAGGLE)\s*=/.test(c) || /^\s*(if|elif)\s+(not\s+)?(DEBUG|FAST|QUICK|SMOKE|SMOKE_TEST|DRY_RUN|LOCAL|IS_LOCAL|ON_KAGGLE)\b/.test(c))
        push(f("det-flag", "high", "Determinism", "Debug or mode flag",
          "Debug and smoke-mode switches are rejected even if they're off in the submitted run.",
          "Delete the flag and keep one fixed path.", "S2 §3.3", n, ln.raw));
      if (/^\s*(if|elif)\s+.*\b[A-Z][A-Z0-9_]*(MODEL|BACKBONE|METHOD|MODE|ARCH|STRATEGY|VARIANT)[A-Z0-9_]*\s*(==|!=|in)\s*/.test(c))
        push(f("det-method-branch", "high", "Determinism", "Branch that chooses between methods",
          "if-branches on a config constant that select a model or method count as unused switches.",
          "Keep only the method you actually use.", "S2 §3.3", n, ln.raw));
      if (/^\s*(if|elif)\b.*(os\.path\.exists|os\.path\.isfile|os\.path\.isdir|\.exists\(\)|\.is_file\(\)|\.is_dir\(\))/.test(c) && !raisesNext(L, i))
        push(f("det-file-branch", "medium", "Determinism", "Behaviour depends on which files exist",
          "Probing for alternative file names creates paths that depend on the environment.",
          "Read the exact file names given in the challenge description and fail loudly if one is missing.", "S2 §3.3", n, ln.raw));
      if (/\bargparse\b|\bclick\.|fire\.Fire/.test(c) && /ArgumentParser|click\.|fire\.Fire/.test(c))
        push(f("io-argparse", "medium", "Contract", "Command-line option parser",
          "The platform passes exactly two positional arguments. Extra required flags exit with code 2, and optional flags read as switches.",
          "Use Path(sys.argv[1]) and Path(sys.argv[2]) directly (or parse_known_args with only those two).", "S3, S2 §3.3", n, ln.raw));
      if (/len\(\s*sys\.argv\s*\)/.test(c))
        push(f("io-argv-default", "info", "Contract", "Local-dev fallback when arguments are missing",
          "Your CLAUDE.md allows a ./dataset/public fallback for local runs. It never triggers on the platform, but keep it a one-liner so it can't read as a switch.",
          "PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/public')", "S3", n, ln.raw));

      // ---- seeds
      if (/cudnn\.benchmark\s*=\s*True/.test(c))
        push(f("det-cudnn-bench", "high", "Determinism", "cudnn.benchmark = True",
          "Autotuning picks kernels per machine and run.", "Set cudnn.benchmark = False and cudnn.deterministic = True.", "S2 §3.4", n, ln.raw));
      if (/use_deterministic_algorithms\(\s*True\s*\)/.test(c))
        push(f("det-strict-mode", "info", "Runtime", "Strict deterministic algorithms (no warn_only)",
          "Raises RuntimeError on ops without a deterministic CUDA kernel, and was measured 4.7× slower on one of your DeBERTa runs.",
          "Prefer use_deterministic_algorithms(True, warn_only=True), and profile one epoch before fixing the work plan.", "", n, ln.raw));
      if (/np\.random\.default_rng\(\s*\)|random\.Random\(\s*\)/.test(c) || (/torch\.Generator\(\s*\)/.test(c) && !/manual_seed/.test(c + " " + nextCode(L, i))))
        push(f("det-unseeded-gen", "high", "Determinism", "Random generator created without a seed", "An unseeded generator draws from OS entropy on every run.",
          "Pass SEED: np.random.default_rng(SEED), torch.Generator().manual_seed(SEED).", "S2 §3.3", n, ln.raw));
      const ctor = SEEDED_CTORS.find((name) => new RegExp("\\b" + name + "\\(").test(c));
      if (ctor && !/^\s*(def|class|from|import)\b/.test(c)) {
        const st = statementAt(L, i);
        if (!/random_state\s*=|\bseed\s*=|random_seed\s*=|\*\*\w+/.test(st))
          push(f("det-no-seed", "high", "Determinism", `${ctor} without a seed`,
            "Every model, split and sampler needs its random_state or seed set, by name.",
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
      if (/(for\s+\w+\s+in|list\(|enumerate\()\s*set\(/.test(c) && !ctx.hashSeed)
        push(f("det-set-order", "low", "Determinism", "Iterating over a set",
          "Set iteration order for strings changes between Python runs unless PYTHONHASHSEED is fixed, which can reorder features or classes.",
          "Use sorted(set(...)), or set os.environ['PYTHONHASHSEED'] = '0' at the very top.", "", n, ln.raw));

      // ---- I/O contract
      const rawNoComment = ln.raw.replace(/#.*$/, "");
      if (/['"](\.\/)?dataset\/public|\/kaggle\/input|\/kaggle\/working|['"](\.\/)?working\/submission\.csv/.test(rawNoComment) && !/len\(\s*sys\.argv/.test(c)) {
        const near = L.slice(Math.max(0, i - 4), i + 2).map((x) => x.code).join(" ");
        if (envRead || /for\s+\w+\s+in\s*[([]/.test(c) || /candidates|roots|search_dirs/.test(c) || /\[[^\]]*,[^\]]*Path\(/.test(c))
          push(f("det-path-chain", "medium", "Contract", "Candidate-folder fallback chain",
            "Searching several folders for the data is a second code path. Your notes record it in several top files; it's harmless to the result but draws questions.",
            "Read Path(sys.argv[1]) directly and fail loudly if a file is missing.", "S3", n, ln.raw));
        else if (/default\s*=|\bor\s+(Path\()?["']|else\s+(Path\()?["']/.test(c) || /sys\.argv|argparse|add_argument/.test(near))
          push(f("io-argv-default", "info", "Contract", "Local-dev default path",
            "A default used only when no argument is given. Fine for local runs.", "Keep the platform path coming from sys.argv.", "S3", n, ln.raw));
        else
          push(f("io-hardcoded-path", "high", "Contract", "Hardcoded data or output path",
            "The platform passes paths as arguments. Hardcoded paths are the old format and fail with FileNotFoundError.",
            "Use Path(sys.argv[1]) / 'train.csv' and Path(sys.argv[2]).", "S3", n, ln.raw));
      }
      if (/(^|[^\w])answers\.csv|['"\/]private\//.test(rawNoComment))
        push(f("io-private", "blocker", "Compliance", "Reads private answers",
          "Private ground truth is never available to solutions; trying to read it is an exploit (the 'Held-out Answer Ingestion' check looks for this).",
          "Remove it. Training labels shipped in public/ (e.g. train_answers.csv) are fine.", "S2 §5.5", n, ln.raw));

      // ---- environment
      if (/\bpip\s+install\b|subprocess\.\w+\(.*pip|os\.system\(|^\s*[!%]\s*(pip|conda)\b|conda\s+install/.test(ln.raw))
        push(f("env-install", "blocker", "Environment", "Installs or shells out",
          "No installs from inside the script, and no approval route for extra packages.",
          "Use only packages already in the Kaggle image.", "S2 §4.1", n, ln.raw));
      else if (/^\s*(!\s*[a-zA-Z]|%%?[a-zA-Z])/.test(ln.raw) && !ln.inDoc && ln.depthBefore === 0 && !(i > 0 && /\\\s*$/.test(L[i - 1].raw)))
        push(f("env-magic", ctx.isNotebook ? "medium" : "high", "Environment", "Notebook magic or shell line",
          "python3 solution.py treats ! and % lines as syntax errors.", "Remove it or replace it with Python.", "S3", n, ln.raw));
      if (/github\.com|raw\.githubusercontent|git\s+clone|torch\.hub\.load\(/.test(rawNoComment))
        push(f("env-github", "blocker", "Environment", "Model or code from GitHub",
          "Installing or loading models through GitHub or other channels is rejected immediately.",
          "Use Hugging Face or timm weights through installed packages.", "S2 §4.3", n, ln.raw));
      if (/requests\.(get|post)\(|urllib\.request|urlretrieve\(|\bwget\b|\bcurl\b|kaggle\s+(datasets|competitions)\s+download/.test(b) ||
          (/load_dataset\(/.test(c) && !/load_dataset\(\s*['"](csv|json|parquet|text|imagefolder)['"]/.test(c)))
        push(f("env-external-data", "high", "Compliance", "Downloads data from outside the challenge",
          "External datasets are banned. Only pretrained backbone weights may come from the internet.",
          "Use only the files under the public directory.", "S2 §5.2.3", n, ln.raw));
      if (/torch\.load\(|joblib\.load\(|pickle\.load\(|\bnp\.load\(|pd\.read_pickle\(|load_state_dict\(/.test(c)) {
        const st = statementAt(L, i);
        const inMemory = /state_dict\(\)|copy\.deepcopy|\bbest_state\b|\bbest_sd\b|\bsnap|\bema\b|\bswa/i.test(st) && /load_state_dict\(/.test(st);
        const fromData = ctx.dataVars.some((v) => new RegExp("\\b" + v + "\\b").test(st)) || /public|PUBLIC|data_dir|DATA_DIR|manifest|row\[|train_dir|test_dir|img|image/.test(st);
        const literalArtifact = /['"][^'"]+\.(pt|pth|bin|npy|npz|pkl|pickle|joblib|safetensors|ckpt)['"]/.test(st);
        const savedInScript = /torch\.save\(|joblib\.dump\(|pickle\.dump\(|np\.save\(|to_pickle\(/.test(ctx.codeText);
        if (!inMemory && !fromData)
          push(f("env-artifact-load", literalArtifact && !savedInScript ? "high" : "info", "Environment",
            literalArtifact && !savedInScript ? "Loads a pre-built artifact" : "Loads a file (check where it comes from)",
            "Every run starts from the raw data. Loading embeddings or weights from a previous run is not allowed.",
            "Make sure the file is produced earlier in the same run, or recompute it.", "S2 §4.7, §5.2.2", n, ln.raw));
      }
      if (/\bopenai\b|\banthropic\b|google\.generativeai|genai\.(Client|GenerativeModel)|ChatCompletion|messages\.create\(/.test(c))
        push(f("env-llm", "blocker", "Compliance", "Calls an LLM API",
          "LLM outputs can't be part of the solution, and calling one is not training a model.",
          "Remove the call.", "S1, S2 §1.1", n, ln.raw));
      const fp = ln.raw.match(/from_pretrained\(\s*["']([^"']+)["']/);
      if (/from_pretrained\(|timm\.create_model\(.*pretrained\s*=\s*True|weights\s*=\s*["']?\w*(DEFAULT|IMAGENET)/i.test(c)) {
        if (ctx.category === "scratch" || R.noPretrained)
          push(f("scr-pretrained", "blocker", "Compliance", "Pretrained weights where the challenge forbids them",
            "From-scratch / 'no pretrained weights' challenges allow nothing pretrained.", "Initialise from scratch.", "S2 §6.5", n, ln.raw));
        else
          push(f("env-pretrained", "info", "Compliance", "Loads pretrained weights" + (fp ? `: ${fp[1]}` : ""),
            "Fine when it's a general-purpose backbone downloaded at run time. It must not be weights you fine-tuned elsewhere. With transformers ≥ 5, pass torch_dtype=torch.float32 when fine-tuning with AMP.",
            "Use public base checkpoints and fine-tune them inside the script.", "S2 §4.2, §5.2.2", n, ln.raw));
      }
      if (/torch\.compile\(/.test(c))
        push(f("rt-compile", "low", "Runtime", "torch.compile",
          "Adds compile time and can introduce run-to-run variation.", "Measure the end-to-end time with and without it.", "", n, ln.raw));
      // decoding the challenge's own data fields (e.g. a *_b64 column) is fine; encoded LITERALS are not
      if (/(^|[^.\w])(exec|eval)\s*\(|marshal\.loads/.test(b) || /(b64decode|b85decode|decompress|codecs\.decode)\(\s*b?["'][A-Za-z0-9+\/=]{40,}/.test(c))
        push(f("src-opaque", "high", "Compliance", "Opaque or generated code",
          "exec/eval and encoded blobs fail the compliance checker with 'source cannot be inspected'.",
          "Write plain, readable code. No encoded data or runtime-generated code.", "S2 §3.1", n, ln.raw));
      if (/json\.loads\(\s*['"]\{/.test(c) && c.length > 200)
        push(f("src-embedded-config", "medium", "Compliance", "Embedded config blob",
          "A frozen JSON config string hides what was tuned and where; reviewers read it as offline-tuned constants.",
          "Write the configuration as plain constants, and search anything tuned inside the script.", "S2 §1.1", n, ln.raw));

      // ---- test-set usage
      const fit = c.match(/\.(fit|fit_transform|partial_fit|adapt)\(\s*([\w.\[\]'"]+)/);
      if (fit && (TEST_NAME.test(fit[2].replace(/^.*\./, "").replace(/\[.*$/, "")) || (/\btest/i.test(fit[2]) && !/train/i.test(fit[2]))))
        push(f("test-fit", "blocker", "Leakage", "Something is fitted on test data",
          "Fitting transforms, vocabularies or models on test inputs adapts the solution to this test set.",
          "Fit on training folds only, and only transform test.", "S2 §5.2.5", n, ln.raw));
      if (/(concat|vstack|hstack|append|cat)\(\s*\[?\s*\(?\s*[\w.]*(train|tr)\w*[\w.\[\]"']*\s*,\s*[\w.]*(test|te)\w*/i.test(c) ||
          /(concat|vstack|hstack|cat)\(\s*\[?\s*\(?\s*[\w.]*(test|te)\w*[\w.\[\]"']*\s*,\s*[\w.]*(train|tr)\w*/i.test(c))
        push(f("test-concat", R.perRow ? "blocker" : "high", "Leakage", "Train and test concatenated",
          "Fine only for identical per-row transforms. Any statistic, vocabulary or fit computed on the combined frame uses test data.",
          "Process train and test separately with transforms fitted on train.", "S2 §5.2.5", n, ln.raw));
      if (/pseudo[_\s-]?label|self[_-]?training|\bsoft[_-]?labels?\b|test[_-]?time[_-]?(adapt|training)|\bTENT\b/i.test(b))
        push(f("test-pseudo", "blocker", "Leakage", "Pseudo-labelling or test-time adaptation",
          "Explicitly prohibited, even without test labels.", "Remove it.", "S2 §5.2.5", n, ln.raw));
      if (/\b(test|x_test|test_df|df_test|te_df|test_pred\w*|pred_test\w*|test_preds?)\b(\[[^\]]+\])*\.(mean|std|median|quantile|describe|value_counts|var|rank|groupby)\(/i.test(c))
        push(f("test-stats", R.perRow ? "blocker" : "high", "Leakage", "Statistic computed over the test set",
          "Test-wide statistics used for features, scaling or post-processing make predictions depend on the rest of the test set.",
          "Use statistics from training data or OOF predictions.", "S2 §5.2.5", n, ln.raw));
      if (/np\.(quantile|percentile)\(\s*\w*(pred|test)/i.test(c) || /len\(\s*(test|x_test|test_df)\w*\s*\)\s*\*\s*0?\.\d+/i.test(c))
        push(f("test-quantile", "high", "Leakage", "Threshold or count derived from the test distribution",
          "Choosing thresholds from test-score quantiles, or assuming a prevalence in test, calibrates to the test set.",
          "Choose thresholds on OOF predictions.", "S2 §5.2.5", n, ln.raw));

      // ---- data
      if (/\bSMOTE\b|\bADASYN\b|\bimblearn\b|make_classification\(|make_regression\(/.test(c) || /back[_-]?translat|\bparaphras\w*\(|\bsynthetic\w*\s*\(|\bsynth_\w+\s*=|generate_(pool|data|samples)\w*\(/i.test(b))
        push(f("data-synthetic", "high", "Compliance", "Synthetic training data",
          "Generating your own synthetic data and training on it is not allowed, unless the description explicitly invites mixing new instances from supplied ingredients (then ask a reviewer).",
          "Use class weights or loss reweighting instead of generated rows.", "S2 §5.2.6", n, ln.raw));
      const nums = (b.match(/-?\b\d+\.?\d*(e-?\d+)?\b/g) || []).length;
      if (nums >= 30 && /[\[{(]/.test(b))
        push(f("data-hardcoded", "high", "Compliance", "Large literal table of numbers",
          "Hardcoded values or labels are removed outright in review.", "Have the model learn the values from data.", "S2 §5.5", n, ln.raw));
      if (/=\s*-?\d+\.\d{9,}\b/.test(c))
        push(f("data-precise-const", "medium", "Compliance", "High-precision constant",
          "A float with many digits usually comes from an offline fit (a calibration or blend weight).",
          "Fit it inside the script on OOF, or explain where it comes from in a comment.", "S2 §1.1", n, ln.raw));

      // ---- correctness: guessed date formats
      if (/\bpd\.to_datetime\(|\bto_datetime\(/.test(c) && !/format\s*=|dayfirst\s*=|unit\s*=/.test(c))
        push(f("data-date-guess", "low", "Correctness", "Date parsing guesses the format",
          "Without format= pandas infers it from the first values; dd/mm and mm/dd dates can be swapped or fail on later rows.",
          "Pass the format the description documents, e.g. pd.to_datetime(col, format=\"%d/%m/%Y\").", "Platform example", n, ln.raw));

      // ---- grey areas
      if (/TfidfVectorizer|CountVectorizer|HashingVectorizer|ngram_range|markov/i.test(b))
        push(f("grey-tfidf", ["nlp", "finetune", "scratch", "retrieval"].includes(ctx.category) ? "medium" : "low", "Grey area", "TF-IDF / n-gram / Markov features",
          ["nlp", "finetune", "scratch", "retrieval"].includes(ctx.category) ? "Reviewer's discretion, depending on how much the solution relies on them."
            : "In a tabular challenge a TF-IDF block on a text column is an ordinary feature (the platform's exemplary tabular solution uses one). The grey area is TF-IDF standing in for the trained model in text challenges.",
          "Keep them as inputs to a trained model, and show the trained model carries most of the signal.", "S2 §5.3.2", n, ln.raw));

      // ---- challenge restrictions
      if (R.cpuOnly && (/["']cuda["']|\.cuda\(\)|device_type\s*=\s*["']cuda|GradScaler\(/.test(c)))
        push(f("rx-cpu", "high", "Restriction", "GPU code in a CPU-only challenge",
          "The description says CPU only, so cuda devices, .cuda() and GradScaler won't exist on the run machine.",
          'Hardcode device = "cpu" and size the fixed plan for the stated cores.', "Challenge text", n, ln.raw));
      if (R.noTTA && /\btta\b|hflip|vflip|torch\.flip\(|np\.flip\(|fliplr|multi[_-]?view|n_views/i.test(b))
        push(f("rx-tta", "high", "Restriction", "Test-time augmentation in a challenge that forbids it",
          "TTA is allowed by default but banned when the description forbids multi-view or test-time augmentation.",
          "Predict each test row once, from a single view.", "Challenge text", n, ln.raw));
      if (R.perRow && /KMeans\(|AgglomerativeClustering|DBSCAN|GaussianMixture/.test(c) && /test/i.test(statementAt(L, i, 4) + nextCode(L, i)))
        push(f("rx-perrow", "blocker", "Restriction", "Pooling across test rows",
          "The description requires each row to be predicted on its own. Clustering or pooling test rows breaks that.",
          "Keep inference row-local.", "Challenge text", n, ln.raw));

      // ---- runtime / version risks
      if (/np\.fromstring\(/.test(c))
        push(f("rt-fromstring", "low", "Runtime", "np.fromstring",
          "Deprecated API. It still parses text with sep= but is fragile across versions.", "Use np.array(s.split(), dtype=int).", "", n, ln.raw));
      if (/\bnp\.(float|int|bool|object)\(|\.ix\[|load_boston\(/.test(c))
        push(f("rt-removed-api", "high", "Runtime", "API removed in current library versions",
          "np.float/np.int, .ix and load_boston no longer exist and will crash.", "Use float/int, .loc, or another dataset.", "", n, ln.raw));
      if (/(early_stopping_rounds|verbose_eval)\s*=/.test(c) && /lgb|lightgbm|LGBM|\.fit\(/.test(statementAt(L, Math.max(0, i - 3), 6)))
        push(f("rt-lgb-api", "medium", "Runtime", "Old LightGBM/XGBoost early-stopping argument",
          "LightGBM 4 removed early_stopping_rounds and verbose_eval from train/fit; XGBoost 2 moved early_stopping_rounds to the constructor.",
          "LightGBM: callbacks=[lgb.early_stopping(N)]. XGBoost: XGBClassifier(early_stopping_rounds=N).", "", n, ln.raw));
      if (/warnings\.(filterwarnings|simplefilter)\(\s*["']ignore/.test(c))
        push(f("rt-warnings", "info", "Runtime", "All warnings silenced",
          "This hides deprecation warnings that predict crashes on a newer image.", "Silence specific categories only.", "", n, ln.raw));

      // ---- custom rules
      (ctx.custom || []).forEach((cr, k) => {
        try {
          if (cr.pattern && new RegExp(cr.pattern, cr.flags || "").test(cr.onRaw ? ln.raw : c))
            push(f("custom-" + k, cr.sev || "high", "Custom", cr.title || "Custom rule", cr.why || "Matches one of your challenge rules.", cr.fix || "", "Your rule", n, ln.raw));
        } catch (_) { /* an invalid pattern is reported once in fileRules */ }
      });
    });
    return out;
  }

  function fileRules(L, ctx, bytes) {
    const out = [];
    const t = ctx.codeText;
    const lineOf = (re) => { const i = L.findIndex((l) => re.test(l.code)); return i >= 0 ? i + 1 : null; };

    if (bytes > SOURCE_LIMIT)
      out.push(f("src-size", "blocker", "Compliance", `Source is ${(bytes / 1000).toFixed(0)} KB (limit 512 KB)`,
        "Sources over 512,000 bytes fail with 'source cannot be inspected'.", "Remove dead code and experiments; never embed data or weights.", "S2 §3.1"));
    else if (bytes > SOURCE_LIMIT * 0.8)
      out.push(f("src-size", "low", "Compliance", `Source is ${(bytes / 1000).toFixed(0)} KB, close to the 512 KB limit`,
        "Very large sources are near the inspection limit.", "Trim dead code.", "S2 §3.1"));

    const positional = /add_argument\(\s*['"][a-z_]\w*['"]/.test(t);
    if (!/sys\.argv\s*\[\s*1\s*\]|sys\.argv\s*\[\s*1\s*:/.test(t) && !positional)
      out.push(f("io-argv-missing", "blocker", "Contract", "Doesn't read the platform's path arguments",
        "The platform runs python3 solution.py <public_dir> <submission_out> and provides the paths only as arguments.",
        "PUBLIC_DIR = Path(sys.argv[1]); SUBMISSION_OUT = Path(sys.argv[2]); SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)", "S3"));
    else if (!positional && !/sys\.argv\s*\[\s*2\s*\]|sys\.argv\s*\[\s*1\s*:/.test(t))
      out.push(f("io-argv-out", "blocker", "Contract", "Output path argument not used",
        "The submission must be written to sys.argv[2].", "SUBMISSION_OUT = Path(sys.argv[2])", "S3"));
    if (ctx.outVar && !/mkdir\(|makedirs\(/.test(t))
      out.push(f("io-mkdir", "medium", "Contract", "Output directory not created",
        "If the parent folder of the output path doesn't exist, to_csv crashes and the credit is consumed.",
        "SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)", "S3"));

    // placeholder / fallback submission writes
    const writeLines = [];
    L.forEach((l, i) => {
      if (/\.to_csv\(|np\.savetxt\(|open\([^)]*,\s*['"]w/.test(l.code) && ((ctx.outVar && l.code.includes(ctx.outVar)) || /submission|sub_path|out_path|submission_out/i.test(l.code))) writeLines.push(i + 1);
    });
    const firstTrain = L.findIndex((l) => TRAINING.test(l.code)) + 1;
    if (writeLines.length >= 2)
      out.push(f("io-placeholder", "high", "Determinism", `Submission written ${writeLines.length} times`,
        "Several writes usually mean a placeholder or fallback file (written first 'to be safe'). The checks flag it and your notes record it in several top files.",
        "Write the submission exactly once, after validation, at the end.", "S2 §3.3, §4.6", writeLines[0]));
    else if (writeLines.length === 1 && firstTrain > 0 && writeLines[0] < firstTrain && /def\s+main|if __name__/.test(t) === false)
      out.push(f("io-placeholder", "medium", "Determinism", "Submission written before any training",
        "A write before training is the placeholder pattern.", "Write once, at the end.", "S2 §4.6", writeLines[0]));

    // dead switches: UPPERCASE constant set to 0/False/None and used in an if
    const consts = {};
    L.forEach((l) => { const m = l.code.match(/^([A-Z][A-Z0-9_]{2,})\s*=\s*(0|False|None|0\.0)\s*$/); if (m) consts[m[1]] = true; });
    const dead = Object.keys(consts).filter((k) => new RegExp("^\\s*(if|elif)\\s+(not\\s+)?" + k + "\\b").test(L.map((l) => l.code).join("\n").replace(/\n/g, "\n")) || L.some((l) => new RegExp("^\\s*(if|elif)\\s+(not\\s+)?" + k + "\\b").test(l.code)));
    if (dead.length)
      out.push(f("det-dead-switch", "high", "Determinism", `Switches set to off: ${dead.slice(0, 6).join(", ")}`,
        "Constants set to 0/False/None that gate if-branches are unused switches. The determinism check rejects them even when the branch never runs.",
        "Delete the switch and the dead branch.", "S2 §3.3", lineOf(new RegExp("^" + dead[0] + "\\s*="))));

    const usesTorch = /\bimport torch\b|\bfrom torch\b/.test(t);
    const usesNpRandom = /np\.random\.(rand|randn|randint|choice|permutation|shuffle|normal|uniform|seed)\(/.test(t);
    if (usesTorch && !/torch\.manual_seed\(|seed_everything|L\.seed_everything|set_seed\(/.test(t)) {
      const localGen = /torch\.Generator\(\)\.manual_seed\(|\.manual_seed\(/.test(t);
      out.push(f("det-torch-seed", localGen ? "high" : "blocker", "Determinism",
        localGen ? "Only a local torch.Generator is seeded" : "torch is used but never seeded",
        localGen ? "Layer initialisation and dropout use torch's global RNG, which stays unseeded." : "Weight initialisation and dropout differ on every run.",
        "torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)", "S2 §3.4", lineOf(/import torch/)));
    }
    if (usesTorch && /nn\.Conv|torchvision|timm/.test(t) && !/cudnn\.deterministic\s*=\s*True/.test(t) && !ctx.restrict.cpuOnly)
      out.push(f("det-cudnn", "medium", "Determinism", "cudnn.deterministic not set", "Convolution kernels can be nondeterministic on GPU.", "torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False", "S2 §3.4"));
    if (usesNpRandom && !/np\.random\.seed\(/.test(t) && !/default_rng\(\s*\w/.test(t))
      out.push(f("det-np-seed", "high", "Determinism", "NumPy random functions used without np.random.seed", "Global NumPy draws differ between runs.", "np.random.seed(SEED) at the top.", "S2 §3.4", lineOf(/np\.random\./)));
    if (/\brandom\.(random|randint|choice|shuffle|sample|uniform)\(/.test(t) && !/\brandom\.seed\(/.test(t))
      out.push(f("det-py-seed", "high", "Determinism", "Python random used without random.seed", "Python's global RNG is unseeded.", "random.seed(SEED)", "S2 §3.4", lineOf(/\brandom\.(random|randint|choice|shuffle|sample|uniform)\(/)));

    if (!TRAINING.test(t))
      out.push(f("ml-no-training", "blocker", "Compliance", "No training step found",
        "Real training or fine-tuning must happen inside the script. Inference-only and rule-based solutions are rejected.",
        "Train or fine-tune a model in the script.", "S2 §1.1, §5.2.1"));
    const hasPretrained = /from_pretrained\(|pretrained\s*=\s*True/.test(t);
    const hasBackward = /\.backward\(|Trainer\(|\.fit\(\s*\w*(loader|dataset|ds)\b|model\.fit\(/i.test(t);
    if (hasPretrained && !hasBackward && TABULAR_MODELS.test(t))
      out.push(f("ft-embed-tabular", ctx.category === "finetune" ? "high" : "medium", "Grey area", "Pretrained embeddings feeding a tabular model, backbone not fine-tuned",
        "Grey area in CV and NLP. In Fine-tuning challenges it's likely rejected because the backbone must be fine-tuned.",
        "Fine-tune the backbone (full or LoRA) and keep the tabular model as a minor member.", "S2 §6.2, §6.4"));
    const images = /cv2\.imread|Image\.open|imread\(|read_image\(|decode_image|load_img\(/.test(t);
    if (images && TABULAR_MODELS.test(t) && !DEEP_MODEL.test(t)) {
      const pixels = /\.flatten\(\)|\.ravel\(\)|reshape\(\s*-1\s*\)|reshape\(\s*\w+\s*,\s*-1\s*\)|reshape\(len\(/.test(t);
      out.push(f(pixels ? "cv-pixels-tabular" : "cv-tabular-only", pixels ? "blocker" : "high", "Compliance",
        pixels ? "Tabular model on raw pixels" : "Images handled only by a tabular model",
        pixels ? "Tabular models on raw pixels are rejected immediately." : "Hand-made image features used only by a tabular model are grey and fragile.",
        "Train or fine-tune a CNN or ViT, and feed any hand features into it as extra inputs.", "S2 §6.2"));
    }
    const ranker = /LGBMRanker|XGBRanker|lambdarank|rank:pairwise|rank:ndcg|BM25|rank_bm25/.test(t);
    if (ranker && !/nn\.Module|Trainer\(|\.backward\(/.test(t))
      out.push(f("rag-offshelf", "high", "Compliance", "Off-the-shelf ranker without a trained deep ranker",
        "Retrieval challenges want a genuinely trained ranking model, usually deep. Hand features plus an off-the-shelf ranking algorithm is called out as not enough.",
        "Make the main ranking signal a trained (cross-)encoder or neural ranker.", "S2 §6.3"));
    const reCalls = (t.match(/\bre\.(sub|findall|search|match|compile|split|fullmatch|finditer)\(/g) || []).length;
    if (reCalls > 8)
      out.push(f("grey-regex", "medium", "Grey area", `Heavy regex use (${reCalls} calls)`,
        "Regex beyond cleaning or deterministic extraction reads as over-reliance on how the data was generated.",
        "Keep regex for cleaning only, let the model learn the patterns, and run the strip-the-model test.", "S2 §5.3.1"));
    if (ctx.outVar && !/assert|raise\b|problems|validate/i.test(t))
      out.push(f("io-validate", "low", "Contract", "No output validation before writing",
        "A malformed output ranks below a score of 0 and the credit is consumed.",
        "Assert columns, row count, ids matching the sample, and finite non-empty values before to_csv.", "S2 §4.6"));
    const firstCode = L.findIndex((l) => l.code.trim() && !/^\s*""/.test(l.code));
    const head = L.slice(0, Math.max(0, firstCode) + 40).map((l) => l.raw).join("\n");
    if (L.length > 150 && !/requirement|compliance|rule[s]?\b|R\d\b|guidebook|description says/i.test(head))
      out.push(f("doc-reqmap", "info", "Compliance", "No requirements map at the top",
        "In your records, a docstring that quotes each challenge rule next to the code satisfying it fixed a Prompt Compliance 'Check Error', and the best-documented top files do this.",
        "Add a short requirements map: rule → where the code satisfies it.", "S2 §3.1"));
    (ctx.custom || []).forEach((cr, k) => { try { new RegExp(cr.pattern, cr.flags || ""); } catch (e) { out.push(f("custom-bad-" + k, "info", "Custom", `Custom rule "${cr.title || k}" has an invalid pattern`, e.message, "Fix the regular expression.", "Your rule")); } });

    // imports
    const imports = new Set();
    L.forEach((l) => {
      const m = l.code.match(/^\s*import\s+([\w.]+(\s*,\s*[\w.]+)*)/);
      const m2 = l.code.match(/^\s*from\s+([\w.]+)\s+import\b/);
      if (m) m[1].split(",").forEach((s) => imports.add(s.trim().split(".")[0].split(" ")[0]));
      if (m2 && !m2[1].startsWith(".")) imports.add(m2[1].split(".")[0]);
    });
    const allowed = new Set([...E.ALLOWED_PACKAGES, ...E.CORE_PACKAGES].map((s) => s.toLowerCase()));
    const unknown = [], verifyList = [];
    [...imports].sort().forEach((mod) => {
      if (E.STDLIB.includes(mod)) return;
      const dist = (E.IMPORT_ALIASES[mod] || mod).toLowerCase();
      if (allowed.has(dist) || allowed.has(mod.toLowerCase()) || allowed.has(dist.replace(/_/g, "-"))) return;
      if (E.LIKELY_IN_IMAGE.map((s) => s.toLowerCase()).includes(dist)) verifyList.push(mod);
      else unknown.push(mod);
    });
    if (unknown.length)
      out.push(f("env-unknown-import", "high", "Environment", "Imports not on the supplied package lists: " + unknown.join(", "),
        "A missing package fails the script run and consumes the credit (peft, for example, couldn't be imported in the image because of a torchao clash).",
        "Confirm each one exists in the Kaggle Docker image or replace it.", "S2 §4.1, S4"));
    if (verifyList.length)
      out.push(f("env-verify-import", "low", "Environment", "Imports to confirm in the image: " + verifyList.join(", "),
        "Usually present in the Kaggle image, but not on the lists you supplied.", "Check the image's pip list once and record the result.", "S4"));
    if (L.length > 1500)
      out.push(f("cx-long", "info", "Complexity", `${L.length} lines`,
        "Very long scripts are harder for reviewers and for the compliance LLM to read.", "Remove dead code and unused experiments; keep one fixed path.", "S2 §3.3"));
    return { findings: out, imports: [...imports].sort() };
  }

  function audit(text, opts = {}) {
    let isNotebook = false;
    let src = String(text || "");
    if (/^\s*\{/.test(src)) { const nb = fromNotebook(src); if (nb !== null) { src = nb; isNotebook = true; } }
    const bytes = typeof TextEncoder !== "undefined" ? new TextEncoder().encode(src).length : src.length;
    const L = prepare(src);
    const codeText = L.map((l) => l.code).join("\n");
    const outM = codeText.match(/(\w+)\s*=\s*(Path\(\s*)?sys\.argv\s*\[\s*2\s*\]/) || codeText.match(/(\w+)\s*=\s*sys\.argv\s*\[\s*2\s*\]\s+if/);
    const dataVars = [];
    const inM = codeText.match(/(\w+)\s*=\s*(Path\(\s*)?sys\.argv\s*\[\s*1\s*\]/); if (inM) dataVars.push(inM[1]);
    const ctx = {
      category: opts.category || "general", isNotebook, text: src, codeText, outVar: outM ? outM[1] : null, dataVars,
      restrict: opts.restrict || {}, custom: opts.custom || [], hashSeed: /PYTHONHASHSEED/.test(codeText),
    };
    const a = lineRules(L, ctx);
    const b = fileRules(L, ctx, bytes);
    const merged = [];
    const key = new Map();
    for (const x of [...a, ...b.findings]) {
      const k = x.id + "|" + x.title;
      if (key.has(k)) { const m = key.get(k); m.count++; m.lines.push(x.line); continue; }
      const m = { ...x, count: 1, lines: [x.line] };
      key.set(k, m); merged.push(m);
    }
    merged.sort((p, q) => SEV_ORDER[p.sev] - SEV_ORDER[q.sev] || (p.line || 0) - (q.line || 0));
    const counts = { blocker: 0, high: 0, medium: 0, low: 0, info: 0 };
    merged.forEach((m) => counts[m.sev]++);
    const verdict = counts.blocker ? "blocked" : counts.high ? "at-risk" : counts.medium ? "review" : "clean";
    const lineSev = {};
    merged.forEach((m) => m.lines.forEach((ln) => { if (ln && (lineSev[ln] === undefined || SEV_ORDER[m.sev] < SEV_ORDER[lineSev[ln]])) lineSev[ln] = m.sev; }));
    return { findings: merged, counts, verdict, imports: b.imports, lines: L.length, bytes, isNotebook, category: ctx.category, lineSev, source: src };
  }

  /* Before/after: which findings were fixed, which are new, which remain. */
  function diff(prev, cur) {
    const k = (x) => x.id + "|" + x.title.replace(/\d+/g, "#");
    const P = new Map(prev.findings.map((x) => [k(x), x])), C = new Map(cur.findings.map((x) => [k(x), x]));
    return {
      fixed: [...P.keys()].filter((x) => !C.has(x)).map((x) => P.get(x)),
      added: [...C.keys()].filter((x) => !P.has(x)).map((x) => C.get(x)),
      remaining: [...C.keys()].filter((x) => P.has(x)).map((x) => C.get(x)),
    };
  }

  E.auditor = { audit, prepare, diff, SEV_ORDER, SOURCE_LIMIT };
})();
