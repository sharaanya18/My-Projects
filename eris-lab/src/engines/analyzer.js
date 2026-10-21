/* Challenge analyzer. Reads a pasted problem description and extracts the
 * header fields (Domain / Difficulty / Scoring / Compute), metric, hardware,
 * restrictions, submission format and hidden-split hints, then maps them to
 * platform rules, a validation plan and candidate strategies. Every inference
 * keeps the phrase that triggered it. Calibrated on ~30 real Eris descriptions. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const CATS = [
    { id: "scratch", name: "From-scratch", re: [/from[\s-]scratch/i, /no pre-?trained (weights|models)/i, /pre-?trained (models?|weights) (are|is) not (allowed|permitted)/i, /must not use (any )?pre-?trained/i, /no downloaded (weights|checkpoints)/i, /downloaded weights\/checkpoints/i] },
    { id: "finetune", name: "Fine-tuning", re: [/fine[\s-]?tun(e|ing) (challenge|category|task)/i, /category:\s*fine[\s-]?tun/i, /must fine[\s-]?tune/i] },
    { id: "retrieval", name: "Retrieval / ranking", re: [/\bretriev/i, /\brerank/i, /\bRAG\b/, /candidate (list|set|passages|documents)/i, /\bMRR\b|NDCG|MAP@|recall@/i, /\brank (the|each|all) (candidate|passage|document)/i] },
    { id: "structured", name: "Structured prediction / matching", re: [/one-to-one|bipartite|permutation|assignment|Hungarian/i, /\bmatch(ing)? (each|every)\b/i, /partition|clusters? of|group the (words|items)/i, /\bspans?\b|named entit|\bNER\b|BIO tags?/i] },
    { id: "cv", name: "Computer vision", re: [/\bimages?\b/i, /\bpixels?\b/i, /segmentation/i, /object detection|bounding box/i, /\.png|\.jpg|\.jpeg/i, /\bchart image|rendered (chart|image)/i] },
    { id: "seqgen", name: "Sequence generation", re: [/translat(e|ion)|transliterat/i, /summari[sz]/i, /\bgenerat(e|ion)\b.{0,30}\b(text|string|note|sequence)/i, /seq2seq|sequence-to-sequence/i, /\bBLEU\b|chrF|ROUGE/i, /caption/i] },
    { id: "signal", name: "Time series / signal", re: [/time[\s-]series|forecast/i, /waveform|spirogram|sensor|accelerometer|motion[\s-]capture|marker trajectories/i, /\bsignal\b|spectrogram|audio/i, /ring widths?|series of (values|measurements)/i] },
    { id: "recsys", name: "Recommendation", re: [/recommend/i, /\busers?\b.{0,40}\bitems?\b/i, /click|purchase|listening (session|history)|co-listening/i, /collaborative filtering/i] },
    { id: "biochem", name: "Biology / chemistry", re: [/\bSMILES\b/, /molecul/i, /protein/i, /\bDNA\b|\bRNA\b|genom/i, /chemi(cal|stry)/i, /amino acid/i] },
    { id: "nlp", name: "NLP", re: [/\btext\b/i, /sentences?/i, /\btokens?\b/i, /\bNLP\b/, /documents?/i, /\bcomments?\b|\bposts?\b|\bdialect|\bwords?\b/i, /\blanguage\b/i] },
  ];

  const DOMAIN_MAP = [[/tabular/i, "general"], [/from scratch/i, "scratch"], [/fine[\s-]?tun/i, "finetune"], [/rag|retriev|rank/i, "retrieval"], [/vision|\bcv\b|image/i, "cv"], [/bio|chem/i, "biochem"], [/nlp|text|language/i, "nlp"]];

  const METRIC_RE = [
    ["tiered", /tiered|0\.5 (credit|for a|if)|partial credit|skeleton/i],
    ["setf1", /set[\s-]F1|mean set F1|set overlap/i],
    ["rps", /ranked probability score|\bRPS\b/],
    ["kendall", /kendall/i],
    ["ap", /average precision|average_precision|\bPR[\s-]AUC\b/i],
    ["auc", /\bAUC\b|ROC[\s-]AUC|area under the (ROC )?curve/i],
    ["logloss", /log[\s-]?loss|cross[\s-]entropy|negative log[\s-]likelihood|-\s*ln\(|−\s*ln\(|-log\(/i],
    ["brier", /brier/i],
    ["qwk", /quadratic weighted kappa|\bQWK\b|cohen'?s kappa/i],
    ["mrr", /\bMRR\b|mean reciprocal rank/i],
    ["ndcg", /NDCG/i],
    ["map", /\bMAP@\d+|mean average precision@|recall@\d+|precision@\d+/i],
    ["f1", /\bF1\b|F-?score|F-?beta/i],
    ["maelog", /MAE\)?\s+in log[\s-]space|mean absolute (log|logarithmic) error|\bMALE\b|absolute error (in|of|on) (the )?log|\|\s*log\(\s*y_?i?\s*\+\s*1\s*\)\s*-\s*log/i],
    ["rmsle", /\bRMSLE\b|root mean squared log/i],
    ["rmse", /\bRMSE\b|root mean squared? error/i],
    ["mae", /\bMAE\b|mean absolute error|normali[sz]ed L1/i],
    ["spearman", /spearman|pearson|correlation coefficient/i],
    ["mapdet", /\bmAP\b(?!@)|COCO/],
    ["bleu", /\bBLEU\b|chrF|ROUGE/i],
    ["iou", /\bIoU\b|\bDice\b|jaccard/i],
    ["accuracy", /\baccuracy\b/i],
  ];

  const BOILERPLATE = [
    [/no internet( access)?/i, "'No internet' is treated as boilerplate. Downloading pretrained weights through allowed packages is fine (§4.2, §5.4), unless the description separately bans downloaded weights."],
    [/(fully )?rule[\s-]based (solutions? )?(are|is) (ok|okay|allowed|fine|acceptable)/i, "Rule-based permissions are boilerplate. Real ML training is still required (§5.2.1, §5.4)."],
    [/decoding tricks (are )?(fine|ok|okay|allowed)/i, "'Decoding tricks are fine' is boilerplate. The model must still be trained (§5.4)."],
  ];

  const ENTITY = "users?|customers?|patients?|participants?|people|persons?|walkers?|subjects?|quer(?:y|ies)|languages?|language pairs?|directions?|domains?|sources?|source systems?|sites?|documents?|headwords?|authors?|annotators?|speakers?|stores?|categories|plants?|machines?|devices?|hospitals?|regions?|cities|countries|schools?|products?|items?|sessions?|groups?|families|varieties|pools?|cells?|comment groups?|threads?|books?|cores?|trees?|series|charts?";
  const GROUP_HINTS = [
    [/\b(user|customer|client)_?id\b|\busers?\b/i, "user"], [/\bpatients?|subject_?id|\bwalkers?\b|\bpeople\b/i, "person"],
    [/\bquery_?id|\bquer(y|ies)\b/i, "query"], [/\bsessions?\b/i, "session"],
    [/\b(store|site|hospital|location|region|plant|source system|cell)s?\b/i, "site / source"],
    [/\b(author|annotator|speaker|writer)s?\b/i, "author"], [/\b(document|article|book|thread)s?\b/i, "document"],
    [/\blanguages?\b|language pair|direction/i, "language / direction"], [/\b(famil(y|ies)|variet(y|ies)|pools?)\b/i, "family / pool"],
    [/\bheadwords?\b/i, "headword"], [/\b(series|cores?|trees?)\b/i, "series"],
  ];

  function evidence(text, re) {
    const m = text.match(re);
    if (!m) return null;
    let s = Math.max(0, m.index - 40);
    if (s > 0) { const sp = text.indexOf(" ", s); if (sp > -1 && sp < m.index) s = sp + 1; }
    let e = Math.min(text.length, m.index + m[0].length + 40);
    if (e < text.length) { const sp = text.lastIndexOf(" ", e); if (sp > m.index + m[0].length) e = sp; }
    return (s > 0 ? "…" : "") + text.slice(s, e).replace(/\s+/g, " ").trim() + (e < text.length ? "…" : "");
  }
  const reEsc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

  function parseSubmissionTable(text) {
    const sec = text.split(/\n#+\s*submission/i)[1] || text;
    const cols = [];
    sec.split("\n").forEach((l) => {
      const m = l.match(/^\s*\|\s*`?([A-Za-z_][\w ]*?)`?\s*\|\s*([\w ]+?)\s*\|/);
      if (m && !/^(column|-+)$/i.test(m[1].trim()) && !/^-+$/.test(m[2])) cols.push({ name: m[1].trim(), type: m[2].trim() });
    });
    if (!cols.length) {
      const m = text.match(/(?:sample_submission\.csv|submission)[^\n]{0,60}?(?:has|columns?|:)\s*:?\s*`?([a-z_]\w*(?:\s*,\s*[a-z_]\w*){1,8})`?/i);
      if (m) m[1].split(",").forEach((c) => cols.push({ name: c.trim(), type: "" }));
    }
    return cols.slice(0, 12);
  }

  function sentencesWith(text, re, max = 6) {
    return text.replace(/\s+/g, " ").split(/(?<=[.;])\s+(?=[A-Z*•-])/).filter((s) => re.test(s)).slice(0, max).map((s) => s.trim().slice(0, 260));
  }

  /* Rubric criteria as the challenge page lists them: a REQUIRED / RECOMMENDED
   * badge, an optional category line, the criterion, then its explanation. */
  const RUBRIC_CATS = /^(Data Handling|Agent Behaviou?r|Modell?ing|Feature Engineering|Evaluation|Validation|Code Quality|Reproducibility|Submission|Documentation|Training|Inference|Compliance|Preprocessing)\b\s*/i;
  function parseRubric(text) {
    const parts = String(text).split(/(?:^|\n|\s)(REQUIRED|RECOMMENDED|OPTIONAL)(?=\s)/);
    const out = [];
    for (let i = 1; i < parts.length; i += 2) {
      let body = parts[i + 1] || "";
      // stop at the next all-caps section heading (e.g. "SOLUTION CODE")
      const stop = body.search(/\n\s*[A-Z][A-Z ()0-9]{5,}\s*\n/);
      if (stop > -1) body = body.slice(0, stop);
      body = body.trim();
      let category = "";
      const cm = body.match(RUBRIC_CATS);
      if (cm) { category = cm[1]; body = body.slice(cm[0].length).trim(); }
      const lines = body.split(/\n+/).map((l) => l.trim()).filter(Boolean);
      if (!lines.length) continue;
      let crit = lines[0];
      let detail = lines.slice(1).join(" ");
      if (lines.length === 1) { const m = crit.match(/^(.+?[.!?])\s+(.+)$/); if (m && m[1].length > 30) { crit = m[1]; detail = m[2]; } }
      if (crit.length < 12) continue;
      out.push({ level: parts[i].toLowerCase(), category, text: crit.slice(0, 400), detail: detail.slice(0, 400) });
    }
    return out.slice(0, 40);
  }

  /* What a pasted grading script enforces. */
  function parseGrader(t) {
    const g = [];
    const add = (re, fn) => { const m = t.match(re); if (m) g.push({ text: fn(m), evidence: evidence(t, re) }); };
    let cols = [];
    const setM = t.match(/set\(\s*submission\.columns\s*\)\s*!=\s*\{([^}]+)\}/);
    if (setM) cols = setM[1].split(",").map((c) => c.trim().replace(/^['"]|['"]$/g, "")).filter(Boolean);
    if (setM) g.push({ text: `Columns must be exactly {${cols.join(", ")}}: no index column, nothing extra.`, evidence: evidence(t, /set\(\s*submission\.columns\s*\)/) });
    add(/len\(\s*submission\s*\)\s*!=\s*len\(\s*answers\s*\)/, () => "Row count must equal the hidden answers (one row per test id).");
    add(/\(\s*submission\[\s*['"](\w+)['"]\s*\]\s*>=\s*0\s*\)/, (m) => `Every ${m[1]} value must be ≥ 0, or the grader raises. Clip after any inverse transform.`);
    add(/sort_values\(\s*by\s*=\s*['"](\w+)['"]/, (m) => `The grader sorts both files by ${m[1]}, so row order doesn't matter, but the ${m[1]} values must match test.csv exactly.`);
    add(/min\(\s*\w+\s*,\s*(\d+(?:\.\d+)?)\s*\)/, (m) => `The score is capped at ${m[1]}.`);
    if (/def grade\(/.test(t) && /raise|assert/.test(t)) g.push({ text: "A failed check raises inside grade(): the submission errors instead of scoring.", evidence: evidence(t, /raise Exception|assert /) });
    if (/def grade\(/.test(t) && !/isfinite|isnan|isna\(|isnull|notna/.test(t)) g.push({ text: "No NaN or infinity check in the grading code: assert finite predictions yourself, or one NaN can wreck the whole score.", evidence: "", prov: "hypothesis" });
    return { items: g, columns: cols };
  }

  /* How a pasted prepare script builds the hidden test. */
  function parsePrepare(t) {
    if (!/def prepare\(|train_test_split|GroupShuffleSplit|answers\.csv/.test(t)) return null;
    const testSize = (t.match(/test_size\s*=\s*(0?\.\d+|\d+)/) || [])[1];
    const seedM = t.match(/random_state\s*=\s*(\d+)/);
    const grouped = t.match(/GroupShuffleSplit|GroupKFold|groups\s*=\s*([\w.\[\]'"]+)/);
    const temporal = /sort_values\([^)]*(date|time|year|timestamp)/i.test(t) && /\.(iloc|head|tail)\[?\(?/.test(t);
    const stratify = (t.match(/stratify\s*=\s*([\w.\[\]'"]+)/) || [])[1];
    const filters = (t.match(/^\s*\w+\s*=\s*\w+\[\s*\w+[.\[][^\n]{0,80}==[^\n]{0,60}\]/gm) || []).map((l) => l.trim()).slice(0, 3);
    const kind = grouped ? "group" : temporal ? "temporal" : /train_test_split/.test(t) ? "random" : "unknown";
    return { kind, testSize: testSize ? Number(testSize) : null, seed: seedM ? Number(seedM[1]) : null, stratify: stratify || null, groupBy: grouped && grouped[1] ? grouped[1] : null, filters,
      evidence: evidence(t, /train_test_split\([^)]*\)|GroupShuffleSplit\([^)]*\)|sort_values\([^)]*\)/) };
  }

  /* Preprocessing the description asks for, one note per documented quirk. */
  const PREP = [
    [/entirely null|100\s*% (null|missing)|all[\s-]null|completely empty/i, "Drop the columns the description says are entirely null, and say so in a comment."],
    [/dd\/mm\/yyyy|DD-MM-YYYY|day\/month\/year/i, "Parse the dates with an explicit format (format=\"%d/%m/%Y\"). pandas' guess can swap day and month."],
    [/mm\/dd\/yyyy|month\/day\/year/i, "Parse the dates with an explicit format (format=\"%m/%d/%Y\")."],
    [/pipe[\s-]separated|\|-separated|separated by (a )?(pipe|\|)|concatenated categoricals?/i, "Split the pipe-separated columns and multi-hot encode them, fitted on training rows. Encoding the raw string treats every combination as its own category."],
    [/right[\s-]skewed|long[\s-]tail|heavy[\s-]tailed|skewed (target|distribution)/i, "Skewed target: model log1p(target), invert with expm1, and clip at the valid range."],
    [/free[\s-]text|requiring NLP|text (column|feature)s?\b/i, "Free-text column: TF-IDF or embeddings fitted on the training fold only, as one feature block next to the tabular features."],
    [/could leak|leak(age|y)? (the )?target|were removed to prevent|leakage columns?/i, "The description mentions target-leaking columns. Drop any that are present, defensively and visibly, with a comment saying why."],
    [/\b(free text|title)\b[^.\n]{0,40}\bstrong signal/i, "The title or text is called a strong signal: give it a real text block, not just its length."],
  ];

  function analyze(text, opts = {}) {
    const t = String(text || "");

    // ---- structured header fields (Domain · Difficulty · Scoring · Compute)
    const hdr = {
      domain: (t.match(/Domain:\s*([^·|\n]+)/i) || [])[1],
      difficulty: (t.match(/Difficulty:\s*([^·|\n]+)/i) || [])[1],
      scoring: (t.match(/Scoring:\s*([^·|\n]+)/i) || [])[1],
      compute: (t.match(/Compute:\s*([^·|\n]+)/i) || [])[1],
    };
    Object.keys(hdr).forEach((k) => { if (hdr[k]) hdr[k] = hdr[k].trim().replace(/[.\s]+$/, ""); });
    // a challenge config block (name / difficulty / domain / grade: direction, minimum, maximum)
    const config = {
      direction: (t.match(/direction:\s*(minimi[sz]e|maximi[sz]e)/i) || [])[1] || null,
      minimum: (t.match(/\bminimum:\s*(-?\d+(?:\.\d+)?)/i) || [])[1] || null,
      maximum: (t.match(/\bmaximum:\s*(-?\d+(?:\.\d+)?)/i) || [])[1] || null,
    };

    // ---- category / task
    const scores = CATS.map((c) => {
      const hits = c.re.map((re) => evidence(t, re)).filter(Boolean);
      return { id: c.id, name: c.name, score: hits.length / c.re.length, hits };
    });
    let category = opts.category && opts.category !== "auto" ? opts.category : null;
    let categorySource = category ? "set manually" : "";
    if (!category && hdr.domain) {
      const hit = DOMAIN_MAP.find(([re]) => re.test(hdr.domain));
      if (hit) { category = hit[1]; categorySource = `Domain: ${hdr.domain}`; }
    }
    const strict = scores.find((s) => (s.id === "scratch" || s.id === "finetune") && s.hits.length);
    if (!category && strict) { category = strict.id; categorySource = strict.hits[0]; }
    const ranked = scores.filter((s) => s.hits.length && s.id !== "scratch" && s.id !== "finetune").sort((a, b) => b.score - a.score || b.hits.length - a.hits.length);
    const tabularHint = (hdr.domain && /tabular/i.test(hdr.domain)) || /\btabular (regression|classification|problem|task|dataset)\b/i.test(t);
    let task = ranked[0] || { id: "tabular", name: "Tabular", hits: [] };
    // a tabular challenge with one text column is still tabular
    if (tabularHint && ["nlp", "structured", "recsys", "signal"].includes(task.id)) task = { id: "tabular", name: "Tabular", hits: [evidence(t, /tabular/i)] };
    if (!category) {
      category = ["retrieval", "cv", "biochem"].includes(task.id) ? task.id : ["nlp", "seqgen"].includes(task.id) ? "nlp" : "general";
      categorySource = task.hits[0] || "no strong domain words; defaulted";
    }

    // the free-text note is only about a text column inside a tabular problem
    const prep = PREP.filter(([, note]) => !note.startsWith("Free-text") || task.id === "tabular").map(([re, note]) => (evidence(t, re) ? { text: note, evidence: evidence(t, re) } : null)).filter(Boolean);

    // ---- metric and direction
    let metric = null;
    // the evaluation section: a "## Evaluation" heading, a bare "Metric"/"Scoring" line, or a "Metric:" sentence
    const evalSec = (t.match(/(^|\n)\s*(#+\s*)?(evaluation|metric|scoring|how (it is|you are) scored)\s*:?\s*\n[\s\S]{0,2500}/i) || [])[0]
      || (t.match(/\n\s*(metric|evaluation)\s*:[\s\S]{0,1500}/i) || [])[0] || (t.match(/\b(metric|scored)\b[\s\S]{0,1500}/i) || [])[0] || t;
    const compositeEval = /weight \d?\.\d+|\bskill\b|adjusted Rand|R-precision|group means|weighted mean of the terms|row score\s*=|score\s*=\s*[\d.]+\s*[×x*]/i.test(evalSec);
    const SPECIFIC = ["tiered", "setf1", "rps", "kendall"];
    for (const [id, re] of METRIC_RE) {
      if (compositeEval && !SPECIFIC.includes(id)) continue;
      const ev = evidence(evalSec, re) || (evalSec !== t ? null : evidence(t, re));
      if (ev) { metric = { ...E.METRICS.find((m) => m.id === id), evidence: ev }; break; }
    }
    if (!metric && compositeEval) metric = { ...E.METRICS.find((m) => m.id === "custom"), evidence: evidence(evalSec, /weight \d?\.\d+|\bskill\b|adjusted Rand|R-precision|group means|weighted mean|row score|score\s*=/i) };
    if (!metric) for (const [id, re] of METRIC_RE) { const ev = evidence(t, re); if (ev) { metric = { ...E.METRICS.find((m) => m.id === id), evidence: ev }; break; } }
    const composite = /score\s*=\s*[\d.]+\s*[×x*]|=\s*mean over|\bweighted (sum|combination|mean)\b|row score/i.test(t);
    if ((!metric || composite) && /score\s*=|=\s*mean over|row score/i.test(t)) metric = { ...E.METRICS.find((m) => m.id === "custom"), evidence: evidence(t, /score\s*=|=\s*mean over|row score/i) };
    if (opts.metric) metric = { ...(E.METRICS.find((m) => m.id === opts.metric) || {}), evidence: "set manually" };
    const dirText = hdr.scoring || (t.match(/\b(lower|higher) is better\b/i) || [])[0] || (config.direction ? (/minimi/i.test(config.direction) ? "lower" : "higher") : "");
    if (metric && /lower/i.test(dirText)) metric.dir = "min";
    else if (metric && /higher/i.test(dirText)) metric.dir = "max";
    const customFormula = composite || /def evaluate|formula|\bmean over\b/i.test(t);

    // ---- hardware and runtime
    const cpuOnly = /\bCPU[\s-]only\b|\bno GPU\b|on CPU\b|CPU within|Compute:\s*CPU/i.test(t) && !/Compute:\s*[^\n·]*GPU/i.test(t);
    const cores = (t.match(/(\d+)\s*(?:CPU\s*)?cores/i) || [])[1];
    const ram = (t.match(/(\d+(?:\.\d+)?)\s*(GiB|GB) (?:of )?RAM/i) || t.match(/(\d+(?:\.\d+)?)\s*(GiB|GB)\b/i) || [])[1];
    const gpu = (t.match(/\b(A10G|A100|T4|V100|L4|H100)\b/) || [])[1] || (/Compute:\s*[^\n·]*GPU/i.test(t) ? "GPU" : null);
    const hardware = cpuOnly ? { kind: "cpu", cores: cores ? Number(cores) : null, ramGb: ram ? Number(ram) : null, evidence: evidence(t, /\bCPU[\s-]only\b|\bno GPU\b|on CPU\b|CPU within|Compute:\s*CPU/i) }
      : gpu ? { kind: "gpu", name: gpu, evidence: evidence(t, new RegExp(reEsc(gpu))) } : { kind: "default", name: "A10G (platform default)", evidence: "" };
    const runtimeM = t.match(/(?:runtime|time limit|must (?:complete|finish|run)|within|under|no more than|at most|budget)[^.\n]{0,40}?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b/i);
    let runtimeLimit = null;
    if (runtimeM) {
      const v = parseFloat(runtimeM[1]);
      runtimeLimit = { minutes: /^h/i.test(runtimeM[2]) ? v * 60 : v, evidence: evidence(t, new RegExp(reEsc(runtimeM[0]))) };
    }

    // ---- restrictions
    const pretrainedAllowed = /(may|can) (be )?(load|use)[a-z]*[^.\n]{0,40}pre-?trained|general-purpose pre-?trained (HF|models|weights)[^.\n]{0,20}(may|can|allowed)/i.test(t);
    const noPretrained = category === "scratch" || (/no (downloaded |pretrained |external )?(weights|checkpoints)|downloaded weights|no pretrained|without pretrained/i.test(t) && !pretrainedAllowed);
    const perRow = /predict each \w+ on its own|do not pool predictions|no (voting|per-batch calibration)|each (row|item|test \w+) (must be|is) predicted (independently|on its own)|row[- ]local|independently of (the )?other (test )?rows/i.test(t);
    const noTTA = /no test[\s-]time augmentation|no TTA\b|without test[\s-]time|no multi[\s-]view|test[\s-]time augmentation (is )?(not allowed|forbidden|banned)/i.test(t);
    const restrict = { cpuOnly, noPretrained, perRow, noTTA };
    const restrictions = [];
    if (cpuOnly) restrictions.push({ text: `CPU only${cores ? `: ${cores} cores` : ""}${ram ? `, ${ram} GB RAM` : ""}. Hardcode device="cpu"; the A10G/AMP advice doesn't apply.`, evidence: hardware.evidence });
    if (noPretrained) restrictions.push({ text: "No pretrained weights or downloaded checkpoints: train everything on the supplied data.", evidence: evidence(t, /from[\s-]scratch|no (downloaded |pretrained |external )?(weights|checkpoints)|downloaded weights|no pretrained/i) });
    if (perRow) restrictions.push({ text: "Predict each row on its own: no pooling, voting, clustering or calibration across test rows.", evidence: evidence(t, /predict each \w+ on its own|do not pool predictions|no (voting|per-batch calibration)|row[- ]local|independently/i) });
    if (noTTA) restrictions.push({ text: "No test-time augmentation or multi-view averaging.", evidence: evidence(t, /no test[\s-]time augmentation|no TTA|without test[\s-]time|no multi[\s-]view/i) });
    const sizeM = t.match(/(\d+(?:\.\d+)?)\s*([MB])\s*(parameters|params)/i);
    if (sizeM) restrictions.push({ text: `Model size limit: ${sizeM[1]}${sizeM[2].toUpperCase()} parameters`, evidence: evidence(t, new RegExp(reEsc(sizeM[0]))) });
    const onlyM = t.match(/(?:only|must use|restricted to)\s+([\w\- ]{3,40}?)\s+models?\b/i);
    if (onlyM) restrictions.push({ text: `Model restriction: only ${onlyM[1]} models`, evidence: evidence(t, new RegExp(reEsc(onlyM[0]))) });
    const notUse = sentencesWith(t, /\b(do not|don't|may not|must not|no) (use|consult|pool|infer|introduce|identify)|what not to use/i, 6);
    const rejectIf = sentencesWith(t, /rejected outright|is rejected|scores? (0|zero) for|invalid (and|submission)|error rather than a score/i, 4);

    const boilerplate = BOILERPLATE.map(([re, note]) => (evidence(t, re) ? { evidence: evidence(t, re), note } : null)).filter(Boolean);

    // ---- validation hints
    const foldsProvided = /folds\.csv|fold column|provided folds/i.test(t) ? evidence(t, /folds\.csv|fold column|provided folds/i) : null;
    const baseM = t.match(/(?:ai[\s-]?baseline|baseline to beat|baseline)[^.\n]{0,160}?[:=]?\s*(?:AP\s*)?(\d?\.\d{2,4})/i);
    const aiBaseline = baseM ? { value: Number(baseM[1]), evidence: evidence(t, new RegExp(reEsc(baseM[0]))) } : null;
    // a reference solution's reported score ("... achieving a MAE of 1.061")
    const refM = t.match(/achiev(?:es|ing|ed)\s+(?:an?\s+)?(?:[\w()-]+\s+){0,4}?(?:of|=|:)\s*(\d+(?:\.\d+)?)/i) || t.match(/(?:reference|exemplary) (?:solution )?scor\w*[^.\n\d]{0,30}(\d+\.\d+)/i);
    const reference = refM ? { value: Number(refM[1]), evidence: evidence(t, new RegExp(reEsc(refM[0]))) } : null;
    const rubric = parseRubric(t);
    const grader = parseGrader(t);
    const prepare = parsePrepare(t);

    const chanceM = t.match(/(?:chance|random|constant|evenly)[^.\n]{0,80}?(?:scores?|=|:)\s*(?:approximately\s*)?(\d?\.\d+|ln\(\d+\)\s*=\s*\d+\.\d+)/i);
    const chance = chanceM ? evidence(t, new RegExp(reEsc(chanceM[0]))) : null;

    const groups = GROUP_HINTS.map(([re, name]) => (evidence(t, re) ? { name, evidence: evidence(t, re) } : null)).filter(Boolean);
    const unseenRe = new RegExp("\\b(unseen|new|held[\\s-]out|withheld|never seen|disjoint|other|different)\\b(?:\\s+\\S+){0,3}?\\s+\\b(" + ENTITY + ")\\b", "i");
    const unseen = evidence(t, unseenRe) || evidence(t, /unseen[\s-]\w+ (split|distribution)|disjoint (comment )?groups/i);
    const temporal = evidence(t, /\bforecast|future (period|weeks|months|years|dates)|later (period|dates|years)|time-based split|chronological(ly)?|split by (date|time)|test (period|window)|most recent (weeks|months|period)/i);
    const imbalance = evidence(t, /imbalanc|\brare\b|minority|(\d{1,2}(\.\d+)?)\s*%\s*(of\s+)?(the\s+)?(samples|rows|positives?|cases)/i);
    const synthetic = evidence(t, /synthetic|simulated|generated (data|dataset|pools)|procedurally|generator/i);
    const missing = evidence(t, /missing values?|\bNaN\b|null values?/i);
    const multilabel = evidence(t, /multi[\s-]label/i);
    const small = opts.nTrain ? Number(opts.nTrain) < 5000 : false;

    const split = [];
    if (prepare && prepare.kind !== "unknown") {
      const cvFor = { random: `KFold (shuffled, seeded${prepare.stratify ? ", stratified like the prepare script" : ""}). Keys such as authors or users can sit on both sides of a random split, so per-entity statistics fitted out-of-fold are legitimate signal.`,
        group: `GroupKFold on ${prepare.groupBy || "the same key"}: the prepare script holds whole groups out.`, temporal: "Forward-chaining split on the same time column: the prepare script holds out the latest rows." };
      split.push({ kind: `${prepare.kind} (prepare script)`, prob: "confirmed", why: `The prepare script builds the test with ${prepare.kind === "random" ? "a random split" : prepare.kind === "group" ? "a group split" : "a time cut"}${prepare.testSize ? ` (test_size=${prepare.testSize})` : ""}`, evidence: prepare.evidence, cv: cvFor[prepare.kind] });
    }
    if (foldsProvided) split.push({ kind: "provided folds", prob: "use it", why: "The description ships the fold assignment", evidence: foldsProvided, cv: "Use folds.csv exactly as given. It encodes the transfer the score asks for." });
    if (unseen) split.push({ kind: "group-held-out", prob: "likely", why: "The description mentions unseen or held-out entities", evidence: unseen, cv: "GroupKFold (or StratifiedGroupKFold) on that entity, plus leave-one-group-out as a stress split" });
    if (temporal) split.push({ kind: "temporal", prob: unseen ? "possible" : "likely", why: "Explicit time-ordering language", evidence: temporal, cv: "Forward-chaining time split. Validate on the latest period in train." });
    if (groups.length && !unseen && !foldsProvided && !(prepare && prepare.kind === "random")) split.push({ kind: "grouped", prob: "possible", why: `Group-like entities mentioned: ${groups.map((g) => g.name).join(", ")}`, evidence: groups[0].evidence, cv: "Check key overlap between train and test. No overlap means GroupKFold." });
    if (!(prepare && prepare.kind !== "unknown")) split.push({ kind: "random", prob: split.length ? "fallback" : "default", why: "No structural hints found", evidence: "", cv: "Stratified KFold (5 folds, seeded). Still check train/test key overlap." });

    // ---- rules that apply
    const ruleIds = ["ML-1", "ENV-8", "ENV-4", "DET-1", "DET-3", "DET-4", "TEST-1", "DATA-1", "DATA-2"];
    const catRules = { retrieval: ["RAG-1", "RAG-2"], cv: ["CV-1", "CV-2", "CV-3"], finetune: ["FT-1", "FT-2"], scratch: ["SCR-1"], biochem: ["FT-2"], nlp: ["GREY-1", "GREY-2"], general: [] };
    const constraints = [...ruleIds, ...(catRules[category] || [])].map((id) => E.RULES.find((r) => r.id === id)).filter(Boolean);

    // ---- difficulty heuristic (the header's own value wins when present)
    const factors = [];
    let d = 2;
    if (category === "scratch") { d++; factors.push("From scratch: no pretrained models within the time limit"); }
    if (cpuOnly) { d += 0.5; factors.push("CPU only: size the model and plan for cores, not a GPU"); }
    if (category === "retrieval" || category === "finetune") { d++; factors.push("Strict category: trained deep model required"); }
    if (customFormula || (metric && ["custom", "tiered", "setf1", "rps"].includes(metric.id))) { d += 0.5; factors.push("Custom metric: re-implement it exactly, then decode to it (expected-metric decoding)"); }
    if (unseen || temporal || foldsProvided) { d += 0.5; factors.push("Distribution shift between train and the hidden test is likely"); }
    if (perRow) { d += 0.25; factors.push("Row-local inference required: no cross-row tricks"); }
    if (small) { d += 0.5; factors.push("Small training set: high CV variance, overfitting risk"); }
    if (synthetic) { d += 0.5; factors.push("Synthetic or generated data: tempting artefacts, and exploiting them is rejected"); }
    if (runtimeLimit && runtimeLimit.minutes < 60) { d += 0.5; factors.push(`Tight runtime: ${runtimeLimit.minutes} min`); }
    let difficulty = Math.max(1, Math.min(5, Math.round(d)));
    if (hdr.difficulty) { const map = { easy: 2, medium: 3, hard: 4, expert: 5, extreme: 5 }; const k = hdr.difficulty.toLowerCase().split(/\s/)[0]; if (map[k]) { difficulty = map[k]; factors.unshift(`Header says Difficulty: ${hdr.difficulty}`); } }

    // ---- opportunities (hypotheses)
    const opp = [];
    if (metric && ["custom", "tiered", "setf1", "rps", "f1", "qwk", "map", "ndcg", "mrr"].includes(metric.id)) opp.push(`Decode to the metric: choose each answer by maximising the expected ${metric.name} under the model's predicted distribution, not by a plain argmax. Fit any temperature or threshold on OOF.`);
    if (foldsProvided) opp.push("The provided folds encode the hidden transfer. Validate every change on them, and use the same folds for blends and decode constants.");
    if ((unseen || groups.length) && !(prepare && prepare.kind === "random")) opp.push("Build corpus statistics (vocabularies, frequencies, encodings) for each training row from the OTHER groups, so features behave on training rows the way they will on unseen test groups.");
    if (metric && ["logloss", "brier"].includes(metric.id)) opp.push("Probability metric: OOF-fitted temperature scaling and a floor on probabilities are low-risk gains.");
    if (metric && metric.id === "auc") opp.push("AUC only depends on order: rank-average diverse members on OOF.");
    if (metric && metric.id === "maelog") opp.push("MAE in log space is minimised by the conditional median of log1p(y), not its mean: train on log1p(y) with an L1 or quantile-0.5 objective, invert with expm1 and clip at 0. Compare it with an MSE-on-log1p model on OOF.");
    if (metric && ["mae", "rmse", "rmsle", "maelog"].includes(metric.id) && task.id === "tabular") opp.push("Add a few meaningful ratio features (rates per day, per item, per author) and keep each one only if it improves CV.");
    const keyCols = [...new Set((t.match(/\b[a-z]+_(?:username|user_?id|author_?id|customer_?id|store_?id|item_?id|group_?id)\b/g) || []))].slice(0, 4);
    if (prepare && prepare.kind === "random" && keyCols.length) opp.push(`The prepare script splits rows at random, so ${keyCols.join(", ")} values recur between train and test. Out-of-fold per-entity statistics (target mean, count) can carry real signal: fit them inside each fold only, and confirm the overlap with the key columns.`);
    if (reference) opp.push(`A reference solution is reported at ${reference.value}. Re-run its recipe on your own folds: the gap between its CV and ${reference.value} calibrates how far your CV is from the leaderboard.`);
    if (rubric.some((r) => r.level === "required")) opp.push(`The rubric lists ${rubric.filter((r) => r.level === "required").length} REQUIRED criteria. Satisfy each one visibly and list it in the requirements docstring with the lines that meet it.`);
    if (synthetic) opp.push("If the description documents how labels or data were produced, model that process inside the network (its exact forward map) rather than learning it from scratch.");
    if (imbalance) opp.push("Imbalance: class weights or focal loss, with thresholds tuned on OOF. No SMOTE (synthetic rows).");
    if (category === "cv") opp.push("Pretrained CNN/ViT fine-tuning, plus exact-geometry augmentation (and TTA if the description allows it).");
    if (category === "nlp" || category === "seqgen") opp.push("Fine-tune a transformer as the core (or, from scratch, a token-level interaction model). Keep TF-IDF baselines for reference only.");
    opp.push("Model-family diversity (not just more seeds of one family) is the cheapest reliable ensemble gain.");

    const risks = [];
    if (synthetic) risks.push({ text: "Synthetic or generated data: the model must learn any generator artefacts. Never hardcode regexes or templates around them.", cite: "S2 §6.1" });
    if (category === "retrieval") risks.push({ text: "Retrieval is the second-highest-violation domain. Feature engineering plus an off-the-shelf ranker gets rejected.", cite: "S2 §6.3" });
    if (category === "cv") risks.push({ text: "CV is the highest-violation domain. No tabular models on pixels.", cite: "S2 §6.2" });
    if (perRow) risks.push({ text: "Any pooling across test rows (clustering, voting, batch calibration) breaks an explicit challenge rule.", cite: "Challenge text" });
    rejectIf.forEach((s) => risks.push({ text: "Format: " + s, cite: "Challenge text" }));
    boilerplate.forEach((b) => risks.push({ text: b.note, cite: "S2 §5.4" }));
    if (rubric.some((r) => r.level === "required")) risks.push({ text: "Review against the rubric is likely: a missing REQUIRED criterion can sink a high-scoring submission (hypothesis: how the rubric is scored isn't documented).", cite: "Platform example" });
    if (grader.items.length) grader.items.filter((x) => /raises|errors/.test(x.text)).forEach((x) => risks.push({ text: "Grader: " + x.text, cite: "Grading script" }));
    if (!metric) risks.push({ text: "No metric found in the text. Read it from the challenge page before building anything.", cite: "" });

    const questions = [];
    if (restrictions.some((r) => /Model restriction|size limit/.test(r.text))) questions.push("Confirm how the model restriction applies to pretrained backbones.");
    if (category === "biochem") questions.push("Is this challenge categorized as NLP or Fine-tuning? The rules differ (§6.4).");
    if (synthetic) questions.push("If you plan to mix new training instances from supplied ingredients, ask whether that counts as synthetic data here.");
    if (boilerplate.length) questions.push("Confirm the boilerplate lines are not intentional overrides.");
    if (!pretrainedAllowed && !noPretrained && category !== "scratch") questions.push("The text doesn't say whether pretrained weights are allowed. Default is yes (§4.2), but confirm if the domain is unusual.");

    const libTask = task.id === "signal" ? "tabular" : task.id;
    const library = E.LIBRARY.filter((x) => x.task === libTask || x.task === category || x.task === "general").slice(0, 14);
    const catName = (E.TASK_TYPES.find((x) => x.id === category) || {}).name || category;
    // keep >= 30% headroom against a stated limit; otherwise the 50-min plan from the owner's solver notes
    const budget = runtimeLimit ? Math.floor(runtimeLimit.minutes * 0.7 + 1e-9) : 50;

    return {
      header: hdr, category, categoryName: catName, categorySource, task: task.id, taskName: task.name, taskEvidence: task.hits || [], categoryScores: ranked.slice(0, 4),
      metric, customFormula, rows: (() => { const m = t.match(/(?:exactly|covering (?:every one of )?(?:the )?)\s*([\d,]+)\s+(?:rows|ids|test ids)/i); return m ? Number(m[1].replace(/,/g, "")) : null; })(),
      submissionColumns: (() => { const c = parseSubmissionTable(t); return c.length ? c : grader.columns.map((name) => ({ name, type: "" })); })(), runtimeLimit, budgetMin: Math.round(budget), hardware, restrict,
      restrictions, notUse, rejectIf, boilerplate, foldsProvided, aiBaseline, reference, chance, config, rubric, grader: grader.items, prepare, prep, groups, temporal, unseen, imbalance, synthetic, missing, multilabel,
      split, constraints, difficulty, factors, opportunities: opp, risks, questions, library,
    };
  }

  E.analyzer = { analyze };
})();
