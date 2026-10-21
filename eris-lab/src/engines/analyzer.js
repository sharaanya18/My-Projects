/* Challenge analyzer. It reads a pasted problem description and extracts
 * category, metric, submission format and constraints, then maps them to
 * platform rules, a validation plan and candidate strategies. Every inference
 * keeps the phrase that triggered it, so the user can check the evidence. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const CATS = [
    { id: "scratch", name: "From-scratch", re: [/from[\s-]scratch/i, /no pre-?trained/i, /pre-?trained (models?|weights) (are|is) not (allowed|permitted)/i, /must not use (any )?pre-?trained/i] },
    { id: "finetune", name: "Fine-tuning", re: [/fine[\s-]?tun(e|ing) (challenge|category|task)/i, /category:\s*fine[\s-]?tun/i, /must fine[\s-]?tune/i] },
    { id: "retrieval", name: "Retrieval / ranking", re: [/\bretriev/i, /\brank(ing|er)?\b/i, /\brerank/i, /\bRAG\b/, /candidate(s| list)/i, /\bquer(y|ies)\b/i, /\bMRR\b|NDCG|MAP@|recall@/i] },
    { id: "cv", name: "Computer vision", re: [/\bimages?\b/i, /\bpixels?\b/i, /segmentation/i, /object detection|bounding box/i, /\.png|\.jpg|\.jpeg/i, /\bmask(s)?\b/i] },
    { id: "seqgen", name: "Sequence generation", re: [/translat(e|ion)/i, /summari[sz]/i, /\bgenerat(e|ion)\b.*\btext\b/i, /seq2seq|sequence-to-sequence/i, /\bBLEU\b|chrF|ROUGE/i, /caption/i] },
    { id: "recsys", name: "Recommendation", re: [/recommend/i, /\busers?\b.*\bitems?\b/i, /click|purchase|session/i, /collaborative filtering/i] },
    { id: "structured", name: "Structured prediction", re: [/named entit|\bNER\b/i, /span(s)? extraction/i, /token[\s-]classification/i, /\bparse|parsing\b/i, /BIO tags?/i] },
    { id: "biochem", name: "Biology / chemistry", re: [/\bSMILES\b/, /molecul/i, /protein/i, /\bDNA\b|\bRNA\b|genom/i, /chemi(cal|stry)/i, /amino acid/i] },
    { id: "nlp", name: "NLP", re: [/\btext\b/i, /sentences?/i, /\btokens?\b/i, /\bNLP\b/, /documents?/i, /reviews?/i, /\blanguage\b/i] },
    { id: "tabular", name: "Tabular", re: [/\bcolumns?\b/i, /\bfeatures?\b/i, /\btabular\b/i, /\bcsv\b/i, /numeric|categorical/i] },
  ];

  const METRIC_RE = [
    ["auc", /\bAUC\b|ROC[\s-]AUC|area under the (ROC )?curve/i],
    ["logloss", /log[\s-]?loss|cross[\s-]entropy|negative log[\s-]likelihood/i],
    ["brier", /brier/i],
    ["qwk", /quadratic weighted kappa|\bQWK\b|cohen'?s kappa/i],
    ["f1", /\bF1\b|F-?score|F-?beta/i],
    ["rmsle", /\bRMSLE\b|root mean squared log/i],
    ["rmse", /\bRMSE\b|root mean squared? error/i],
    ["mae", /\bMAE\b|mean absolute error/i],
    ["spearman", /spearman|pearson|correlation coefficient/i],
    ["mrr", /\bMRR\b|mean reciprocal rank/i],
    ["ndcg", /NDCG/i],
    ["map", /\bMAP@\d+|mean average precision@|recall@\d+|precision@\d+/i],
    ["mapdet", /\bmAP\b(?!@)|COCO/],
    ["bleu", /\bBLEU\b|chrF|ROUGE/i],
    ["iou", /\bIoU\b|\bDice\b|jaccard/i],
    ["accuracy", /\baccuracy\b/i],
  ];

  const BOILERPLATE = [
    [/no internet( access)?/i, "'No internet' is treated as boilerplate. Downloading pretrained weights through allowed packages is fine (§4.2, §5.4)."],
    [/(fully )?rule[\s-]based (solutions? )?(are|is) (ok|okay|allowed|fine|acceptable)/i, "Rule-based permissions are boilerplate. Real ML training is still required (§5.2.1, §5.4)."],
    [/decoding tricks (are )?(fine|ok|okay|allowed)/i, "'Decoding tricks are fine' is boilerplate. The model must still be trained (§5.4)."],
  ];

  const GROUP_HINTS = [
    [/\b(user|customer|client)_?id\b|\busers?\b/i, "user"],
    [/\bpatient|subject_?id|\bsubjects?\b/i, "patient/subject"],
    [/\bquery_?id|\bquer(y|ies)\b/i, "query"],
    [/\bsession/i, "session"],
    [/\b(store|shop|site|hospital|location|region|plant|factory|farm|well|city|country|school|compan(y|ie))s?\b/i, "site/location"],
    [/\b(machine|device|sensor|vehicle|product|item)_?(id)?s?\b/i, "device/item"],
    [/\b(author|writer|speaker)s?\b/i, "author/speaker"],
    [/\b(document|article|book|case)_?id\b/i, "document"],
    [/\blanguages?\b|language pair|source_lang|target_lang/i, "language"],
    [/\bdomains?\b|\bsources?\b/i, "domain/source"],
    [/\bgroup_?id\b|\bgroups?\b/i, "group"],
  ];

  function evidence(text, re) {
    const m = text.match(re);
    if (!m) return null;
    // ~40 characters of context on each side, snapped to word boundaries
    let s = Math.max(0, m.index - 40);
    if (s > 0) { const sp = text.indexOf(" ", s); if (sp > -1 && sp < m.index) s = sp + 1; }
    let e = Math.min(text.length, m.index + m[0].length + 40);
    if (e < text.length) { const sp = text.lastIndexOf(" ", e); if (sp > m.index + m[0].length) e = sp; }
    return (s > 0 ? "…" : "") + text.slice(s, e).replace(/\s+/g, " ").trim() + (e < text.length ? "…" : "");
  }

  function parseSubmissionTable(text) {
    const sec = text.split(/\n#+\s*submission/i)[1] || text;
    const cols = [];
    sec.split("\n").forEach((l) => {
      const m = l.match(/^\s*\|\s*`?([A-Za-z_][\w ]*?)`?\s*\|\s*([\w ]+?)\s*\|/);
      if (m && !/^(column|-+)$/i.test(m[1].trim()) && !/^-+$/.test(m[2])) cols.push({ name: m[1].trim(), type: m[2].trim() });
    });
    return cols.slice(0, 12);
  }

  function analyze(text, opts = {}) {
    const t = String(text || "");
    const scores = CATS.map((c) => {
      const hits = c.re.map((re) => evidence(t, re)).filter(Boolean);
      return { id: c.id, name: c.name, score: hits.length / c.re.length, hits };
    });
    // strict categories win when explicitly stated
    let category = opts.category && opts.category !== "auto" ? opts.category : null;
    const strict = scores.find((s) => (s.id === "scratch" || s.id === "finetune") && s.hits.length);
    if (!category) category = strict ? strict.id : null;
    const ranked = scores.filter((s) => s.hits.length).sort((a, b) => b.score - a.score || b.hits.length - a.hits.length);
    const task = ranked.find((s) => s.id !== "scratch" && s.id !== "finetune") || { id: "tabular", name: "Tabular", hits: [] };
    if (!category) category = ["retrieval", "cv", "biochem"].includes(task.id) ? task.id : ["nlp", "seqgen", "structured"].includes(task.id) ? "nlp" : "general";

    let metric = null;
    for (const [id, re] of METRIC_RE) {
      const ev = evidence(t, re);
      if (ev) { metric = { ...E.METRICS.find((m) => m.id === id), evidence: ev }; break; }
    }
    if (!metric && /def evaluate|custom metric|our metric|weighted (sum|combination)/i.test(t)) metric = { ...E.METRICS.find((m) => m.id === "custom"), evidence: evidence(t, /def evaluate|custom metric|our metric|weighted (sum|combination)/i) };
    if (opts.metric) metric = { ...(E.METRICS.find((m) => m.id === opts.metric) || {}), evidence: "set manually" };
    const customFormula = /def evaluate|formula|weighted|0\.\d+\s*[\*×]/.test(t);

    const rowsM = t.match(/exactly\s+([\d,]+)\s+rows/i);
    const runtimeM = t.match(/(?:runtime|time limit|must (?:complete|finish|run)|within|under|no more than|at most)[^.\n]{0,40}?(\d+(?:\.\d+)?)\s*(minutes?|mins?|hours?|hrs?|h)\b/i);
    let runtimeLimit = null;
    if (runtimeM) {
      const v = parseFloat(runtimeM[1]);
      runtimeLimit = { minutes: /^h/i.test(runtimeM[2]) ? v * 60 : v, evidence: evidence(t, new RegExp(runtimeM[0].replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))) };
    }
    const sizeM = t.match(/(\d+(?:\.\d+)?)\s*([MB])\s*(parameters|params)/i) || t.match(/(?:at most|no more than|under|max(?:imum)?)\s*(\d+(?:\.\d+)?)\s*([MB])\b/i);
    const restrictions = [];
    if (sizeM) restrictions.push({ text: `Model size limit: ${sizeM[1]}${sizeM[2].toUpperCase()} parameters`, evidence: evidence(t, new RegExp(sizeM[0].replace(/[.*+?^${}()|[\]\\]/g, "\\$&"))) });
    const onlyM = t.match(/(?:only|must use|restricted to)\s+([\w\- ]{3,40}?)\s+models?/i);
    if (onlyM) restrictions.push({ text: `Model restriction: only ${onlyM[1]} models`, evidence: evidence(t, /(?:only|must use|restricted to)\s+[\w\- ]{3,40}?\s+models?/i) });
    if (/no (external|additional) (data|datasets)/i.test(t)) restrictions.push({ text: "No external data (already a platform rule)", evidence: evidence(t, /no (external|additional) (data|datasets)/i) });

    const boilerplate = BOILERPLATE.map(([re, note]) => (evidence(t, re) ? { evidence: evidence(t, re), note } : null)).filter(Boolean);

    const groups = GROUP_HINTS.map(([re, name]) => (evidence(t, re) ? { name, evidence: evidence(t, re) } : null)).filter(Boolean);
    const temporal = evidence(t, /\b(date|timestamp|time[\s-]series|forecast|future|later period|chronolog|month|year)\b/i);
    const unseen = evidence(t, /\b(unseen|new|held[\s-]out|different|novel|never (seen|appear))\b[^.\n]{0,40}\b(users?|customers?|patients?|quer(y|ies)|languages?|domains?|sites?|documents?|authors?|speakers?|directions?|stores?|subjects?|categories|plants?|factories|machines?|devices?|hospitals?|regions?|cities|countries|schools?|products?|items?|sessions?|groups?|companies)\b/i);
    const imbalance = evidence(t, /imbalanc|rare|minority|(\d{1,2}(\.\d+)?)\s*%\s*(of\s+)?(the\s+)?(samples|rows|positives?|cases)/i);
    const synthetic = evidence(t, /synthetic|simulated|generated (data|dataset)|procedurally/i);
    const missing = evidence(t, /missing values?|\bNaN\b|null values?/i);
    const multilabel = evidence(t, /multi[\s-]label/i);
    const small = opts.nTrain ? Number(opts.nTrain) < 5000 : false;

    // hidden-test construction hypotheses
    const split = [];
    if (unseen) split.push({ kind: "group-held-out", prob: "likely", why: "The description mentions unseen entities", evidence: unseen, cv: "GroupKFold (or StratifiedGroupKFold) on that entity, plus leave-one-group-out as a stress split" });
    if (temporal) split.push({ kind: "temporal", prob: unseen ? "possible" : "likely", why: "Time-related wording", evidence: temporal, cv: "Forward-chaining time split. Validate on the latest period in train." });
    if (groups.length && !unseen) split.push({ kind: "grouped", prob: "possible", why: `Group-like entities mentioned: ${groups.map((g) => g.name).join(", ")}`, evidence: groups[0].evidence, cv: "Check key overlap between train and test. No overlap means GroupKFold." });
    split.push({ kind: "random", prob: split.length ? "fallback" : "default", why: "No structural hints found", evidence: "", cv: "Stratified KFold (5 folds, seeded). Still check train/test key overlap." });

    // constraints from platform rules
    const ruleIds = ["ML-1", "ENV-8", "ENV-4", "DET-1", "DET-3", "DET-4", "TEST-1", "DATA-1", "DATA-2"];
    const catRules = { retrieval: ["RAG-1", "RAG-2"], cv: ["CV-1", "CV-2", "CV-3"], finetune: ["FT-1", "FT-2"], scratch: ["SCR-1"], biochem: ["FT-2"], nlp: ["GREY-1", "GREY-2"], general: [] };
    const constraints = [...ruleIds, ...(catRules[category] || [])].map((id) => E.RULES.find((r) => r.id === id)).filter(Boolean);

    // difficulty heuristic
    const factors = [];
    let d = 2;
    if (category === "scratch") { d++; factors.push("From-scratch: no pretrained models within 1.5 h"); }
    if (category === "retrieval" || category === "finetune") { d++; factors.push("Strict category: trained deep model required"); }
    if (customFormula || (metric && metric.id === "custom")) { d += 0.5; factors.push("Custom or composite metric: re-implement and unit-test it"); }
    if (unseen || temporal) { d += 0.5; factors.push("Distribution shift between train and hidden test is likely"); }
    if (small) { d += 0.5; factors.push("Small training set: high CV variance, overfitting risk"); }
    if (synthetic) { d += 0.5; factors.push("Synthetic data: tempting artefacts, and exploiting them is rejected"); }
    if (runtimeLimit && runtimeLimit.minutes < 60) { d += 0.5; factors.push(`Tight runtime: ${runtimeLimit.minutes} min`); }
    const difficulty = Math.max(1, Math.min(5, Math.round(d)));

    // opportunities (hypotheses)
    const opp = [];
    if (unseen || groups.length) opp.push("Most solvers validate randomly. A split that matches the test construction is the cheapest edge on the private LB.");
    if (metric && ["f1", "qwk", "accuracy"].includes(metric.id)) opp.push(`${metric.name}: threshold or cut-point optimisation on OOF predictions often adds measurable gains.`);
    if (metric && ["logloss", "brier"].includes(metric.id)) opp.push("Probability metric: OOF-fitted temperature scaling is a low-risk gain.");
    if (metric && ["mrr", "ndcg", "map"].includes(metric.id)) opp.push("List metric: a listwise loss plus hard negatives mined inside folds.");
    if (metric && metric.id === "auc") opp.push("AUC only depends on order: rank-average diverse members on OOF.");
    if (imbalance) opp.push("Imbalance: class weights or focal loss, with thresholds tuned on OOF. No SMOTE (synthetic rows).");
    if (missing) opp.push("Missing values: GBDTs handle NaN natively. Missingness indicators can carry signal.");
    if (category === "cv") opp.push("Pretrained CNN/ViT fine-tuning with TTA (allowed per sample).");
    if (category === "nlp" || category === "seqgen") opp.push("Fine-tuned transformer as the core. Keep TF-IDF baselines for reference only (grey area).");
    opp.push("Seed bagging the strongest member is a reliable variance reduction if runtime allows.");

    const risks = [];
    if (synthetic) risks.push({ text: "Synthetic data: the model must learn any generator artefacts. Never hardcode regexes or templates around them.", cite: "S2 §6.1" });
    if (category === "retrieval") risks.push({ text: "Retrieval is the second-highest-violation domain. Feature engineering plus an off-the-shelf ranker gets rejected.", cite: "S2 §6.3" });
    if (category === "cv") risks.push({ text: "CV is the highest-violation domain. No tabular models on pixels.", cite: "S2 §6.2" });
    boilerplate.forEach((b) => risks.push({ text: b.note, cite: "S2 §5.4" }));
    if (!metric) risks.push({ text: "No metric found in the text. Read it from the challenge page before building anything.", cite: "" });

    const questions = [];
    if (restrictions.length) questions.push("Confirm how the model restriction applies to pretrained backbones.");
    if (category === "biochem") questions.push("Is this challenge categorized as NLP or Fine-tuning? The rules differ (§6.4).");
    if (synthetic) questions.push("Are structure-aware features acceptable, given the data is synthetic?");
    if (boilerplate.length) questions.push("Confirm the boilerplate lines are not intentional overrides.");

    const library = E.LIBRARY.filter((x) => x.task === task.id || x.task === category || x.task === "general").slice(0, 14);
    const catName = (E.TASK_TYPES.find((x) => x.id === category) || {}).name || category;

    return {
      category, categoryName: catName, task: task.id, taskName: task.name, taskEvidence: task.hits || [], categoryScores: ranked.slice(0, 4),
      metric, customFormula, rows: rowsM ? Number(rowsM[1].replace(/,/g, "")) : null,
      submissionColumns: parseSubmissionTable(t), runtimeLimit, budgetMin: runtimeLimit ? Math.min(75, Math.floor(runtimeLimit.minutes * 0.83)) : 75,
      restrictions, boilerplate, groups, temporal, unseen, imbalance, synthetic, missing, multilabel,
      split, constraints, difficulty, factors, opportunities: opp, risks, questions, library,
    };
  }

  E.analyzer = { analyze };
})();
