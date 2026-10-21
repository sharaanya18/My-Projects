/* Training-log parser. Pulls fold scores, CV mean/std, stress-split score,
 * train score, public LB, runtime and peak memory out of pasted log text.
 * Each extracted value keeps the line it came from so the user can check it. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const NUM = "(-?\\d+(?:\\.\\d+)?(?:e-?\\d+)?)";
  const METRIC_WORDS = "(?:score|auc|roc_auc|acc(?:uracy)?|f1|rmse|rmsle|mae|log_?loss|logloss|loss|mrr|ndcg(?:@\\d+)?|map(?:@\\d+)?|util|metric|kappa|qwk|r2|spearman|pearson|iou|dice|bleu|val_\\w+|valid_\\w+)";

  function lastMatch(lines, re) {
    let hit = null;
    for (const l of lines) { const m = l.match(re); if (m) hit = { m, line: l.trim() }; }
    return hit;
  }

  /* Kaggle kernel logs arrive as JSON lines: {"stream_name":"stdout","time":12.3,"data":"[ 12s] fold 1 ...\n"}.
   * Unwrap them to plain text and remember the largest time stamp (seconds). */
  function unwrap(text) {
    const raw = String(text || "").replace(/\r\n?/g, "\n").split("\n");
    let maxTime = null, kaggle = false;
    const lines = [];
    for (const l of raw) {
      const m = l.match(/"data"\s*:\s*"((?:[^"\\]|\\.)*)"/);
      if (m && /"stream_name"/.test(l)) {
        kaggle = true;
        const tm = l.match(/"time"\s*:\s*([\d.]+)/);
        if (tm) maxTime = Math.max(maxTime || 0, Number(tm[1]));
        if (/"stream_name"\s*:\s*"stderr"/.test(l)) continue;
        let d = m[1];
        try { d = JSON.parse('"' + m[1] + '"'); } catch (_) { /* keep the raw text */ }
        d.split("\n").forEach((x) => { if (x.trim()) lines.push(x); });
      } else lines.push(l);
    }
    return { lines, maxTime, kaggle };
  }

  function parse(text) {
    const U = unwrap(text);
    const lines = U.lines;
    const out = { folds: [], evidence: {}, kaggle: U.kaggle };

    // 1) explicit list: "folds=[0.81, 0.82]" or "fold scores: 0.81 0.82 ..."
    const listRe = /fold[_ ]?scores?\s*[:=]\s*\[?\s*((?:-?\d*\.\d+(?:e-?\d+)?[\s,;]*){2,})\]?/i;
    const listRe2 = /\bfolds\s*=\s*\[\s*((?:-?\d*\.\d+(?:e-?\d+)?[\s,;]*){2,})\]/i;
    const list = lastMatch(lines, listRe) || lastMatch(lines, listRe2);
    if (list) {
      out.folds = list.m[1].split(/[\s,;]+/).filter(Boolean).map(Number);
      out.evidence.folds = list.line;
    } else {
      // 2) one line per fold: "Fold 3 ... auc: 0.812" (the last value per fold wins)
      // On each "fold N" line take the LAST metric/value pair that isn't a train metric
      // ("train_loss=.. val_loss=.. val_auc=0.81" → 0.81).
      const per = new Map();
      const foldRe = /\bfold\s*[#:=]?\s*(\d+)(?:\s*\/\s*\d+)?\b/i;
      const pairRe = new RegExp("(\\w*)\\b" + METRIC_WORDS + "\\b\\s*[:=]?\\s*" + NUM, "gi");
      const ev = [];
      for (const l of lines) {
        const f = l.match(foldRe);
        if (!f) continue;
        const rest = l.slice(f.index + f[0].length);
        let val = null;
        for (const m of rest.matchAll(pairRe)) {
          const before = rest.slice(Math.max(0, m.index - 7), m.index + m[1].length + 1).toLowerCase();
          if (/train/.test(before) || /^train/i.test(m[0])) continue;
          val = Number(m[2]);
        }
        if (val !== null) { per.set(Number(f[1]), val); ev.push(l.trim()); }
      }
      if (per.size >= 2) {
        out.folds = [...per.entries()].sort((a, b) => a[0] - b[0]).map(([, v]) => v);
        out.evidence.folds = ev.slice(-per.size).join("\n");
      }
    }

    // CV mean ± std
    const cv = lastMatch(lines, new RegExp("\\b(?:cv|oof|validation|valid|val|mean)\\b[^\\n]{0,40}?" + NUM + "\\s*(?:\\+/-|±|\\+-)\\s*" + NUM, "i"));
    if (cv) { out.cv = Number(cv.m[1]); out.cvStd = Number(cv.m[2]); out.evidence.cv = cv.line; }
    else {
      // "validation (held-out headwords…): score 0.3768", "HELD-OUT validation, official metric: TOTAL 0.5816",
      // "OOF skill (mean over cells) at thr 0.55: 0.4669", "CV auc 0.81"
      const kw = "(?:score|skill|total|util|metric|auc|f1|acc(?:uracy)?|mrr|ndcg|rmse|mae|logloss|loss|map|ap)";
      const cv2 = lastMatch(lines, new RegExp("\\b(?:cv|oof|validation|valid|held[- ]out)\\b[^\\n]*?\\b" + kw + "\\b\\s*[:=]?\\s*" + NUM, "i"))
        || lastMatch(lines, /\b(?:cv|oof|validation|held[- ]out)\b[^\n]*?[:=]\s*(-?\d*\.\d+)(?![^\n|]*:\s*\d)/i);
      if (cv2 && !/\bfold\s*\d/i.test(cv2.line)) { out.cv = Number(cv2.m[cv2.m.length - 1]); out.evidence.cv = cv2.line; }
    }

    // per-group scores ("per-held-out-person scores: 0.379 0.496 …") → subgroup field
    const per = lastMatch(lines, /\bper[- ]([\w-]+?)\s+scores?\s*[:=]\s*((?:-?\d*\.\d+[\s,]*){2,})/i);
    if (per) {
      const label = per.m[1].replace(/^held[- ]out[- ]/i, "");
      out.groups = per.m[2].trim().split(/[\s,]+/).map(Number);
      out.groupLabel = label;
      out.evidence.groups = per.line;
    }
    const base = lastMatch(lines, new RegExp("\\b(?:best constant|constant|baseline|random)\\b[^\\n]*?(?:scores?|=|:)\\s*" + NUM, "i"));
    if (base) { out.baseline = Number(base.m[1]); out.evidence.baseline = base.line; }

    // stress split (leave-one-group-out / LODO / unseen …)
    // a stress split needs stress wording (plain "held-out" validation is ordinary CV) and a decimal score
    const ood = lastMatch(lines.filter((l) => !/\bper[- ]/i.test(l)), /\b(?:lodo|logo|stress(?:[- ]split)?|ood|leave[- ]one[- ][\w-]+[- ]out|unseen[- ][\w-]+ (?:util|score|auc|f1|acc\w*|skill))\b[^\n]{0,30}?(-?\d*\.\d+)/i);
    if (ood) { out.ood = Number(ood.m[1]); out.evidence.ood = ood.line; }

    const tr = lastMatch(lines, new RegExp("\\btrain(?:ing)?[ _]" + METRIC_WORDS + "\\s*[:=]?\\s*" + NUM, "i"));
    if (tr) { out.train = Number(tr.m[1]); out.evidence.train = tr.line; }

    const lb = lastMatch(lines, new RegExp("\\b(?:public[ _]?lb|public leaderboard|public score|lb score)\\b\\s*[:=]?\\s*" + NUM, "i"));
    if (lb) { out.lb = Number(lb.m[1]); out.evidence.lb = lb.line; }

    // runtime, in minutes
    // the label itself contains colons ("h:mm:ss or m:ss"), so anchor on the time value at line end
    const wall = lastMatch(lines, /Elapsed \(wall clock\) time.*?:\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s*$/i);
    const rt = lastMatch(lines, new RegExp("\\b(?:runtime|total time|elapsed|took|wall time|time taken)\\b\\s*[:=]?\\s*" + NUM + "\\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)\\b", "i"));
    const stamp = lastMatch(lines, /^\s*\[\s*(\d+(?:\.\d+)?)\s*s\]/);
    if (wall) { out.runtime = (Number(wall.m[1] || 0) * 3600 + Number(wall.m[2]) * 60 + Number(wall.m[3])) / 60; out.evidence.runtime = wall.line; }
    else if (rt) {
      const v = Number(rt.m[1]), u = rt.m[2].toLowerCase();
      out.runtime = u.startsWith("h") ? v * 60 : u.startsWith("m") ? v : v / 60;
      out.evidence.runtime = rt.line;
    } else if (U.maxTime) { out.runtime = U.maxTime / 60; out.evidence.runtime = `Kaggle log: last "time" field ${U.maxTime.toFixed(1)} s`; }
    else if (stamp) { out.runtime = Number(stamp.m[1]) / 60; out.evidence.runtime = stamp.line + "  (last timestamp)"; }

    // peak memory, in GB
    const rss = lastMatch(lines, /Maximum resident set size \(kbytes\):\s*(\d+)/i);
    const mem = lastMatch(lines, new RegExp("\\b(?:peak|max(?:imum)?)[ _](?:gpu[ _])?(?:memory|mem|vram|rss)(?:[ _]allocated)?\\b[^\\n]{0,15}?" + NUM + "\\s*(gb|gib|mb|mib)\\b", "i"));
    if (rss) { out.mem = Number(rss.m[1]) / 1048576; out.evidence.mem = rss.line; }
    else if (mem) { out.mem = Number(mem.m[1]) / (/^m/i.test(mem.m[2]) ? 1024 : 1); out.evidence.mem = mem.line; }

    // model family hint
    const fam = [["lightgbm|lgbm", "lgbm"], ["xgboost|xgb", "xgboost"], ["catboost", "catboost"], ["deberta", "deberta"], ["roberta", "roberta"], ["bert", "bert"], ["efficientnet", "efficientnet"], ["convnext", "convnext"], ["resnet", "resnet"], ["vit\\b", "vit"], ["t5\\b|mt5", "t5"], ["mlp", "mlp"]];
    const lower = String(text || "").toLowerCase();
    let bestFam = null, bestCount = 0;
    fam.forEach(([re, name]) => { const c = (lower.match(new RegExp("\\b(?:" + re + ")", "g")) || []).length; if (c > bestCount) { bestCount = c; bestFam = name; } });
    if (bestFam) out.model = bestFam;

    if (out.folds.length && out.cv === undefined) {
      const m = out.folds.reduce((s, x) => s + x, 0) / out.folds.length;
      out.cvFromFolds = m;
    }
    out.found = ["folds", "cv", "ood", "train", "lb", "runtime", "mem", "groups", "baseline"].filter((k) => (k === "folds" ? out.folds.length : out[k] !== undefined));
    return out;
  }

  E.logparse = { parse };
})();
