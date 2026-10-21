/* Submission CSV validation, runtime budgeting and credit planning. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  /* RFC 4180-style CSV parser: quotes, escaped quotes, CRLF, embedded newlines. */
  function parseCSV(text) {
    const rows = [];
    let row = [];
    let field = "";
    let q = false;
    const s = String(text || "").replace(/^﻿/, "");
    for (let i = 0; i < s.length; i++) {
      const ch = s[i];
      if (q) {
        if (ch === '"') {
          if (s[i + 1] === '"') { field += '"'; i++; } else q = false;
        } else field += ch;
      } else if (ch === '"') q = true;
      else if (ch === ",") { row.push(field); field = ""; }
      else if (ch === "\n" || ch === "\r") {
        if (ch === "\r" && s[i + 1] === "\n") i++;
        row.push(field); rows.push(row); row = []; field = "";
      } else field += ch;
    }
    if (field !== "" || row.length) { row.push(field); rows.push(row); }
    while (rows.length && rows[rows.length - 1].length === 1 && rows[rows.length - 1][0] === "") rows.pop();
    return { header: rows[0] || [], rows: rows.slice(1), unterminatedQuote: q };
  }

  const isNum = (v) => v.trim() !== "" && Number.isFinite(Number(v));
  const isNanLike = (v) => /^(nan|NaN|NAN|inf|-inf|Infinity|-Infinity|None|null|)$/.test(v.trim());

  function checkSubmission(subText, sampleText, testText) {
    const F = [];
    const add = (sev, title, detail) => F.push({ sev, title, detail });
    const sub = parseCSV(subText);
    if (!sub.header.length) { add("blocker", "Submission is empty", "No header row found."); return { findings: F, stats: {} }; }
    if (sub.unterminatedQuote) add("blocker", "Unterminated quote", "A quoted field never closes. The file will not parse.");
    const sample = sampleText ? parseCSV(sampleText) : null;
    const test = testText ? parseCSV(testText) : null;

    const hdr = sub.header.map((h) => h.trim());
    if (hdr.some((h, i) => h !== sub.header[i])) add("medium", "Whitespace in header names", `Header: ${JSON.stringify(sub.header)}`);
    if (/^(Unnamed: 0|)$/.test(hdr[0]) && hdr.length > 1) add("blocker", "Leading index column", "The first column is unnamed. That usually means to_csv was called without index=False.");
    if (new Set(hdr).size !== hdr.length) add("blocker", "Duplicate column names", hdr.join(", "));

    const width = hdr.length;
    const ragged = sub.rows.map((r, i) => [i + 2, r.length]).filter(([, n]) => n !== width);
    if (ragged.length) add("blocker", `${ragged.length} rows have the wrong number of fields`, `First: line ${ragged[0][0]} has ${ragged[0][1]} fields, header has ${width}.`);

    let idCol = 0;
    if (sample && sample.header.length) {
      const sh = sample.header.map((h) => h.trim());
      if (sh.join("|") !== hdr.join("|")) {
        const missing = sh.filter((h) => !hdr.includes(h));
        const extra = hdr.filter((h) => !sh.includes(h));
        if (missing.length || extra.length) add("blocker", "Columns differ from sample_submission.csv", `Missing: [${missing.join(", ")}] Extra: [${extra.join(", ")}]`);
        else add("high", "Column order differs from sample_submission.csv", `Expected ${sh.join(", ")}`);
      }
      idCol = Math.max(0, hdr.indexOf(sh[0]));
      if (sample.rows.length !== sub.rows.length) add("blocker", "Row count differs from sample_submission.csv", `Submission ${sub.rows.length} rows, sample ${sample.rows.length}.`);
    } else {
      const guess = hdr.findIndex((h) => /^(id|.*_id|row_id|image_id|query_id)$/i.test(h));
      idCol = guess >= 0 ? guess : 0;
    }

    const ids = sub.rows.map((r) => (r[idCol] || "").trim());
    const seen = new Map();
    ids.forEach((v) => seen.set(v, (seen.get(v) || 0) + 1));
    const dups = [...seen].filter(([, n]) => n > 1);
    if (dups.length) add("blocker", `${dups.length} duplicate ids`, `e.g. ${dups.slice(0, 5).map(([k, n]) => `${k} ×${n}`).join(", ")}`);
    if (ids.some((v) => v === "")) add("blocker", "Empty id values", `${ids.filter((v) => !v).length} rows have no id.`);

    const ref = sample || test;
    if (ref && ref.header.length) {
      const rh = ref.header.map((h) => h.trim());
      const rIdx = Math.max(0, rh.indexOf(hdr[idCol]));
      const refIds = new Set(ref.rows.map((r) => (r[rIdx] || "").trim()));
      const subIds = new Set(ids);
      const missing = [...refIds].filter((v) => !subIds.has(v));
      const extra = [...subIds].filter((v) => !refIds.has(v));
      const label = sample ? "sample_submission.csv" : "test.csv";
      if (missing.length) add("blocker", `${missing.length} ids from ${label} are missing`, `e.g. ${missing.slice(0, 5).join(", ")}`);
      if (extra.length) add("blocker", `${extra.length} ids not in ${label}`, `e.g. ${extra.slice(0, 5).join(", ")}`);
      if (sample && test && test.header.length) {
        const th = test.header.map((h) => h.trim());
        const ti = th.indexOf(hdr[idCol]);
        if (ti >= 0 && test.rows.length !== sub.rows.length) add("high", "Row count differs from test.csv", `test.csv has ${test.rows.length} rows.`);
      }
    }

    const stats = { rows: sub.rows.length, columns: hdr, idColumn: hdr[idCol], valueColumns: [] };
    hdr.forEach((h, ci) => {
      if (ci === idCol) return;
      const vals = sub.rows.map((r) => r[ci] === undefined ? "" : r[ci]);
      const sampleVals = sample && sample.header.map((x) => x.trim()).indexOf(h) >= 0
        ? sample.rows.map((r) => r[sample.header.map((x) => x.trim()).indexOf(h)] || "") : null;
      const expectNumeric = sampleVals ? sampleVals.filter((v) => v.trim()).every(isNum) : vals.filter((v) => v.trim()).every(isNum);
      const nan = vals.filter(isNanLike).length;
      const col = { name: h, nan, numeric: expectNumeric };
      if (nan) add("blocker", `${nan} empty/NaN/inf values in "${h}"`, "The grader may reject the file or score those rows as wrong.");
      if (expectNumeric) {
        const bad = vals.filter((v) => !isNanLike(v) && !isNum(v));
        if (bad.length) add("blocker", `${bad.length} non-numeric values in numeric column "${h}"`, `e.g. ${bad.slice(0, 3).join(", ")}`);
        const nums = vals.filter(isNum).map(Number);
        if (nums.length) {
          let min = Infinity, max = -Infinity, sum = 0;
          nums.forEach((x) => { if (x < min) min = x; if (x > max) max = x; sum += x; });
          const distinct = new Set(nums).size;
          Object.assign(col, { min, max, mean: sum / nums.length, distinct });
          if (distinct === 1) add("high", `"${h}" is constant`, `Every row is ${nums[0]}. This usually means a pipeline bug.`);
          const sampleProb = sampleVals && sampleVals.filter(isNum).map(Number).every((x) => x >= 0 && x <= 1);
          const nameProb = /prob|pred|score|target|label|proba/i.test(h);
          if ((sampleProb || (!sampleVals && nameProb && min >= 0)) && (min < 0 || max > 1))
            add("high", `"${h}" falls outside [0, 1]`, `Range ${min} to ${max}. Check whether probabilities are expected.`);
          const sampleInts = sampleVals && sampleVals.filter(isNum).every((v) => /^-?\d+$/.test(v.trim()));
          if (sampleInts && nums.some((x) => !Number.isInteger(x))) add("high", `"${h}" has decimals where the sample has integers`, "The sample suggests class labels or counts.");
        }
      } else {
        const distinct = new Set(vals).size;
        Object.assign(col, { distinct });
        if (sampleVals) {
          const allowedSet = new Set(sampleVals);
          if (allowedSet.size <= 50) {
            const unseen = [...new Set(vals)].filter((v) => !allowedSet.has(v));
            if (unseen.length) add("medium", `"${h}" contains values not seen in the sample`, `e.g. ${unseen.slice(0, 5).join(", ")}. Fine for open label sets, a bug for fixed classes.`);
          }
        }
      }
      stats.valueColumns.push(col);
    });

    if (!F.some((x) => x.sev === "blocker" || x.sev === "high"))
      add("ok", "No structural problems found", sample ? "Matches sample_submission.csv structure." : "Add sample_submission.csv for a full comparison.");
    return { findings: F, stats };
  }

  /* Runtime budget on the target hardware. speedup = A10G throughput relative to
   * the local machine (1 = same; a hypothesis the user enters). */
  function runtimeBudget(o) {
    const n = (v, d = 0) => (Number.isFinite(Number(v)) && v !== "" ? Number(v) : d);
    const speed = Math.max(0.05, n(o.speedup, 1));
    const train = (n(o.epochMin) * n(o.epochs) * Math.max(1, n(o.folds, 1)) + n(o.finalFitMin)) / speed;
    const total = n(o.loadMin) + n(o.featMin) + train + n(o.hpoMin) / speed + n(o.inferMin) / speed + n(o.overheadMin, 2);
    const limit = n(o.limitMin, 90);
    const target = limit * 0.83;
    let status = "ok";
    if (total > limit + 10) status = "over";
    else if (total > limit) status = "grace";
    else if (total > target) status = "tight";
    const maxEpochs = n(o.epochMin) > 0 ? Math.floor(((target - (total - train)) * speed - n(o.finalFitMin)) / (n(o.epochMin) * Math.max(1, n(o.folds, 1)))) : null;
    return { total, train, limit, target, status, maxEpochs: maxEpochs !== null ? Math.max(0, maxEpochs) : null };
  }

  /* Rough training-memory estimate for dense models (hypothesis-level).
   * params in millions; bytes per param: weights + grads + Adam moments. */
  function memoryEstimate(o) {
    const p = Number(o.paramsM || 0) * 1e6;
    const prec = o.precision === "fp32" ? 4 : 2;
    const weights = p * (o.precision === "fp32" ? 4 : 2 + 4); // mixed precision keeps fp32 master weights
    const grads = p * prec;
    const optim = o.optimizer === "sgd" ? p * 4 : p * 8;
    const states = o.trainable === false ? 0 : grads + optim;
    const gb = (weights + states) / 1e9;
    return { gb, fits: gb * 1.6 < 24, note: "Excludes activations, which scale with batch size × sequence length (or image size). Treat the ×1.6 margin as a rough guide." };
  }

  /* Credit plan for one challenge: pool capped at 6, +1 every 4 h (S2 §4.6).
   * Simulated hour by hour, spreading the planned scripted submissions evenly so
   * there is time to learn from each run. The refill clock is approximated as
   * running continuously; the global daily pool (~15–25, each credit back 24 h
   * after use) is shown as a separate caution. */
  function creditPlan(o) {
    const hours = Math.max(0, Math.floor(Number(o.hoursLeft || 0)));
    const planned = Math.max(0, Math.floor(Number(o.planned || 0)));
    let credits = Math.min(6, Math.max(0, Math.floor(Number(o.localNow || 0))));
    const maxPossible = credits + Math.floor(hours / 4);
    const schedule = [];
    let wasted = 0;
    for (let h = 0; h <= hours && schedule.length < planned; h++) {
      if (h > 0 && h % 4 === 0) {
        if (credits === 6) wasted++;
        credits = Math.min(6, credits + 1);
      }
      const target = planned > 1 ? Math.round((schedule.length * hours) / Math.max(1, planned - 1)) : 0;
      if (h >= target && credits > 0) {
        credits--;
        schedule.push({ atHour: h, left: credits });
      }
    }
    return { maxPossible, schedule, achieved: schedule.length, enough: schedule.length >= planned, wasted, globalCaution: planned > 15 };
  }

  E.checker = { parseCSV, checkSubmission, runtimeBudget, memoryEstimate, creditPlan };
})();
