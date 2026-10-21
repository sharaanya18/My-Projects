/* OOF analysis engine: metrics, per-fold and per-subgroup scores, prediction
 * correlation, blend-weight search with an honest held-out estimate,
 * threshold scan and calibration. Operates on training-data predictions only. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const METRICS = {
    auc: { name: "AUC", dir: "max", binary: true },
    logloss: { name: "Log loss", dir: "min", binary: true },
    brier: { name: "Brier", dir: "min", binary: true },
    accuracy: { name: "Accuracy @0.5", dir: "max", binary: true },
    f1: { name: "F1 @0.5", dir: "max", binary: true },
    rmse: { name: "RMSE", dir: "min", binary: false },
    mae: { name: "MAE", dir: "min", binary: false },
  };

  function auc(y, p) {
    const n = y.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => p[a] - p[b]);
    const rank = new Float64Array(n);
    for (let i = 0; i < n;) {
      let j = i;
      while (j + 1 < n && p[idx[j + 1]] === p[idx[i]]) j++;
      const r = (i + j) / 2 + 1; // average rank for ties
      for (let k = i; k <= j; k++) rank[idx[k]] = r;
      i = j + 1;
    }
    let pos = 0, sum = 0;
    for (let i = 0; i < n; i++) if (y[i] === 1) { pos++; sum += rank[i]; }
    const neg = n - pos;
    if (!pos || !neg) return null;
    return (sum - (pos * (pos + 1)) / 2) / (pos * neg);
  }

  function score(metric, y, p, thr = 0.5) {
    const n = y.length;
    if (!n) return null;
    let s = 0;
    switch (metric) {
      case "auc": return auc(y, p);
      case "logloss":
        for (let i = 0; i < n; i++) { const q = Math.min(1 - 1e-15, Math.max(1e-15, p[i])); s -= y[i] * Math.log(q) + (1 - y[i]) * Math.log(1 - q); }
        return s / n;
      case "brier": for (let i = 0; i < n; i++) s += (p[i] - y[i]) ** 2; return s / n;
      case "accuracy": for (let i = 0; i < n; i++) s += (p[i] >= thr ? 1 : 0) === y[i] ? 1 : 0; return s / n;
      case "f1": {
        let tp = 0, fp = 0, fn = 0;
        for (let i = 0; i < n; i++) { const h = p[i] >= thr; if (h && y[i] === 1) tp++; else if (h) fp++; else if (y[i] === 1) fn++; }
        return tp ? (2 * tp) / (2 * tp + fp + fn) : 0;
      }
      case "rmse": for (let i = 0; i < n; i++) s += (p[i] - y[i]) ** 2; return Math.sqrt(s / n);
      case "mae": for (let i = 0; i < n; i++) s += Math.abs(p[i] - y[i]); return s / n;
      default: return null;
    }
  }

  const better = (metric, a, b) => (METRICS[metric].dir === "max" ? a > b : a < b);
  const pick = (arr, ix) => ix.map((i) => arr[i]);

  function groupIndex(keys) {
    const m = new Map();
    keys.forEach((k, i) => { if (!m.has(k)) m.set(k, []); m.get(k).push(i); });
    return m;
  }

  /* Score of `p` within each value of `keys` (folds or subgroups). */
  function byGroup(metric, y, p, keys, limit = 30) {
    const out = [];
    for (const [k, ix] of groupIndex(keys)) out.push({ key: k, n: ix.length, score: score(metric, pick(y, ix), pick(p, ix)) });
    out.sort((a, b) => b.n - a.n);
    return out.slice(0, limit);
  }

  function pearson(a, b) {
    const n = a.length;
    let ma = 0, mb = 0;
    for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const da = a[i] - ma, db = b[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
    return saa && sbb ? sab / Math.sqrt(saa * sbb) : null;
  }

  /* All weight vectors on a simplex grid with the given step. */
  function simplex(k, step) {
    const units = Math.round(1 / step);
    const out = [];
    const rec = (left, parts) => {
      if (parts.length === k - 1) { out.push([...parts, left].map((u) => u / units)); return; }
      for (let u = 0; u <= left; u++) rec(left - u, [...parts, u]);
    };
    rec(units, []);
    return out;
  }

  function blendPreds(models, w, ix) {
    const n = ix ? ix.length : models[0].length;
    const out = new Float64Array(n);
    for (let m = 0; m < models.length; m++) {
      if (!w[m]) continue;
      const col = models[m];
      for (let i = 0; i < n; i++) out[i] += w[m] * col[ix ? ix[i] : i];
    }
    return out;
  }

  function bestWeights(metric, y, models, ix, grid) {
    const yy = ix ? pick(y, ix) : y;
    let best = null;
    for (const w of grid) {
      const s = score(metric, yy, blendPreds(models, w, ix));
      if (s !== null && (!best || better(metric, s, best.score))) best = { w, score: s };
    }
    return best;
  }

  /* Blend search. The honest estimate fits weights on half of the folds and
   * scores the other half (and vice versa), so it isn't tuned on the rows it reports. */
  function blend(metric, y, models, folds) {
    const k = models.length;
    if (k < 2) return null;
    const step = k === 2 ? 0.05 : k <= 4 ? 0.1 : 0.25;
    const grid = simplex(k, step);
    const all = bestWeights(metric, y, models, null, grid);
    let A = [], B = [];
    if (folds) {
      const ids = [...new Set(folds)].sort();
      const half = new Set(ids.filter((_, i) => i % 2 === 0));
      folds.forEach((f, i) => (half.has(f) ? A : B).push(i));
    } else {
      let s = 12345;
      y.forEach((_, i) => { s = (s * 1103515245 + 12345) % 2147483648; (s / 2147483648 < 0.5 ? A : B).push(i); });
    }
    let honest = null;
    if (A.length && B.length) {
      const wA = bestWeights(metric, y, models, A, grid).w;
      const wB = bestWeights(metric, y, models, B, grid).w;
      const sB = score(metric, pick(y, B), blendPreds(models, wA, B));
      const sA = score(metric, pick(y, A), blendPreds(models, wB, A));
      honest = (sA * A.length + sB * B.length) / (A.length + B.length);
    }
    const singles = models.map((m) => score(metric, y, m));
    let bestSingle = 0;
    singles.forEach((s, i) => { if (better(metric, s, singles[bestSingle])) bestSingle = i; });
    return { weights: all.w, score: all.score, honest, singles, bestSingle, step, gridSize: grid.length };
  }

  function thresholdScan(metric, y, p, steps = 100) {
    const pts = [];
    let best = null;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const s = score(metric, y, p, t);
      pts.push({ t, s });
      if (!best || s > best.s) best = { t, s };
    }
    return { pts, best, at05: score(metric, y, p, 0.5) };
  }

  function calibration(y, p, bins = 10) {
    const B = Array.from({ length: bins }, () => ({ n: 0, sp: 0, sy: 0 }));
    for (let i = 0; i < y.length; i++) {
      const b = Math.min(bins - 1, Math.max(0, Math.floor(p[i] * bins)));
      B[b].n++; B[b].sp += p[i]; B[b].sy += y[i];
    }
    let ece = 0;
    const out = B.map((b, i) => {
      if (!b.n) return { bin: i, n: 0, pred: null, obs: null };
      const pred = b.sp / b.n, obs = b.sy / b.n;
      ece += (b.n / y.length) * Math.abs(pred - obs);
      return { bin: i, n: b.n, pred, obs };
    });
    return { bins: out, ece };
  }

  /* Build a dataset from parsed CSV rows. */
  function prepare(table, cols) {
    const h = table.header.map((x) => x.trim());
    const col = (name) => h.indexOf(name);
    const iy = col(cols.target);
    const ips = cols.preds.map(col);
    const ifold = cols.fold ? col(cols.fold) : -1;
    const igroup = cols.group ? col(cols.group) : -1;
    const y = [], P = ips.map(() => []), folds = ifold >= 0 ? [] : null, groups = igroup >= 0 ? [] : null;
    let skipped = 0;
    for (const r of table.rows) {
      const yv = Number(r[iy]);
      const pv = ips.map((i) => Number(r[i]));
      if (!Number.isFinite(yv) || pv.some((v) => !Number.isFinite(v))) { skipped++; continue; }
      y.push(yv); pv.forEach((v, j) => P[j].push(v));
      if (folds) folds.push(String(r[ifold]).trim());
      if (groups) groups.push(String(r[igroup]).trim());
    }
    const binary = y.length > 0 && y.every((v) => v === 0 || v === 1);
    const probs = P.every((col) => col.every((v) => v >= 0 && v <= 1));
    return { y, P, folds, groups, skipped, binary, probs, names: cols.preds };
  }

  function analyze(ds, metric, opts = {}) {
    const res = { metric, info: METRICS[metric], n: ds.y.length, models: [] };
    ds.P.forEach((p, j) => {
      const m = { name: ds.names[j], score: score(metric, ds.y, p) };
      if (ds.folds) {
        const f = byGroup(metric, ds.y, p, ds.folds, 100).sort((a, b) => String(a.key).localeCompare(String(b.key), undefined, { numeric: true }));
        m.folds = f;
        const v = f.map((x) => x.score).filter((x) => x !== null);
        const mean = v.reduce((s, x) => s + x, 0) / v.length;
        m.foldMean = mean;
        m.foldSd = v.length > 1 ? Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (v.length - 1)) : null;
      }
      res.models.push(m);
    });
    let best = 0;
    res.models.forEach((m, i) => { if (m.score !== null && better(metric, m.score, res.models[best].score)) best = i; });
    res.best = best;
    const focus = opts.focus !== undefined ? opts.focus : best;
    if (ds.groups) res.subgroups = byGroup(metric, ds.y, ds.P[focus], ds.groups, 25);
    res.corr = ds.P.map((a) => ds.P.map((b) => pearson(a, b)));
    if (ds.P.length >= 2 && ds.P.length <= 6) res.blend = blend(metric, ds.y, ds.P, ds.folds);
    if (ds.binary && ds.probs) {
      res.calibration = calibration(ds.y, ds.P[focus]);
      res.threshold = { f1: thresholdScan("f1", ds.y, ds.P[focus]), accuracy: thresholdScan("accuracy", ds.y, ds.P[focus]) };
    }
    res.focus = focus;
    return res;
  }

  /* A labelled fictional OOF file for trying the lab: 3 models, 5 folds, 2 regions. */
  function sampleCSV() {
    let s = 7;
    const r = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648; };
    const g = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
    const sig = (x) => 1 / (1 + Math.exp(-x));
    const rows = ["id,fold,region,target,lgbm,catboost,mlp"];
    for (let i = 0; i < 1500; i++) {
      const region = r() < 0.7 ? "north" : "south";
      const z = g();
      const y = sig(1.6 * z - 1.2 + (region === "south" ? 0.3 : 0)) > r() ? 1 : 0;
      const noise = region === "south" ? 1.3 : 0.9;
      const a = sig(1.5 * z - 1.1 + noise * g() * 0.8);
      const b = sig(1.4 * z - 1.0 + noise * g() * 0.85);
      const c = sig(1.1 * z - 1.0 + noise * g() * 1.1);
      rows.push([i + 1, i % 5, region, y, a.toFixed(4), b.toFixed(4), c.toFixed(4)].join(","));
    }
    return rows.join("\n");
  }

  E.oof = { METRICS, score, auc, byGroup, pearson, blend, simplex, thresholdScan, calibration, prepare, analyze, sampleCSV };
})();
