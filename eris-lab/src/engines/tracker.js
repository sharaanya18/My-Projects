/* Experiment tracker logic: statistics, paired comparisons and evidence-based
 * next-step recommendations. Pure functions; storage lives in the UI layer.
 * Nothing here invents numbers. Every recommendation names the entries and
 * values it was derived from. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const num = (v) => (v === "" || v === null || v === undefined || !Number.isFinite(Number(v)) ? null : Number(v));
  const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
  const sd = (a) => { if (a.length < 2) return null; const m = mean(a); return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length - 1)); };

  function parseFolds(v) {
    if (Array.isArray(v)) return v.map(num).filter((x) => x !== null);
    return String(v || "").split(/[\s,;|]+/).map(num).filter((x) => x !== null);
  }

  /* Normalised view of one experiment. cv comes from folds when given. */
  function stats(e) {
    const folds = parseFolds(e.folds);
    const m = folds.length ? mean(folds) : num(e.cv);
    const s = folds.length >= 2 ? sd(folds) : num(e.cvStd);
    const n = folds.length || (num(e.nFolds) || null);
    const se = s !== null && n ? s / Math.sqrt(n) : null;
    return { mean: m, sd: s, n, se, folds, ood: num(e.ood), train: num(e.train), lb: num(e.lb), runtime: num(e.runtime), mem: num(e.mem) };
  }

  const better = (dir, a, b) => (dir === "min" ? a < b : a > b);
  const gain = (dir, a, b) => (dir === "min" ? b - a : a - b); // positive = a better than b

  /* Paired comparison of a vs b (b = parent/baseline). */
  function compare(a, b, dir = "max") {
    const A = stats(a), B = stats(b);
    if (A.mean === null || B.mean === null) return { verdict: "unknown", delta: null };
    const delta = gain(dir, A.mean, B.mean);
    let se = null, paired = false;
    if (A.folds.length >= 2 && A.folds.length === B.folds.length) {
      const d = A.folds.map((x, i) => gain(dir, x, B.folds[i]));
      const s = sd(d);
      se = s !== null ? s / Math.sqrt(d.length) : null;
      paired = true;
    } else if (A.se !== null && B.se !== null) {
      se = Math.sqrt(A.se ** 2 + B.se ** 2);
    }
    let verdict = "unknown";
    if (se !== null && se > 0) verdict = delta > 2 * se ? "better" : delta < -2 * se ? "worse" : "noise";
    else if (se === 0) verdict = delta > 0 ? "better" : delta < 0 ? "worse" : "noise";
    const oodDelta = A.ood !== null && B.ood !== null ? gain(dir, A.ood, B.ood) : null;
    return { delta, se, paired, verdict, z: se ? delta / se : null, oodDelta };
  }

  function pearson(xs, ys) {
    if (xs.length < 3) return null;
    const mx = mean(xs), my = mean(ys);
    let sxy = 0, sxx = 0, syy = 0;
    xs.forEach((x, i) => { sxy += (x - mx) * (ys[i] - my); sxx += (x - mx) ** 2; syy += (ys[i] - my) ** 2; });
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : null;
  }

  function best(exps, dir) {
    let b = null;
    exps.forEach((e) => { const s = stats(e); if (s.mean !== null && (!b || better(dir, s.mean, stats(b).mean))) b = e; });
    return b;
  }

  const fmt = (x, d = 4) => (x === null || x === undefined ? "–" : Number(x).toFixed(d));

  function recommend(exps, settings = {}) {
    const dir = settings.direction || "max";
    const budget = Number(settings.budgetMin || 50);
    const R = [];
    const add = (priority, title, evidence, action) => R.push({ priority, title, evidence, action });
    const scored = exps.filter((e) => stats(e).mean !== null);

    if (!exps.length) {
      add(1, "Establish a baseline", "No experiments recorded yet.", "Run the compliant skeleton with one simple model and log per-fold scores plus a constant or random baseline.");
      return R;
    }
    if (!scored.length) {
      add(1, "Record validation scores", "None of the entries has a CV score or fold scores.", "Enter per-fold scores from your logs. The tracker needs them to judge noise.");
      return R;
    }
    const noFolds = scored.filter((e) => stats(e).folds.length < 2 && stats(e).sd === null);
    if (noFolds.length) add(2, "Record per-fold scores", `${noFolds.map((e) => e.name).join(", ")} ${noFolds.length > 1 ? "have" : "has"} no fold scores or std, so gains can't be separated from noise.`, "Paste the per-fold scores from the training log.");

    const top = best(scored, dir);
    const T = stats(top);
    const parent = top.parent ? exps.find((e) => e.name === top.parent || e.id === top.parent) : null;
    if (parent) {
      const c = compare(top, parent, dir);
      if (c.verdict === "noise") add(1, `"${top.name}" is not distinguishable from "${parent.name}"`, `Δ ${fmt(c.delta)} vs ${c.paired ? "paired " : ""}SE ${fmt(c.se)} (|z| = ${fmt(Math.abs(c.z), 2)} < 2).`, "Prefer the simpler of the two, or add seeds or folds to tighten the estimate before building on it.");
      if (c.verdict === "better" && c.oodDelta !== null && c.oodDelta < 0) add(1, `"${top.name}" gains CV but loses on the stress split`, `CV Δ ${fmt(c.delta)}, stress Δ ${fmt(c.oodDelta)}.`, "It is likely fitting group-specific patterns. Reduce capacity or regularise, and keep the parent if the stress loss persists.");
    }

    // overfitting signal
    if (T.train !== null && T.mean !== null) {
      const gap = gain(dir, T.train, T.mean);
      const rel = Math.abs(T.mean) > 1e-9 ? gap / Math.abs(T.mean) : gap;
      if (rel > 0.1) add(2, "Large train-vs-CV gap", `"${top.name}": train ${fmt(T.train)} vs CV ${fmt(T.mean)} (${(rel * 100).toFixed(1)}% relative).`, "Try stronger regularisation, fewer epochs or lower capacity. Then compare with a learning curve at 50% data.");
      else if (rel < 0.01 && scored.length >= 2) add(3, "Train ≈ CV: possible underfitting", `"${top.name}": train ${fmt(T.train)} vs CV ${fmt(T.mean)}.`, "There is headroom for capacity: more trees, larger model, more epochs within budget, richer features.");
    }

    // stress split vs CV
    if (T.ood !== null && T.mean !== null) {
      const drop = gain(dir, T.mean, T.ood);
      const rel = Math.abs(T.mean) > 1e-9 ? drop / Math.abs(T.mean) : drop;
      if (rel > 0.05) add(1, "Stress-split score well below CV", `"${top.name}": CV ${fmt(T.mean)} vs stress ${fmt(T.ood)} (${(rel * 100).toFixed(1)}% lower).`, "If the hidden test resembles the stress split, the private LB will look like the stress number. Prioritise experiments that raise it, even at small CV cost.");
    } else add(3, "Add a stress split", "No stress-split (leave-one-group-out / temporal) score on the best run.", "Score the best config on a split that mimics the hidden-test construction.");

    // CV ↔ LB agreement
    const both = scored.filter((e) => stats(e).lb !== null);
    if (both.length >= 3) {
      const r = pearson(both.map((e) => stats(e).mean), both.map((e) => stats(e).lb));
      if (r !== null && r < 0.3) add(1, "CV and public LB disagree", `Correlation across ${both.length} submissions: r = ${fmt(r, 2)}.`, "Either the public split is small and noisy, or your CV doesn't match the test construction. Check key overlap and rebuild folds. Don't chase LB.");
      else if (r !== null) add(4, "CV tracks the public LB", `r = ${fmt(r, 2)} over ${both.length} submissions.`, "Keep selecting by CV. The public LB is consistent with it.");
    } else if (both.length) add(4, "Too few LB points to judge CV↔LB agreement", `${both.length} submission(s) with a public LB score.`, "Keep logging LB next to CV. Three or more points give a first read.");

    // subgroups
    if (top.subgroups && typeof top.subgroups === "object") {
      const entries = Object.entries(top.subgroups).map(([k, v]) => [k, num(v)]).filter(([, v]) => v !== null);
      if (entries.length >= 2) {
        entries.sort((a, b) => (better(dir, a[1], b[1]) ? 1 : -1));
        const worst = entries[0];
        const bestG = entries[entries.length - 1];
        add(2, `Weakest subgroup: ${worst[0]}`, `"${top.name}": ${worst[0]} = ${fmt(worst[1])} vs ${bestG[0]} = ${fmt(bestG[1])}.`, `Run error analysis on ${worst[0]}: read the worst examples, then test one fix the model can learn.`);
      }
    }

    // runtime
    if (T.runtime !== null) {
      if (T.runtime > budget) add(1, "Runtime over budget", `"${top.name}" takes ${fmt(T.runtime, 1)} min, budget ${budget} min.`, "Use the runtime prompt: profile stages, cut epochs, folds or HPO trials, and use AMP. Fix the new values as constants.");
      else if (T.runtime < budget * 0.4) add(3, "Runtime headroom available", `"${top.name}" uses ${fmt(T.runtime, 1)} of ${budget} min.`, "Spend it on seed bagging or a second diverse ensemble member, and validate the gain against noise.");
    }

    // plateau
    const chron = scored.slice().sort((a, b) => (a.created || 0) - (b.created || 0));
    if (chron.length >= 4) {
      const last = chron.slice(-3);
      const gains = last.map((e) => { const p = e.parent && exps.find((x) => x.name === e.parent || x.id === e.parent); return p ? compare(e, p, dir).verdict : null; }).filter(Boolean);
      if (gains.length === 3 && gains.every((v) => v !== "better")) add(2, "Plateau: the last 3 changes didn't beat their parents", last.map((e) => e.name).join(" → "), "Switch levers: a different model family for ensemble diversity, a different loss aligned with the metric, or error-analysis-driven features.");
    }

    // ensembling opportunity
    const families = new Set(scored.map((e) => (e.model || "").toLowerCase()).filter(Boolean));
    if (families.size >= 2 && !scored.some((e) => /blend|ensemble|stack/i.test(e.name + " " + (e.change || "")))) add(3, "Try an OOF blend", `${families.size} model families recorded (${[...families].join(", ")}) and no blend yet.`, "Blend on OOF with a coarse grid, and check the stress split before keeping it.");

    const unverified = exps.filter((e) => e.status !== "verified");
    if (unverified.length) add(4, "Unverified entries", `${unverified.length} entr${unverified.length > 1 ? "ies are" : "y is"} not marked verified.`, "Mark an entry verified only when its numbers come from a log of an end-to-end run of the current script.");

    return R.sort((a, b) => a.priority - b.priority);
  }

  E.tracker = { stats, compare, recommend, best, parseFolds, pearson };
})();
