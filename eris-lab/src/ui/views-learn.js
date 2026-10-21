/* Views for reference and learning content. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;
  const V = (E.views = E.views || {});

  const pageHead = (eyebrow, title, lede) => `<header class="page-head"><p class="eyebrow">${esc(eyebrow)}</p><h1>${esc(title)}</h1>${lede ? `<p class="lede">${lede}</p>` : ""}</header>`;
  UI.pageHead = pageHead;

  /* ---------------- Dashboard ---------------- */
  V.dashboard = {
    title: "Dashboard",
    render(s) {
      const real = s.experiments.filter((e) => !e.demo);
      const dir = s.settings.direction || "max";
      const best = E.tracker.best(real, dir);
      const bs = best ? E.tracker.stats(best) : null;
      const recs = E.tracker.recommend(real, s.settings).slice(0, 3);
      const wfDone = E.WORKFLOW.reduce((n, w) => n + w.steps.filter((_, i) => s.workflow[w.id + ":" + i]).length, 0);
      const wfTotal = E.WORKFLOW.reduce((n, w) => n + w.steps.length, 0);
      const fcDone = E.FINAL_CHECKLIST.filter((c) => s.checklist[c.id]).length;
      const audit = s.audit;
      return `${pageHead("Project Eris · Shipd", "Solver dashboard", "Platform limits, the state of your current challenge, and what to do next. Everything cites the document it came from, and every number in the panels is one you entered.")}
      <section class="panel-grid status-row" aria-label="Current challenge status">
        <article class="stat">
          <h2 class="stat-label">Best validated run</h2>
          ${best ? `<p class="stat-value num">${UI.fmt(bs.mean)}<small>${bs.sd !== null ? ` ± ${UI.fmt(bs.sd)}` : ""}</small></p><p class="stat-sub">${esc(best.name)} · ${best.status === "verified" ? UI.prov("verified") : '<span class="tag tag-verify">Unverified</span>'}</p>`
              : `<p class="stat-empty">No runs logged yet</p><p class="stat-sub"><a href="#tracker">Log your first run →</a></p>`}
        </article>
        <article class="stat">
          <h2 class="stat-label">Latest audit</h2>
          ${audit ? `<p class="stat-value verdict-${esc(audit.verdict)}">${esc({ blocked: "Blocked", "at-risk": "At risk", review: "Review", clean: "Clean" }[audit.verdict])}</p><p class="stat-sub">${esc(audit.fileName || "pasted code")}: ${audit.counts.blocker} blocker · ${audit.counts.high} high · ${audit.counts.medium} medium</p>`
              : `<p class="stat-empty">Not audited</p><p class="stat-sub"><a href="#auditor">Audit a script →</a></p>`}
        </article>
        <article class="stat">
          <h2 class="stat-label">Workflow progress</h2>
          <p class="stat-value num">${wfDone}<small> / ${wfTotal}</small></p>
          <div class="meter" role="img" aria-label="${wfDone} of ${wfTotal} steps done"><span style="width:${(100 * wfDone / wfTotal).toFixed(1)}%"></span></div>
          <p class="stat-sub"><a href="#workflow">Open the workflow →</a></p>
        </article>
        <article class="stat">
          <h2 class="stat-label">Final checklist</h2>
          <p class="stat-value num">${fcDone}<small> / ${E.FINAL_CHECKLIST.length}</small></p>
          <div class="meter" role="img" aria-label="${fcDone} of ${E.FINAL_CHECKLIST.length} items done"><span style="width:${(100 * fcDone / E.FINAL_CHECKLIST.length).toFixed(1)}%"></span></div>
          <p class="stat-sub"><a href="#review">Final review →</a></p>
        </article>
      </section>

      <section class="section">
        <div class="section-head"><h2>Next steps from your evidence</h2><a class="btn btn-ghost btn-small" href="#tracker">Tracker</a></div>
        <ol class="rec-list">${recs.map((r) => `<li class="rec p${r.priority}"><div><strong>${esc(r.title)}</strong><p class="muted">${esc(r.evidence)}</p><p>${esc(r.action)}</p></div></li>`).join("")}</ol>
      </section>

      <section class="section">
        <div class="section-head"><h2>Platform limits</h2><span class="muted small">From your platform documents</span></div>
        <dl class="facts">${E.KEY_FACTS.map((k) => `<div class="fact"><dt>${esc(k.label)}</dt><dd>${esc(k.value)}</dd><dd class="fact-cite">${UI.cite(k.cite)}</dd></div>`).join("")}</dl>
      </section>

      <section class="section">
        <h2>What happens when you submit a script</h2>
        <ol class="pipeline">${E.CHECKS.map((c) => `<li><span class="pipe-n">${c.n}</span><div><strong>${esc(c.name)}</strong><p>${esc(c.what)}</p><p class="pipe-credit ${c.n === 4 ? "spent" : ""}">${esc(c.credit)}</p></div></li>`).join("")}</ol>
        <p class="muted small">Checks 1–3 run together; only scripts that pass them reach check 4. ${UI.cite("S2 §3.1–3.2, §4.6")}</p>
      </section>

      <section class="section two-col">
        <div>
          <h2>Sources analysed</h2>
          <ul class="sources">${E.SOURCES.map((s2) => `<li><span class="src-id">${s2.id}</span><div><strong>${esc(s2.title)}</strong> <span class="muted small mono">${esc(s2.file)}</span><p>${esc(s2.role)}</p><p class="muted small">Used for: ${esc(s2.used)}</p></div></li>`).join("")}</ul>
        </div>
        <div>
          <h2>Not in your Drive folder</h2>
          <p>These inputs weren't present, so the app doesn't contain any of their numbers. The tracker and review fill in from what you enter.</p>
          <ul class="checks-plain">${E.MISSING_INPUTS.map((m) => `<li>${esc(m)}</li>`).join("")}</ul>
          <h3>How to read the tags</h3>
          <ul class="legend">
            <li>${UI.prov("rule")} stated in a platform document, with its section</li>
            <li>${UI.prov("verified")} observed in your own files (code behaviour, not its score)</li>
            <li>${UI.prov("hypothesis")} general practice; validate it on your data</li>
            <li>${UI.prov("verify")} documents disagree or are unclear; confirm before relying on it</li>
          </ul>
        </div>
      </section>

      <section class="section">
        <h2>Where your documents disagree</h2>
        <div class="conflicts">${E.CONFLICTS.map((c, i) => `<details id="conf-${i}" class="conflict"><summary>${UI.prov("verify")} <strong>${esc(c.topic)}</strong></summary><dl><dt>One source</dt><dd>${esc(c.a)}</dd><dt>Another source</dt><dd>${esc(c.b)}</dd><dt>Working resolution</dt><dd>${esc(c.resolution)}</dd></dl></details>`).join("")}</div>
      </section>`;
    },
  };

  /* ---------------- Rules ---------------- */
  V.rules = {
    title: "Platform rules",
    render() {
      const groups = [...new Set(E.RULES.map((r) => r.group))];
      return `${pageHead("Reference", "Platform rules", "Paraphrased from the Solver Guidebook (S2), the Shipd docs (S1) and the submission instructions (S3). When a challenge's own text adds a real restriction, it overrides these.")}
      <div class="toolbar" role="group" aria-label="Filter rules">
        <label class="field-inline"><span class="sr-only">Filter</span><input id="rules-q" type="search" placeholder="Filter rules…" autocomplete="off"></label>
        <div class="chips" id="rules-sev">${["all", "blocker", "high", "grey", "info"].map((s) => `<button type="button" class="chip${s === "all" ? " on" : ""}" data-sev="${s}">${s === "all" ? "All" : UI.sev(s)}</button>`).join("")}</div>
      </div>
      ${groups.map((g) => `<section class="section rule-group" data-group="${esc(g)}"><h2>${esc(g)}</h2><div class="rule-list">${E.RULES.filter((r) => r.group === g).map((r) => `
        <article class="rule" id="rule-${esc(r.id)}" data-sev="${r.sev}" data-text="${esc((r.title + " " + r.text + " " + r.id).toLowerCase())}">
          <div class="rule-top">${UI.sev(r.sev)}<span class="rule-id mono">${esc(r.id)}</span>${UI.cite(r.cite)}</div>
          <h3>${esc(r.title)}</h3><p>${esc(r.text)}</p></article>`).join("")}</div></section>`).join("")}`;
    },
    mount(root) {
      let sev = "all";
      const apply = () => {
        const q = root.querySelector("#rules-q").value.trim().toLowerCase();
        root.querySelectorAll(".rule").forEach((el) => { el.hidden = !((sev === "all" || el.dataset.sev === sev) && (!q || el.dataset.text.includes(q))); });
        root.querySelectorAll(".rule-group").forEach((g) => { g.hidden = ![...g.querySelectorAll(".rule")].some((r) => !r.hidden); });
      };
      root.querySelector("#rules-q").addEventListener("input", apply);
      root.querySelector("#rules-sev").addEventListener("click", (e) => {
        const b = e.target.closest("[data-sev]"); if (!b) return;
        sev = b.dataset.sev; root.querySelectorAll("#rules-sev .chip").forEach((c) => c.classList.toggle("on", c === b)); apply();
      });
    },
  };

  /* ---------------- Workflow ---------------- */
  V.workflow = {
    title: "Solution workflow",
    render(s) {
      return `${pageHead("Plan", "Solution engineering workflow", "The order matters: validation comes before modelling, and hardening comes before spending credits. Ticks are saved in this browser.")}
      <ol class="workflow">${E.WORKFLOW.map((w, wi) => {
        const done = w.steps.filter((_, i) => s.workflow[w.id + ":" + i]).length;
        return `<li class="wf" id="wf-${w.id}">
          <div class="wf-n" aria-hidden="true">${wi}</div>
          <div class="wf-body">
            <div class="wf-head"><h2>${esc(w.name)}</h2><span class="wf-count num">${done}/${w.steps.length}</span></div>
            <p class="lede-sm">${esc(w.goal)}</p>
            <ul class="checklist">${w.steps.map((st, i) => `<li><label><input type="checkbox" data-wf="${w.id}:${i}" ${s.workflow[w.id + ":" + i] ? "checked" : ""}> <span>${esc(st)}</span></label></li>`).join("")}</ul>
            <p class="wf-done"><strong>Done when:</strong> ${esc(w.done)}</p>
            <a class="btn btn-small btn-ghost" href="#${w.tool}">Open the matching tool →</a>
          </div></li>`;
      }).join("")}</ol>
      <button type="button" class="btn btn-ghost" id="wf-reset">Clear all ticks</button>`;
    },
    mount(root) {
      root.addEventListener("change", (e) => {
        const k = e.target.dataset.wf; if (!k) return;
        UI.store.update((s) => { s.workflow[k] = e.target.checked; });
        const li = e.target.closest(".wf");
        const boxes = li.querySelectorAll("input[data-wf]");
        li.querySelector(".wf-count").textContent = `${[...boxes].filter((b) => b.checked).length}/${boxes.length}`;
      });
      root.querySelector("#wf-reset").addEventListener("click", () => { UI.store.update((s) => { s.workflow = {}; }); E.app.render(); UI.toast("Workflow ticks cleared"); });
    },
  };

  /* ---------------- Leaderboard optimisation ---------------- */
  const TECH_TAGS = [["all", "All"], ["auc", "AUC"], ["logloss", "Log loss"], ["f1", "F1 / labels"], ["rmse", "Regression"], ["ranking", "Ranking"], ["cv", "Vision"], ["nlp", "NLP"], ["banned", "Banned"]];
  V.leaderboard = {
    title: "Leaderboard optimization",
    render() {
      return `${pageHead("Improve", "Leaderboard optimization", "Every technique says when it helps, when it hurts, and how to prove the gain on out-of-fold data before spending a credit.")}
      <section class="section">
        <h2>Metric playbook</h2>
        <div class="metric-picker"><label for="metric-sel">Metric</label>
          <select id="metric-sel">${E.METRICS.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("")}</select></div>
        <div id="metric-card" class="metric-card"></div>
      </section>
      <section class="section">
        <h2>Public-LB noise estimator</h2>
        <p class="muted">How much of a public-LB difference is just sampling noise? This assumes independent rows, so treat the output as an estimate. ${UI.prov("hypothesis")}</p>
        <form class="calc" id="noise-form" autocomplete="off">
          <label>Metric type<select id="nz-type"><option value="prop">Proportion (accuracy, top-1, hit@k)</option><option value="auc">AUC</option><option value="mean">Per-row mean (MRR, MAE, log loss)</option></select></label>
          <label>Public test rows<input id="nz-n" type="number" min="1" value="2000"></label>
          <label>Score<input id="nz-score" type="number" step="any" value="0.85"></label>
          <label class="nz-auc">Positive share (AUC)<input id="nz-pos" type="number" step="any" min="0" max="1" value="0.3"></label>
          <label class="nz-mean">Per-row std (from OOF)<input id="nz-sd" type="number" step="any" value="0.35"></label>
        </form>
        <div id="noise-out" class="calc-out" aria-live="polite"></div>
      </section>
      <section class="section">
        <div class="section-head"><h2>Techniques</h2>
          <div class="chips" id="tech-tags">${TECH_TAGS.map(([t, l]) => `<button type="button" class="chip${t === "all" ? " on" : ""}" data-tag="${t}">${esc(l)}</button>`).join("")}</div></div>
        <div class="tech-grid">${E.TECHNIQUES.map((t) => `<article class="tech${t.tags.includes("banned") ? " banned" : ""}" id="tech-${t.id}" data-tags="${t.tags.join(" ")}">
          <div class="tech-top"><h3>${esc(t.name)}</h3>${UI.prov(t.prov)}</div>
          <dl><dt>Helps when</dt><dd>${esc(t.helps)}</dd><dt>Hurts when</dt><dd>${esc(t.hurts)}</dd><dt>Prove it</dt><dd>${esc(t.validate)}</dd><dt>Compliance</dt><dd>${esc(t.compliance)}</dd></dl></article>`).join("")}</div>
      </section>
      <section class="section callout">
        <h2>Choosing your final submission</h2>
        <p>Choose by matched-split CV plus stability across seeds and the stress split. Use the public LB only to catch pipeline bugs, such as a gap far larger than its noise. ${UI.cite("S2 §1.2")}</p>
      </section>`;
    },
    mount(root) {
      const sel = root.querySelector("#metric-sel");
      const card = root.querySelector("#metric-card");
      const showMetric = () => {
        const m = E.METRICS.find((x) => x.id === sel.value);
        card.innerHTML = `<div class="metric-dir">${m.dir === "max" ? "Higher is better" : "Lower is better"}</div>
          <dl class="kv"><dt>Train with</dt><dd>${esc(m.loss)}</dd><dt>Post-processing (fit on OOF only)</dt><dd>${esc(m.post)}</dd><dt>Watch out</dt><dd>${esc(m.pitfall)}</dd></dl>`;
        card.id = "metric-card";
      };
      sel.addEventListener("change", showMetric); showMetric();

      const out = root.querySelector("#noise-out");
      const calc = () => {
        const type = root.querySelector("#nz-type").value;
        root.querySelector(".nz-auc").hidden = type !== "auc";
        root.querySelector(".nz-mean").hidden = type !== "mean";
        const n = Number(root.querySelector("#nz-n").value);
        const a = Number(root.querySelector("#nz-score").value);
        let se = null;
        if (type === "prop" && a >= 0 && a <= 1) se = Math.sqrt((a * (1 - a)) / n);
        if (type === "auc") {
          const pos = Number(root.querySelector("#nz-pos").value);
          const np = Math.max(1, Math.round(n * pos)), nn = Math.max(1, n - np);
          const q1 = a / (2 - a), q2 = (2 * a * a) / (1 + a);
          se = Math.sqrt(Math.max(0, (a * (1 - a) + (np - 1) * (q1 - a * a) + (nn - 1) * (q2 - a * a)) / (np * nn)));
        }
        if (type === "mean") se = Number(root.querySelector("#nz-sd").value) / Math.sqrt(n);
        if (!(se >= 0) || !n) { out.innerHTML = `<p class="muted">Enter valid values.</p>`; return; }
        out.innerHTML = `<div class="calc-figs"><div><span class="calc-k">Standard error</span><span class="calc-v num">±${se.toFixed(4)}</span></div>
          <div><span class="calc-k">95% interval</span><span class="calc-v num">${(a - 1.96 * se).toFixed(4)} – ${(a + 1.96 * se).toFixed(4)}</span></div></div>
          <p>A public-LB move smaller than about <strong class="num">${se.toFixed(4)}</strong> is weak evidence on its own. Two of your submissions scored on the same rows are correlated, so their true difference is less noisy than this. That's still no reason to prefer the LB over a well-built CV.</p>`;
      };
      root.querySelector("#noise-form").addEventListener("input", calc); calc();

      root.querySelector("#tech-tags").addEventListener("click", (e) => {
        const b = e.target.closest("[data-tag]"); if (!b) return;
        root.querySelectorAll("#tech-tags .chip").forEach((c) => c.classList.toggle("on", c === b));
        const t = b.dataset.tag;
        const map = { rmse: ["rmse", "rmsle", "mae"], f1: ["f1", "qwk", "accuracy"], ranking: ["ranking", "recsys", "contrastive"], cv: ["cv", "audio"], nlp: ["nlp", "finetune"] };
        const want = map[t] || [t];
        root.querySelectorAll(".tech").forEach((el) => { const tags = el.dataset.tags.split(" "); el.hidden = !(t === "all" || tags.includes("all") && t !== "banned" || want.some((w) => tags.includes(w))); });
      });
    },
  };

  /* ---------------- Generalisation ---------------- */
  V.generalization = {
    title: "Unseen-data generalization",
    render(s) {
      return `${pageHead("Improve", "Unseen-data generalization", "The private leaderboard rewards a CV that behaves like the hidden test. Build that first, then check every gain against it.")}
      <section class="section two-col">
        <div>
          <h2>Validation split chooser</h2>
          <form id="split-form" class="stack" autocomplete="off">
            <fieldset><legend>Does the test share entities with train (same users, queries, patients…)?</legend>
              <label><input type="radio" name="ov" value="no"> No: test entities are unseen</label>
              <label><input type="radio" name="ov" value="yes"> Yes: same entities appear</label>
              <label><input type="radio" name="ov" value="unk" checked> Not checked yet</label></fieldset>
            <fieldset><legend>Is the data ordered in time, with the test later?</legend>
              <label><input type="radio" name="tm" value="yes"> Yes</label>
              <label><input type="radio" name="tm" value="no" checked> No or unknown</label></fieldset>
            <fieldset><legend>Is there a named shift axis (language pair, site, domain)?</legend>
              <label><input type="radio" name="ax" value="yes"> Yes</label>
              <label><input type="radio" name="ax" value="no" checked> No</label></fieldset>
            <fieldset><legend>Training size</legend>
              <label><input type="radio" name="sz" value="small"> Under ~5k rows or ~300 groups</label>
              <label><input type="radio" name="sz" value="large" checked> Larger</label></fieldset>
          </form>
        </div>
        <div><div id="split-out" class="split-out" aria-live="polite"></div></div>
      </section>

      <section class="section">
        <h2>Overfitting / underfitting diagnostic</h2>
        <form class="calc" id="diag-form" autocomplete="off">
          <label>Direction<select id="dg-dir"><option value="max">Higher is better</option><option value="min">Lower is better</option></select></label>
          <label>Train score<input id="dg-train" type="number" step="any" placeholder="e.g. 0.95"></label>
          <label>CV mean<input id="dg-cv" type="number" step="any" placeholder="e.g. 0.88"></label>
          <label>CV fold std<input id="dg-sd" type="number" step="any" placeholder="e.g. 0.01"></label>
          <label>Stress split<input id="dg-ood" type="number" step="any" placeholder="optional"></label>
          <label>Public LB<input id="dg-lb" type="number" step="any" placeholder="optional"></label>
        </form>
        <div id="diag-out" class="calc-out" aria-live="polite"><p class="muted">Enter your own numbers from a run log. The rules below are heuristics. ${UI.prov("hypothesis")}</p></div>
      </section>

      <section class="section two-col">
        <div>
          <h2>Leakage checklist</h2>
          <ul class="checklist">${E.LEAKAGE_CHECKS.map((c, i) => `<li id="leak-${i}"><label><input type="checkbox" data-leak="${i}" ${s.leakage[i] ? "checked" : ""}> <span>${esc(c.text)}</span></label>${c.cite ? UI.cite(c.cite) : ""}</li>`).join("")}</ul>
        </div>
        <div>
          <h2>Robustness tests</h2>
          <ul class="rob">${E.ROBUSTNESS_TESTS.map((r, i) => `<li id="rob-${i}"><strong>${esc(r.name)}</strong><p>${esc(r.how)}</p>${r.from ? `<p class="small muted">${esc(r.from)}</p>` : ""}</li>`).join("")}</ul>
        </div>
      </section>

      <section class="section callout">
        <h2>Subgroup evaluation</h2>
        <p>Report the worst subgroup next to the mean, by class, group, length bucket or confidence decile, using OOF predictions. The tracker accepts subgroup scores (<code>name=score, …</code>) and points the next experiment at the weakest one. Shift analysis uses training data only: compare train-internal slices, not test-set statistics. ${UI.cite("S2 §5.2.5")}</p>
      </section>`;
    },
    mount(root) {
      const out = root.querySelector("#split-out");
      const chooser = () => {
        const f = new FormData(root.querySelector("#split-form"));
        const ov = f.get("ov"), tm = f.get("tm"), ax = f.get("ax"), sz = f.get("sz");
        const rec = [];
        if (tm === "yes") rec.push(["Forward-chaining time split", "Train on earlier periods and validate on the latest one, with a gap if features use lagged windows. Random folds leak the future."]);
        if (ov === "no") rec.push(["GroupKFold on the entity", "Every entity lives in exactly one fold. Use StratifiedGroupKFold if the label is imbalanced."]);
        if (ov === "unk") rec.push(["First, measure key overlap", "Count how many test ids or keys also appear in train, looking only at key columns. The answer decides between grouped and stratified folds."]);
        if (ov === "yes" && tm !== "yes") rec.push(["Stratified KFold", "Shared entities make random folds representative. Still group near-duplicates."]);
        if (ax === "yes") rec.push(["Leave-one-group-out stress split", "Hold out each value of the shift axis in turn. Pick models and blend weights on a mix of CV and this score, as your high_score.py did with its 0.5·CV + 0.5·LODO objective."]);
        if (sz === "small") rec.push(["Repeated folds and seed averaging", "With small data, use 3× repeated 5-fold or more seeds, and judge gains against the paired fold SE."]);
        out.innerHTML = `<h3>Recommended validation</h3><ol class="split-rec">${rec.map(([a, b]) => `<li><strong>${esc(a)}</strong><p>${esc(b)}</p></li>`).join("")}</ol>
          <p class="small muted">Fix the fold indices once (seeded) and reuse them for every model so comparisons are paired.</p>`;
      };
      root.querySelector("#split-form").addEventListener("change", chooser); chooser();

      const dout = root.querySelector("#diag-out");
      root.querySelector("#diag-form").addEventListener("input", () => {
        const g = (id) => { const v = root.querySelector(id).value; return v === "" ? null : Number(v); };
        const dir = root.querySelector("#dg-dir").value;
        const tr = g("#dg-train"), cv = g("#dg-cv"), sd = g("#dg-sd"), ood = g("#dg-ood"), lb = g("#dg-lb");
        if (cv === null) { dout.innerHTML = `<p class="muted">Enter at least the CV mean.</p>`; return; }
        const better = (a, b) => (dir === "max" ? a - b : b - a);
        const notes = [];
        if (tr !== null) {
          const gap = better(tr, cv); const rel = gap / Math.max(1e-9, Math.abs(cv));
          if (rel > 0.1) notes.push(["warn", "Overfitting likely", `Train beats CV by ${(rel * 100).toFixed(1)}%. Regularise, reduce capacity or epochs, and add data augmentation where it's allowed.`]);
          else if (rel < 0.01) notes.push(["info", "Possible underfitting", "Train ≈ CV. There is room for more capacity or features, within the runtime budget."]);
          else notes.push(["ok", "Train/CV gap looks normal", `${(rel * 100).toFixed(1)}% relative gap.`]);
        }
        if (ood !== null) {
          const drop = better(cv, ood); const rel = drop / Math.max(1e-9, Math.abs(cv));
          notes.push(rel > 0.05 ? ["warn", "Shift-sensitive", `The stress split is ${(rel * 100).toFixed(1)}% worse than CV. The private LB may sit closer to the stress number.`] : ["ok", "Stable under shift", `The stress split is within ${(Math.abs(rel) * 100).toFixed(1)}% of CV.`]);
        }
        if (lb !== null && sd !== null) {
          const z = Math.abs(lb - cv) / Math.max(1e-9, sd);
          notes.push(z > 3 ? ["warn", "LB far from CV", `|LB − CV| is ${z.toFixed(1)}× the fold std. Check the split construction, metric implementation and preprocessing mismatch.`] : ["ok", "LB consistent with CV", `|LB − CV| = ${z.toFixed(1)}× fold std.`]);
        }
        if (sd !== null) notes.push(["info", "Noise floor", `Changes under about ${(2 * sd / Math.sqrt(5)).toFixed(4)} (2 × SE for 5 folds) are hard to tell from noise without paired fold comparison.`]);
        dout.innerHTML = `<ul class="diag">${notes.map(([k, t, d]) => `<li class="diag-${k}"><strong>${esc(t)}</strong><span>${esc(d)}</span></li>`).join("")}</ul><p class="small muted">Heuristic reading of your numbers. ${UI.prov("hypothesis")}</p>`;
      });
      root.addEventListener("change", (e) => {
        const k = e.target.dataset.leak; if (k === undefined) return;
        UI.store.update((s) => { s.leakage[k] = e.target.checked; });
      });
    },
  };

  /* ---------------- Library ---------------- */
  V.library = {
    title: "Strategy library",
    render() {
      const kinds = [["all", "All"], ["lesson", "Your lessons"], ["strategy", "Strategies"], ["pitfall", "Pitfalls"]];
      return `${pageHead("Improve", "Reusable ML strategy library", "Organised by task type. Lessons come from your high_score.py: the code does what they describe, but no score came with the file, so their effect is unmeasured.")}
      <div class="toolbar">
        <label class="field-inline"><span class="sr-only">Search library</span><input id="lib-q" type="search" placeholder="Search strategies…" autocomplete="off"></label>
        <label class="field-inline">Task <select id="lib-task">${E.TASK_TYPES.map((t) => `<option value="${t.id}">${esc(t.name)}</option>`).join("")}</select></label>
        <div class="chips" id="lib-kind">${kinds.map(([k, l]) => `<button type="button" class="chip${k === "all" ? " on" : ""}" data-kind="${k}">${esc(l)}</button>`).join("")}</div>
      </div>
      <p class="muted small" id="lib-count"></p>
      <div class="lib-grid">${E.LIBRARY.map((l, i) => `<article class="lib kind-${l.kind}" id="lib-${i}" data-task="${l.task}" data-kind="${l.kind}" data-text="${esc((l.title + " " + l.body + " " + l.validate).toLowerCase())}">
        <div class="lib-top"><span class="lib-kind">${esc({ lesson: "Lesson", strategy: "Strategy", pitfall: "Pitfall" }[l.kind])}</span><span class="lib-task">${esc((E.TASK_TYPES.find((t) => t.id === l.task) || {}).name)}</span></div>
        <h3>${esc(l.title)}</h3><p>${esc(l.body)}</p>
        <p class="lib-validate"><strong>Validate:</strong> ${esc(l.validate)}</p>
        <div class="lib-foot">${UI.prov(l.prov, l.cite)}</div></article>`).join("")}</div>`;
    },
    mount(root) {
      let kind = "all";
      const apply = () => {
        const q = root.querySelector("#lib-q").value.trim().toLowerCase();
        const task = root.querySelector("#lib-task").value;
        let n = 0;
        root.querySelectorAll(".lib").forEach((el) => {
          const show = (task === "general" || el.dataset.task === task || el.dataset.task === "general") && (kind === "all" || el.dataset.kind === kind) && (!q || el.dataset.text.includes(q));
          el.hidden = !show; if (show) n++;
        });
        root.querySelector("#lib-count").textContent = `${n} entr${n === 1 ? "y" : "ies"}${task !== "general" ? " (including entries for all tasks)" : ""}`;
      };
      root.querySelector("#lib-q").addEventListener("input", apply);
      root.querySelector("#lib-task").addEventListener("change", apply);
      root.querySelector("#lib-kind").addEventListener("click", (e) => {
        const b = e.target.closest("[data-kind]"); if (!b) return; kind = b.dataset.kind;
        root.querySelectorAll("#lib-kind .chip").forEach((c) => c.classList.toggle("on", c === b)); apply();
      });
      apply();
    },
  };

  /* ---------------- Templates ---------------- */
  V.templates = {
    title: "Templates",
    render() {
      const ext = { python: "py", bash: "sh", markdown: "md" };
      return `${pageHead("Build with Claude", "Reusable templates", "Starting points that already satisfy the runtime, determinism and output rules. The tabular skeleton passes this app's auditor with no blocker or high findings.")}
      ${E.TEMPLATES.map((t) => `<section class="section" id="tpl-${t.id}"><h2>${esc(t.name)}</h2><p class="muted">${esc(t.note)}</p>${UI.codeBlock(t.code, t.lang, { download: t.id + "." + ext[t.lang], tall: t.code.length > 2500 })}</section>`).join("")}`;
    },
  };
})();
