/* Interactive tool views. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;
  const V = (E.views = E.views || {});
  const ph = (...a) => UI.pageHead(...a);

  const CATEGORY_OPTS = [["auto", "Detect from text"], ["general", "General"], ["nlp", "NLP"], ["cv", "Computer vision"], ["retrieval", "RAG / retrieval"], ["finetune", "Fine-tuning"], ["scratch", "From scratch"], ["biochem", "Bio / chem"]];
  const catSelect = (id, val, withAuto = true) => `<select id="${id}">${CATEGORY_OPTS.filter(([k]) => withAuto || k !== "auto").map(([k, l]) => `<option value="${k}" ${k === val ? "selected" : ""}>${esc(l)}</option>`).join("")}</select>`;

  const EXAMPLE_DESC = `# Pump Failure Forecasting
## Overview
Predict whether an industrial pump fails within the next 30 days from daily sensor summaries. The training data covers 140 plants; the test set contains readings from unseen plants. About 4% of rows are positive (class imbalance). Some sensors have missing values.

## Evaluation
Submissions are scored using **AUC-ROC**. Your solution must complete within 60 minutes. No internet access.

## Submission
| Column | Type | Description |
|--------|------|-------------|
| id | int | Row identifier from test.csv |
| prediction | float | Predicted probability from 0 to 1 |

Must contain exactly 48,210 rows.`;

  /* ---------------- Challenge analyzer ---------------- */
  V.analyzer = {
    title: "Challenge analyzer",
    render(s) {
      const text = s.analyzerText !== null && s.analyzerText !== undefined ? s.analyzerText : EXAMPLE_DESC;
      const isExample = text === EXAMPLE_DESC;
      return `${ph("Plan", "Challenge analyzer", "Paste the problem description. The analyzer quotes the phrases behind every inference, maps them to platform rules, and proposes a validation plan and strategies.")}
      <div class="tool-split">
        <form id="an-form" class="stack tool-input" autocomplete="off">
          ${isExample ? `<p class="example-flag" id="an-example">Example description (fictional). Replace it with your challenge text.</p>` : ""}
          <label for="an-text">Problem description</label>
          <textarea id="an-text" rows="16" spellcheck="false">${esc(text)}</textarea>
          <div class="row-fields">
            <label>Category ${catSelect("an-cat", "auto")}</label>
            <label>Metric <select id="an-metric"><option value="">Detect</option>${E.METRICS.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("")}</select></label>
            <label>Training rows <input id="an-n" type="number" min="0" placeholder="optional"></label>
          </div>
          <div class="btn-row"><button type="submit" class="btn btn-primary">Analyze</button><button type="button" class="btn btn-ghost" id="an-clear">Clear</button></div>
        </form>
        <div id="an-out" class="tool-output" aria-live="polite"></div>
      </div>`;
    },
    mount(root) {
      const form = root.querySelector("#an-form");
      const out = root.querySelector("#an-out");
      const run = () => {
        const text = root.querySelector("#an-text").value;
        const a = E.analyzer.analyze(text, { category: root.querySelector("#an-cat").value, metric: root.querySelector("#an-metric").value, nTrain: root.querySelector("#an-n").value });
        const titleM = text.match(/^#\s+(.+)$/m);
        a.title = titleM ? titleM[1].trim() : "";
        // the fictional example is never saved as "your" analysis
        if (text !== EXAMPLE_DESC) UI.store.update((s) => { s.analysis = a; s.analyzerText = text; });
        out.innerHTML = renderAnalysis(a);
        out.querySelector("#an-to-prompt").addEventListener("click", () => {
          UI.store.update((s) => {
            s.promptCtx = Object.assign({}, s.promptCtx, {
              challenge: a.title || s.promptCtx.challenge, category: a.category, task: a.task,
              metric: a.metric ? a.metric.name : "", direction: a.metric ? a.metric.dir : "max", budgetMin: a.budgetMin,
              constraints: [...a.restrictions.map((r) => r.text), a.runtimeLimit ? `Challenge runtime limit ${a.runtimeLimit.minutes} min` : ""].filter(Boolean).join("; "),
            });
            if (a.title) s.settings.challenge = a.title;
            if (a.metric) { s.settings.metric = a.metric.name; s.settings.direction = a.metric.dir; }
            s.settings.budgetMin = a.budgetMin;
          });
          location.hash = "prompts";
        });
      };
      form.addEventListener("submit", (e) => { e.preventDefault(); run(); const ex = root.querySelector("#an-example"); if (ex && root.querySelector("#an-text").value !== EXAMPLE_DESC) ex.remove(); });
      root.querySelector("#an-clear").addEventListener("click", () => { root.querySelector("#an-text").value = ""; root.querySelector("#an-text").focus(); const ex = root.querySelector("#an-example"); if (ex) ex.remove(); });
      run();
    },
  };

  function renderAnalysis(a) {
    const ev = (t) => (t ? `<q class="evidence">${esc(t)}</q>` : "");
    return `
      <div class="an-summary">
        <div><span class="k">Task</span><span class="v">${esc(a.taskName)}</span></div>
        <div><span class="k">Category</span><span class="v">${esc(a.categoryName)}</span></div>
        <div><span class="k">Metric</span><span class="v">${a.metric ? esc(a.metric.name) + `<small> ${a.metric.dir === "max" ? "↑" : "↓"}</small>` : '<span class="warn-text">Not found</span>'}</span></div>
        <div><span class="k">Difficulty</span><span class="v"><span class="dots" aria-label="${a.difficulty} of 5">${[1, 2, 3, 4, 5].map((i) => `<i class="${i <= a.difficulty ? "on" : ""}"></i>`).join("")}</span></span></div>
      </div>
      <p class="small muted">The difficulty dots are a heuristic from detected factors. ${UI.prov("hypothesis")}</p>
      ${a.metric && a.metric.evidence ? `<p class="small">Metric evidence: ${ev(a.metric.evidence)}</p>` : ""}
      ${a.factors.length ? `<ul class="small factor-list">${a.factors.map((f) => `<li>${esc(f)}</li>`).join("")}</ul>` : ""}
      <div class="btn-row"><button type="button" class="btn btn-primary" id="an-to-prompt">Use in prompt generator →</button>${UI.copyBtn(analysisMarkdown(a), "Copy analysis")}</div>

      <h3>How the hidden test was probably built</h3>
      <ol class="hyp-list">${a.split.map((sp) => `<li><div class="hyp-top"><strong>${esc(sp.kind)}</strong><span class="prob prob-${esc(sp.prob)}">${esc(sp.prob)}</span></div><p>${esc(sp.why)} ${ev(sp.evidence)}</p><p class="small"><strong>Validate with:</strong> ${esc(sp.cv)}</p></li>`).join("")}</ol>
      <p class="small muted">Confirm by checking whether test.csv shares key values with train.csv. Look only at the key columns and never fit on test. ${UI.prov("verify")}</p>

      <h3>Submission format</h3>
      ${a.submissionColumns.length ? `<div class="table-wrap"><table><thead><tr><th>Column</th><th>Type</th></tr></thead><tbody>${a.submissionColumns.map((c) => `<tr><td class="mono">${esc(c.name)}</td><td>${esc(c.type)}</td></tr>`).join("")}</tbody></table></div>` : `<p class="muted">No submission table found. Use sample_submission.csv as the reference.</p>`}
      ${a.rows ? `<p>Required rows: <strong class="num">${a.rows.toLocaleString()}</strong></p>` : ""}
      ${a.runtimeLimit ? `<p>Challenge runtime limit: <strong>${a.runtimeLimit.minutes} min</strong> ${ev(a.runtimeLimit.evidence)}. This overrides the 1.5 h default. ${UI.cite("S2 §5.4")} Plan for about ${a.budgetMin} min.</p>` : ""}

      ${a.restrictions.length ? `<h3>Challenge-specific restrictions</h3><ul>${a.restrictions.map((r) => `<li>${esc(r.text)} ${ev(r.evidence)}</li>`).join("")}</ul>` : ""}
      ${a.boilerplate.length ? `<h3>Boilerplate to ignore</h3><ul class="boiler">${a.boilerplate.map((b) => `<li>${ev(b.evidence)}<p>${esc(b.note)}</p></li>`).join("")}</ul>` : ""}

      <h3>Rules that apply</h3>
      <ul class="rule-mini">${a.constraints.map((r) => `<li>${UI.sev(r.sev)} <a href="#rules" data-anchor="rule-${esc(r.id)}"><strong>${esc(r.title)}</strong></a> ${UI.cite(r.cite)}</li>`).join("")}</ul>

      <h3>Improvement opportunities</h3>
      <ul>${a.opportunities.map((o) => `<li>${esc(o)}</li>`).join("")}</ul>
      <p class="small muted">${UI.prov("hypothesis")} Validate each one on matched-split CV.</p>

      ${a.risks.length ? `<h3>Risks</h3><ul class="risk-list">${a.risks.map((r) => `<li>${esc(r.text)} ${UI.cite(r.cite)}</li>`).join("")}</ul>` : ""}
      ${a.questions.length ? `<h3>Ask a reviewer before building on</h3><ul>${a.questions.map((q) => `<li>${esc(q)}</li>`).join("")}</ul>` : ""}

      <h3>Relevant library entries</h3>
      <ul class="lib-mini">${a.library.map((l) => `<li><a href="#library">${esc(l.title)}</a> <span class="small muted">${esc(l.kind)}</span></li>`).join("")}</ul>`;
  }

  function analysisMarkdown(a) {
    return [`# Challenge analysis: ${a.title || ""}`, `- Task: ${a.taskName}; category: ${a.categoryName}; metric: ${a.metric ? a.metric.name : "not found"}`,
      `- Difficulty (heuristic): ${a.difficulty}/5. ${a.factors.join("; ")}`,
      "## Hidden-test hypotheses", ...a.split.map((s) => `- ${s.kind} (${s.prob}): ${s.why}. Validate with: ${s.cv}`),
      "## Restrictions", ...(a.restrictions.length ? a.restrictions.map((r) => `- ${r.text}`) : ["- none detected"]),
      "## Boilerplate to ignore", ...(a.boilerplate.length ? a.boilerplate.map((b) => `- "${b.evidence}": ${b.note}`) : ["- none"]),
      "## Rules", ...a.constraints.map((r) => `- [${r.sev}] ${r.title} (${r.cite})`),
      "## Opportunities (hypotheses)", ...a.opportunities.map((o) => `- ${o}`)].join("\n");
  }

  /* ---------------- Prompt generator ---------------- */
  V.prompts = {
    title: "Claude Code prompt generator",
    render(s) {
      const c = Object.assign({ challenge: "", category: "general", task: "", metric: "", direction: "max", publicDir: "./dataset/public", solutionPath: "solution.py", budgetMin: 75, bestCv: "", constraints: "", focus: "", failureLog: "", template: "recon" }, s.promptCtx);
      return `${ph("Build with Claude", "Claude Code prompt generator", "Every prompt carries the platform contract, the evidence standard and the prohibited shortcuts, so Claude inspects your files before acting, labels what it verified, and won't take banned shortcuts.")}
      <div class="tool-split prompt-split">
        <form id="pg-form" class="stack tool-input" autocomplete="off">
          <fieldset class="tpl-pick"><legend>Stage</legend>
            ${E.PROMPT_ORDER.map((id) => `<label class="tpl-opt" id="prompt-${id}"><input type="radio" name="template" value="${id}" ${c.template === id ? "checked" : ""}><span><strong>${esc(E.PROMPTS[id].name)}</strong><small>${esc(E.PROMPTS[id].when)}</small></span></label>`).join("")}
          </fieldset>
          <label>Challenge name<input id="pg-challenge" value="${esc(c.challenge)}" placeholder="e.g. Cross-lingual provision ranking"></label>
          <div class="row-fields">
            <label>Category ${catSelect("pg-category", c.category, false)}</label>
            <label>Task type <select id="pg-task">${E.TASK_TYPES.filter((t) => t.id !== "general").map((t) => `<option value="${t.id}" ${c.task === t.id ? "selected" : ""}>${esc(t.name)}</option>`).join("")}</select></label>
          </div>
          <div class="row-fields">
            <label>Metric<input id="pg-metric" value="${esc(c.metric)}" placeholder="e.g. MRR"></label>
            <label>Direction<select id="pg-direction"><option value="max" ${c.direction === "max" ? "selected" : ""}>Higher is better</option><option value="min" ${c.direction === "min" ? "selected" : ""}>Lower is better</option></select></label>
          </div>
          <div class="row-fields">
            <label>Data dir<input id="pg-publicDir" value="${esc(c.publicDir)}"></label>
            <label>Solution file<input id="pg-solutionPath" value="${esc(c.solutionPath)}"></label>
            <label>Budget (min)<input id="pg-budgetMin" type="number" min="5" max="120" value="${esc(c.budgetMin)}"></label>
          </div>
          <label>Best validated score so far <small class="muted">(from your logs; leave empty if none)</small><input id="pg-bestCv" value="${esc(c.bestCv)}" placeholder="e.g. CV MRR 0.7412 ± 0.0061 (5 folds)"></label>
          <label>Known challenge constraints<textarea id="pg-constraints" rows="2" placeholder="e.g. model ≤ 100M parameters; 60-min runtime">${esc(c.constraints)}</textarea></label>
          <label class="pg-focus">Focus area for experiments<input id="pg-focus" value="${esc(c.focus)}" placeholder="e.g. hard-negative mining"></label>
          <label class="pg-fail">Failure log (for debugging)<textarea id="pg-failureLog" rows="4" spellcheck="false" placeholder="Paste the error and the last log lines">${esc(c.failureLog)}</textarea></label>
        </form>
        <div class="tool-output"><div id="pg-out"></div></div>
      </div>`;
    },
    mount(root) {
      const form = root.querySelector("#pg-form");
      const out = root.querySelector("#pg-out");
      const fields = ["challenge", "category", "task", "metric", "direction", "publicDir", "solutionPath", "budgetMin", "bestCv", "constraints", "focus", "failureLog"];
      const update = () => {
        const ctx = {};
        fields.forEach((f) => { ctx[f] = root.querySelector("#pg-" + f).value; });
        ctx.template = form.querySelector("input[name=template]:checked").value;
        ctx.categoryName = (CATEGORY_OPTS.find(([k]) => k === ctx.category) || [])[1];
        ctx.taskName = (E.TASK_TYPES.find((t) => t.id === ctx.task) || {}).name;
        root.querySelector(".pg-focus").hidden = ctx.template !== "experiment";
        root.querySelector(".pg-fail").hidden = ctx.template !== "debug";
        UI.store.update((s) => { s.promptCtx = ctx; });
        const text = E.buildPrompt(ctx.template, ctx);
        out.innerHTML = `<div class="section-head"><h2>${esc(E.PROMPTS[ctx.template].name)}</h2><span class="muted small num">${text.length.toLocaleString()} chars</span></div>${UI.codeBlock(text, "prompt", { download: `prompt-${ctx.template}.md`, tall: true })}
          <p class="small muted">Paste this into Claude Code from the challenge folder. Add the CLAUDE.md template there too, so the rules persist across sessions.</p>`;
      };
      form.addEventListener("input", update);
      // Text fields already update on "input". Re-rendering again on their blur
      // ("change") would replace the Copy button mid-click and swallow the click.
      form.addEventListener("change", (e) => { if (e.target.matches("select, input[type=radio]")) update(); });
      update();
    },
  };

  /* ---------------- Solution auditor ---------------- */
  const EXAMPLE_CODE = `# Example script (fictional) with typical problems. Replace it with your solution.
import os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
train = pd.read_csv("./dataset/public/train.csv")
test = pd.read_csv("./dataset/public/test.csv")

scaler = StandardScaler().fit(pd.concat([train, test])[["x1", "x2"]])
folds = KFold(n_splits=5, shuffle=True)
start = time.time()
for epoch in range(100):
    if time.time() - start > 50 * 60:
        break
pred = np.zeros(len(test))
thr = np.quantile(pred, 0.96)
pd.DataFrame({"id": test.id, "prediction": pred > thr}).to_csv("working/submission.csv")
`;
  const CASE = [
    ["blocker", "Device chosen with torch.cuda.is_available()", "Named as a rejection in the determinism check (§3.3). Fix: DEVICE = torch.device(\"cuda\")."],
    ["medium", "try/except around torch.use_deterministic_algorithms", "A version-compatibility fallback. Call it once with warn_only=True, which current torch supports."],
    ["medium", "read_inputs probes several file names and merges targets conditionally", "Use the exact file names from the description. Probing creates paths that depend on the environment."],
    ["medium", "'Established offline' capacity grid and fixed GBDT parameters", "§1.1 prefers in-script search. Widen the seeded in-script grid with a fixed trial count."],
    ["low", "argv defaults (`if len(sys.argv) > 1 else …`)", "A second code path. Read sys.argv[1] and sys.argv[2] unconditionally."],
    ["low", "np.fromstring(…, sep=\" \")", "Deprecated API. np.array(s.split(), dtype=int) is safer across NumPy versions."],
    ["info", "Slate features use sibling candidates of the same test query", "One query's candidate list is the natural inference unit for ranking, like one batch, so this is likely within §5.1.3. Say so in a comment so reviewers see the reasoning. Check with a reviewer if unsure."],
    ["ok", "Strengths", "Train-only vocabulary, seeds throughout, thread pinning, slate-grouped CV plus leave-one-direction-out, a neural-major blend constraint, logged baselines, and output validation before writing."],
  ];
  V.auditor = {
    title: "Solution auditor",
    render(s) {
      return `${ph("Verify", "Solution auditor", "Static checks for the patterns behind compliance rejections, determinism failures, leakage and crashes. It flags lines to inspect; it can't prove a script compliant. Your code stays in this browser.")}
      <div class="tool-split">
        <form id="au-form" class="stack tool-input" autocomplete="off">
          <p class="example-flag" id="au-example">Example script (fictional) with typical problems. Paste yours or load a .py or .ipynb file.</p>
          <div class="row-fields">
            <label>Load file<input id="au-file" type="file" accept=".py,.ipynb,.txt"></label>
            <label>Category ${catSelect("au-cat", (s.analysis && s.analysis.category) || "general", false)}</label>
          </div>
          <label for="au-code">Solution code</label>
          <textarea id="au-code" rows="20" spellcheck="false" class="mono">${esc(EXAMPLE_CODE)}</textarea>
          <div class="btn-row"><button type="submit" class="btn btn-primary">Run audit</button></div>
        </form>
        <div id="au-out" class="tool-output" aria-live="polite"></div>
      </div>
      <section class="section case">
        <h2>Case study: your high_score.py</h2>
        <p>Result of this auditor plus a manual read of the ranking solution in your Drive folder. Load the file above to reproduce it. The repository is public, so the file isn't stored in it.</p>
        <ul class="case-list">${CASE.map(([sv, t, d]) => `<li>${UI.sev(sv)}<div><strong>${esc(t)}</strong><p>${esc(d)}</p></div></li>`).join("")}</ul>
        <p class="small muted">${UI.prov("verified", "S5")} These code patterns are in the file. No score or log came with it, so whether they affected the score is unknown.</p>
      </section>`;
    },
    mount(root) {
      let fileName = "example.py";
      const out = root.querySelector("#au-out");
      let filter = "all";
      const run = () => {
        const code = root.querySelector("#au-code").value;
        const r = E.auditor.audit(code, { category: root.querySelector("#au-cat").value });
        r.fileName = fileName; r.at = Date.now();
        if (fileName !== "example.py") UI.store.update((s) => { s.audit = r; });
        const cats = ["all", ...new Set(r.findings.map((f) => f.cat))];
        const md = auditMarkdown(r);
        const fixPrompt = E.buildPrompt("audit", Object.assign({}, UI.store.load().promptCtx)) + "\n\n## Findings from the static audit (verify each one; some may be false positives)\n" + r.findings.filter((x) => x.sev !== "info").map((x) => `- [${x.sev}] line ${x.lines.filter(Boolean).join(",") || "–"}: ${x.title}. ${x.fix}`).join("\n");
        out.innerHTML = `<div class="verdict verdict-${r.verdict}"><strong>${esc({ blocked: "Blocked: fix the blockers before submitting", "at-risk": "At risk: high-severity findings", review: "Review the medium findings", clean: "No blocker, high or medium findings" }[r.verdict])}</strong>
            <span class="counts">${["blocker", "high", "medium", "low", "info"].map((k) => `${UI.sev(k)}<b class="num">${r.counts[k]}</b>`).join(" ")}</span></div>
          <p class="small muted">${r.lines} lines${r.isNotebook ? " (notebook code cells)" : ""} · imports: ${esc(r.imports.join(", ") || "none")}${fileName === "example.py" ? " · example audits aren't saved to the review" : ""}</p>
          <div class="btn-row">${UI.copyBtn(md, "Copy findings (Markdown)")}${UI.copyBtn(fixPrompt, "Copy fix prompt for Claude")}</div>
          <div class="chips" id="au-cats">${cats.map((c) => `<button type="button" class="chip${c === filter ? " on" : ""}" data-cat="${esc(c)}">${esc(c === "all" ? "All" : c)}</button>`).join("")}</div>
          <ol class="findings">${r.findings.map((x) => `<li class="finding f-${x.sev}" data-cat="${esc(x.cat)}" ${filter !== "all" && x.cat !== filter ? "hidden" : ""}>
            <div class="finding-top">${UI.sev(x.sev)}<span class="f-cat">${esc(x.cat)}</span>${x.lines.filter(Boolean).length ? `<span class="f-line mono">line ${x.lines.filter(Boolean).slice(0, 6).join(", ")}${x.lines.length > 6 ? "…" : ""}</span>` : ""}${UI.cite(x.cite)}</div>
            <h3>${esc(x.title)}</h3>${x.snippet ? `<pre class="snippet"><code>${esc(x.snippet)}</code></pre>` : ""}
            <p>${esc(x.why)}</p><p class="fix"><strong>Fix:</strong> ${esc(x.fix)}</p></li>`).join("")}</ol>`;
        out.querySelector("#au-cats").addEventListener("click", (e) => {
          const b = e.target.closest("[data-cat]"); if (!b) return; filter = b.dataset.cat;
          out.querySelectorAll("#au-cats .chip").forEach((c) => c.classList.toggle("on", c === b));
          out.querySelectorAll(".finding").forEach((li) => { li.hidden = filter !== "all" && li.dataset.cat !== filter; });
        });
      };
      root.querySelector("#au-form").addEventListener("submit", (e) => { e.preventDefault(); if (root.querySelector("#au-code").value !== EXAMPLE_CODE && fileName === "example.py") fileName = "pasted code"; const ex = root.querySelector("#au-example"); if (ex && fileName !== "example.py") ex.remove(); run(); });
      root.querySelector("#au-file").addEventListener("change", async (e) => {
        const f = e.target.files[0]; if (!f) return;
        fileName = f.name;
        root.querySelector("#au-code").value = await UI.readFile(f);
        const ex = root.querySelector("#au-example"); if (ex) ex.remove();
        run();
      });
      root.querySelector("#au-cat").addEventListener("change", run);
      run();
    },
  };
  function auditMarkdown(r) {
    return [`# Audit: ${r.fileName}`, `Verdict: ${r.verdict}. ${Object.entries(r.counts).map(([k, v]) => `${v} ${k}`).join(", ")}`, "",
      "| Severity | Category | Line | Finding | Rule | Fix |", "|---|---|---|---|---|---|",
      ...r.findings.map((x) => `| ${x.sev} | ${x.cat} | ${x.lines.filter(Boolean).join(", ") || "–"} | ${x.title.replace(/\|/g, "/")} | ${x.cite || "–"} | ${x.fix.replace(/\|/g, "/")} |`)].join("\n");
  }

  /* ---------------- Experiment tracker ---------------- */
  const DEMO = [
    { name: "demo-baseline", model: "lgbm", change: "LightGBM defaults, grouped 5-fold", folds: "0.781 0.774 0.790 0.769 0.785", train: 0.86, ood: 0.741, runtime: 6, mem: 3.1, status: "unverified" },
    { name: "demo-hpo", parent: "demo-baseline", model: "lgbm", change: "In-script Optuna, 20 trials", folds: "0.792 0.781 0.799 0.776 0.795", train: 0.88, ood: 0.748, lb: 0.787, runtime: 24, mem: 3.1, status: "unverified" },
    { name: "demo-nn", parent: "demo-hpo", model: "mlp", change: "Add a seeded MLP member", folds: "0.790 0.786 0.795 0.780 0.791", train: 0.83, ood: 0.761, lb: 0.784, runtime: 41, mem: 5.2, subgroups: "large plants=0.81, small plants=0.73", status: "unverified" },
  ];
  V.tracker = {
    title: "Experiment tracker",
    render(s) {
      const set = s.settings;
      const hasDemo = s.experiments.some((e) => e.demo);
      return `${ph("Verify", "Experiment tracker", "Log each run's real numbers. The tracker compares runs on paired folds, separates gains from noise, and recommends the next experiment from that evidence. Data is saved in this browser; export it to keep a copy.")}
      <form class="settings-row" id="tr-settings" autocomplete="off">
        <label>Challenge<input id="ts-challenge" value="${esc(set.challenge || "")}" placeholder="name"></label>
        <label>Metric<input id="ts-metric" value="${esc(set.metric || "")}" placeholder="e.g. AUC"></label>
        <label>Direction<select id="ts-direction"><option value="max" ${set.direction !== "min" ? "selected" : ""}>Higher is better</option><option value="min" ${set.direction === "min" ? "selected" : ""}>Lower is better</option></select></label>
        <label>Runtime budget (min)<input id="ts-budget" type="number" value="${esc(set.budgetMin || 75)}"></label>
      </form>
      ${hasDemo ? `<p class="example-flag">Demo rows (names start with "demo-") are invented to show how the tracker works. The dashboard and the final review ignore them. <button type="button" class="btn btn-small" id="tr-clear-demo">Remove demo rows</button></p>` : ""}
      <div id="tr-main"></div>
      <details class="section add-exp" id="tr-add" ${s.experiments.length ? "" : "open"}>
        <summary><span class="btn btn-primary">Add a run</span></summary>
        <form id="tr-form" class="exp-form" autocomplete="off">
          <input type="hidden" id="te-id">
          <label>Run name *<input id="te-name" required placeholder="exp-07-hardneg"></label>
          <label>Parent run<select id="te-parent"><option value="">(none: baseline)</option></select></label>
          <label>Model family<input id="te-model" placeholder="lgbm / deberta / cnn…"></label>
          <label class="wide">Change (exactly one)<input id="te-change" placeholder="What changed vs the parent"></label>
          <label class="wide">Hypothesis<input id="te-hypothesis" placeholder="Why it should help on unseen data"></label>
          <label class="wide">Per-fold scores<input id="te-folds" placeholder="0.812 0.807 0.815 0.809 0.811" class="mono"></label>
          <label>…or CV mean<input id="te-cv" type="number" step="any"></label>
          <label>CV std<input id="te-cvStd" type="number" step="any"></label>
          <label>Folds (n)<input id="te-nFolds" type="number" min="1"></label>
          <label>Stress split<input id="te-ood" type="number" step="any" placeholder="leave-one-group-out"></label>
          <label>Train score<input id="te-train" type="number" step="any"></label>
          <label>Public LB<input id="te-lb" type="number" step="any"></label>
          <label>Runtime (min)<input id="te-runtime" type="number" step="any"></label>
          <label>Peak memory (GB)<input id="te-mem" type="number" step="any"></label>
          <label class="wide">Subgroup scores<input id="te-subgroups" placeholder="short texts=0.79, long texts=0.71"></label>
          <label>Status<select id="te-status"><option value="unverified">Unverified</option><option value="verified">Verified: from a log of this script</option></select></label>
          <label class="wide">Notes<input id="te-notes" placeholder="log path, commit, observations"></label>
          <div class="btn-row wide"><button type="submit" class="btn btn-primary" id="te-submit">Save run</button><button type="button" class="btn btn-ghost" id="te-cancel">Cancel</button></div>
        </form>
      </details>
      <section class="section io-row">
        <h2>Import / export</h2>
        <div class="btn-row">
          <button type="button" class="btn" id="tr-export-json">Download JSON</button>
          <button type="button" class="btn" id="tr-export-csv">Download CSV</button>
          <span id="tr-copy-json"></span>
          <label class="btn btn-ghost file-btn">Import JSON<input type="file" id="tr-import" accept=".json"></label>
          ${s.experiments.length ? "" : `<button type="button" class="btn btn-ghost" id="tr-demo">Load demo rows</button>`}
        </div>
      </section>`;
    },
    mount(root) {
      const S = () => UI.store.load();
      const dirOf = () => S().settings.direction || "max";
      const main = root.querySelector("#tr-main");
      const form = root.querySelector("#tr-form");
      const fields = ["id", "name", "parent", "model", "change", "hypothesis", "folds", "cv", "cvStd", "nFolds", "ood", "train", "lb", "runtime", "mem", "subgroups", "status", "notes"];

      const parseSub = (txt) => { const o = {}; String(txt || "").split(",").forEach((p) => { const m = p.split("="); if (m.length === 2 && m[0].trim()) o[m[0].trim()] = m[1].trim(); }); return o; };
      const subText = (o) => (o && typeof o === "object" ? Object.entries(o).map(([k, v]) => `${k}=${v}`).join(", ") : o || "");

      const draw = () => {
        const s = S();
        const exps = s.experiments;
        const dir = dirOf();
        const parentSel = root.querySelector("#te-parent");
        const cur = parentSel.value;
        parentSel.innerHTML = `<option value="">(none: baseline)</option>` + exps.map((e) => `<option value="${esc(e.name)}">${esc(e.name)}</option>`).join("");
        parentSel.value = cur;
        UI.setCopy("tr-json", JSON.stringify(exps, null, 2));
        root.querySelector("#tr-copy-json").innerHTML = exps.length ? `<button type="button" class="btn" data-copy="tr-json">Copy JSON</button>` : "";
        if (!exps.length) {
          main.innerHTML = `<div class="empty"><h2>No runs yet</h2><p>Add your baseline first: its per-fold scores, runtime, and whether the numbers come from a log of the current script. Every later run names a parent, so the tracker can compare them fold by fold.</p><p class="muted small">Want to see how it works? Use <strong>Load demo rows</strong> below. Demo numbers are invented and marked as such.</p></div>`;
          return;
        }
        const recs = E.tracker.recommend(exps.filter((e) => !e.demo).length ? exps.filter((e) => !e.demo) : exps, s.settings);
        const best = E.tracker.best(exps, dir);
        main.innerHTML = `
          <section class="section"><div class="section-head"><h2>Runs</h2><span class="muted small">${exps.length} logged · best: <strong>${esc(best ? best.name : "–")}</strong></span></div>
          <div class="table-wrap"><table class="exp-table"><thead><tr><th>Run</th><th>Change</th><th class="r">CV mean</th><th class="r">± sd</th><th class="r">n</th><th>vs parent</th><th class="r">Stress</th><th class="r">Train</th><th class="r">LB</th><th class="r">Min</th><th class="r">GB</th><th>Status</th><th><span class="sr-only">Actions</span></th></tr></thead><tbody>
          ${exps.map((e) => {
            const t = E.tracker.stats(e);
            const p = e.parent ? exps.find((x) => x.name === e.parent) : null;
            const c = p ? E.tracker.compare(e, p, dir) : null;
            const pill = c ? `<span class="vs vs-${c.verdict}" title="${c.se !== null ? `Δ ${UI.fmt(c.delta)} · SE ${UI.fmt(c.se)}${c.paired ? " (paired)" : ""}` : "No fold data for an SE"}">${esc({ better: "Better", worse: "Worse", noise: "Within noise", unknown: "No SE" }[c.verdict])} <span class="num">${c.delta >= 0 ? "+" : ""}${UI.fmt(c.delta)}</span></span>` : '<span class="muted small">baseline</span>';
            return `<tr class="${e === best ? "is-best" : ""}"><td><strong>${esc(e.name)}</strong>${e.demo ? ' <span class="tag tag-demo">demo</span>' : ""}${e.parent ? `<div class="small muted">← ${esc(e.parent)}</div>` : ""}</td>
              <td class="change">${esc(e.change || "")}${e.model ? `<div class="small muted">${esc(e.model)}</div>` : ""}</td>
              <td class="r num">${UI.fmt(t.mean)}</td><td class="r num">${UI.fmt(t.sd)}</td><td class="r num">${t.n || "–"}</td><td>${pill}</td>
              <td class="r num">${UI.fmt(t.ood)}</td><td class="r num">${UI.fmt(t.train)}</td><td class="r num">${UI.fmt(t.lb)}</td><td class="r num">${UI.fmt(t.runtime, 1)}</td><td class="r num">${UI.fmt(t.mem, 1)}</td>
              <td>${e.status === "verified" ? '<span class="tag tag-verified">Verified</span>' : '<span class="tag tag-verify">Unverified</span>'}</td>
              <td class="actions"><button type="button" class="btn btn-small btn-ghost" data-edit="${esc(e.id)}">Edit</button><button type="button" class="btn btn-small btn-ghost" data-dup="${esc(e.id)}">Duplicate</button><button type="button" class="btn btn-small btn-ghost danger" data-del="${esc(e.id)}">Delete</button></td></tr>`;
          }).join("")}</tbody></table></div>
          <p class="small muted">"Better" or "Worse" means the paired fold delta exceeds 2 × its standard error. Anything smaller is reported as within noise.</p></section>
          <section class="section"><h2>Scores by run</h2>${chart(exps, dir, S().settings.metric)}</section>
          <section class="section"><h2>Recommended next experiments</h2>
            <ol class="rec-list">${recs.map((r) => `<li class="rec p${r.priority}"><div><strong>${esc(r.title)}</strong><p class="muted"><span class="ev-k">Evidence:</span> ${esc(r.evidence)}</p><p><span class="ev-k">Next:</span> ${esc(r.action)}</p></div></li>`).join("")}</ol>
            <p class="small muted">Each recommendation is derived from the numbers shown. Whether the next change helps is a hypothesis until it is measured.</p></section>`;
        bindChart(main);
      };

      const fill = (e) => {
        fields.forEach((f) => { const el = root.querySelector("#te-" + f); if (el) el.value = f === "subgroups" ? subText(e[f]) : e[f] === undefined || e[f] === null ? (f === "status" ? "unverified" : "") : e[f]; });
        root.querySelector("#tr-add").open = true;
        root.querySelector("#te-submit").textContent = e.id ? "Update run" : "Save run";
        root.querySelector("#te-name").focus();
      };
      form.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const rec = {};
        fields.forEach((f) => { rec[f] = root.querySelector("#te-" + f).value.trim(); });
        rec.subgroups = parseSub(rec.subgroups);
        const s = S();
        const dupName = s.experiments.find((x) => x.name === rec.name && x.id !== rec.id);
        if (dupName) { UI.toast(`A run named "${rec.name}" already exists. Pick a different name.`); return; }
        UI.store.update((st) => {
          if (rec.id) {
            const i = st.experiments.findIndex((x) => x.id === rec.id);
            if (i >= 0) st.experiments[i] = Object.assign(st.experiments[i], rec);
          } else { rec.id = UI.uid(); rec.created = Date.now(); st.experiments.push(rec); }
        });
        form.reset(); root.querySelector("#te-id").value = ""; root.querySelector("#te-submit").textContent = "Save run";
        UI.toast(`Saved "${rec.name}"`);
        E.app.render();
      });
      root.querySelector("#te-cancel").addEventListener("click", () => { form.reset(); root.querySelector("#te-id").value = ""; root.querySelector("#tr-add").open = false; });
      main.addEventListener("click", (ev) => {
        const b = ev.target.closest("[data-edit],[data-dup],[data-del]"); if (!b) return;
        const s = S();
        if (b.dataset.edit) fill(s.experiments.find((x) => x.id === b.dataset.edit));
        if (b.dataset.dup) { const src = s.experiments.find((x) => x.id === b.dataset.dup); fill(Object.assign({}, src, { id: "", name: src.name + "-copy", parent: src.name, folds: "", cv: "", cvStd: "", lb: "", ood: "", train: "", status: "unverified" })); }
        if (b.dataset.del) {
          if (b.dataset.confirm !== "1") { b.dataset.confirm = "1"; b.textContent = "Confirm delete"; setTimeout(() => { if (b.isConnected) { b.dataset.confirm = ""; b.textContent = "Delete"; } }, 4000); return; }
          UI.store.update((st) => { st.experiments = st.experiments.filter((x) => x.id !== b.dataset.del); });
          UI.toast("Run deleted"); E.app.render();
        }
      });
      root.querySelector("#tr-settings").addEventListener("input", () => {
        UI.store.update((s) => {
          s.settings.challenge = root.querySelector("#ts-challenge").value;
          s.settings.metric = root.querySelector("#ts-metric").value;
          s.settings.direction = root.querySelector("#ts-direction").value;
          s.settings.budgetMin = Number(root.querySelector("#ts-budget").value) || 75;
        });
        draw();
      });
      const demoBtn = root.querySelector("#tr-demo");
      if (demoBtn) demoBtn.addEventListener("click", () => {
        UI.store.update((s) => { DEMO.forEach((d, i) => s.experiments.push(Object.assign({ id: UI.uid(), created: Date.now() + i, demo: true }, d, { subgroups: parseSub(d.subgroups) }))); });
        E.app.render();
      });
      const clr = root.querySelector("#tr-clear-demo");
      if (clr) clr.addEventListener("click", () => { UI.store.update((s) => { s.experiments = s.experiments.filter((e) => !e.demo); }); E.app.render(); });
      root.querySelector("#tr-export-json").addEventListener("click", () => UI.download("eris-experiments.json", JSON.stringify(S().experiments, null, 2), "application/json"));
      root.querySelector("#tr-export-csv").addEventListener("click", () => {
        const cols = ["name", "parent", "model", "change", "hypothesis", "folds", "cv_mean", "cv_sd", "n", "stress", "train", "lb", "runtime_min", "mem_gb", "status", "notes"];
        const q = (v) => `"${String(v === null || v === undefined ? "" : v).replace(/"/g, '""')}"`;
        const rows = S().experiments.map((e) => { const t = E.tracker.stats(e); return [e.name, e.parent, e.model, e.change, e.hypothesis, t.folds.join(" "), t.mean, t.sd, t.n, t.ood, t.train, t.lb, t.runtime, t.mem, e.status, e.notes].map(q).join(","); });
        UI.download("eris-experiments.csv", [cols.join(","), ...rows].join("\n"), "text/csv");
      });
      root.querySelector("#tr-import").addEventListener("change", async (ev) => {
        const f = ev.target.files[0]; if (!f) return;
        try {
          const data = JSON.parse(await UI.readFile(f));
          if (!Array.isArray(data) || !data.every((x) => x && typeof x.name === "string")) throw new Error("Expected a JSON array of runs, each with a name.");
          let added = 0;
          UI.store.update((s) => { data.forEach((x) => { const i = s.experiments.findIndex((y) => y.id === x.id); if (i >= 0) s.experiments[i] = x; else { s.experiments.push(Object.assign({ id: UI.uid(), created: Date.now() }, x)); added++; } }); });
          UI.toast(`Imported ${data.length} runs (${added} new)`); E.app.render();
        } catch (err) { UI.toast("Import failed: " + err.message); }
      });
      draw();
    },
  };

  /* Score chart: CV mean ± sd, stress split and public LB per run. One y-axis. */
  function chart(exps, dir, metric) {
    const pts = exps.map((e) => ({ e, t: E.tracker.stats(e) })).filter((p) => p.t.mean !== null);
    if (!pts.length) return `<p class="muted">No scores to chart yet.</p>`;
    const vals = [];
    pts.forEach(({ t }) => { vals.push(t.mean - (t.sd || 0), t.mean + (t.sd || 0)); if (t.ood !== null) vals.push(t.ood); if (t.lb !== null) vals.push(t.lb); });
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo || Math.abs(hi) * 0.05 || 0.01) * 0.15; lo -= pad; hi += pad;
    const W = 720, H = 280, L = 58, R = 16, T = 16, B = 56;
    const band = (W - L - R) / pts.length;
    const x = (i) => L + (i + 0.5) * band; // each run sits at the centre of its own band
    const y = (v) => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
    const ticks = 5;
    const step = (hi - lo) / ticks;
    const dec = step < 0.001 ? 4 : step < 0.01 ? 3 : 2;
    const has = (k) => pts.some((p) => p.t[k] !== null);
    let svg = `<svg viewBox="0 0 ${W} ${H}" class="chart-svg" role="img" aria-label="Scores by run: CV mean with standard deviation, stress split and public LB">`;
    for (let i = 0; i <= ticks; i++) { const v = lo + step * i; svg += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="axis" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(dec)}</text>`; }
    pts.forEach((p, i) => {
      const cx = x(i);
      svg += `<g class="col" data-i="${i}"><rect class="hit" x="${cx - band / 2}" y="${T}" width="${band}" height="${H - T - B}" fill="transparent"/>`;
      if (p.t.sd !== null) svg += `<line class="err" x1="${cx}" x2="${cx}" y1="${y(p.t.mean - p.t.sd)}" y2="${y(p.t.mean + p.t.sd)}"/><line class="err" x1="${cx - 5}" x2="${cx + 5}" y1="${y(p.t.mean - p.t.sd)}" y2="${y(p.t.mean - p.t.sd)}"/><line class="err" x1="${cx - 5}" x2="${cx + 5}" y1="${y(p.t.mean + p.t.sd)}" y2="${y(p.t.mean + p.t.sd)}"/>`;
      svg += `<circle class="m-cv" cx="${cx}" cy="${y(p.t.mean)}" r="5"/>`;
      if (p.t.ood !== null) svg += `<rect class="m-ood" x="${cx + 9 - 4.5}" y="${y(p.t.ood) - 4.5}" width="9" height="9" rx="1.5"/>`;
      if (p.t.lb !== null) svg += `<path class="m-lb" d="M ${cx - 9} ${y(p.t.lb) - 5.5} l 5.5 5.5 l -5.5 5.5 l -5.5 -5.5 z"/>`;
      const maxChars = Math.max(4, Math.floor(band / 7));
      const label = p.e.name.length > maxChars ? p.e.name.slice(0, maxChars - 1) + "…" : p.e.name;
      svg += `<text class="axis xlab" x="${cx}" y="${H - B + 18}" text-anchor="middle">${esc(label)}</text></g>`;
    });
    svg += `<text class="axis" x="${L}" y="${H - 10}">${esc(metric || "score")} · ${dir === "min" ? "lower" : "higher"} is better</text></svg>`;
    const legend = `<ul class="legend-row"><li><i class="lg lg-cv"></i>CV mean ± sd</li>${has("ood") ? '<li><i class="lg lg-ood"></i>Stress split</li>' : ""}${has("lb") ? '<li><i class="lg lg-lb"></i>Public LB</li>' : ""}</ul>`;
    const data = JSON.stringify(pts.map((p) => ({ n: p.e.name, m: p.t.mean, s: p.t.sd, o: p.t.ood, l: p.t.lb, r: p.t.runtime })));
    const table = `<details class="chart-table"><summary>Show as table</summary><div class="table-wrap"><table><thead><tr><th>Run</th><th class="r">CV mean</th><th class="r">sd</th><th class="r">Stress</th><th class="r">LB</th></tr></thead><tbody>${pts.map((p) => `<tr><td>${esc(p.e.name)}</td><td class="r num">${UI.fmt(p.t.mean)}</td><td class="r num">${UI.fmt(p.t.sd)}</td><td class="r num">${UI.fmt(p.t.ood)}</td><td class="r num">${UI.fmt(p.t.lb)}</td></tr>`).join("")}</tbody></table></div></details>`;
    return `<div class="chart viz-root" data-points='${esc(data)}'>${legend}<div class="chart-box">${svg}<div class="tip" hidden></div></div>${table}</div>`;
  }
  function bindChart(root) {
    const c = root.querySelector(".chart"); if (!c) return;
    const pts = JSON.parse(c.dataset.points);
    const tip = c.querySelector(".tip");
    const box = c.querySelector(".chart-box");
    c.querySelectorAll(".col").forEach((g) => {
      const show = () => {
        const p = pts[Number(g.dataset.i)];
        c.querySelectorAll(".col").forEach((o) => o.classList.toggle("active", o === g));
        tip.innerHTML = `<strong>${esc(p.n)}</strong><span>CV <b class="num">${UI.fmt(p.m)}</b>${p.s !== null ? ` ± ${UI.fmt(p.s)}` : ""}</span>${p.o !== null ? `<span>Stress <b class="num">${UI.fmt(p.o)}</b></span>` : ""}${p.l !== null ? `<span>LB <b class="num">${UI.fmt(p.l)}</b></span>` : ""}${p.r !== null ? `<span>${UI.fmt(p.r, 1)} min</span>` : ""}`;
        tip.hidden = false;
        // keep the tooltip inside the chart box, which clips overflow
        const br = box.getBoundingClientRect(), gr = g.querySelector(".m-cv").getBoundingClientRect();
        const tw = tip.offsetWidth, th = tip.offsetHeight;
        let left = gr.left - br.left + 14;
        if (left + tw > br.width - 4) left = gr.left - br.left - tw - 14;
        const top = Math.min(br.height - th - 4, Math.max(4, gr.top - br.top - th / 2));
        tip.style.left = Math.max(4, left) + "px"; tip.style.top = top + "px";
      };
      g.addEventListener("mouseenter", show);
      g.addEventListener("click", show);
    });
    box.addEventListener("mouseleave", () => { tip.hidden = true; c.querySelectorAll(".col").forEach((o) => o.classList.remove("active")); });
  }

  /* ---------------- Runtime & submission checker ---------------- */
  const FAILURES = [
    ["ModuleNotFoundError", "Package isn't in the Kaggle image", "Run the auditor's import check. Swap to an allowed package. Never pip install."],
    ["CUDA out of memory", "Batch size, sequence length or model too large for the A10G", "Lower the batch size or max length as constants, use AMP/bf16, use gradient accumulation, and free per-fold models (del + empty_cache)."],
    ["Runtime over 1.5 h", "Epochs, folds or HPO trials sized for a faster machine", "Profile the stages, fix smaller counts, cut folds in the final script, or reduce trials."],
    ["IndexError / KeyError on argv or paths", "Hardcoded paths or the wrong argument order", "Path(sys.argv[1]) is the public dir and Path(sys.argv[2]) is the output file. Create the parent dir."],
    ["Malformed submission", "Index column, wrong order, missing ids, NaN", "Validate against sample_submission.csv before writing: to_csv(…, index=False)."],
    ["Determinism check failed", "Unseeded RNG, cuda-if-available, clock or cpu_count logic, try/except fallback", "Run the auditor and fix every Determinism finding. Fix the script before resubmitting."],
    ["Prompt compliance check failed", "Script contradicts the challenge text or platform rules", "Re-read the overview. Fix the approach, not the wording. Resubmitting unchanged won't help."],
    ["Server crash (rare)", "Underlying AWS host failure", "Report it, or simply resubmit. It's random and rare."],
  ];
  V.checker = {
    title: "Runtime & submission checker",
    render() {
      return `${ph("Verify", "Runtime and submission checker", "Catch the failures that cost a credit, or place you below a score of 0, before you submit.")}
      <section class="section">
        <h2>Submission CSV validator</h2>
        <p class="muted">Files are read in this browser only. Add sample_submission.csv (and optionally test.csv) for a full comparison.</p>
        <form id="ck-form" class="ck-files" autocomplete="off">
          <label>submission.csv *<input type="file" id="ck-sub" accept=".csv,text/csv"></label>
          <label>sample_submission.csv<input type="file" id="ck-sample" accept=".csv,text/csv"></label>
          <label>test.csv<input type="file" id="ck-test" accept=".csv,text/csv"></label>
          <label class="wide">…or paste submission CSV text<textarea id="ck-paste" rows="4" class="mono" placeholder="id,prediction&#10;1,0.73"></textarea></label>
          <div class="btn-row wide"><button type="submit" class="btn btn-primary">Check submission</button></div>
        </form>
        <div id="ck-out" aria-live="polite"></div>
      </section>

      <section class="section">
        <h2>Runtime budget</h2>
        <form class="calc" id="rt-form" autocomplete="off">
          <label>Limit (min)<input id="rt-limitMin" type="number" value="90"></label>
          <label>Load + features (min)<input id="rt-featMin" type="number" step="any" value="3"></label>
          <label>Minutes per epoch (local)<input id="rt-epochMin" type="number" step="any" value="2.5"></label>
          <label>Epochs<input id="rt-epochs" type="number" value="4"></label>
          <label>Folds trained<input id="rt-folds" type="number" value="5"></label>
          <label>Final full fit (min)<input id="rt-finalFitMin" type="number" step="any" value="8"></label>
          <label>HPO (min, local)<input id="rt-hpoMin" type="number" step="any" value="0"></label>
          <label>Inference (min, local)<input id="rt-inferMin" type="number" step="any" value="2"></label>
          <label>A10G speed vs local ×<input id="rt-speedup" type="number" step="any" value="1"></label>
        </form>
        <div id="rt-out" aria-live="polite"></div>
        <p class="small muted">The speed factor is your estimate (1 = same speed as where you timed it). Measure one epoch locally and keep the factor conservative. Expected max is 1.5 h, and grace beyond that is usually only 5–10 min. ${UI.cite("S2 §4.4")}</p>
      </section>

      <section class="section two-col">
        <div>
          <h2>GPU memory estimate</h2>
          <form class="calc calc-tight" id="mem-form" autocomplete="off">
            <label>Parameters (millions)<input id="mem-paramsM" type="number" value="184"></label>
            <label>Precision<select id="mem-precision"><option value="amp">Mixed (AMP/bf16)</option><option value="fp32">fp32</option></select></label>
            <label>Optimizer<select id="mem-optimizer"><option value="adam">Adam/AdamW</option><option value="sgd">SGD + momentum</option></select></label>
          </form>
          <div id="mem-out" aria-live="polite"></div>
        </div>
        <div>
          <h2>Credit planner</h2>
          <form class="calc calc-tight" id="cr-form" autocomplete="off">
            <label>Credits left on this challenge<input id="cr-localNow" type="number" min="0" max="6" value="6"></label>
            <label>Hours until close<input id="cr-hoursLeft" type="number" min="0" value="36"></label>
            <label>Scripted submissions planned<input id="cr-planned" type="number" min="0" value="4"></label>
          </form>
          <div id="cr-out" aria-live="polite"></div>
        </div>
      </section>

      <section class="section">
        <h2>Script preflight</h2>
        <p class="muted">Runs the auditor's runtime, contract, environment and determinism rules on a pasted script. Use the <a href="#auditor">auditor</a> for the full compliance review.</p>
        <textarea id="pf-code" rows="8" class="mono" spellcheck="false" placeholder="Paste solution.py here"></textarea>
        <div class="btn-row"><button type="button" class="btn btn-primary" id="pf-run">Run preflight</button></div>
        <div id="pf-out" aria-live="polite"></div>
      </section>

      <section class="section">
        <h2>Failure catalogue</h2>
        <div class="table-wrap"><table><thead><tr><th>Symptom</th><th>Usual cause</th><th>Fix</th></tr></thead><tbody>${FAILURES.map(([a, b, c]) => `<tr><td><strong>${esc(a)}</strong></td><td>${esc(b)}</td><td>${esc(c)}</td></tr>`).join("")}</tbody></table></div>
        <p class="small muted">Only the script run (check 4) consumes a credit, and a malformed output still consumes it. ${UI.cite("S2 §3.2, §4.6")} The <a href="#templates">local check script</a> runs the exact command twice and compares hashes.</p>
      </section>`;
    },
    mount(root) {
      const out = root.querySelector("#ck-out");
      root.querySelector("#ck-form").addEventListener("submit", async (e) => {
        e.preventDefault();
        const get = async (id) => { const f = root.querySelector(id).files[0]; return f ? UI.readFile(f) : null; };
        const sub = (await get("#ck-sub")) || root.querySelector("#ck-paste").value;
        if (!sub.trim()) { out.innerHTML = `<p class="warn-text">Choose a submission file or paste its contents.</p>`; return; }
        const r = E.checker.checkSubmission(sub, await get("#ck-sample"), await get("#ck-test"));
        out.innerHTML = `<ul class="ck-list">${r.findings.map((f) => `<li>${UI.sev(f.sev)}<div><strong>${esc(f.title)}</strong><p>${esc(f.detail)}</p></div></li>`).join("")}</ul>
          ${r.stats.rows !== undefined ? `<p class="small muted">${r.stats.rows.toLocaleString()} rows · id column <code>${esc(r.stats.idColumn)}</code> · ${r.stats.valueColumns.map((c) => `${esc(c.name)}${c.min !== undefined ? ` [${UI.fmt(c.min)} … ${UI.fmt(c.max)}], ${c.distinct} distinct` : ""}`).join("; ")}</p>` : ""}`;
      });

      const rt = root.querySelector("#rt-out");
      const runRt = () => {
        const o = {}; ["limitMin", "featMin", "epochMin", "epochs", "folds", "finalFitMin", "hpoMin", "inferMin", "speedup"].forEach((k) => { o[k] = root.querySelector("#rt-" + k).value; });
        const r = E.checker.runtimeBudget(o);
        const scale = Math.max(120, r.limit + 30, r.total * 1.05);
        const pct = (v) => Math.min(100, (100 * v) / scale);
        const label = { ok: "Fits with margin", tight: "Tight: under the limit but past the 83% target", grace: "In the grace zone. Usually only 5–10 min are allowed", over: "Over the limit plus grace" }[r.status];
        rt.innerHTML = `<div class="rt-meter rt-${r.status}" role="img" aria-label="Estimated ${r.total.toFixed(0)} of ${r.limit} minutes">
            <div class="rt-bar" style="width:${pct(r.total)}%"></div>
            <div class="rt-mark" style="left:${pct(r.target)}%"><span>target ${r.target.toFixed(0)}</span></div>
            <div class="rt-mark rt-limit" style="left:${pct(r.limit)}%"><span>limit ${r.limit}</span></div>
          </div>
          <p class="rt-read"><strong class="num">${r.total.toFixed(1)} min</strong> estimated (training ${r.train.toFixed(1)}). ${esc(label)}.${r.maxEpochs !== null ? ` Max epochs for the target: <strong class="num">${r.maxEpochs}</strong>.` : ""}</p>`;
      };
      root.querySelector("#rt-form").addEventListener("input", runRt); runRt();

      const mem = () => {
        const r = E.checker.memoryEstimate({ paramsM: root.querySelector("#mem-paramsM").value, precision: root.querySelector("#mem-precision").value, optimizer: root.querySelector("#mem-optimizer").value });
        root.querySelector("#mem-out").innerHTML = `<p><strong class="num">${r.gb.toFixed(1)} GB</strong> for weights, gradients and optimizer states. ${r.fits ? "Leaves room for activations on a 24 GB A10G." : "Unlikely to fit a 24 GB A10G once activations are added. Consider a smaller model, LoRA or 8-bit optimizers if the image has them."}</p><p class="small muted">${esc(r.note)} The 24 GB figure is the A10G hardware spec, not from your documents. ${UI.prov("hypothesis")}</p>`;
      };
      root.querySelector("#mem-form").addEventListener("input", mem); mem();

      const cr = () => {
        const r = E.checker.creditPlan({ localNow: root.querySelector("#cr-localNow").value, hoursLeft: root.querySelector("#cr-hoursLeft").value, planned: root.querySelector("#cr-planned").value });
        root.querySelector("#cr-out").innerHTML = `<p>Up to <strong class="num">${r.maxPossible}</strong> scripted submissions possible before close if refills aren't wasted at the 6-credit cap.</p>
          ${r.schedule.length ? `<ol class="cr-plan">${r.schedule.map((x) => `<li><span class="num">+${x.atHour} h</span><span class="pips" aria-label="${x.left} credits left">${Array.from({ length: 6 }, (_, i) => `<i class="${i < x.left ? "on" : ""}"></i>`).join("")}</span></li>`).join("")}</ol>` : ""}
          <p class="${r.enough ? "" : "warn-text"}">${r.enough ? "The plan fits the local credit pool." : `Only ${r.achieved} of the planned submissions fit. Start sooner or plan fewer.`}${r.globalCaution ? " Your global daily pool (~15–25, each credit back 24 h after use) may bind first." : ""}</p>
          <p class="small muted">Spaced evenly so each run can inform the next. The refill timing is approximated. CSV self-checks are free. ${UI.cite("S2 §4.6")}</p>`;
      };
      root.querySelector("#cr-form").addEventListener("input", cr); cr();

      root.querySelector("#pf-run").addEventListener("click", () => {
        const code = root.querySelector("#pf-code").value;
        const o = root.querySelector("#pf-out");
        if (!code.trim()) { o.innerHTML = `<p class="warn-text">Paste a script first.</p>`; return; }
        const r = E.auditor.audit(code);
        const keep = r.findings.filter((x) => ["Runtime", "Contract", "Environment", "Determinism"].includes(x.cat) && x.sev !== "info");
        o.innerHTML = keep.length ? `<ul class="ck-list">${keep.map((x) => `<li>${UI.sev(x.sev)}<div><strong>${esc(x.title)}</strong>${x.lines.filter(Boolean).length ? ` <span class="mono small">line ${x.lines.filter(Boolean).join(", ")}</span>` : ""}<p>${esc(x.fix)}</p></div></li>`).join("")}</ul>` : `<p>No runtime, contract, environment or determinism problems found by the static checks.</p>`;
      });
    },
  };

  /* ---------------- Final review ---------------- */
  V.review = {
    title: "Final solution review",
    render(s) {
      const realExps = s.experiments.filter((e) => !e.demo);
      return `${ph("Verify", "Final solution review", "A prioritised report built from your analysis, your latest audit and the runs you logged. Sections without data say so; nothing is filled in by assumption.")}
      <div class="review-sources">
        <span class="${s.analysis ? "have" : "miss"}">${s.analysis ? "✓" : "–"} Challenge analysis</span>
        <span class="${s.audit ? "have" : "miss"}">${s.audit ? "✓" : "–"} Audit${s.audit ? `: ${esc(s.audit.fileName)}` : ""}</span>
        <span class="${realExps.length ? "have" : "miss"}">${realExps.length ? "✓" : "–"} ${realExps.length} run${realExps.length === 1 ? "" : "s"} logged</span>
      </div>
      <div class="tool-split review-split">
        <section class="tool-input">
          <h2>Final submission checklist</h2>
          <ul class="checklist" id="fc-list">${E.FINAL_CHECKLIST.map((c) => `<li><label><input type="checkbox" data-fc="${c.id}" ${s.checklist[c.id] ? "checked" : ""}> <span>${esc(c.text).replace(/`([^`]+)`/g, "<code>$1</code>")}</span></label>${UI.cite(c.cite)}</li>`).join("")}</ul>
        </section>
        <section class="tool-output">
          <div class="section-head"><h2>Report</h2><span id="rv-actions" class="btn-row"></span></div>
          <div id="rv-out" class="report"></div>
        </section>
      </div>`;
    },
    mount(root) {
      const draw = () => {
        const s = UI.store.load();
        const md = E.report.build(Object.assign({}, s, { experiments: s.experiments.filter((e) => !e.demo) }));
        root.querySelector("#rv-out").innerHTML = UI.md(md);
        root.querySelector("#rv-actions").innerHTML = UI.copyBtn(md, "Copy Markdown") + `<button type="button" class="btn btn-small btn-ghost" id="rv-dl">Download .md</button>`;
        root.querySelector("#rv-dl").addEventListener("click", () => UI.download("final-review.md", md, "text/markdown"));
      };
      root.querySelector("#fc-list").addEventListener("change", (e) => {
        const k = e.target.dataset.fc; if (!k) return;
        UI.store.update((s) => { s.checklist[k] = e.target.checked; });
        draw();
      });
      draw();
    },
  };
})();
