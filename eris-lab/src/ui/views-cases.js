/* Case studies from the private corpus pack: your past challenges, what the
 * leaders did, your gaps, and real runs that can open as a tracker workspace.
 * Without a pack this view explains how to load one; the public build has none. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;
  const V = (E.views = E.views || {});

  const STATUS = { closed: "Closed", open: "Open" };

  /* One score strip per case on its own scale: your CV, public and private
   * against the AI baseline (dashed) and the board top. */
  function strip(c) {
    const s = c.scores || {};
    const pts = [
      ["cv", "Your CV", s.ourCV], ["pub", "Your public", s.ourPublic], ["priv", "Your private", s.ourPrivate],
      ["top", "Board top", s.boardTopPrivate !== undefined ? s.boardTopPrivate : s.boardTopPublic],
    ].filter((p) => typeof p[2] === "number");
    const base = typeof s.aiBaseline === "number" ? s.aiBaseline : null;
    if (!pts.length && base === null) return `<p class="small muted">No scores were supplied for this challenge.</p>`;
    const vals = pts.map((p) => p[2]).concat(base !== null ? [base] : []);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = Math.max((hi - lo) * 0.18, 0.01); lo -= pad; hi += pad;
    const W = 520, H = 64, L = 12, R = 12;
    const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    let g = `<svg viewBox="0 0 ${W} ${H}" class="strip" role="img" aria-label="${esc(c.title)} scores: ${pts.map((p) => `${p[1]} ${p[2]}`).join(", ")}${base !== null ? `, AI baseline ${base}` : ""}">`;
    g += `<line x1="${L}" x2="${W - R}" y1="30" y2="30" class="strip-axis"/>`;
    if (base !== null) g += `<line x1="${x(base)}" x2="${x(base)}" y1="10" y2="50" class="strip-base"/><text x="${x(base)}" y="60" text-anchor="middle" class="strip-lab">baseline ${base}</text>`;
    pts.forEach(([k, , v], i) => {
      const cx = x(v);
      g += k === "top" ? `<path d="M ${cx} 22 l 7 8 l -7 8 l -7 -8 z" class="sm-${k}"><title>Board top ${v}</title></path>` : `<circle cx="${cx}" cy="30" r="7" class="sm-${k}"><title>${esc(k)} ${v}</title></circle>`;
      g += `<text x="${cx}" y="${i % 2 ? 16 : 13}" text-anchor="middle" class="strip-val">${v}</text>`;
    });
    return g + "</svg>";
  }

  function caseCard(c, pack) {
    const lessons = (c.lessons || []).map((id) => pack.lessons.find((l) => l.id === id)).filter(Boolean);
    const s = c.scores || {};
    const metric = E.METRICS.find((m) => m.id === c.metric);
    const beat = typeof s.aiBaseline === "number" && [s.ourPrivate, s.ourPublic].find((v) => typeof v === "number");
    const verdict = beat === undefined || beat === false ? "" : (c.dir === "min" ? beat < s.aiBaseline : beat > s.aiBaseline) ? '<span class="vs vs-better">Above baseline</span>' : '<span class="vs vs-worse">Below baseline</span>';
    return `<article class="case-card cat-${esc(c.category)}" id="case-${esc(c.slug)}" data-cat="${esc(c.category)}" data-task="${esc(c.task)}">
      <header class="cc-head"><div><h3>${esc(c.title)}</h3><p class="small muted">${esc(c.family)}</p></div>
        <div class="cc-tags"><span class="tag tag-rule">${esc(STATUS[c.status] || c.status)}</span>${metric ? `<span class="tag tag-hypothesis">${esc(metric.name)} ${c.dir === "min" ? "↓" : "↑"}</span>` : ""}${verdict}</div></header>
      ${strip(c)}
      ${s.cvKind ? `<p class="small muted">CV: ${esc(s.cvKind)}${s.ourRank ? ` · your rank ${esc(s.ourRank)}` : ""}${s.note ? ` · ${esc(s.note)}` : ""}</p>` : s.note ? `<p class="small muted">${esc(s.note)}</p>` : ""}
      <p>${esc(c.summary)}</p>
      <p class="cc-lesson"><strong>Key lesson:</strong> ${esc(c.keyLesson)}</p>
      <div class="cc-cols">
        ${c.leadersDid && c.leadersDid.length ? `<div><h4>What the leaders did ${UI.prov("observed")}</h4><ul>${c.leadersDid.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : ""}
        ${c.ourGaps && c.ourGaps.length ? `<div><h4>Your gaps</h4><ul>${c.ourGaps.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></div>` : ""}
      </div>
      ${lessons.length ? `<p class="cc-lessons">${lessons.map((l) => `<a href="#library" data-anchor="lib-pack-${esc(l.id)}" class="chip">${esc(l.id)} · ${esc(l.title)}</a>`).join("")}</p>` : ""}
      <div class="btn-row">${(c.runs || []).length ? `<button type="button" class="btn btn-small btn-primary" data-open-case="${esc(c.slug)}">Open as workspace (${c.runs.length} real run${c.runs.length === 1 ? "" : "s"})</button>` : ""}<a class="btn btn-small btn-ghost" href="#prompts" data-prompt-case="${esc(c.slug)}">Use its lessons in a prompt</a></div>
    </article>`;
  }

  /* The platform's own worked examples: public, so shown with or without a pack. */
  function exampleCard(x) {
    const metric = E.METRICS.find((m) => m.id === x.metric);
    const req = x.rubric.filter((r) => r[0] === "required").length;
    return `<article class="case-card example-card cat-${esc(x.category)}" id="example-${esc(x.slug)}">
      <header class="cc-head"><div><h3>${esc(x.title)}</h3><p class="small muted">${esc(x.domain)} · difficulty ${esc(x.difficulty)}</p></div>
        <div class="cc-tags">${UI.prov("example")}${metric ? `<span class="tag tag-hypothesis">${esc(metric.name)} ${x.dir === "min" ? "↓" : "↑"}</span>` : ""}<span class="tag tag-rule">Reference ${x.reference.score}</span></div></header>
      <p>${esc(x.summary)}</p>
      <h4>What each part of the challenge tells you</h4>
      <dl class="anatomy">${x.anatomy.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join("")}</dl>
      <details class="ex-rubric"><summary>The rubric: ${x.rubric.length} criteria, ${req} required</summary>
        <ul class="rubric-list">${x.rubric.map(([lvl, cat, text]) => `<li class="rubric-${lvl}"><span class="rubric-level">${esc(lvl)}</span><span class="rubric-cat">${esc(cat)}</span><span class="rubric-text">${esc(text)}</span></li>`).join("")}</ul></details>
      <div class="cc-cols">
        <div><h4>What the reference solution does</h4><ol>${x.recipe.map((r) => `<li>${esc(r)}</li>`).join("")}</ol><p class="small muted">${esc(x.reference.note)}</p></div>
        <div><h4>Where it leaves room ${UI.prov("hypothesis")}</h4><ul>${x.room.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>
      </div>
      <div class="btn-row"><button type="button" class="btn btn-small btn-primary" data-practise="${esc(x.slug)}">Practise it in the analyzer</button><a class="btn btn-small btn-ghost" href="#learn" data-anchor="lesson-anatomy">Lesson: read a challenge like its author</a></div>
    </article>`;
  }
  const examplesHTML = () => (E.EXAMPLES || []).length ? `<section class="section"><div class="section-head"><h2>Platform worked examples</h2><span class="small muted">Published by the platform to show how a challenge is built and solved</span></div>
    <div class="case-grid">${E.EXAMPLES.map(exampleCard).join("")}</div></section>` : "";

  /* A practice workspace with the example's description loaded in the analyzer
   * and its reference score in the tracker (reported, not measured by you). */
  E.openExampleWorkspace = (slug) => {
    const x = (E.EXAMPLES || []).find((e) => e.slug === slug);
    if (!x) return;
    const existing = UI.store.list().find((w) => w.exampleSlug === slug);
    if (existing) { UI.store.switchTo(existing.id); E.app.refreshChrome(); location.hash = "analyzer"; UI.toast(`Opened "${existing.name}"`); return; }
    UI.store.create(x.title + " (example)");
    UI.store.active().exampleSlug = slug;
    UI.store.update((s) => {
      const metric = E.METRICS.find((m) => m.id === x.metric);
      s.analyzerText = x.description;
      s.settings.challenge = x.title;
      s.settings.metric = metric ? metric.name : x.metric;
      s.settings.direction = x.dir;
      s.experiments.push({ id: UI.uid(), created: Date.now(), name: "reference-rf", parent: "", model: "RandomForest on log1p(y)", change: "Platform reference solution",
        folds: "", cv: "", lb: String(x.reference.score), ood: "", runtime: "", subgroups: {}, status: "unverified",
        notes: "Score reported on the platform's example page. Re-run the recipe on your own folds to get its CV." });
    });
    E.app.refreshChrome();
    location.hash = "analyzer";
    UI.toast(`Created practice workspace "${x.title} (example)"`);
  };

  function loaderHTML(compact) {
    return `<form class="pack-load${compact ? " compact" : ""}" id="pack-form" autocomplete="off">
      <label>Load private pack<input type="file" id="pack-file" accept=".json,application/json"></label>
      <p class="small muted">The file is <code>tools/eris-solver-lab/corpus-pack.json</code> in your private repository. It stays in this browser only.</p>
    </form>`;
  }

  V.cases = {
    title: "Case studies",
    render() {
      const pack = UI.pack();
      if (!pack)
        return `${UI.pageHead("Learn", "Case studies", "Your own past challenges: what scored, what the leaders did, where you lost points, and the runs behind each number.")}
          <div class="empty">${UI.pip("think", 72)}<div><h2>No private pack loaded</h2>
          <p>Case studies come from your private corpus (other solvers' top solutions and your own results), so they aren't part of this public app. Load the pack to see 12 challenges, 33 lessons and extra rules-game scenarios.</p>
          ${loaderHTML()}</div></div>
          ${examplesHTML()}`;
      const cats = [...new Set(pack.cases.map((c) => c.category))];
      const measured = pack.lessons.filter((l) => l.prov === "measured").length;
      const cross = pack.lessons.filter((l) => /^X-/.test(l.id));
      return `${UI.pageHead("Learn", "Case studies", "Your own past challenges: what scored, what the leaders did, where you lost points, and the runs behind each number.")}
        <section class="hello">
          ${UI.speech("wink", `<p><b>${pack.cases.length}</b> challenges and <b>${pack.lessons.length}</b> lessons (${measured} measured on your runs) from your private corpus, ${esc(UI.packSource())}.</p><p class="small muted">${esc(pack.calibration ? pack.calibration.note : "")}</p>`)}
          <div class="legend-card"><p class="lc-k">Score strip</p><ul class="legend-row"><li><i class="lg sm-cv"></i>Your CV</li><li><i class="lg sm-pub"></i>Your public</li><li><i class="lg sm-priv"></i>Your private</li><li><i class="lg sm-top-d"></i>Board top</li><li><i class="lg lg-base"></i>AI baseline</li></ul><p class="small muted">Each strip has its own scale.</p></div>
        </section>
        ${examplesHTML()}
        ${cross.length ? `<section class="section"><h2>Lessons that repeat across challenges</h2><div class="cross-grid">${cross.map((l) => `<article class="cross" id="lib-pack-${esc(l.id)}"><h3>${esc(l.title)}</h3><p>${esc(l.body)}</p><p class="small"><strong>Check:</strong> ${esc(l.validate)}</p>${UI.prov(l.prov)}</article>`).join("")}</div></section>` : ""}
        <section class="section"><div class="section-head"><h2>Challenges</h2>
          <div class="chips" id="case-cats"><button type="button" class="chip on" data-cat="all">All</button>${cats.map((k) => `<button type="button" class="chip" data-cat="${esc(k)}">${esc((E.TASK_TYPES.find((t) => t.id === k) || {}).name || k)}</button>`).join("")}</div></div>
          <div class="case-grid">${pack.cases.map((c) => caseCard(c, pack)).join("")}</div></section>
        <section class="section"><h2>Pack</h2><p class="small muted">Built ${esc(pack.built || "?")} from ${esc(pack.source || "")}. ${esc(pack.privacy || "")}</p>
          <div class="btn-row">${E.PACK ? "" : `<button type="button" class="btn btn-small btn-ghost danger" id="pack-remove">Remove pack from this browser</button>`}</div>${E.PACK ? "" : loaderHTML(true)}</section>`;
    },
    mount(root) {
      const file = root.querySelector("#pack-file");
      if (file) file.addEventListener("change", async (e) => {
        const f = e.target.files[0]; if (!f) return;
        try { const p = UI.importPack(await UI.readFile(f)); UI.toast(`Loaded ${p.cases.length} case studies`); UI.award("pack:load", 25, "private pack loaded"); E.app.resetIndex(); E.app.render(); }
        catch (err) { UI.toast("Couldn't load the pack: " + err.message); }
      });
      const rm = root.querySelector("#pack-remove");
      if (rm) rm.addEventListener("click", () => {
        if (rm.dataset.confirm !== "1") { rm.dataset.confirm = "1"; rm.textContent = "Confirm: remove pack"; return; }
        UI.removePack(); E.app.resetIndex(); E.app.render(); UI.toast("Pack removed from this browser");
      });
      const cats = root.querySelector("#case-cats");
      if (cats) cats.addEventListener("click", (e) => {
        const b = e.target.closest("[data-cat]"); if (!b) return;
        cats.querySelectorAll(".chip").forEach((c) => c.classList.toggle("on", c === b));
        root.querySelectorAll(".case-card").forEach((el) => { el.hidden = b.dataset.cat !== "all" && el.dataset.cat !== b.dataset.cat; });
      });
      root.addEventListener("click", (e) => {
        const px = e.target.closest("[data-practise]");
        if (px) { E.openExampleWorkspace(px.dataset.practise); return; }
        const o = e.target.closest("[data-open-case]");
        if (o) { E.openCaseWorkspace(o.dataset.openCase); return; }
        const pr = e.target.closest("[data-prompt-case]");
        if (pr) {
          const c = UI.pack().cases.find((x) => x.slug === pr.dataset.promptCase);
          UI.store.update((s) => { s.promptCtx = Object.assign({}, s.promptCtx, { relatedCase: c.slug, category: c.category, task: c.task === "signal" ? "tabular" : c.task, metric: (E.METRICS.find((m) => m.id === c.metric) || {}).name || s.promptCtx.metric, direction: c.dir }); });
        }
      });
    },
  };

  /* Create a workspace holding a case's real runs. */
  E.openCaseWorkspace = (slug) => {
    const c = UI.pack().cases.find((x) => x.slug === slug);
    if (!c) return;
    const existing = UI.store.list().find((w) => w.caseSlug === slug);
    if (existing) { UI.store.switchTo(existing.id); E.app.refreshChrome(); location.hash = "tracker"; UI.toast(`Opened "${existing.name}"`); return; }
    const id = UI.store.create(c.title);
    const ws = UI.store.active();
    ws.caseSlug = slug;
    UI.store.update((s) => {
      const metric = E.METRICS.find((m) => m.id === c.metric);
      s.settings.metric = metric ? metric.name : c.metric;
      s.settings.direction = c.dir || "max";
      const t0 = Date.now();
      (c.runs || []).forEach((r, i) => s.experiments.push({
        id: UI.uid(), created: t0 + i, name: r.name, parent: r.parent || "", model: r.model || "", change: r.change || "",
        folds: r.folds || "", cv: r.cv !== undefined ? String(r.cv) : "", lb: r.lb !== undefined ? String(r.lb) : "",
        ood: r.ood !== undefined ? String(r.ood) : "", runtime: r.runtime !== undefined ? String(r.runtime) : "",
        subgroups: r.subgroups ? Object.fromEntries(r.subgroups.split(",").map((p) => p.split("=").map((x) => x.trim()))) : {},
        status: r.cv !== undefined || r.lb !== undefined || r.folds ? "verified" : "unverified",
        notes: (r.notes || "") + " (from your private corpus pack)",
      }));
    });
    E.app.refreshChrome();
    location.hash = "tracker";
    UI.toast(`Created workspace "${c.title}" with ${c.runs.length} real run${c.runs.length === 1 ? "" : "s"}`);
    return id;
  };
})();
