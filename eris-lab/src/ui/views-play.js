/* Beginner learning views: lessons with live demos and quizzes, and the
 * "Allowed or banned?" rules game. Progress is global (shared by all workspaces). */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;
  const V = (E.views = E.views || {});

  /* deterministic pseudo-random numbers so demos look the same every time */
  const rng = (seed) => () => { seed = (seed * 1103515245 + 12345) % 2147483648; return seed / 2147483648; };

  /* ---------------- interactive demos ---------------- */
  const DEMOS = {
    threshold(el) {
      const r = rng(11);
      const pts = Array.from({ length: 44 }, (_, i) => { const y = i % 3 === 0 ? 1 : 0; const p = Math.min(0.97, Math.max(0.03, (y ? 0.62 : 0.36) + (r() - 0.5) * 0.62)); return { y, p, row: r() }; });
      el.innerHTML = `<div class="demo-head"><strong>Try it:</strong> move the threshold</div>
        <svg viewBox="0 0 600 150" class="demo-svg" role="img" aria-label="Predicted probabilities with a movable threshold"><g id="thr-g"></g></svg>
        <label class="demo-slider">Threshold <input type="range" id="thr" min="0.05" max="0.95" step="0.01" value="0.5"><output id="thr-v" class="num">0.50</output></label>
        <div class="demo-stats" id="thr-stats"></div>
        <p class="demo-legend"><span><i class="dot dot-pos"></i>positive</span><span><i class="dot dot-neg"></i>negative</span><span>Dots right of the line are predicted positive</span></p>`;
      const draw = () => {
        const t = Number(el.querySelector("#thr").value);
        el.querySelector("#thr-v").textContent = t.toFixed(2);
        const x = (p) => 20 + p * 560;
        let tp = 0, fp = 0, fn = 0, tn = 0;
        pts.forEach((d) => { const h = d.p >= t; if (h && d.y) tp++; else if (h) fp++; else if (d.y) fn++; else tn++; });
        el.querySelector("#thr-g").innerHTML = `<rect x="${x(t)}" y="8" width="${580 - x(t)}" height="118" rx="8" class="demo-zone"/>
          <line x1="20" x2="580" y1="126" y2="126" class="demo-axis"/>
          ${pts.map((d) => `<circle cx="${x(d.p)}" cy="${18 + d.row * 100}" r="6" class="${d.y ? "dot-pos" : "dot-neg"}"/>`).join("")}
          <line x1="${x(t)}" x2="${x(t)}" y1="4" y2="130" class="demo-thr"/>
          <text x="20" y="145" class="demo-tick">0</text><text x="572" y="145" class="demo-tick">1</text>`;
        const acc = (tp + tn) / pts.length, prec = tp + fp ? tp / (tp + fp) : 0, rec = tp + fn ? tp / (tp + fn) : 0, f1 = prec + rec ? (2 * prec * rec) / (prec + rec) : 0;
        el.querySelector("#thr-stats").innerHTML = [["Accuracy", acc], ["Precision", prec], ["Recall", rec], ["F1", f1]].map(([k, v]) => `<div><span>${k}</span><b class="num">${v.toFixed(2)}</b></div>`).join("");
      };
      el.querySelector("#thr").addEventListener("input", draw); draw();
    },

    split(el) {
      const groups = ["A", "B", "C", "D", "E", "F"];
      const r = rng(5);
      const rows = [];
      groups.forEach((g, gi) => { const n = 3 + (gi % 3); for (let i = 0; i < n; i++) rows.push({ g, gi }); });
      const shuffled = rows.map((x) => ({ ...x, k: r() })).sort((a, b) => a.k - b.k);
      el.innerHTML = `<div class="demo-head"><strong>Try it:</strong> how is fold 1 built?</div>
        <div class="seg" role="group" aria-label="Split type"><button type="button" class="seg-btn on" data-mode="random">Random KFold</button><button type="button" class="seg-btn" data-mode="group">GroupKFold</button></div>
        <div class="split-viz" id="sv"></div><p class="split-msg" id="sm"></p>`;
      const draw = (mode) => {
        const assign = mode === "random" ? shuffled.map((x, i) => ({ ...x, fold: i % 3 })) : rows.map((x) => ({ ...x, fold: x.gi % 3 }));
        const val = assign.filter((x) => x.fold === 0), tr = assign.filter((x) => x.fold !== 0);
        const trG = new Set(tr.map((x) => x.g));
        const leaked = [...new Set(val.map((x) => x.g))].filter((g) => trG.has(g));
        el.querySelector("#sv").innerHTML = `<div><p class="sv-label">Train (folds 2–3)</p><div class="chips-g">${tr.map((x) => `<span class="gchip g${x.gi}${leaked.includes(x.g) ? " leak" : ""}">${x.g}</span>`).join("")}</div></div>
          <div><p class="sv-label">Validation (fold 1)</p><div class="chips-g">${val.map((x) => `<span class="gchip g${x.gi}${leaked.includes(x.g) ? " leak" : ""}">${x.g}</span>`).join("")}</div></div>`;
        el.querySelector("#sm").innerHTML = leaked.length
          ? `<b class="bad-text">${leaked.length} group${leaked.length > 1 ? "s" : ""} (${leaked.join(", ")}) appear on both sides.</b> The model gets to peek at validation users while training, so the score is inflated.`
          : `<b class="good-text">No leaks.</b> Every validation group is unseen during training, just like the hidden test.`;
      };
      el.querySelector(".seg").addEventListener("click", (e) => { const b = e.target.closest("[data-mode]"); if (!b) return; el.querySelectorAll(".seg-btn").forEach((x) => x.classList.toggle("on", x === b)); draw(b.dataset.mode); });
      draw("random");
    },

    oofstep(el) {
      const N = 10, K = 5;
      el.innerHTML = `<div class="demo-head"><strong>Try it:</strong> fill the OOF column</div>
        <div class="oof-grid" id="og"></div><div class="btn-row"><button type="button" class="btn btn-primary btn-small" id="og-next">Train next fold</button><button type="button" class="btn btn-ghost btn-small" id="og-reset">Reset</button></div><p id="og-msg" class="small"></p>`;
      let k = 0;
      const draw = () => {
        el.querySelector("#og").innerHTML = `<div class="og-row og-headrow"><span>Row</span>${Array.from({ length: K }, (_, f) => `<span>Fold ${f + 1}</span>`).join("")}<span>OOF</span></div>` + Array.from({ length: N }, (_, i) => {
          const fold = i % K;
          return `<div class="og-row"><span class="num">${i + 1}</span>${Array.from({ length: K }, (_, f) => {
            const cls = f >= k ? "" : f === fold ? "og-pred" : "og-train";
            return `<span class="og-cell ${cls}" title="${cls === "og-pred" ? "predicted (unseen)" : cls ? "used for training" : ""}"></span>`;
          }).join("")}<span class="og-cell ${fold < k ? "og-done" : ""}">${fold < k ? (0.3 + ((i * 37) % 60) / 100).toFixed(2) : ""}</span></div>`;
        }).join("");
        el.querySelector("#og-msg").textContent = k === 0 ? "Each fold model trains on the other folds (green) and predicts its own fold (gold)." : k < K ? `${k} of ${K} folds done. Rows in finished folds now have an honest prediction.` : "Done. Every row has exactly one prediction from a model that never saw it.";
        el.querySelector("#og-next").disabled = k >= K;
      };
      el.querySelector("#og-next").addEventListener("click", () => { k = Math.min(K, k + 1); draw(); });
      el.querySelector("#og-reset").addEventListener("click", () => { k = 0; draw(); });
      draw();
    },

    capacity(el) {
      const train = (c) => 0.32 * Math.exp(-0.33 * c) + 0.02;
      const val = (c) => 0.1 + 0.22 * Math.exp(-0.55 * c) + 0.0045 * (c - 4.5) ** 2;
      el.innerHTML = `<div class="demo-head"><strong>Try it:</strong> make the model bigger</div>
        <svg viewBox="0 0 600 200" class="demo-svg" role="img" aria-label="Train and validation error by model capacity"><g id="cap-g"></g></svg>
        <label class="demo-slider">Capacity <input type="range" id="cap" min="1" max="12" step="0.1" value="2"><output id="cap-v" class="num"></output></label>
        <p class="demo-legend"><span><i class="line line-train"></i>train error</span><span><i class="line line-val"></i>validation error</span></p>
        <p id="cap-msg" class="cap-msg"></p>`;
      const x = (c) => 40 + ((c - 1) / 11) * 540, y = (v) => 180 - v * 420;
      const path = (f) => Array.from({ length: 111 }, (_, i) => { const c = 1 + i * 0.1; return `${i ? "L" : "M"}${x(c).toFixed(1)} ${y(f(c)).toFixed(1)}`; }).join(" ");
      const draw = () => {
        const c = Number(el.querySelector("#cap").value);
        el.querySelector("#cap-v").textContent = c.toFixed(1);
        el.querySelector("#cap-g").innerHTML = `<line x1="40" x2="580" y1="180" y2="180" class="demo-axis"/><path d="${path(train)}" class="curve-train"/><path d="${path(val)}" class="curve-val"/>
          <line x1="${x(c)}" x2="${x(c)}" y1="10" y2="180" class="demo-thr"/><circle cx="${x(c)}" cy="${y(train(c))}" r="6" class="pt-train"/><circle cx="${x(c)}" cy="${y(val(c))}" r="6" class="pt-val"/>
          <text x="40" y="196" class="demo-tick">small model</text><text x="580" y="196" text-anchor="end" class="demo-tick">huge model</text>`;
        const gap = val(c) - train(c);
        const msg = c < 3.2 ? ["under", "Underfitting: both errors are high. Add capacity or features."] : c <= 6 ? ["sweet", "Sweet spot: validation error is near its lowest."] : ["over", `Overfitting: train keeps improving but validation gets worse (gap ${gap.toFixed(2)}).`];
        const m = el.querySelector("#cap-msg"); m.className = "cap-msg cap-" + msg[0]; m.textContent = msg[1];
      };
      el.querySelector("#cap").addEventListener("input", draw); draw();
    },

    lbnoise(el) {
      el.innerHTML = `<div class="demo-head"><strong>Try it:</strong> two models, many possible public leaderboards</div>
        <p class="small">Model A is truly better: 81% accuracy vs B's 80% on the full test set.</p>
        <label class="demo-slider">Public rows <input type="range" id="lbn" min="100" max="10000" step="100" value="1000"><output id="lbn-v" class="num"></output></label>
        <div class="btn-row"><button type="button" class="btn btn-primary btn-small" id="lb-run">Draw 400 public leaderboards</button></div>
        <div id="lb-out" class="lb-out"></div>`;
      const run = () => {
        const n = Number(el.querySelector("#lbn").value);
        const r = rng(n + 3);
        const draws = 400;
        let flips = 0;
        // normal approximation to the binomial accuracy of each model on n rows
        const gauss = () => Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
        for (let i = 0; i < draws; i++) {
          const a = 0.81 + gauss() * Math.sqrt((0.81 * 0.19) / n);
          const b = 0.80 + gauss() * Math.sqrt((0.8 * 0.2) / n);
          if (b > a) flips++;
        }
        const pct = (100 * flips) / draws;
        el.querySelector("#lb-out").innerHTML = `<div class="lb-bar" role="img" aria-label="${pct.toFixed(0)} percent of draws ranked the worse model first"><span style="width:${pct}%"></span></div>
          <p><b class="num">${pct.toFixed(0)}%</b> of public leaderboards ranked the <b>worse</b> model first with ${n.toLocaleString()} public rows.${pct > 20 ? " That's why a small public LB gap is weak evidence." : " With this many rows the LB is more trustworthy, but still not perfect."}</p>`;
      };
      el.querySelector("#lbn").addEventListener("input", () => { el.querySelector("#lbn-v").textContent = Number(el.querySelector("#lbn").value).toLocaleString(); });
      el.querySelector("#lbn-v").textContent = "1,000";
      el.querySelector("#lb-run").addEventListener("click", run);
      run();
    },

    checks(el) {
      el.innerHTML = `<div class="demo-head"><strong>The four checks</strong> (click each)</div><ol class="check-steps">${E.CHECKS.map((c) => `<li><button type="button" class="check-step" data-n="${c.n}"><span class="cs-n">${c.n}</span>${esc(c.name)}</button></li>`).join("")}</ol><div class="check-detail" id="cd"></div>`;
      const show = (n) => { const c = E.CHECKS[n - 1]; el.querySelectorAll(".check-step").forEach((b) => b.classList.toggle("on", Number(b.dataset.n) === n)); el.querySelector("#cd").innerHTML = `<p>${esc(c.what)}</p><p class="${n === 4 ? "bad-text" : "good-text"}"><b>${esc(c.credit)}</b></p>`; };
      el.querySelector(".check-steps").addEventListener("click", (e) => { const b = e.target.closest("[data-n]"); if (b) show(Number(b.dataset.n)); });
      show(1);
    },
  };

  /* ---------------- Learn ---------------- */
  let currentLesson = null;
  V.learn = {
    title: "Learn",
    render() {
      const g = UI.store.global();
      const done = E.LESSONS.filter((l) => g.lessons[l.id]).length;
      if (currentLesson) return lessonHTML(E.LESSONS.find((l) => l.id === currentLesson), g);
      const next = E.LESSONS.find((l) => !g.lessons[l.id]);
      const lv = UI.level(g.xp);
      return `${UI.pageHead("Learn", "Learn the craft", "Eight short lessons, each with a live demo and a three-question quiz. Every lesson ends at the tool you'll use for real.")}
        <section class="learn-hero">
          ${UI.speech(done === E.LESSONS.length ? "wink" : "happy", done === 0 ? "Hi, I'm <b>Pip</b>, the golden apple of Eris. Start with lesson 1. It takes four minutes." : done === E.LESSONS.length ? "You finished every lesson! Keep your edge with the <a href=\"#game\">rules game</a>." : `Nice progress! Next up: <b>${esc(next.title)}</b>.`)}
          <div class="learn-progress">
            <div class="ring" style="--p:${(done / E.LESSONS.length) * 100}"><span class="num">${done}/${E.LESSONS.length}</span></div>
            <div><p class="lp-k">Lessons done</p><p class="lp-v">${esc(lv.name)} · <span class="num">${g.xp}</span> XP</p><div class="xpbar"><span style="width:${(lv.pct * 100).toFixed(1)}%"></span></div><p class="small muted">${lv.next ? `${lv.next - g.xp} XP to ${esc(lv.nextName)}` : "Top level reached"}</p></div>
          </div>
        </section>
        <ol class="trail">${E.LESSONS.map((l, i) => {
          const ok = !!g.lessons[l.id];
          const correct = l.quiz.filter((q, qi) => g.quiz[l.id + ":" + qi] === q.a).length;
          return `<li id="lesson-${l.id}"><button type="button" class="trail-card${ok ? " done" : ""}${next && next.id === l.id ? " next" : ""}" data-lesson="${l.id}">
            <span class="trail-n">${ok ? "✓" : i + 1}</span>
            <span class="trail-body"><strong>${esc(l.title)}</strong><small>${l.mins} min${l.demo ? " · interactive demo" : ""} · quiz ${correct}/3</small></span>
          </button></li>`;
        }).join("")}</ol>
        <section class="section">
          <div class="section-head"><h2>Glossary</h2><input id="gl-q" type="search" placeholder="Filter terms…" autocomplete="off" aria-label="Filter glossary"></div>
          <dl class="glossary">${E.GLOSSARY.map(([t, d], i) => `<div class="gl-item" id="gloss-${i}" data-t="${esc((t + " " + d).toLowerCase())}"><dt>${esc(t)}</dt><dd>${esc(d)}</dd></div>`).join("")}</dl>
        </section>`;
    },
    mount(root) {
      if (currentLesson) return mountLesson(root, E.LESSONS.find((l) => l.id === currentLesson));
      root.querySelector(".trail").addEventListener("click", (e) => { const b = e.target.closest("[data-lesson]"); if (!b) return; currentLesson = b.dataset.lesson; E.app.render(); });
      const q = root.querySelector("#gl-q");
      q.addEventListener("input", () => { const v = q.value.trim().toLowerCase(); root.querySelectorAll(".gl-item").forEach((d) => { d.hidden = !!v && !d.dataset.t.includes(v); }); });
    },
    leave() { currentLesson = null; },
  };
  E.openLesson = (id) => { currentLesson = id; };

  function lessonHTML(l, g) {
    const i = E.LESSONS.indexOf(l);
    return `<div class="lesson">
      <button type="button" class="btn btn-ghost btn-small" id="ls-back">← All lessons</button>
      <header class="page-head"><p class="eyebrow">Lesson ${i + 1} of ${E.LESSONS.length} · ${l.mins} min</p><h1>${esc(l.title)}</h1></header>
      <div class="lesson-body">${l.body.map((p) => `<p>${esc(p)}</p>`).join("")}</div>
      ${l.demo ? `<div class="demo" id="demo"></div>` : ""}
      <section class="quiz"><h2>Quick check</h2>${l.quiz.map((q, qi) => {
        const ans = g.quiz[l.id + ":" + qi];
        return `<div class="qcard" data-qi="${qi}"><p class="qq">${esc(q.q)}</p><div class="qopts">${q.options.map((o, oi) => `<button type="button" class="qopt${ans !== undefined ? (oi === q.a ? " right" : oi === ans ? " wrong" : "") : ""}" data-oi="${oi}" ${ans !== undefined ? "disabled" : ""}>${esc(o)}</button>`).join("")}</div>
          <p class="qwhy" ${ans === undefined ? "hidden" : ""}>${ans === q.a ? "Correct. " : "Not quite. "}${esc(q.why)}</p></div>`;
      }).join("")}</section>
      <div class="lesson-foot">
        <button type="button" class="btn btn-primary" id="ls-done" ${g.lessons[l.id] ? "disabled" : ""}>${g.lessons[l.id] ? "Lesson complete ✓" : "Finish lesson"}</button>
        <a class="btn" href="#${l.tool}">Open the tool →</a>
        ${E.LESSONS[i + 1] ? `<button type="button" class="btn btn-ghost" id="ls-next">Next: ${esc(E.LESSONS[i + 1].title)} →</button>` : ""}
      </div>
    </div>`;
  }

  function mountLesson(root, l) {
    if (l.demo && DEMOS[l.demo]) DEMOS[l.demo](root.querySelector("#demo"));
    root.querySelector("#ls-back").addEventListener("click", () => { currentLesson = null; E.app.render(); });
    const nx = root.querySelector("#ls-next");
    if (nx) nx.addEventListener("click", () => { currentLesson = E.LESSONS[E.LESSONS.indexOf(l) + 1].id; E.app.render(); });
    root.querySelector(".quiz").addEventListener("click", (e) => {
      const b = e.target.closest(".qopt"); if (!b || b.disabled) return;
      const card = b.closest(".qcard"), qi = Number(card.dataset.qi), oi = Number(b.dataset.oi), q = l.quiz[qi];
      UI.store.updateGlobal((g) => { g.quiz[l.id + ":" + qi] = oi; });
      card.querySelectorAll(".qopt").forEach((o, k) => { o.disabled = true; if (k === q.a) o.classList.add("right"); else if (k === oi) o.classList.add("wrong"); });
      const why = card.querySelector(".qwhy"); why.hidden = false; why.textContent = (oi === q.a ? "Correct. " : "Not quite. ") + q.why;
      if (oi === q.a) UI.award(`quiz:${l.id}:${qi}`, 10, "correct answer");
    });
    root.querySelector("#ls-done").addEventListener("click", (e) => {
      const unanswered = l.quiz.some((_, qi) => UI.store.global().quiz[l.id + ":" + qi] === undefined);
      if (unanswered) { UI.toast("Answer the three quiz questions first."); return; }
      UI.store.updateGlobal((g) => { g.lessons[l.id] = Date.now(); });
      UI.award("lesson:" + l.id, 50, "lesson complete");
      UI.badge("first-lesson");
      if (E.LESSONS.every((x) => UI.store.global().lessons[x.id])) UI.badge("all-lessons");
      UI.confetti();
      e.target.disabled = true; e.target.textContent = "Lesson complete ✓";
    });
  }

  /* ---------------- Allowed or banned? ---------------- */
  const ROUND = 12;
  let game = null; // { order, i, score, streak, bestStreak, answers }
  const LABEL = { allowed: "Allowed", grey: "Grey area", banned: "Banned" };

  V.game = {
    title: "Allowed or banned?",
    render() {
      const g = UI.store.global();
      if (!game) {
        return `${UI.pageHead("Learn", "Allowed or banned?", "Rejections usually come from rules, not bad models. Judge real situations against the Eris rulebook. Each round is 12 random scenarios.")}
          <section class="game-intro">
            ${UI.speech("think", "I'll describe something a solver did. You decide: <b>Allowed</b>, <b>Grey area</b> (reviewer's call) or <b>Banned</b>. Every answer comes with the rule behind it.")}
            <div class="game-stats"><div><span>Best round</span><b class="num">${g.game.best === null ? "–" : g.game.best + "/" + ROUND}</b></div><div><span>Rounds played</span><b class="num">${g.game.plays}</b></div><div><span>Scenarios</span><b class="num">${E.SCENARIOS.length}</b></div></div>
            <button type="button" class="btn btn-primary btn-big" id="gm-start">Start a round</button>
          </section>`;
      }
      if (game.i >= game.order.length) {
        const pct = game.score / ROUND;
        const missed = game.answers.filter((a) => !a.ok);
        return `${UI.pageHead("Learn", "Round complete", "")}
          <section class="game-end">
            ${UI.speech(pct === 1 ? "wink" : pct >= 0.8 ? "happy" : pct >= 0.5 ? "think" : "sad", `You scored <b class="num">${game.score}/${ROUND}</b>, with a best streak of ${game.bestStreak}. ${pct === 1 ? "Flawless!" : pct >= 0.8 ? "You know the rules well." : "Review the misses below. They're the ones that cost real submissions."}`)}
            ${missed.length ? `<h2>Review</h2><ul class="review-list">${missed.map((a) => { const s = E.SCENARIOS[a.idx]; return `<li><p>${esc(s.s)}</p><p><span class="verdict-chip vc-${s.a}">${LABEL[s.a]}</span> ${esc(s.why)} ${UI.cite(s.cite)}</p><p class="small muted">You said: ${LABEL[a.pick]}</p></li>`; }).join("")}</ul>` : ""}
            <div class="btn-row"><button type="button" class="btn btn-primary" id="gm-start">Play again</button><a class="btn" href="#rules">Read the rules</a></div>
          </section>`;
      }
      const s = E.SCENARIOS[game.order[game.i]];
      const ans = game.answers[game.i];
      return `${UI.pageHead("Learn", "Allowed or banned?", "")}
        <div class="game-bar"><span>Scenario <b class="num">${game.i + 1}</b> of ${ROUND}</span><span>Score <b class="num">${game.score}</b></span><span class="streak${game.streak >= 3 ? " hot" : ""}">Streak <b class="num">${game.streak}</b></span></div>
        <div class="game-dots">${game.order.map((_, k) => `<i class="${k < game.i ? (game.answers[k].ok ? "ok" : "no") : k === game.i ? "cur" : ""}"></i>`).join("")}</div>
        <article class="scenario${ans ? (ans.ok ? " is-ok" : " is-no") : ""}">
          <p class="scenario-text">${esc(s.s)}</p>
          <div class="verdict-btns">${["allowed", "grey", "banned"].map((v) => `<button type="button" class="vbtn vb-${v}${ans && ans.pick === v ? " picked" : ""}${ans && s.a === v ? " truth" : ""}" data-v="${v}" ${ans ? "disabled" : ""}>${LABEL[v]}</button>`).join("")}</div>
          ${ans ? `<div class="feedback">${UI.pip(ans.ok ? "happy" : "sad", 44)}<div><p><b>${ans.ok ? "Right!" : `It's ${LABEL[s.a].toLowerCase()}.`}</b> ${esc(s.why)} ${UI.cite(s.cite)}</p><button type="button" class="btn btn-primary btn-small" id="gm-next">${game.i + 1 < ROUND ? "Next scenario →" : "See results"}</button></div></div>` : ""}
        </article>`;
    },
    mount(root) {
      const start = root.querySelector("#gm-start");
      if (start) start.addEventListener("click", () => {
        const idx = E.SCENARIOS.map((_, i) => i).sort(() => Math.random() - 0.5).slice(0, ROUND);
        game = { order: idx, i: 0, score: 0, streak: 0, bestStreak: 0, answers: [] };
        E.app.render();
      });
      const btns = root.querySelector(".verdict-btns");
      if (btns) btns.addEventListener("click", (e) => {
        const b = e.target.closest("[data-v]"); if (!b || b.disabled) return;
        const idx = game.order[game.i];
        const ok = E.SCENARIOS[idx].a === b.dataset.v;
        game.answers[game.i] = { idx, pick: b.dataset.v, ok };
        if (ok) { game.score++; game.streak++; game.bestStreak = Math.max(game.bestStreak, game.streak); UI.award("game:" + idx, 5, "rule judged right"); }
        else game.streak = 0;
        E.app.render();
        const nb = document.getElementById("gm-next"); if (nb) nb.focus();
      });
      const nx = root.querySelector("#gm-next");
      if (nx) nx.addEventListener("click", () => {
        game.i++;
        if (game.i >= game.order.length) {
          UI.store.updateGlobal((g) => { g.game.plays++; g.game.best = Math.max(g.game.best || 0, game.score); });
          UI.award("game:first-round", 20, "first round finished");
          if (game.score / ROUND >= 0.8) UI.badge("rule-keeper");
          if (game.score === ROUND) UI.badge("perfect-game");
          if (game.score / ROUND >= 0.8) UI.confetti();
        }
        E.app.render();
      });
    },
    leave() { if (game && game.i >= game.order.length) game = null; },
  };
})();
