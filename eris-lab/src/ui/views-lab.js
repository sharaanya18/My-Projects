/* OOF analysis lab: load out-of-fold predictions and get per-fold and
 * per-subgroup scores, model correlation, blend weights with an honest
 * estimate, threshold scan and calibration. Everything runs in the browser. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;
  const V = (E.views = E.views || {});

  let table = null;      // parsed CSV { header, rows }
  let source = "";       // file name or "sample"
  let last = null;       // { ds, res, cols }

  const guess = (h, re) => h.find((c) => re.test(c)) || "";

  V.oof = {
    title: "OOF lab",
    render() {
      return `${UI.pageHead("Improve", "OOF lab", "Load your out-of-fold predictions and see where each model wins, how well they'd blend, and which threshold and calibration hold up. All of it uses training data only. Files stay in this browser.")}
        <section class="lab-load">
          ${UI.speech("think", "Export one CSV from your script with the <b>true target</b>, a <b>fold</b> column, any <b>subgroup</b> column, and one column per model's OOF prediction. No file yet? Try the fictional sample.")}
          <form id="lab-form" class="lab-files" autocomplete="off">
            <label>OOF predictions CSV<input type="file" id="lab-file" accept=".csv,text/csv"></label>
            <label>Labels CSV <small class="muted">(optional, if the target lives in another file; joined on the first column)</small><input type="file" id="lab-labels" accept=".csv,text/csv"></label>
            <div class="btn-row"><button type="button" class="btn" id="lab-sample">Load fictional sample</button></div>
          </form>
          <div id="lab-cols"></div>
        </section>
        <div id="lab-out" aria-live="polite"></div>`;
    },
    mount(root) {
      const cols = root.querySelector("#lab-cols");
      const out = root.querySelector("#lab-out");

      const showPicker = () => {
        if (!table) { cols.innerHTML = ""; return; }
        const h = table.header.map((x) => x.trim());
        const numeric = h.filter((c, i) => table.rows.slice(0, 200).every((r) => r[i] === undefined || r[i].trim() === "" || Number.isFinite(Number(r[i]))));
        const tgt = guess(h, /^(target|label|y|truth|answer)s?$/i) || numeric[1] || "";
        const fold = guess(h, /^(fold|kfold|fold_id|cv_fold)$/i);
        const grp = h.find((c) => c !== fold && c !== tgt && !numeric.includes(c)) || "";
        const preds = numeric.filter((c) => c !== tgt && c !== fold && !/^(id|row_id|index)$/i.test(c)).slice(0, 6);
        cols.innerHTML = `<div class="lab-picker">
          <p class="small muted">${esc(source)} · ${table.rows.length.toLocaleString()} rows · ${h.length} columns${source === "sample" ? ' · <span class="tag tag-demo">fictional sample</span>' : ""}</p>
          <div class="row-fields">
            <label>Target column<select id="lc-target">${h.map((c) => `<option ${c === tgt ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>
            <label>Fold column<select id="lc-fold"><option value="">(none)</option>${h.map((c) => `<option ${c === fold ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>
            <label>Subgroup column<select id="lc-group"><option value="">(none)</option>${h.map((c) => `<option ${c === grp ? "selected" : ""}>${esc(c)}</option>`).join("")}</select></label>
            <label>Metric<select id="lc-metric">${Object.entries(E.oof.METRICS).map(([k, m]) => `<option value="${k}">${esc(m.name)}</option>`).join("")}</select></label>
          </div>
          <fieldset><legend>Model prediction columns (up to 6)</legend><div class="pred-picks">${numeric.map((c) => `<label><input type="checkbox" value="${esc(c)}" ${preds.includes(c) ? "checked" : ""}> ${esc(c)}</label>`).join("")}</div></fieldset>
          <button type="button" class="btn btn-primary" id="lab-run">Analyze</button>
        </div>`;
        cols.querySelector("#lab-run").addEventListener("click", run);
      };

      const run = () => {
        const picked = [...cols.querySelectorAll(".pred-picks input:checked")].map((i) => i.value);
        if (!picked.length) { UI.toast("Tick at least one prediction column."); return; }
        if (picked.length > 6) { UI.toast("Pick at most 6 prediction columns."); return; }
        const c = { target: cols.querySelector("#lc-target").value, preds: picked, fold: cols.querySelector("#lc-fold").value, group: cols.querySelector("#lc-group").value };
        const metric = cols.querySelector("#lc-metric").value;
        const ds = E.oof.prepare(table, c);
        if (!ds.y.length) { out.innerHTML = `<p class="warn-text">No usable rows: check that the target and prediction columns are numeric.</p>`; return; }
        if (E.oof.METRICS[metric].binary && !ds.binary) { out.innerHTML = `<p class="warn-text">${esc(E.oof.METRICS[metric].name)} needs a 0/1 target. This target has other values, so pick RMSE or MAE.</p>`; return; }
        const res = E.oof.analyze(ds, metric);
        last = { ds, res, cols: c };
        out.innerHTML = results(res, ds);
        UI.award("oof:first", 30, "OOF analysis");
        if (source !== "sample") UI.badge("oof-scientist");
      };

      root.querySelector("#lab-sample").addEventListener("click", () => {
        table = E.checker.parseCSV(E.oof.sampleCSV()); source = "sample"; showPicker();
        cols.querySelector("#lc-metric").value = "auc";
        run();
      });
      const loadFiles = async () => {
        const f = root.querySelector("#lab-file").files[0];
        if (!f) return;
        const t = E.checker.parseCSV(await UI.readFile(f));
        const lf = root.querySelector("#lab-labels").files[0];
        if (lf) {
          const L = E.checker.parseCSV(await UI.readFile(lf));
          const key = L.header[0].trim();
          const ki = t.header.map((x) => x.trim()).indexOf(key);
          if (ki < 0) { UI.toast(`The labels file's first column "${key}" isn't in the OOF file.`); return; }
          const map = new Map(L.rows.map((r) => [String(r[0]).trim(), r.slice(1)]));
          const extra = L.header.slice(1).map((h) => (t.header.includes(h) ? h + "_label" : h));
          t.header = [...t.header, ...extra];
          t.rows = t.rows.map((r) => [...r, ...(map.get(String(r[ki]).trim()) || extra.map(() => ""))]);
        }
        table = t; source = f.name + (lf ? " + " + lf.name : ""); showPicker(); out.innerHTML = "";
      };
      root.querySelector("#lab-file").addEventListener("change", loadFiles);
      root.querySelector("#lab-labels").addEventListener("change", loadFiles);
      bind(out); // one delegated listener per mount; results re-render inside it
      if (table) { showPicker(); if (last) out.innerHTML = results(last.res, last.ds); }
    },
  };

  const f4 = (x) => UI.fmt(x, 4);

  function results(res, ds) {
    const m = res.info;
    const best = res.models[res.best];
    const dirWord = m.dir === "max" ? "higher" : "lower";
    const b = res.blend;
    const gain = b ? (m.dir === "max" ? b.honest - b.singles[b.bestSingle] : b.singles[b.bestSingle] - b.honest) : null;
    return `
      <section class="tiles">
        <div class="tile"><span>Rows</span><b class="num">${res.n.toLocaleString()}</b>${ds.skipped ? `<small>${ds.skipped} skipped</small>` : ""}</div>
        <div class="tile"><span>Best single model</span><b>${esc(best.name)}</b><small class="num">${esc(m.name)} ${f4(best.score)}</small></div>
        ${b ? `<div class="tile ${gain > 0 ? "t-good" : "t-meh"}"><span>Blend (honest)</span><b class="num">${f4(b.honest)}</b><small>${gain > 0 ? "+" : ""}${f4(gain)} vs best single</small></div>` : ""}
        ${res.calibration ? `<div class="tile"><span>Calibration error</span><b class="num">${res.calibration.ece.toFixed(3)}</b><small>ECE of ${esc(best.name)}</small></div>` : ""}
      </section>

      <section class="section"><h2>Models</h2>
        <div class="table-wrap"><table><thead><tr><th>Model</th><th class="r">${esc(m.name)} (all OOF)</th>${ds.folds ? `<th class="r">Fold mean</th><th class="r">Fold sd</th><th>Per fold</th>` : ""}<th><span class="sr-only">Actions</span></th></tr></thead><tbody>
        ${res.models.map((x, i) => `<tr class="${i === res.best ? "is-best" : ""}"><td><strong>${esc(x.name)}</strong>${i === res.best ? ' <span class="tag tag-verified">best</span>' : ""}</td><td class="r num">${f4(x.score)}</td>
          ${ds.folds ? `<td class="r num">${f4(x.foldMean)}</td><td class="r num">${f4(x.foldSd)}</td><td><div class="fold-mini">${x.folds.map((fd) => `<span title="fold ${esc(fd.key)}: n=${fd.n}" class="num">${f4(fd.score)}</span>`).join("")}</div></td>` : ""}
          <td><button type="button" class="btn btn-small btn-ghost" data-send="${i}">Log in tracker</button></td></tr>`).join("")}
        </tbody></table></div>
        <p class="small muted">${esc(m.name)}: ${dirWord} is better. ${ds.folds ? "Fold scores are what the tracker needs for paired comparisons." : "Add a fold column to get per-fold scores and paired comparisons."}</p>
      </section>

      ${ds.P.length > 1 ? `<section class="section two-col"><div><h2>Prediction correlation</h2>${heatmap(res.corr, ds.names)}<p class="small muted">Highly correlated models (above about 0.97) add little to a blend. Diversity is what makes ensembles work.</p></div>
        <div>${b ? `<h2>Blend weights</h2>
          <ul class="wbars">${ds.names.map((n, i) => `<li><span>${esc(n)}</span><div class="wbar"><i style="width:${(b.weights[i] * 100).toFixed(0)}%"></i></div><b class="num">${b.weights[i].toFixed(2)}</b></li>`).join("")}</ul>
          <dl class="kv small"><dt>In-sample blend</dt><dd class="num">${f4(b.score)} <span class="muted">(tuned on the rows it scores, so optimistic)</span></dd>
          <dt>Honest blend</dt><dd class="num">${f4(b.honest)} <span class="muted">(weights fit on half the ${ds.folds ? "folds" : "rows"}, scored on the other half)</span></dd>
          <dt>Best single</dt><dd class="num">${f4(b.singles[b.bestSingle])} (${esc(ds.names[b.bestSingle])})</dd></dl>
          <p class="${gain > 0 ? "good-text" : "muted"}">${gain > 0 ? `The blend beats the best single model by ${f4(gain)} on held-out halves. Confirm on your stress split before keeping it.` : "The blend doesn't beat the best single model on held-out halves. Keep it simple."}</p>
          <button type="button" class="btn btn-small" data-send="blend">Log blend in tracker</button>` : ""}</div></section>` : ""}

      ${res.subgroups ? `<section class="section"><h2>Subgroups</h2>${subgroups(res)}</section>` : ""}

      ${res.threshold ? `<section class="section two-col"><div><h2>Threshold scan</h2>${thresholdChart(res.threshold)}
        <p>Best F1 <b class="num">${res.threshold.f1.best.s.toFixed(4)}</b> at <b class="num">${res.threshold.f1.best.t.toFixed(2)}</b> (vs ${res.threshold.f1.at05.toFixed(4)} at 0.50). Best accuracy at <b class="num">${res.threshold.accuracy.best.t.toFixed(2)}</b>.</p>
        <p class="small muted">Use thresholds only for label metrics (F1, accuracy). Fix the chosen value in the script, and never derive it from the test distribution (§5.2.5).</p></div>
        <div><h2>Calibration</h2>${calChart(res.calibration)}<p class="small muted">Points on the diagonal mean "predicted 0.7 really is 70%". Below it means overconfident. This matters for log loss and Brier, not for AUC. Fit any calibrator on OOF only.</p></div></section>` : ""}

      <p class="small muted">Computed from your file in this browser. ${UI.prov("verified")} for these predictions. Whether a change helps on the hidden test is still a hypothesis until a matched-split CV confirms it.</p>`;
  }

  function subgroups(res) {
    const dir = res.info.dir;
    const vals = res.subgroups.filter((g) => g.score !== null);
    if (!vals.length) return `<p class="muted">Not enough positives and negatives inside the subgroups to score them.</p>`;
    const worst = vals.reduce((a, b) => (dir === "max" ? (b.score < a.score ? b : a) : (b.score > a.score ? b : a)));
    const lo = Math.min(...vals.map((v) => v.score)), hi = Math.max(...vals.map((v) => v.score));
    return `<p class="small">Model: <b>${esc(res.models[res.focus].name)}</b>. Weakest: <b>${esc(worst.key)}</b> (${f4(worst.score)}, n=${worst.n}).</p>
      <ul class="sg-bars">${res.subgroups.map((g) => `<li class="${g === worst ? "worst" : ""}"><span class="sg-k">${esc(g.key)}</span><div class="sg-bar"><i style="width:${g.score === null ? 0 : (hi === lo ? 100 : 15 + (85 * (g.score - lo)) / (hi - lo)).toFixed(1)}%"></i></div><span class="num">${f4(g.score)}</span><span class="small muted num">n=${g.n.toLocaleString()}</span></li>`).join("")}</ul>
      <p class="small muted">Point your next error-analysis prompt at the weakest subgroup.</p>`;
  }

  function heatmap(corr, names) {
    const k = names.length, cell = 46, L = 90, T = 8;
    const W = L + k * cell + 8, H = T + k * cell + 70;
    let s = `<div class="chart-box"><svg viewBox="0 0 ${W} ${H}" class="heat" style="max-width:${Math.round(W * 1.2)}px" role="img" aria-label="Correlation between model predictions">`;
    corr.forEach((row, i) => row.forEach((v, j) => {
      const a = v === null ? 0 : Math.max(0, Math.min(1, (v - 0.5) / 0.5));
      s += `<rect x="${L + j * cell}" y="${T + i * cell}" width="${cell - 2}" height="${cell - 2}" rx="4" class="heat-cell" style="fill-opacity:${(0.12 + a * 0.88).toFixed(2)}"/><text x="${L + j * cell + cell / 2 - 1}" y="${T + i * cell + cell / 2 + 4}" text-anchor="middle" class="heat-t${a > 0.55 ? " on" : ""}">${v === null ? "–" : v.toFixed(2)}</text>`;
    }));
    names.forEach((n, i) => {
      const lab = n.length > 11 ? n.slice(0, 10) + "…" : n;
      s += `<text x="${L - 6}" y="${T + i * cell + cell / 2 + 4}" text-anchor="end" class="axis">${esc(lab)}</text><text x="${L + i * cell + cell / 2}" y="${T + k * cell + 14}" text-anchor="end" transform="rotate(-35 ${L + i * cell + cell / 2} ${T + k * cell + 14})" class="axis">${esc(lab)}</text>`;
    });
    return s + "</svg></div>";
  }

  function thresholdChart(th) {
    const W = 420, H = 210, L = 40, R = 10, T = 10, B = 30;
    const x = (t) => L + t * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
    const line = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.s).toFixed(1)}`).join(" ");
    let s = `<div class="chart-box"><ul class="legend-row"><li><i class="lg lg-cv"></i>F1</li><li><i class="lg lg-lb"></i>Accuracy</li></ul><svg viewBox="0 0 ${W} ${H}" class="chart-svg mini" role="img" aria-label="F1 and accuracy by threshold">`;
    [0, 0.25, 0.5, 0.75, 1].forEach((v) => { s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" class="axis">${v.toFixed(2)}</text>`; });
    [0, 0.5, 1].forEach((t) => { s += `<text x="${x(t)}" y="${H - 10}" text-anchor="middle" class="axis">${t.toFixed(1)}</text>`; });
    s += `<path d="${line(th.f1.pts)}" class="ln-1"/><path d="${line(th.accuracy.pts)}" class="ln-2"/>`;
    s += `<line x1="${x(th.f1.best.t)}" x2="${x(th.f1.best.t)}" y1="${T}" y2="${H - B}" class="demo-thr"/><circle cx="${x(th.f1.best.t)}" cy="${y(th.f1.best.s)}" r="5" class="m-cv"/>`;
    return s + "</svg></div>";
  }

  function calChart(cal) {
    const W = 300, H = 230, L = 40, R = 10, T = 10, B = 34;
    const x = (v) => L + v * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
    let s = `<div class="chart-box"><svg viewBox="0 0 ${W} ${H}" class="chart-svg mini" role="img" aria-label="Reliability diagram">`;
    [0, 0.5, 1].forEach((v) => { s += `<line x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end" class="axis">${v.toFixed(1)}</text><text x="${x(v)}" y="${H - 16}" text-anchor="middle" class="axis">${v.toFixed(1)}</text>`; });
    s += `<line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" class="diag"/>`;
    const pts = cal.bins.filter((b) => b.n);
    s += `<path d="${pts.map((b, i) => `${i ? "L" : "M"}${x(b.pred).toFixed(1)} ${y(b.obs).toFixed(1)}`).join(" ")}" class="ln-1"/>`;
    pts.forEach((b) => { s += `<circle cx="${x(b.pred)}" cy="${y(b.obs)}" r="${Math.min(8, 3 + Math.sqrt(b.n) / 6).toFixed(1)}" class="m-cv"><title>predicted ${b.pred.toFixed(2)}, observed ${b.obs.toFixed(2)}, n=${b.n}</title></circle>`; });
    s += `<text x="${(L + W - R) / 2}" y="${H - 2}" text-anchor="middle" class="axis">predicted probability</text>`;
    return s + "</svg></div>";
  }

  function bind(out) {
    out.addEventListener("click", (e) => {
      const b = e.target.closest("[data-send]"); if (!b || !last) return;
      const { ds, res } = last;
      let folds = [], name, model;
      if (b.dataset.send === "blend") {
        const w = res.blend.weights;
        const p = ds.y.map((_, i) => ds.P.reduce((s, col, m) => s + w[m] * col[i], 0));
        if (ds.folds) folds = E.oof.byGroup(res.metric, ds.y, p, ds.folds, 100).sort((a, c) => String(a.key).localeCompare(String(c.key), undefined, { numeric: true })).map((f) => f.score);
        name = "blend-" + ds.names.map((n, i) => `${n}${w[i].toFixed(2)}`).join("-");
        model = "blend";
      } else {
        const m = res.models[Number(b.dataset.send)];
        folds = m.folds ? m.folds.map((f) => f.score) : [];
        name = "oof-" + m.name; model = m.name;
      }
      const s = UI.store.load();
      let nm = name.slice(0, 40), k = 2;
      while (s.experiments.some((x) => x.name === nm)) nm = name.slice(0, 36) + "-" + k++;
      const all = model === "blend" ? res.blend.score : res.models[Number(b.dataset.send)].score;
      UI.store.update((st) => {
        st.experiments.push({ id: UI.uid(), created: Date.now(), name: nm, model, change: "Logged from the OOF lab", folds: folds.map((v) => v.toFixed(6)).join(" "), cv: folds.length ? "" : String(all), status: "unverified", notes: `OOF lab, ${res.info.name}, ${source}` });
        if (!st.settings.metric) st.settings.metric = res.info.name;
        st.settings.direction = res.info.dir;
      });
      UI.toast(`Logged "${nm}" in the tracker. Set its parent there.`);
    });
  }
})();
