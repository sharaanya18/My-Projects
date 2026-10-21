/* Final solution review: composes a Markdown report from what the user has
 * actually entered (analysis, latest audit, tracked experiments, checklist).
 * Sections without data say so instead of filling in assumptions. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  E.FINAL_CHECKLIST = [
    { id: "fc-argv", text: "Script reads Path(sys.argv[1]) and writes Path(sys.argv[2]), creating the parent directory", cite: "S3" },
    { id: "fc-local", text: "Ran `python3 solution.py ./dataset/public ./working/submission.csv` end to end from a clean folder", cite: "S3" },
    { id: "fc-hash", text: "Two local runs produce identical submission hashes", cite: "S2 §3.3" },
    { id: "fc-runtime", text: "Total runtime fits ≤ 75 min on an A10G (or the challenge's own limit) with a margin", cite: "S2 §4.4" },
    { id: "fc-seeds", text: "Every RNG, model, split, sampler and DataLoader is seeded; device and worker counts are constants", cite: "S2 §3.3" },
    { id: "fc-nobranch", text: "No try/except fallbacks, env or debug switches, clock or machine checks", cite: "S2 §3.3" },
    { id: "fc-train", text: "Real training happens inside the script; the strip-the-model test passes", cite: "S2 §1.1, §5.3" },
    { id: "fc-test", text: "Test data used only for per-sample (or per-query-list) inference", cite: "S2 §5.2.5" },
    { id: "fc-data", text: "No external or synthetic data, no self-fine-tuned weights, no GitHub models", cite: "S2 §4.3, §5.2" },
    { id: "fc-libs", text: "Every import exists in the Kaggle image; no installs", cite: "S2 §4.1" },
    { id: "fc-cat", text: "Category rules respected (CV / RAG / Fine-tuning / From-scratch)", cite: "S2 §6" },
    { id: "fc-csv", text: "Submission checker: columns, rows, ids and values match sample_submission.csv", cite: "S2 §3.1, §4.6" },
    { id: "fc-csvupload", text: "Uploaded the CSV self-check before spending a script credit", cite: "S2 §2" },
    { id: "fc-comments", text: "Comments explain the reasoning at each step", cite: "S1" },
    { id: "fc-select", text: "Final chosen by matched-split CV and stability, not public LB alone", cite: "S2 §1.2" },
  ];

  const f4 = (x, d = 4) => (x === null || x === undefined || Number.isNaN(x) ? "–" : Number(x).toFixed(d));

  function build(state) {
    const s = state || {};
    const dir = (s.settings && s.settings.direction) || "max";
    const exps = s.experiments || [];
    const A = s.analysis;
    const audit = s.audit;
    const checks = s.checklist || {};
    const md = [];
    const date = new Date().toISOString().slice(0, 10);

    md.push(`# Final solution review: ${s.settings && s.settings.challenge ? s.settings.challenge : "(challenge name)"}`);
    md.push(`_Generated ${date} by Eris Solver Lab. VERIFIED marks values you entered from your own runs. HYPOTHESIS marks recommendations not yet tested. No private-leaderboard performance is implied._`);

    md.push("\n## 1. Summary");
    const top = E.tracker.best(exps, dir);
    if (top) {
      const t = E.tracker.stats(top);
      md.push(`- Best validated run: **${top.name}**, CV ${f4(t.mean)}${t.sd !== null ? ` ± ${f4(t.sd)}` : ""}${t.n ? ` (${t.n} folds)` : ""}${t.ood !== null ? `, stress split ${f4(t.ood)}` : ""}${t.lb !== null ? `, public LB ${f4(t.lb)}` : ""}. Status: ${top.status === "verified" ? "VERIFIED" : "UNVERIFIED"}.`);
      if (top.change) md.push(`- Approach: ${top.change}`);
    } else md.push("- No experiments recorded in the tracker.");
    if (A) md.push(`- Challenge reading: ${A.taskName} · category ${A.categoryName} · metric ${A.metric ? A.metric.name : "not found"} · estimated difficulty ${A.difficulty}/5 (heuristic).`);

    md.push("\n## 2. Compliance findings");
    if (audit && audit.findings) {
      const c = audit.counts;
      md.push(`Static audit of \`${audit.fileName || "solution"}\` (${audit.lines} lines): ${c.blocker} blocker, ${c.high} high, ${c.medium} medium, ${c.low} low. Static checks can't prove compliance; they flag patterns reviewers look for.`);
      const rel = audit.findings.filter((x) => x.sev !== "info");
      if (rel.length) {
        md.push("\n| Severity | Line | Finding | Rule | Fix |\n|---|---|---|---|---|");
        rel.forEach((x) => md.push(`| ${x.sev} | ${x.lines.filter(Boolean).join(", ") || "–"} | ${x.title.replace(/\|/g, "/")} | ${x.cite || "–"} | ${x.fix.replace(/\|/g, "/")} |`));
      } else md.push("No blocker, high, medium or low findings.");
    } else md.push("No audit run yet. Run the Solution Auditor on your final script.");

    md.push("\n## 3. Validation evidence");
    if (A && A.split && A.split.length) md.push(`- Planned split (from the challenge reading, HYPOTHESIS until checked): ${A.split[0].kind}. ${A.split[0].cv}.`);
    if (exps.length) {
      md.push("\n| Run | Parent | Change | CV mean | CV sd | Folds | Stress | Public LB | Runtime (min) | Status |\n|---|---|---|---|---|---|---|---|---|---|");
      exps.forEach((e) => {
        const t = E.tracker.stats(e);
        md.push(`| ${e.name} | ${e.parent || "–"} | ${(e.change || "–").replace(/\|/g, "/")} | ${f4(t.mean)} | ${f4(t.sd)} | ${t.n || "–"} | ${f4(t.ood)} | ${f4(t.lb)} | ${f4(t.runtime, 1)} | ${e.status === "verified" ? "VERIFIED" : "UNVERIFIED"} |`);
      });
      const comps = exps.filter((e) => e.parent).map((e) => {
        const p = exps.find((x) => x.name === e.parent || x.id === e.parent);
        return p ? [e, p, E.tracker.compare(e, p, dir)] : null;
      }).filter(Boolean);
      if (comps.length) {
        md.push("\nPaired comparisons (a gain counts only when Δ > 2 × SE):");
        comps.forEach(([e, p, c]) => md.push(`- ${e.name} vs ${p.name}: Δ ${f4(c.delta)}${c.se !== null ? `, SE ${f4(c.se)}` : ""} → **${c.verdict}**${c.oodDelta !== null ? `; stress Δ ${f4(c.oodDelta)}` : ""}`));
      }
    } else md.push("- No validation results recorded.");

    md.push("\n## 4. Prioritised improvements (HYPOTHESIS: test each one)");
    const recs = E.tracker.recommend(exps, s.settings || {});
    recs.slice(0, 8).forEach((r, i) => md.push(`${i + 1}. **${r.title}**. Evidence: ${r.evidence} Next: ${r.action}`));
    if (A && A.opportunities) A.opportunities.slice(0, 3).forEach((o) => md.push(`- ${o}`));

    md.push("\n## 5. Execution instructions");
    md.push("```bash\npython3 solution.py ./dataset/public ./working/submission.csv\n```");
    md.push(`- Platform command: \`python3 solution.py <public_dir> <submission_out>\`. Expected max runtime is 1.5 h on one NVIDIA A10G${A && A.runtimeLimit ? `; this challenge states ${A.runtimeLimit.minutes} min` : ""}.`);

    md.push("\n## 6. Final submission checklist");
    E.FINAL_CHECKLIST.forEach((c) => md.push(`- [${checks[c.id] ? "x" : " "}] ${c.text} _(${c.cite})_`));
    const open = E.FINAL_CHECKLIST.filter((c) => !checks[c.id]).length;
    md.push(`\n${open ? `**${open} checklist item(s) still open.**` : "All checklist items ticked."}`);
    return md.join("\n");
  }

  E.report = { build };
})();
