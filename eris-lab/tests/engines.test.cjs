// Unit tests for the pure engines. Run: node --test eris-lab/tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");

const root = path.join(__dirname, "..", "src");
for (const f of ["data/platform.js", "data/library.js", "data/techniques.js", "data/prompts.js", "data/templates.js", "data/learn.js",
  "engines/auditor.js", "engines/checker.js", "engines/analyzer.js", "engines/tracker.js", "engines/report.js", "engines/search.js", "engines/oof.js", "engines/logparse.js"]) {
  require(path.join(root, f));
}
const E = globalThis.Eris;
const fixture = (n) => fs.readFileSync(path.join(__dirname, "fixtures", n), "utf8");
const ids = (r) => r.findings.map((x) => x.id);

test("auditor: violating fixture raises the expected findings", () => {
  const r = E.auditor.audit(fixture("violations.py"));
  const got = ids(r);
  for (const id of ["det-cuda-available", "det-machine", "det-clock", "det-try", "det-env-switch", "det-flag",
    "det-no-seed", "io-hardcoded-path", "env-install", "env-github", "test-fit", "test-concat", "test-pseudo",
    "test-quantile", "data-synthetic", "env-artifact-load", "det-hpo-timeout", "io-argv-missing", "grey-tfidf", "det-set-order", "io-private"]) {
    assert.ok(got.includes(id), `expected ${id}; got ${got.join(", ")}`);
  }
  assert.equal(r.verdict, "blocked");
});

test("auditor: the compliant skeleton template has no blocker or high findings", () => {
  const tpl = E.TEMPLATES.find((t) => t.id === "skeleton-tabular").code;
  const r = E.auditor.audit(tpl);
  const bad = r.findings.filter((x) => x.sev === "blocker" || x.sev === "high");
  assert.deepEqual(bad.map((x) => x.id + ": " + x.title), []);
});

test("auditor: comments and docstrings don't trigger code rules", () => {
  const code = `"""Uses torch.cuda.is_available() in docs only.
pip install nothing
"""
import sys
from pathlib import Path
# DEV = "cuda" if torch.cuda.is_available() else "cpu"
PUBLIC_DIR = Path(sys.argv[1]); OUT = Path(sys.argv[2])
`;
  const r = E.auditor.audit(code);
  assert.ok(!ids(r).includes("det-cuda-available"));
  assert.ok(!ids(r).includes("env-install"));
});

test("auditor: notebook JSON is parsed into code cells", () => {
  const nb = JSON.stringify({ cells: [{ cell_type: "markdown", source: ["# title"] }, { cell_type: "code", source: ["import os\n", "n = os.cpu_count()\n"] }] });
  const r = E.auditor.audit(nb);
  assert.equal(r.isNotebook, true);
  assert.ok(ids(r).includes("det-machine"));
});

test("auditor: seeded constructors spanning lines are accepted", () => {
  const code = `import sys\nfrom sklearn.ensemble import RandomForestClassifier\nm = RandomForestClassifier(\n    n_estimators=10,\n    random_state=0,\n)\nm.fit(X, y)\nP = sys.argv[1]; O = sys.argv[2]\n`;
  assert.ok(!ids(E.auditor.audit(code)).includes("det-no-seed"));
});

test("auditor: from-scratch category escalates pretrained weights", () => {
  const code = `import sys\nfrom transformers import AutoModel\nm = AutoModel.from_pretrained("bert-base-uncased")\nloss.backward()\nP=sys.argv[1];O=sys.argv[2]`;
  assert.ok(ids(E.auditor.audit(code, { category: "scratch" })).includes("scr-pretrained"));
  assert.ok(ids(E.auditor.audit(code)).includes("env-pretrained"));
});

test("auditor: unknown imports are reported", () => {
  const r = E.auditor.audit("import sys\nimport some_private_pkg\nimport numpy as np\nP=sys.argv[1];O=sys.argv[2]\nmodel.fit(X)");
  const f = r.findings.find((x) => x.id === "env-unknown-import");
  assert.ok(f && f.title.includes("some_private_pkg") && !f.title.includes("numpy"));
});

test("checker: CSV parser handles quotes and CRLF", () => {
  const p = E.checker.parseCSV('id,text\r\n1,"a, b"\r\n2,"say ""hi"""\r\n');
  assert.deepEqual(p.header, ["id", "text"]);
  assert.deepEqual(p.rows, [["1", "a, b"], ["2", 'say "hi"']]);
});

test("checker: catches row, id, NaN and range problems against the sample", () => {
  const sample = "id,prediction\n1,0.5\n2,0.5\n3,0.5\n";
  const sub = "id,prediction\n1,0.2\n1,1.4\n4,nan\n";
  const r = E.checker.checkSubmission(sub, sample);
  const t = r.findings.map((x) => x.title).join(" | ");
  assert.match(t, /duplicate ids/);
  assert.match(t, /missing/);
  assert.match(t, /not in sample_submission/);
  assert.match(t, /NaN/);
  assert.match(t, /outside \[0, 1\]/);
});

test("checker: a matching submission passes", () => {
  const sample = "id,prediction\n1,0.5\n2,0.5\n";
  const r = E.checker.checkSubmission("id,prediction\n2,0.1\n1,0.9\n", sample);
  assert.equal(r.findings.length, 1);
  assert.equal(r.findings[0].sev, "ok");
});

test("checker: index column is detected", () => {
  const r = E.checker.checkSubmission(",id,prediction\n0,1,0.3\n", null);
  assert.ok(r.findings.some((x) => /index column/.test(x.title)));
});

test("checker: runtime budget statuses", () => {
  // 3 min × 5 epochs × 5 folds + 2 min overhead = 77 min: above the 83% target (74.7) but under 90
  assert.equal(E.checker.runtimeBudget({ epochMin: 3, epochs: 5, folds: 5, limitMin: 90 }).status, "tight");
  assert.equal(E.checker.runtimeBudget({ epochMin: 1, epochs: 5, folds: 5, limitMin: 90 }).status, "ok");
  assert.equal(E.checker.runtimeBudget({ epochMin: 3, epochs: 10, folds: 5, limitMin: 90 }).status, "over");
});

test("checker: credit plan never exceeds the 6-credit cap", () => {
  const p = E.checker.creditPlan({ localNow: 6, hoursLeft: 24, planned: 5 });
  assert.equal(p.achieved, 5);
  assert.ok(p.schedule.every((s) => s.left <= 5));
  const q = E.checker.creditPlan({ localNow: 0, hoursLeft: 3, planned: 2 });
  assert.equal(q.achieved, 0);
  assert.equal(q.enough, false);
});

test("analyzer: retrieval description with unseen groups and boilerplate", () => {
  const desc = `# Legal provision ranking
## Overview
For each query, rank candidate provisions. The test set contains unseen language directions.
No internet access. Fully rule-based solutions are okay.
## Evaluation
Submissions are scored using **MRR** (mean reciprocal rank). Runtime must complete within 60 minutes.
## Submission
| Column | Type | Description |
|--------|------|-------------|
| id | int | Row identifier |
| prediction | float | Score |
Must contain exactly 12,000 rows.`;
  const a = E.analyzer.analyze(desc);
  assert.equal(a.category, "retrieval");
  assert.equal(a.metric.id, "mrr");
  assert.equal(a.rows, 12000);
  assert.equal(a.runtimeLimit.minutes, 60);
  assert.equal(a.boilerplate.length, 2);
  assert.equal(a.split[0].kind, "group-held-out");
  assert.deepEqual(a.submissionColumns.map((c) => c.name), ["id", "prediction"]);
  assert.ok(a.constraints.some((r) => r.id === "RAG-1"));
});

test("analyzer: explicit from-scratch wins over task keywords", () => {
  const a = E.analyzer.analyze("Classify images of leaves. This is a from-scratch challenge: pretrained models are not allowed. Metric: accuracy.");
  assert.equal(a.category, "scratch");
  assert.equal(a.task, "cv");
});

test("tracker: paired comparison separates noise from gains", () => {
  const base = { name: "b", folds: "0.80 0.81 0.79 0.80 0.80" };
  const noise = { name: "n", parent: "b", folds: "0.801 0.808 0.792 0.801 0.799" };
  const gain = { name: "g", parent: "b", folds: "0.82 0.83 0.81 0.82 0.82" };
  assert.equal(E.tracker.compare(noise, base).verdict, "noise");
  assert.equal(E.tracker.compare(gain, base).verdict, "better");
  assert.equal(E.tracker.compare(base, gain).verdict, "worse");
  // lower-is-better metric flips the sign
  assert.equal(E.tracker.compare(base, gain, "min").verdict, "better");
});

test("tracker: recommendations cite the entries they come from", () => {
  const exps = [
    { name: "base", folds: "0.70 0.71 0.69", train: 0.95, ood: 0.6, runtime: 100, model: "lgbm", created: 1 },
  ];
  const recs = E.tracker.recommend(exps, { direction: "max", budgetMin: 75 });
  const titles = recs.map((r) => r.title).join(" | ");
  assert.match(titles, /train-vs-CV gap/);
  assert.match(titles, /Stress-split score well below CV/);
  assert.match(titles, /Runtime over budget/);
  assert.ok(recs.every((r) => r.evidence && r.action));
  assert.equal(E.tracker.recommend([], {})[0].title, "Establish a baseline");
});

test("report: builds markdown without inventing scores", () => {
  const md = E.report.build({ experiments: [], settings: { challenge: "X" } });
  assert.match(md, /No experiments recorded/);
  assert.match(md, /No audit run yet/);
  assert.doesNotMatch(md, /\d\.\d{4}/);
});

test("prompts: every template includes the contract, evidence and prohibitions", () => {
  for (const id of E.PROMPT_ORDER) {
    const p = E.buildPrompt(id, { challenge: "C", category: "retrieval", metric: "MRR", direction: "max" });
    assert.match(p, /Platform contract/);
    assert.match(p, /Evidence standard/);
    assert.match(p, /Prohibited shortcuts/);
    assert.match(p, /§6\.3/);
    assert.match(p, /sys\.argv\[1\]/);
  }
});

test("search: finds rules and techniques", () => {
  const idx = E.search.buildIndex();
  const r = E.search.search(idx, "pseudo label");
  assert.ok(r.length && r.some((x) => x.type === "Rule" || x.type === "Technique"));
  assert.deepEqual(E.search.search(idx, ""), []);
});

test("oof: AUC handles ties and matches the rank definition", () => {
  assert.equal(E.oof.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75);
  assert.equal(E.oof.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5);
  assert.equal(E.oof.auc([1, 1], [0.2, 0.3]), null);
});

test("oof: metrics and direction", () => {
  const y = [0, 1, 1, 0], p = [0.2, 0.7, 0.4, 0.1];
  assert.equal(E.oof.score("accuracy", y, p), 0.75);
  assert.ok(Math.abs(E.oof.score("brier", y, p) - (0.04 + 0.09 + 0.36 + 0.01) / 4) < 1e-12);
  assert.ok(Math.abs(E.oof.score("rmse", [1, 2], [2, 4]) - Math.sqrt(2.5)) < 1e-12);
  assert.equal(E.oof.METRICS.logloss.dir, "min");
});

test("oof: simplex grid sums to one", () => {
  const g = E.oof.simplex(3, 0.1);
  assert.equal(g.length, 66);
  assert.ok(g.every((w) => Math.abs(w.reduce((a, b) => a + b, 0) - 1) < 1e-9));
});

test("oof: sample analysis gives fold scores, subgroups, honest blend and calibration", () => {
  const t = E.checker.parseCSV(E.oof.sampleCSV());
  const ds = E.oof.prepare(t, { target: "target", preds: ["lgbm", "catboost", "mlp"], fold: "fold", group: "region" });
  assert.equal(ds.y.length, 1500);
  assert.ok(ds.binary && ds.probs);
  const a = E.oof.analyze(ds, "auc");
  assert.equal(a.models[0].folds.length, 5);
  assert.equal(a.subgroups.length, 2);
  assert.ok(a.blend.honest <= a.blend.score + 1e-12, "honest estimate can't beat the in-sample optimum");
  assert.ok(a.calibration.ece >= 0 && a.calibration.ece < 1);
  assert.ok(a.threshold.f1.best.t >= 0 && a.threshold.f1.best.t <= 1);
});

test("logparse: per-fold lines, CV, stress split, runtime, memory", () => {
  const log = `[   0.4s] device=cuda
Fold 1/5 | epoch 3 | train_loss 0.21 | val_loss 0.41 | val_auc 0.8123
Fold 2/5 | train auc 0.99 valid auc 0.8051
fold 3: auc=0.8190
Fold 4 - score: 0.8010
Fold 5 AUC 0.8122
CV 0.80992 +/- 0.00700
  CV util 0.6412 | LODO util 0.5901
Public LB: 0.807
peak gpu memory allocated: 7.5 GB
runtime 2765.2s`;
  const r = E.logparse.parse(log);
  assert.deepEqual(r.folds, [0.8123, 0.8051, 0.819, 0.801, 0.8122]);
  assert.equal(r.cv, 0.80992);
  assert.equal(r.cvStd, 0.007);
  assert.equal(r.ood, 0.5901);
  assert.equal(r.lb, 0.807);
  assert.ok(Math.abs(r.runtime - 2765.2 / 60) < 1e-9);
  assert.equal(r.mem, 7.5);
  assert.ok(r.evidence.folds.includes("Fold 1/5"));
});

test("logparse: skeleton-template output and /usr/bin/time", () => {
  const r = E.logparse.parse(`CV 0.81234 +/- 0.00412 folds=[0.811, 0.808, 0.818, 0.809, 0.816]
\tElapsed (wall clock) time (h:mm:ss or m:ss): 1:02:30
\tMaximum resident set size (kbytes): 4194304`);
  assert.deepEqual(r.folds, [0.811, 0.808, 0.818, 0.809, 0.816]);
  assert.equal(r.runtime, 62.5);
  assert.equal(r.mem, 4);
  assert.deepEqual(E.logparse.parse("nothing useful here").found, []);
});

test("learn: lessons, scenarios and glossary are well-formed", () => {
  assert.equal(E.LESSONS.length, 8);
  E.LESSONS.forEach((l) => { assert.equal(l.quiz.length, 3); l.quiz.forEach((q) => assert.ok(q.a >= 0 && q.a < q.options.length)); });
  assert.ok(E.SCENARIOS.length >= 30);
  E.SCENARIOS.forEach((s) => { assert.ok(["allowed", "grey", "banned"].includes(s.a)); assert.ok(s.cite.startsWith("§")); });
  const idx = E.search.buildIndex();
  assert.ok(E.search.search(idx, "overfitting").some((x) => x.type === "Lesson"));
});
