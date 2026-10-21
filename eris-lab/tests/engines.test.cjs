// Unit tests for the pure engines. Run: node --test eris-lab/tests/
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const fs = require("node:fs");

const root = path.join(__dirname, "..", "src");
for (const f of ["data/platform.js", "data/library.js", "data/techniques.js", "data/prompts.js", "data/templates.js", "data/learn.js", "data/examples.js",
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
  assert.equal(E.LESSONS.length, 9);
  E.LESSONS.forEach((l) => { assert.equal(l.quiz.length, 3); l.quiz.forEach((q) => assert.ok(q.a >= 0 && q.a < q.options.length)); });
  assert.ok(E.SCENARIOS.length >= 30);
  E.SCENARIOS.forEach((s) => { assert.ok(["allowed", "grey", "banned"].includes(s.a)); assert.ok(s.cite.startsWith("§")); });
  const idx = E.search.buildIndex();
  assert.ok(E.search.search(idx, "overfitting").some((x) => x.type === "Lesson"));
});

/* Calibration cases from auditing real Eris solutions (top-5 files and the
 * owner's own scripts). Snippets are synthetic re-statements of the patterns. */
test("auditor calibration: legitimate patterns are not flagged", () => {
  const code = `"""Requirements map: R1 argv paths, R2 train only."""
import os, sys, base64, json
from pathlib import Path
import numpy as np, pandas as pd, torch
import argparse
PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")
torch.manual_seed(0); np.random.seed(0)
ans = pd.read_csv(PUBLIC_DIR / "train_answers.csv")
arr = np.load(PUBLIC_DIR / "train_inputs.npz")
raw = np.frombuffer(base64.b64decode(row["flow_b64"]), np.uint8)
assert torch.cuda.is_available()
print(f"threads={torch.get_num_threads()}")
msg = ("%d rows"
% (len(ans),))
model.eval()
ema.load_state_dict(net.state_dict())
model.fit(X, y)
OUT.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(OUT, index=False)
`;
  const r = E.auditor.audit(code);
  const bad = r.findings.filter((x) => x.sev === "blocker" || x.sev === "high").map((x) => x.id);
  assert.deepEqual(bad, [], "unexpected: " + bad.join(", "));
  assert.ok(r.findings.some((x) => x.id === "det-cuda-guard" && x.sev === "low"));
});

test("auditor calibration: patterns seen in real top files are flagged", () => {
  const code = `import os, sys, signal
from pathlib import Path
import learner
EPOCHS = int(os.environ.get("EPOCHS", "14"))
for cand in (sys.argv[1], os.environ.get("DATA_DIR"), "dataset/public"):
    pass
USE_EMA = 0
if USE_EMA:
    pass
signal.alarm(3000)
OUT = Path(sys.argv[2])
pd.DataFrame({"id": test.id, "note": "x"}).to_csv(OUT, index=False)
model.fit(X, y)
sub.to_csv(OUT, index=False)
CFG = json.loads('{"align": {"chunk_elems": 128000000, "ctx": false, "dim": 32, "heads": 4, "layers": 2, "lr": 0.0003, "steps": 12000, "warmup": 500, "wd": 0.01, "drop": 0.1, "seed": 42, "batch": 64, "x": 1}}')
`;
  const ids = E.auditor.audit(code).findings.map((x) => x.id);
  for (const id of ["det-env-switch", "det-path-chain", "det-dead-switch", "det-alarm", "io-placeholder", "env-unknown-import", "src-embedded-config"])
    assert.ok(ids.includes(id), `expected ${id}; got ${ids.join(", ")}`);
});

test("auditor: source size limit and restrictions", () => {
  const big = "x = 1\n".repeat(90000);
  assert.ok(E.auditor.audit(big).findings.some((x) => x.id === "src-size" && x.sev === "blocker"));
  const gpu = `import sys\nDEVICE = "cuda"\npred = (model(x) + model(torch.flip(x, [3]))) / 2\nmodel.fit(X)\nP=sys.argv[1]; O=sys.argv[2]`;
  const ids = E.auditor.audit(gpu, { restrict: { cpuOnly: true, noTTA: true } }).findings.map((x) => x.id);
  assert.ok(ids.includes("rx-cpu") && ids.includes("rx-tta"));
  const custom = E.auditor.audit(gpu, { custom: [{ pattern: "torch\\.flip", title: "No flips", sev: "medium" }] });
  assert.ok(custom.findings.some((x) => x.id === "custom-0" && x.sev === "medium"));
});

test("auditor: before/after diff", () => {
  const a = E.auditor.audit(`import torch\nDEV = "cuda" if torch.cuda.is_available() else "cpu"\nmodel.fit(X)`);
  const b = E.auditor.audit(`import sys, torch\ntorch.manual_seed(0)\nDEV = torch.device("cuda")\nmodel.fit(X)\nP=sys.argv[1];O=sys.argv[2]`);
  const d = E.auditor.diff(a, b);
  assert.ok(d.fixed.some((x) => x.id === "det-cuda-available"));
  assert.ok(d.fixed.some((x) => x.id === "io-argv-missing"));
});

test("logparse calibration: real log formats (Kaggle JSON lines, held-out lines, per-group scores)", () => {
  const kaggle = [
    '[{"stream_name":"stdout","time":12.5,"data":"[      0s] fold 0: train 6559 rows / 52 authors; held-out 1626 rows / 11 authors\\n"}',
    ',{"stream_name":"stdout","time":1300.2,"data":"[   1290s]   epoch 4: greedy chrF held-out authors (600 rows) = 0.2179 | train rows = 0.6920\\n"}',
    ',{"stream_name":"stderr","time":1310.0,"data":"SyntaxWarning: invalid escape sequence 0.99\\n"}]',
  ].join("\n");
  const a = E.logparse.parse(kaggle);
  assert.equal(a.cv, 0.2179);
  assert.ok(Math.abs(a.runtime - 1310 / 60) < 1e-9, "runtime from the last time field");
  assert.equal(a.ood, undefined, "plain held-out validation is not a stress split");

  const b = E.logparse.parse(`[  201s] validation (held-out headwords, one CTC model, n=3201): score 0.3768 full-match 0.3077
[  382.5s] HELD-OUT-PERSON validation, official metric: TOTAL 0.5816  x 0.5085  y 0.7365
[  382.5s]   per-held-out-person scores: 0.379 0.496 0.546
[  382.5s]   reference: best constant (median) waveform scores 0.2469 -> model adds +0.3347
[ 1791.8s] wrote working_run/submission.csv shape=(322, 4)`);
  assert.equal(b.cv, 0.5816);
  assert.deepEqual(b.groups, [0.379, 0.496, 0.546]);
  assert.equal(b.groupLabel, "person");
  assert.equal(b.baseline, 0.2469);
  assert.ok(Math.abs(b.runtime - 1791.8 / 60) < 1e-9);
});

test("analyzer calibration: real description header and rules", () => {
  const desc = `# Dating Segments
Creator x · Domain: From Scratch · Difficulty: Medium · Scoring: lower is better · Compute: CPU · Tags: features
## Overview
Every model must be trained from zero on the provided data, on CPU. No pretrained weights of any kind.
The training data are complete series from other sites; test items come from unseen sites.
## Evaluation
q = 0.999 * p + 0.001 / 421
score = mean over test segments of -ln(q)
Spreading the probability evenly over the 421 candidate years scores ln(421) = 6.04.
Your full solution must run on CPU only, on 10 cores with 62.5 GiB of RAM and no GPU, within 90 minutes.
Do not pool predictions across test charts: predict each chart on its own, with no voting.
folds.csv - five folds over the training charts.
A submission is rejected outright if it has duplicate ids.`;
  const a = E.analyzer.analyze(desc);
  assert.equal(a.category, "scratch");
  assert.equal(a.header.difficulty, "Medium");
  assert.equal(a.metric.dir, "min");
  assert.equal(a.hardware.kind, "cpu");
  assert.equal(a.hardware.cores, 10);
  assert.equal(a.runtimeLimit.minutes, 90);
  assert.equal(a.budgetMin, 63);
  assert.deepEqual(a.restrict, { cpuOnly: true, noPretrained: true, perRow: true, noTTA: false });
  assert.equal(a.split[0].kind, "provided folds");
  assert.ok(a.rejectIf.length >= 1);
  assert.equal(a.difficulty, 3);
  const b = E.analyzer.analyze("Metric: mean set F1 (prec, rec). Gold has exactly two. A prediction may contain a different count. Environment: CPU.");
  assert.equal(b.metric.id, "setf1");
  assert.ok(b.opportunities.some((o) => /Decode to the metric/.test(o)));
});

test("analyzer: the platform worked example (rubric, grader, prepare script, log-space MAE)", () => {
  const x = E.EXAMPLES.find((e) => e.slug === "notebook-upvotes");
  const a = E.analyzer.analyze(x.description);
  assert.equal(a.task, "tabular");
  assert.equal(a.category, "general");
  assert.equal(a.metric.id, "maelog");
  assert.equal(a.metric.dir, "min");
  assert.equal(a.rubric.length, 11);
  assert.equal(a.rubric.filter((r) => r.level === "required").length, 4);
  assert.equal(a.rubric[0].category, "Data Handling");
  assert.ok(a.grader.some((g) => /exactly \{content_id, upvotes\}/.test(g.text)));
  assert.ok(a.grader.some((g) => /≥ 0/.test(g.text)));
  assert.equal(a.prepare.kind, "random");
  assert.equal(a.split[0].prob, "confirmed");
  assert.ok(a.prep.some((p) => /%d\/%m\/%Y/.test(p.text)));
  assert.ok(a.prep.some((p) => /pipe-separated/.test(p.text)));
  assert.equal(a.reference.value, 1.061);
  assert.deepEqual(a.submissionColumns.map((c) => c.name), ["content_id", "upvotes"]);
  assert.ok(a.opportunities.some((o) => /median of log1p/.test(o)));
  assert.ok(a.opportunities.some((o) => /author_username/.test(o)));
});

test("analyzer: rubric pasted as one line per criterion, and no rubric in ordinary text", () => {
  const a = E.analyzer.analyze("Some challenge.\nREQUIRED Data Handling Remove the leaking columns before training. They hold future information.\nRECOMMENDED Modeling Use a gradient boosting model for the regression.");
  assert.equal(a.rubric.length, 2);
  assert.equal(a.rubric[0].category, "Data Handling");
  assert.equal(a.rubric[0].text, "Remove the leaking columns before training.");
  assert.equal(a.rubric[1].level, "recommended");
  assert.equal(E.analyzer.analyze("All fields are required. Scoring: lower is better").rubric.length, 0);
  // a text column inside a non-tabular challenge is not a tabular free-text note
  assert.ok(!E.analyzer.analyze("Rank the candidate passages for each query. Free text answers are not scored.").prep.some((p) => /Free-text/.test(p.text)));
});

test("auditor calibration: the platform's exemplary tabular solution patterns", () => {
  const code = `import sys
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])
train = pd.read_csv(public_dir / "train.csv")
train["d"] = pd.to_datetime(train["created_date"])
tfidf = TfidfVectorizer(max_features=256)
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(train[["d"]], train["y"])
pd.DataFrame({"a": [1]}).to_csv(submission_out, index=False)
`;
  const r = E.auditor.audit(code, { category: "general" });
  const by = (id) => r.findings.find((x) => x.id === id);
  assert.equal(by("det-all-cores").sev, "low");
  assert.equal(by("grey-tfidf").sev, "low");
  assert.ok(by("data-date-guess"));
  assert.equal(E.auditor.audit(code, { category: "nlp" }).findings.find((x) => x.id === "grey-tfidf").sev, "medium");
  assert.ok(!ids(E.auditor.audit(code.replace('pd.to_datetime(train["created_date"])', 'pd.to_datetime(train["created_date"], format="%d/%m/%Y")'))).includes("data-date-guess"));
});

test("prompts: analyzer evidence (rubric, grader, metric decode) reaches every prompt", () => {
  const ctx = { challenge: "Upvotes", metric: "MAE in log space", metricId: "maelog", direction: "min",
    rubric: [{ level: "required", category: "Modeling", text: "No significant overfitting." }, { level: "recommended", category: "", text: "Clip predictions at 0." }],
    grader: ["Every upvotes value must be ≥ 0."], prep: ["Parse dates with format=\"%d/%m/%Y\"."], reference: "1.061" };
  for (const id of E.PROMPT_ORDER) {
    const p = E.buildPrompt(id, ctx);
    assert.ok(/## Challenge rubric/.test(p) && /REQUIRED \(Modeling\): No significant overfitting/.test(p), id);
    assert.ok(/conditional median of log1p/.test(p) && /Grader contract/.test(p) && /Reference solution score.*1\.061/.test(p), id);
  }
  assert.ok(E.PROMPT_ORDER.includes("rubric"));
  assert.ok(/train-fit score next to the validation score/.test(E.buildPrompt("baseline", {})));
  assert.ok(!/## Challenge rubric/.test(E.buildPrompt("recon", {})));
});

test("templates: the rubric-ready regression template audits without blocker or high findings", () => {
  const r = E.auditor.audit(E.TEMPLATES.find((t) => t.id === "skeleton-tabular-reg").code);
  assert.deepEqual(r.findings.filter((x) => x.sev === "blocker" || x.sev === "high").map((x) => x.id), []);
});

test("report: unticked REQUIRED rubric criteria are listed", () => {
  const a = E.analyzer.analyze(E.EXAMPLES[0].description);
  const md = E.report.build({ analysis: a, rubricChecks: { [a.rubric[0].text]: true }, experiments: [], settings: {} });
  assert.ok(/Rubric coverage/.test(md));
  assert.ok(/3 REQUIRED rubric criterion\/criteria not yet ticked/.test(md));
});
