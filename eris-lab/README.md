# Eris Solver Lab

A browser workbench for engineering strong, compliant solutions to Shipd / Project Eris ML challenges with Claude Code. It aims for genuine unseen-data performance, not public-leaderboard overfitting.

No build step and no dependencies. Open `index.html` directly in a browser, or serve the folder:

```bash
cd eris-lab
python3 -m http.server 8000     # then open http://localhost:8000
```

## What's inside

| Section | What it does |
|---|---|
| Dashboard | Platform limits with citations, the four submission checks, sources analysed, where the documents disagree, live status from your tracker, audit and checklists, full backup and restore, and optional sync to your account (on the published page) |
| Platform rules | 49 searchable rules paraphrased from the Solver Guidebook, Shipd docs and submission instructions, filterable by severity |
| Lessons | 9 beginner lessons (metrics, validation, OOF, over/underfitting, leakage, public vs private LB, Eris rules, reading a challenge like its author, working with Claude Code), each with a live demo, a 3-question quiz and a link to the matching tool, plus a glossary |
| Allowed or banned? | A rules game: 12 random scenarios per round from 36 (more with a private pack), each answered with the guidebook section behind it |
| Case studies | The platform's worked example (Notebook Upvote Prediction): what each part of a challenge tells you, its 11-criterion rubric, what the reference solution does and where it leaves room, and a one-click practice workspace. With your private pack loaded: your past challenges, what the leaders did, your gaps, and the real runs behind each number |
| OOF lab | Load out-of-fold predictions to get per-fold and per-subgroup scores, prediction correlation, blend weights with an honest held-out estimate, a threshold scan and a calibration curve. Sends any model or the blend to the tracker |
| Challenge analyzer | Paste a problem description (plus the prepare script, grading script, config and rubric when the page shows them) to get category, metric, hardware, submission format, the grader's own checks, how the prepare script splits the test, data quirks to handle, a tickable rubric, runtime overrides, boilerplate to ignore, hidden-test hypotheses (with the quoted evidence), applicable rules, opportunities and risks |
| Workflow | 11 stages from reading the rules to submitting, with persistent checklists |
| Leaderboard optimization | Metric playbook, techniques (helps / hurts / how to prove it / compliance) and a public-LB noise estimator |
| Generalization | Validation split chooser, overfit/underfit diagnostic, leakage checklist, robustness tests |
| Strategy library | Lessons from your previous solution plus strategies and pitfalls by task type |
| Prompt generator | 11 Claude Code prompts (recon → final review, plus a rubric coverage check). Each embeds the platform contract, an evidence standard (VERIFIED vs HYPOTHESIS) and the prohibited-shortcut list, and carries what the analyzer found: rubric, grader checks, data quirks, metric decode notes and the reference score |
| Templates | A compliant tabular skeleton, a rubric-ready tabular regression for log-space metrics, a PyTorch seed header, a local pre-submission check script, a CLAUDE.md for challenge folders, an experiment log entry and a reviewer question |
| Solution auditor | Static checks for determinism, compliance, leakage, I/O contract, environment and version risks. Accepts `.py` and `.ipynb`. Code never leaves the browser |
| Experiment tracker | Paste a training log to fill a run automatically (each number shown with its source line). Logs runs with fold scores, stress-split, LB, runtime, memory and subgroups. Paired fold comparisons (gain only if Δ > 2·SE), chart, evidence-based next steps, JSON/CSV import and export |
| Runtime & submission | CSV validator against the sample and test files, runtime budget meter, GPU memory estimate, credit planner, script preflight, failure catalogue |
| Final review | Assembles a Markdown report (summary, compliance findings, rubric coverage, validation evidence, prioritised improvements, run command, checklist) from what you entered |

Global search: `Ctrl K` or `/`.

**Workspaces.** Each challenge gets its own workspace (switcher in the top bar, management on the dashboard) with separate runs, audit, analysis, prompts and checklists. Learning progress is shared across workspaces.

**Progress.** Lessons, quizzes, the rules game and real milestones (first logged run, a clean audit, an OOF analysis, a parsed log) earn XP once each, and unlock levels from Seed to Golden Apple, plus badges. Pip, the golden apple of Eris, is your guide.

## Provenance

Every recommendation carries a tag:

- **Platform rule**: stated in your documents, with the source and section (S1–S5, see the dashboard).
- **Verified in your files**: observed in your own code (behaviour only, not its score effect).
- **Hypothesis**: general practice, to be validated on your data.
- **Check first**: the documents conflict or are unclear.
- **From the challenge page**: read directly from the text you pasted into the analyzer.
- **Platform example**: from the platform's own worked example challenge.
- **Measured on your runs** / **Seen in top solutions**: from a private corpus pack, when one is loaded.

The app contains no scores you didn't enter. The tracker's optional demo rows are labelled as invented and are excluded from the dashboard and the report.

## Privacy

The tracker, checklists and the latest audit are stored in your browser's `localStorage` (key `eris-lab:v2`; older `v1` data migrates into the first workspace automatically). Use **Back up and restore** on the dashboard to keep a copy or move to another browser. This repository is public, so the app doesn't bundle your solution files or any corpus. A private corpus pack (`corpus-pack.json`, kept in your private repository) can be loaded from Case studies; it stays in that browser only. `node tools/build-single.cjs out.html --pack <pack.json>` bakes one into a single file, which is then private data.

## Tests

```bash
cd eris-lab
node --test tests/engines.test.cjs                     # 39 unit tests for the engines
NODE_PATH="$(npm root -g)" node tests/e2e.cjs          # Playwright: 79 checks across every view, light and dark, phone width, all tools
ERIS_PACK=<path to corpus-pack.json> NODE_PATH="$(npm root -g)" node tests/e2e.cjs   # adds the private-pack flow (80)
```

`node tools/build-single.cjs out.html` bundles everything into one HTML file.

## Layout

```
index.html            app shell
assets/styles.css     design tokens, light and dark themes, responsive layout
src/data/             platform rules, library, techniques, prompts, templates, lessons and game, worked examples
src/engines/          pure logic (auditor, analyzer, checker, tracker, report, search, oof, logparse), tested under Node
src/ui/               views and UI helpers
src/app.js            routing, navigation, search, theme
tests/                unit tests, e2e test and fixtures
tools/                single-file bundler
```
