/* Claude Code prompt templates. buildPrompt(id, ctx) returns plain text ready
 * to paste. Every template inherits the platform contract, the evidence
 * standard and the prohibited-shortcut list, so no prompt can omit them. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  const CATEGORY_RULES = {
    general: [],
    nlp: ["Regex only for cleaning or deterministic extraction (§5.3.1). TF-IDF, n-grams and Markov chains are grey area (§5.3.2). The core model must be trained.",
      "If the data looks synthetic, the model must learn any generator artefacts. Never hardcode templates or regexes around them (§6.1)."],
    cv: ["No tabular models on raw pixels: immediate rejection (§6.2). Train or fine-tune a CNN or ViT.",
      "Hand-made image features go into the deep model as extra inputs. A tabular-only use of them is grey (§6.2)."],
    retrieval: ["The ranking or generation stage must be a genuinely trained model, usually deep. Hand features plus an off-the-shelf ranker is not enough (§6.3).",
      "A frozen retriever is fine, but a fully frozen pipeline is not. Fine-tuning the retriever is recommended (§6.3)."],
    finetune: ["The pretrained backbone must actually be fine-tuned. Embeddings plus a tabular model is grey and likely rejected (§6.4).",
      "For bio/chem Fine-tuning challenges, fine-tune a domain-appropriate model. XGBoost or LightGBM alone is not tolerated (§6.4)."],
    scratch: ["Nothing pretrained: no weights, embeddings or distillation teachers (§6.5). Standard tokenizers are fine unless the challenge says otherwise.",
      "The same 1.5 h limit applies. Size the architecture to fit."],
    biochem: ["Check the category. NLP-categorized: tabular models on frozen embeddings are fine. Fine-tuning categorized: fine-tune a domain model (§6.4, §6.6)."],
  };

  function contextBlock(c) {
    const lines = [
      `Challenge: ${c.challenge || "(name not given)"}`,
      `Task type: ${c.taskName || c.task || "(not given)"} | Platform category: ${c.categoryName || c.category || "(not given)"}`,
      `Metric: ${c.metric || "(read it from the problem description)"}${c.direction ? ` (${c.direction === "min" ? "lower" : "higher"} is better)` : ""}`,
      `Data: ${c.publicDir || "./dataset/public"} | Solution file: ${c.solutionPath || "solution.py"}`,
      `Runtime budget for the whole script: ${c.budgetMin || 50} min${c.cpuOnly ? " on CPU only (" + (c.cores || "?") + " cores)" : " on one NVIDIA A10G"}, with at least 30% headroom`,
    ];
    if (c.bestCv) lines.push(`Current best validated score (from my own logs): ${c.bestCv}`);
    if (c.reference) lines.push(`Reference solution score reported on the challenge page: ${c.reference} (a test-set number, not CV)`);
    if (c.constraints) lines.push(`Challenge-specific constraints I know of: ${c.constraints}`);
    const m = c.metricId && (E.METRICS || []).find((x) => x.id === c.metricId);
    if (m) lines.push(`Optimise for the metric (HYPOTHESIS until checked on OOF): train with ${m.loss.replace(/\.$/, "")}. Decode: ${m.post.replace(/\.$/, "")}. Pitfall: ${m.pitfall}`);
    return "## Context\n" + lines.map((l) => "- " + l).join("\n") + rubricBlock(c) + notesBlock("Grader contract (from the grading script)", c.grader) + notesBlock("Data quirks the description documents", c.prep);
  }

  function notesBlock(title, items) {
    return items && items.length ? `\n\n## ${title}\n` + items.map((x) => "- " + x).join("\n") : "";
  }

  /* The challenge rubric, when one was pasted into the analyzer. REQUIRED items
   * must be visibly met; the requirements docstring maps each one to code. */
  function rubricBlock(c) {
    if (!c.rubric || !c.rubric.length) return "";
    const req = c.rubric.filter((r) => r.level === "required"), rec = c.rubric.filter((r) => r.level !== "required");
    return "\n\n## Challenge rubric (from the challenge page)\n" +
      req.map((r) => `- REQUIRED${r.category ? ` (${r.category})` : ""}: ${r.text}`).concat(rec.map((r) => `- ${r.level.toUpperCase()}${r.category ? ` (${r.category})` : ""}: ${r.text}`)).join("\n") +
      "\nEvery REQUIRED criterion must be visibly satisfied in the code and listed in the requirements docstring at the top of the solution, with the lines that meet it. Do RECOMMENDED ones when they don't hurt validated CV; if you skip one, say why in a comment.";
  }

  function contractBlock(c) {
    const cat = CATEGORY_RULES[c.category] || [];
    const rules = [
      "Entry point: `python3 solution.py <public_dir> <submission_out>`. Read `Path(sys.argv[1])` and `Path(sys.argv[2])`, with no defaults or fallbacks. Create the output's parent directory.",
      "One self-contained script. Preprocessing, features, training and inference all run fresh every time. Never load cached artifacts from earlier runs.",
      "Kaggle Docker image libraries only. Never pip install. No models from GitHub. Downloading general-purpose pretrained weights from Hugging Face or timm is fine (when the category allows pretrained models).",
      "Real training or fine-tuning must happen inside the script. No inference-only or rule-based solutions.",
      "Determinism (checked by code review): seed random, numpy, torch (CPU and CUDA) and every random_state or seed argument. Seed DataLoader workers and use a seeded generator. Set the device as the constant `torch.device(\"cuda\")`.",
      "Never branch on the clock, os.cpu_count(), torch.cuda.device_count(), cuda availability, free memory or environment variables. No try/except fallbacks, debug flags or method-switching if-branches. Keep one fixed path.",
      "Fixed epochs or steps measured to fit the budget. In-script HPO must use a seeded sampler and a fixed n_trials, never a timeout.",
      "Challenge restrictions in the problem description (model limits, shorter runtime) override these rules. Ignore boilerplate such as 'no internet', 'rule-based OK' or 'decoding tricks fine'.",
    ].concat(cat);
    return "## Platform contract (do not violate)\n" + rules.map((r) => "- " + r).join("\n");
  }

  const PROHIBITED = `## Prohibited shortcuts. Refuse them even if they raise the score.
- Using test data for anything beyond one-sample (or one-query-list) inference: pseudo-labelling, reweighting, test-time adaptation, fitting scalers, vocabularies or encoders on test, or calibrating or thresholding to test-set statistics.
- External datasets, self-generated synthetic training data (including SMOTE-style row synthesis and LLM-written text), or weights I fine-tuned elsewhere.
- Hardcoded labels, lookup tables, or hand-written rules that encode a pattern the model should learn.
- Reading private/answers files, or probing the leaderboard to infer labels.
- Calling LLM APIs from the solution.
If something you want to try is in a grey area (regex beyond cleaning, TF-IDF, tabular-on-embeddings), stop and tell me, with the reason, before you use it.`;

  const EVIDENCE = `## Evidence standard
- Inspect the actual files before you assume anything about columns, sizes, ids, file names or the metric. Quote what you found: \`head\`, shapes, dtypes.
- Never state a score you did not compute in this session. Every number you report must come from a command you ran, and you must show the log line.
- Label every claim VERIFIED (you ran it, with output shown) or HYPOTHESIS (reasoning or prior experience, not yet tested).
- Never claim or imply anything about private-leaderboard performance. The best you can report is validated CV on a split that mimics the test construction.
- When a result is within fold-to-fold noise, say so and don't call it an improvement.`;

  const T = {};

  T.recon = {
    name: "1 · Recon & constraint sheet",
    when: "Start of a new challenge. No code changes yet.",
    body: (c) => `${contextBlock(c)}

## Task
Do not write the solution yet. Build a constraint sheet and a validation plan from the actual files.
1. List the files under the data directory with sizes. Read the problem description, the metric definition and the sample submission.
2. Profile train and test: shapes, dtypes, missing values, target distribution, and cardinality of id-like columns.
3. Find the entity the hidden test is most likely built on (users, queries, documents, languages, dates, sites). Check overlap of those keys between train.csv and test.csv using only the key columns, and report the counts. Describe the likely split construction as a HYPOTHESIS with the evidence.
4. Re-implement the metric exactly as described. Unit-test it on the sample submission so it matches the expected format and range.
5. Propose a CV scheme (and one stress split) that mimics the hidden-test construction, with the reasons.
6. List every rule from the platform contract and the challenge text that constrains model choice, and note any line that looks like boilerplate.
7. Build the rubric. If the challenge page shows one (REQUIRED / RECOMMENDED criteria), copy it. If not, derive the criteria the description implies: every column it flags (target-leaking, entirely null, dates and their format, multi-value or pipe-separated, free text), the target transform the metric implies, the grader's format checks, and a train-vs-validation overfitting check. Mark each derived item HYPOTHESIS.
8. If a prepare script or grading script is shown, read it: say how it splits train from test (random, grouped, by time) and list every check the grader raises on.
9. Finish with: constraint sheet, validation plan, 3–5 candidate approaches ranked by expected value and compliance risk, and the questions I should ask a reviewer.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.baseline = {
    name: "2 · Compliant baseline (submission 1)",
    when: "You have the constraint sheet and want a first end-to-end submission.",
    body: (c) => `${contextBlock(c)}

## Task
Write ${c.solutionPath || "solution.py"}: a correct, compliant, end-to-end baseline. Correctness and compliance matter more than score at this stage.
1. Skeleton: thread pinning, then seeds, then constants (SEED, DEVICE, N_FOLDS, EPOCHS) at the top, then argv paths. No fallbacks.
2. Load data with the exact file names from the description. Fail loudly if a column is missing.
3. Build the validation folds from the agreed scheme with a fixed seed and use them for every model.
4. Train one simple, appropriate model (category rules apply). Produce OOF and test predictions.
5. Log the challenge metric per fold, the mean ± std, and trivial baselines (constant, random, one simple heuristic) on the same folds. Also print the train-fit score next to the validation score, so the overfitting gap is visible in the log.
6. Final fit on all training data with a fixed iteration count derived from CV, then predict test.
7. Before writing, validate the submission against sample_submission.csv: same columns and order, row count, unique ids matching test, finite values, valid range, not constant. Raise on failure.
8. Run it locally: \`python3 ${c.solutionPath || "solution.py"} ${c.publicDir || "./dataset/public"} ./working/submission.csv\`. Report wall time and peak memory. Run it twice and compare the CSV hashes.
Start the file with a requirements-map docstring: each rubric criterion and challenge constraint, and the lines that satisfy it. Comment the reasoning at each step. Reviewers read the comments.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.experiment = {
    name: "3 · Controlled experiment loop",
    when: "A working baseline exists and you want genuine improvements.",
    body: (c) => `${contextBlock(c)}

## Task
Improve the validated score through controlled experiments. Each round:
1. State one hypothesis and the single change that tests it${c.focus ? ` (focus area: ${c.focus})` : ""}. Say why you expect it to help on unseen data, not just on CV.
2. Keep folds, seeds and everything else fixed. Run the parent and the variant on identical folds.
3. Report a table: experiment, per-fold scores, mean, std, paired delta vs parent, stress-split score, runtime, and peak memory.
4. Decide: KEEP only if the paired delta beats about 2× its standard error AND the stress split doesn't get worse. Otherwise REVERT and record it as a negative result.
5. Propose the next experiment from the evidence (error slices, fold variance, runtime headroom), not from a generic list.
Stop after 3 rounds, or sooner if the runtime budget would be exceeded, and summarise what is VERIFIED vs still HYPOTHESIS.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.error = {
    name: "4 · Error analysis & subgroups",
    when: "Scores plateau or a fold is unusually weak.",
    body: (c) => `${contextBlock(c)}

## Task
Diagnose where the current model fails, using OOF predictions only.
1. Produce OOF predictions if they don't exist yet (same folds as the tracker).
2. Slice the metric by: class or target bucket, the group key, input length or size buckets, missingness patterns, and confidence deciles. Show each slice with its row count.
3. List the worst-performing slices. For each one, show 5–10 concrete examples with prediction vs truth.
4. For each failure pattern, write a hypothesis, a proposed fix that the model learns (not a hand rule), expected gain, compliance risk, and how to validate it.
5. Rank the fixes by expected gain divided by cost. Don't implement anything yet.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.generalize = {
    name: "5 · Generalisation stress test",
    when: "Before trusting a gain, and before choosing finals.",
    body: (c) => `${contextBlock(c)}

## Task
Estimate how the current solution behaves on unseen data, without using the test set beyond inspecting key overlap.
1. Compare three splits on the current best config: random KFold, the matched split (grouped or temporal), and leave-one-group-out on the most plausible shift axis. Report the per-fold or per-group scores.
2. Check train vs OOF score gap (overfitting) and learning curves at 25/50/100% of training data (under- or over-capacity).
3. Run seed stability with 3 seeds and report the spread.
4. Run a leakage audit: every fit/fit_transform call, every feature computed from aggregates, any id- or order-derived feature. Name the file and line.
5. Verdict: which number best predicts private-LB behaviour and why, plus concrete changes that would reduce the gap (lower capacity, different grouping, regularisation).

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.audit = {
    name: "6 · Compliance & determinism audit",
    when: "Before every scripted submission.",
    body: (c) => `${contextBlock(c)}

## Task
Audit ${c.solutionPath || "solution.py"} as the platform's four checks would. Don't change behaviour silently.
1. Determinism review: list every RNG source and its seed, any clock, machine or env-dependent logic, every try/except, and every branch that could choose a different method. Give line numbers.
2. Compliance review against the challenge text and the platform contract: test-set usage, external or synthetic data, pretrained-model rules for this category, hardcoded values, regex or TF-IDF reliance (strip-the-model test), and library imports not in the Kaggle image.
3. I/O contract: argv usage, files written outside the output path, and output validation.
4. Runtime: estimate wall time on an A10G from a local timing, and list the biggest costs.
5. Output a findings table: severity (blocker/high/medium/low), line, issue, rule, and minimal fix. Then apply only the blocker and high fixes, re-run, and show the CSV hash is unchanged between two runs.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.debug = {
    name: "7 · Debug a failed run",
    when: "A local or platform run crashed, timed out, ran out of memory, or wrote a malformed CSV.",
    body: (c) => `${contextBlock(c)}

## Failure evidence
${c.failureLog ? "```\n" + c.failureLog + "\n```" : "(paste the error output or log tail here)"}

## Task
1. Reproduce the failure locally with the exact runtime command, and show the error.
2. Identify the root cause (don't guess). Show the line and the state that triggers it.
3. Fix it with the smallest change that keeps one fixed code path. No try/except fallbacks, no "if it fails, use X".
4. If it is memory or time: measure the peak memory and time per stage, then reduce batch size, sequence length or epochs to constants that fit with a margin. Never make them conditional on the machine.
5. Re-run end to end, validate the CSV, and run twice to confirm identical hashes.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.runtime = {
    name: "8 · Fit the runtime budget",
    when: "The script runs too long or too close to the limit.",
    body: (c) => `${contextBlock(c)}

## Task
Bring the total runtime under ${c.budgetMin || 50} min on the target hardware with at least a 15% margin, losing as little validated score as possible.
1. Profile each stage (load, features, each fold, final fit, inference) with timestamps. Logging only; nothing may depend on the clock.
2. Rank the stages by cost and propose cuts: fewer folds in the final script, AMP/bf16, shorter sequences, smaller models, fewer HPO trials, cached tokenisation within the run, and vectorised feature code.
3. For each cut, measure the score cost on the same folds and the time saved.
4. Apply the best set as constants and report the before/after time and score.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.ensemble = {
    name: "9 · Ensemble on OOF",
    when: "Two or more validated models exist.",
    body: (c) => `${contextBlock(c)}

## Task
Build an ensemble only if it genuinely helps on unseen data.
1. Collect OOF and test predictions for each member on identical folds. Show the pairwise correlation of the OOF predictions.
2. Fit blend weights on OOF with a coarse grid or non-negative least squares. Validate them by fitting on half the folds and scoring the other half.
3. Check the stress split as well. Reject weightings that win CV but lose the stress split.
4. Keep any category rule (for example, the trained deep ranker stays the major component in retrieval).
5. Report member scores, blend score, delta vs best single model with its standard error, and the extra runtime.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.final = {
    name: "10 · Final review & submission package",
    when: "Choosing and preparing the final submission.",
    body: (c) => `${contextBlock(c)}

## Task
Prepare the final submission and a review report.
1. Re-run the final script from a clean working directory with the exact runtime command. Record wall time, peak memory, and the CSV hash over two runs.
2. Validate the CSV against sample_submission.csv and test.csv.
3. Compliance pass: determinism, test-set usage, data and weights origin, category rules, imports, and argv contract. Fix any blocker.
4. Write a report with: summary of approach; validation evidence (split design, per-fold scores, stress-split scores, seed spread), all VERIFIED with log lines; ranked improvement ideas not yet done (HYPOTHESIS); known risks; and the exact run command.
5. Recommend which submission to select as final, based on matched-split CV and stability, not public LB alone. Make no claims about private-LB performance.

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  T.rubric = {
    name: "11 · Rubric coverage check",
    when: "After the baseline and before each scripted submission: is every rubric criterion visibly met?",
    body: (c) => `${contextBlock(c)}

## Task
Check ${c.solutionPath || "solution.py"} against the challenge rubric${c.rubric && c.rubric.length ? " above" : " (copy it from the challenge page; if there is none, derive the implied criteria from the description and mark them HYPOTHESIS)"}. Don't change behaviour silently.
1. For each criterion, quote the line numbers that satisfy it, or write MISSING. "The model handles it implicitly" doesn't count: a reviewer must be able to see it.
2. For any overfitting criterion, run the script and show the train-fit and validation scores side by side from the log.
3. For leakage criteria, list the columns that are dropped and show the drop is defensive (only when present) and commented.
4. Propose the smallest change for each MISSING item. Apply the REQUIRED ones. For RECOMMENDED ones, run each as a controlled experiment on the same folds and keep it only if CV doesn't get worse, or keep it with a comment if it is free.
5. Update the requirements-map docstring at the top, re-run end to end, validate the CSV, and show the coverage table: criterion, level, lines, status (VERIFIED with log line, or HYPOTHESIS).

${contractBlock(c)}

${EVIDENCE}

${PROHIBITED}`,
  };

  E.PROMPTS = T;
  E.PROMPT_ORDER = ["recon", "baseline", "experiment", "error", "generalize", "audit", "debug", "runtime", "ensemble", "final", "rubric"];
  E.buildPrompt = (id, ctx) => (T[id] ? T[id].body(ctx || {}) : "");
  E.CATEGORY_RULES = CATEGORY_RULES;
})();
