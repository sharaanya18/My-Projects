// End-to-end browser test with Playwright (Chromium). Opens the app straight
// from file://, visits every view, and drives each interactive tool.
// Run: NODE_PATH="$(npm root -g)" node eris-lab/tests/e2e.cjs [--shots <dir>]
const { chromium } = require("playwright");
const path = require("node:path");
const fs = require("node:fs");
const assert = require("node:assert/strict");

const url = "file://" + path.join(__dirname, "..", "index.html");
const shotDir = process.argv.includes("--shots") ? process.argv[process.argv.indexOf("--shots") + 1] : null;
const ROUTES = ["dashboard", "rules", "learn", "game", "analyzer", "workflow", "leaderboard", "generalization", "oof", "library", "prompts", "templates", "auditor", "tracker", "checker", "review"];

(async () => {
  const browser = await chromium.launch();
  const errors = [];
  const results = [];
  const step = async (name, fn) => {
    try { await fn(); results.push(["ok", name]); } catch (e) { results.push(["FAIL", name + ": " + e.message.split("\n")[0]]); }
  };

  for (const scheme of ["light", "dark"]) {
    const ctx = await browser.newContext({ colorScheme: scheme, viewport: { width: 1280, height: 900 } });
    const page = await ctx.newPage();
    page.on("pageerror", (e) => errors.push(`[${scheme}] pageerror: ${e.message}`));
    page.on("console", (m) => { if (m.type() === "error") errors.push(`[${scheme}] console: ${m.text()}`); });
    await page.goto(url);
    for (const r of ROUTES) {
      await step(`${scheme}: render #${r}`, async () => {
        await page.evaluate((h) => { location.hash = h; }, r);
        await page.waitForFunction((h) => document.querySelector(`.nav a[href="#${h}"]`)?.classList.contains("active"), r);
        const h1 = await page.textContent("main h1");
        assert.ok(h1 && h1.trim().length > 3, "missing h1");
        const overflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
        assert.equal(overflow, false, "horizontal overflow");
        if (shotDir && scheme === "light") await page.screenshot({ path: path.join(shotDir, `desktop-${r}.png`), fullPage: false });
      });
    }
    if (shotDir && scheme === "dark") {
      for (const r of ["dashboard", "auditor"]) { await page.evaluate((h) => { location.hash = h; }, r); await page.waitForTimeout(150); await page.screenshot({ path: path.join(shotDir, `dark-${r}.png`) }); }
    }
    await ctx.close();
  }

  // Interactive flows
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 900 }, permissions: ["clipboard-read", "clipboard-write"] });
  const page = await ctx.newPage();
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") errors.push(`console: ${m.text()}`); });
  await page.goto(url + "#analyzer");

  await step("analyzer: example produces analysis with evidence", async () => {
    await page.waitForSelector("#an-out .an-summary");
    const t = await page.textContent("#an-out");
    assert.match(t, /AUC-ROC/);
    assert.match(t, /group-held-out/);
    assert.match(t, /48,210/);
    assert.match(t, /Boilerplate to ignore/);
  });
  await step("analyzer: custom text, then hand off to the prompt generator", async () => {
    await page.fill("#an-text", "# Clause Ranker\nRank candidate clauses for each query. Scored by NDCG@10. Test queries come from unseen documents.");
    await page.click("#an-form button[type=submit]");
    await page.waitForFunction(() => document.querySelector("#an-out").textContent.includes("NDCG"));
    await page.click("#an-to-prompt");
    await page.waitForSelector("#pg-out .code");
    assert.equal(await page.inputValue("#pg-challenge"), "Clause Ranker");
    assert.equal(await page.inputValue("#pg-category"), "retrieval");
    const prompt = await page.textContent("#pg-out pre");
    assert.match(prompt, /Clause Ranker/);
    assert.match(prompt, /§6\.3/);
  });
  await step("prompts: switching template and copy button", async () => {
    await page.check("input[name=template][value=debug]");
    await page.fill("#pg-failureLog", "CUDA out of memory at epoch 2");
    await page.waitForFunction(() => document.querySelector("#pg-out pre").textContent.includes("CUDA out of memory"));
    await page.click("#pg-out [data-copy]");
    await page.waitForFunction(() => document.querySelector("#toast")?.classList.contains("show"));
    assert.match(await page.textContent("#toast"), /Copied/);
    const clip = await page.evaluate(() => navigator.clipboard.readText());
    assert.match(clip, /## Failure evidence[\s\S]*CUDA out of memory/);
  });
  await step("auditor: example audit shows blockers; pasted clean code shows none", async () => {
    await page.goto(url + "#auditor");
    await page.waitForSelector("#au-out .verdict");
    assert.match(await page.textContent("#au-out .verdict"), /Blocked/);
    const skeleton = await page.evaluate(() => Eris.TEMPLATES.find((t) => t.id === "skeleton-tabular").code);
    await page.fill("#au-code", skeleton);
    await page.click("#au-form button[type=submit]");
    await page.waitForFunction(() => !document.querySelector("#au-out .verdict").className.includes("blocked"));
    const counts = await page.evaluate(() => Eris.ui.store.load().audit.counts);
    assert.equal(counts.blocker, 0);
    assert.equal(counts.high, 0);
  });
  await step("auditor: file upload path", async () => {
    await page.setInputFiles("#au-file", path.join(__dirname, "fixtures", "violations.py"));
    await page.waitForFunction(() => Eris.ui.store.load().audit && Eris.ui.store.load().audit.fileName === "violations.py");
    assert.match(await page.textContent("#au-out .verdict"), /Blocked/);
  });
  await step("tracker: add runs, paired comparison, chart and recommendations", async () => {
    await page.goto(url + "#tracker");
    await page.fill("#ts-metric", "AUC");
    await page.fill("#te-name", "base");
    await page.fill("#te-folds", "0.80 0.81 0.79 0.80 0.80");
    await page.fill("#te-runtime", "20");
    await page.selectOption("#te-status", "verified");
    await page.click("#te-submit");
    await page.waitForSelector(".exp-table");
    await page.click("#tr-add summary");
    await page.fill("#te-name", "hardneg");
    await page.selectOption("#te-parent", "base");
    await page.fill("#te-folds", "0.82 0.83 0.81 0.82 0.82");
    await page.fill("#te-ood", "0.70");
    await page.fill("#te-lb", "0.815");
    await page.click("#te-submit");
    await page.waitForFunction(() => document.querySelectorAll(".exp-table tbody tr").length === 2);
    assert.match(await page.textContent(".exp-table"), /Better/);
    assert.ok(await page.$(".chart-svg .m-cv"));
    assert.match(await page.textContent(".rec-list"), /Stress-split score well below CV/);
    await page.hover(".chart-svg .col[data-i='1'] .hit");
    assert.equal(await page.isVisible(".tip"), true);
  });
  await step("tracker: delete needs confirmation", async () => {
    const del = page.locator("[data-del]").first();
    await del.click();
    assert.equal(await page.locator(".exp-table tbody tr").count(), 2);
    await page.locator("[data-del]").first().click();
    await page.waitForFunction(() => document.querySelectorAll(".exp-table tbody tr").length === 1);
  });
  await step("dashboard reflects saved state", async () => {
    await page.goto(url + "#dashboard");
    const t = await page.textContent(".status-row");
    assert.match(t, /0\.8200/);
    assert.match(t, /violations\.py/);
  });
  await step("checker: CSV validation against sample", async () => {
    await page.goto(url + "#checker");
    const tmp = fs.mkdtempSync(path.join(require("os").tmpdir(), "eris-"));
    fs.writeFileSync(path.join(tmp, "sample.csv"), "id,prediction\n1,0.5\n2,0.5\n3,0.5\n");
    fs.writeFileSync(path.join(tmp, "sub.csv"), "id,prediction\n1,0.1\n2,nan\n2,0.4\n");
    await page.setInputFiles("#ck-sub", path.join(tmp, "sub.csv"));
    await page.setInputFiles("#ck-sample", path.join(tmp, "sample.csv"));
    await page.click("#ck-form button[type=submit]");
    await page.waitForSelector("#ck-out .ck-list");
    const t = await page.textContent("#ck-out");
    assert.match(t, /duplicate ids/);
    assert.match(t, /missing/);
  });
  await step("checker: runtime, memory and credit calculators react", async () => {
    await page.fill("#rt-epochMin", "6");
    await page.fill("#rt-epochs", "5");
    await page.waitForFunction(() => document.querySelector(".rt-meter").className.includes("over"));
    assert.match(await page.textContent("#mem-out"), /GB/);
    await page.fill("#cr-planned", "20");
    await page.fill("#cr-hoursLeft", "8");
    await page.waitForFunction(() => document.querySelector("#cr-out").textContent.includes("fit"));
    assert.match(await page.textContent("#cr-out"), /Only \d+ of the planned/);
  });
  await step("review: checklist ticks update the report", async () => {
    await page.goto(url + "#review");
    await page.waitForSelector("#rv-out h2");
    const before = await page.textContent("#rv-out");
    assert.match(before, /15 checklist item\(s\) still open/);
    await page.check("input[data-fc=fc-argv]");
    await page.waitForFunction(() => document.querySelector("#rv-out").textContent.includes("14 checklist item(s) still open"));
    assert.match(await page.textContent("#rv-out"), /violations\.py/);
  });
  await step("search: Ctrl+K, query, navigate to a rule anchor", async () => {
    await page.keyboard.press("Control+k");
    await page.keyboard.type("pseudo");
    await page.waitForSelector("#search-results li[data-i]");
    await page.keyboard.press("Enter");
    await page.waitForFunction(() => document.querySelector(".flash"));
  });
  await step("library: filter by task and kind", async () => {
    await page.goto(url + "#library");
    await page.selectOption("#lib-task", "retrieval");
    await page.click("#lib-kind [data-kind=lesson]");
    const visible = await page.$$eval(".lib", (els) => els.filter((e) => !e.hidden).map((e) => e.dataset.kind));
    assert.ok(visible.length > 0 && visible.every((k) => k === "lesson"));
  });

  await step("learn: demo reacts, quiz answers, lesson completes with XP", async () => {
    await page.goto(url + "#learn");
    await page.click("[data-lesson=metric]");
    await page.waitForSelector("#thr-stats");
    const before = await page.textContent("#thr-stats");
    await page.evaluate(() => { const r = document.querySelector("#thr"); r.value = "0.8"; r.dispatchEvent(new Event("input")); });
    assert.notEqual(await page.textContent("#thr-stats"), before);
    for (let i = 0; i < 3; i++) await page.click(`.qcard[data-qi='${i}'] .qopt[data-oi='0']`);
    assert.equal(await page.locator(".qwhy:not([hidden])").count(), 3);
    await page.click("#ls-done");
    const g = await page.evaluate(() => Eris.ui.store.global());
    assert.ok(g.lessons.metric && g.xp >= 50 && g.badges["first-lesson"]);
  });
  await step("learn: split and capacity demos", async () => {
    await page.click("#ls-back");
    await page.click("[data-lesson=cv]");
    await page.waitForSelector(".gchip");
    assert.match(await page.textContent("#sm"), /appear on both sides/);
    await page.click(".seg-btn[data-mode=group]");
    assert.match(await page.textContent("#sm"), /No leaks/);
    await page.click("#ls-back");
    await page.click("[data-lesson=fit]");
    await page.evaluate(() => { const r = document.querySelector("#cap"); r.value = "11"; r.dispatchEvent(new Event("input")); });
    assert.match(await page.textContent("#cap-msg"), /Overfitting/);
  });
  await step("game: a full round records the score", async () => {
    await page.goto(url + "#game");
    await page.click("#gm-start");
    for (let i = 0; i < 12; i++) {
      await page.click(".vbtn.vb-banned");
      await page.click("#gm-next");
    }
    await page.waitForFunction(() => document.querySelector("main h1").textContent.includes("Round complete"));
    const g = await page.evaluate(() => Eris.ui.store.global().game);
    assert.equal(g.plays, 1);
    assert.ok(g.best >= 0 && g.best <= 12);
  });
  await step("oof lab: sample analysis, blend and send to tracker", async () => {
    await page.goto(url + "#oof");
    await page.click("#lab-sample");
    await page.waitForSelector(".tiles");
    assert.ok(await page.$(".heat"));
    assert.equal(await page.locator(".wbars li").count(), 3);
    assert.equal(await page.locator(".sg-bars li").count(), 2);
    assert.match(await page.textContent("#lab-out"), /Best F1/);
    await page.click("[data-send=blend]");
    const runs = await page.evaluate(() => Eris.ui.store.load().experiments.map((e) => e.name));
    assert.ok(runs.some((n) => n.startsWith("blend-")));
    await page.click("[data-send='0']");
    const n = await page.evaluate(() => Eris.ui.store.load().experiments.filter((e) => e.name.startsWith("oof-") || e.name.startsWith("blend-")).length);
    assert.equal(n, 2, "each click logs exactly one run");
  });
  await step("tracker: log parser fills the run form", async () => {
    await page.goto(url + "#tracker");
    await page.click("#tr-log summary");
    await page.fill("#lg-text", "Fold 1 val_auc 0.81\nFold 2 val_auc 0.83\nFold 3 val_auc 0.82\nruntime 30 min\nLODO auc 0.74");
    await page.click("#lg-parse");
    await page.waitForSelector("#lg-fill");
    await page.click("#lg-fill");
    assert.equal(await page.inputValue("#te-folds"), "0.81 0.83 0.82");
    assert.equal(await page.inputValue("#te-ood"), "0.74");
    assert.equal(await page.inputValue("#te-runtime"), "30.0");
  });
  await step("workspaces: create, isolate and switch back", async () => {
    await page.goto(url + "#dashboard");
    const firstId = await page.evaluate(() => Eris.ui.store.active().id);
    await page.fill("#ws-name", "Second challenge");
    await page.click("#ws-new button[type=submit]");
    await page.waitForFunction(() => Eris.ui.store.active().name === "Second challenge");
    assert.equal(await page.evaluate(() => Eris.ui.store.load().experiments.length), 0);
    await page.goto(url + "#tracker");
    assert.match(await page.textContent("#tr-main"), /No runs in "Second challenge"/);
    await page.selectOption("#ws-select", firstId);
    await page.waitForSelector(".exp-table");
    assert.ok(await page.evaluate(() => Eris.ui.store.load().experiments.length) > 0);
    assert.ok(await page.evaluate(() => Eris.ui.store.global().badges.organised));
  });
  await step("workflow ticks persist across reload", async () => {
    await page.goto(url + "#workflow");
    await page.check("input[data-wf='w2:0']");
    await page.reload();
    assert.equal(await page.isChecked("input[data-wf='w2:0']"), true);
  });
  await ctx.close();


  // v1 → v2 migration keeps existing data
  const mctx = await browser.newContext();
  await mctx.addInitScript(() => {
    if (!localStorage.getItem("eris-lab:v2") && !localStorage.getItem("eris-lab:v1"))
      localStorage.setItem("eris-lab:v1", JSON.stringify({ experiments: [{ id: "x1", name: "legacy-run", folds: "0.7 0.71" }], settings: { direction: "max", challenge: "Old challenge" } }));
  });
  const mp = await mctx.newPage();
  mp.on("pageerror", (e) => errors.push(`migration pageerror: ${e.message}`));
  await step("storage: v1 data migrates into the first workspace", async () => {
    await mp.goto(url + "#tracker");
    await mp.waitForSelector(".exp-table");
    assert.match(await mp.textContent(".exp-table"), /legacy-run/);
    assert.equal(await mp.evaluate(() => Eris.ui.store.active().name), "Old challenge");
  });
  await mctx.close();
  // Phone width
  const phone = await browser.newContext({ viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true });
  const pp = await phone.newPage();
  pp.on("pageerror", (e) => errors.push(`phone pageerror: ${e.message}`));
  await pp.goto(url);
  for (const r of ROUTES) {
    await step(`phone: #${r} has no horizontal overflow`, async () => {
      await pp.evaluate((h) => { location.hash = h; }, r);
      await pp.waitForFunction((h) => document.querySelector(`.nav a[href="#${h}"]`)?.classList.contains("active"), r);
      const over = await pp.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
      assert.ok(over <= 1, `overflow ${over}px`);
      if (shotDir && ["dashboard", "tracker", "learn", "game", "oof"].includes(r)) await pp.screenshot({ path: path.join(shotDir, `phone-${r}.png`) });
    });
  }
  await step("phone: menu opens the navigation drawer", async () => {
    await pp.click("#menu-btn");
    await pp.waitForFunction(() => document.body.classList.contains("nav-open"));
    await pp.click(".nav a[href='#rules']");
    await pp.waitForFunction(() => location.hash === "#rules" && !document.body.classList.contains("nav-open"));
  });
  await phone.close();
  await browser.close();

  results.forEach(([s, n]) => console.log(`${s === "ok" ? "ok  " : "FAIL"} ${n}`));
  if (errors.length) console.log("\nBrowser errors:\n" + errors.join("\n"));
  const failed = results.filter(([s]) => s !== "ok").length;
  console.log(`\n${results.length - failed}/${results.length} passed, ${errors.length} browser errors`);
  process.exit(failed || errors.length ? 1 : 0);
})();
