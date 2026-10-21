/* Global search over every piece of content in the app. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});

  function buildIndex() {
    const items = [];
    const add = (type, title, text, route, anchor) => items.push({ type, title, text, route, anchor, hay: (title + " " + text).toLowerCase() });
    (E.RULES || []).forEach((r) => add("Rule", r.title, `${r.text} ${r.group} ${r.cite} ${r.id}`, "rules", "rule-" + r.id));
    (E.LIBRARY || []).forEach((l, i) => add("Library", l.title, `${l.body} ${l.validate} ${l.task}`, "library", "lib-" + i));
    (E.TECHNIQUES || []).forEach((t) => add("Technique", t.name, `${t.helps} ${t.hurts} ${t.validate} ${t.compliance} ${t.tags.join(" ")}`, "leaderboard", "tech-" + t.id));
    (E.METRICS || []).forEach((m) => add("Metric", m.name, `${m.loss} ${m.post} ${m.pitfall}`, "leaderboard", "metric-" + m.id));
    (E.WORKFLOW || []).forEach((w) => add("Workflow", w.name, `${w.goal} ${w.steps.join(" ")} ${w.done}`, "workflow", "wf-" + w.id));
    (E.PROMPT_ORDER || []).forEach((id) => add("Prompt", E.PROMPTS[id].name, E.PROMPTS[id].when, "prompts", "prompt-" + id));
    (E.TEMPLATES || []).forEach((t) => add("Template", t.name, t.note + " " + t.code.slice(0, 400), "templates", "tpl-" + t.id));
    (E.LEAKAGE_CHECKS || []).forEach((c, i) => add("Leakage check", c.text, c.cite, "generalization", "leak-" + i));
    (E.ROBUSTNESS_TESTS || []).forEach((r, i) => add("Robustness", r.name, r.how, "generalization", "rob-" + i));
    (E.CONFLICTS || []).forEach((c, i) => add("Doc conflict", c.topic, `${c.a} ${c.b} ${c.resolution}`, "dashboard", "conf-" + i));
    (E.KEY_FACTS || []).forEach((k) => add("Fact", k.label, `${k.value} ${k.cite}`, "dashboard", ""));
    return items;
  }

  function search(index, q, limit = 30) {
    const terms = String(q || "").toLowerCase().split(/\s+/).filter((t) => t.length > 1);
    if (!terms.length) return [];
    return index
      .map((it) => {
        let score = 0;
        for (const t of terms) {
          const inTitle = it.title.toLowerCase().includes(t);
          const inHay = it.hay.includes(t);
          if (!inHay) return null;
          score += inTitle ? 3 : 1;
        }
        return { ...it, score };
      })
      .filter(Boolean)
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  E.search = { buildIndex, search };
})();
