/* App shell: navigation, hash routing, global search, theme toggle. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const { esc } = UI;

  const NAV = [
    ["Overview", [["dashboard", "Dashboard"], ["rules", "Platform rules"]]],
    ["Plan", [["analyzer", "Challenge analyzer"], ["workflow", "Workflow"]]],
    ["Improve", [["leaderboard", "Leaderboard optimization"], ["generalization", "Generalization"], ["library", "Strategy library"]]],
    ["Build with Claude", [["prompts", "Prompt generator"], ["templates", "Templates"]]],
    ["Verify", [["auditor", "Solution auditor"], ["tracker", "Experiment tracker"], ["checker", "Runtime & submission"], ["review", "Final review"]]],
  ];
  const ROUTES = NAV.flatMap(([, items]) => items.map(([id]) => id));

  let index = null;
  let pendingAnchor = null;

  function route() {
    const h = (location.hash || "").replace(/^#/, "");
    return ROUTES.includes(h) ? h : "dashboard";
  }

  function render() {
    const id = route();
    const view = E.views[id];
    const main = document.getElementById("main");
    const state = UI.store.load();
    main.innerHTML = view.render(state);
    document.title = `${view.title} · Eris Solver Lab`;
    document.querySelectorAll(".nav a").forEach((a) => {
      const on = a.getAttribute("href") === "#" + id;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    if (view.mount) view.mount(main, state);
    document.body.classList.remove("nav-open");
    if (pendingAnchor) {
      const el = document.getElementById(pendingAnchor);
      pendingAnchor = null;
      if (el) {
        if (el.hidden) el.hidden = false;
        if (el.tagName === "DETAILS") el.open = true;
        el.scrollIntoView({ block: "center" });
        el.classList.add("flash");
        setTimeout(() => el.classList.remove("flash"), 1800);
        return;
      }
    }
    main.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }
  E.app = { render };

  function navHTML() {
    return NAV.map(([group, items]) => `<div class="nav-group"><p class="nav-label">${esc(group)}</p>${items.map(([id, label]) => `<a href="#${id}">${esc(label)}</a>`).join("")}</div>`).join("");
  }

  function go(r, anchor) {
    pendingAnchor = anchor || null;
    if (location.hash === "#" + r) render(); else location.hash = r;
  }

  function setupSearch() {
    const input = document.getElementById("search");
    const list = document.getElementById("search-results");
    let active = -1;
    let results = [];
    const close = () => { list.hidden = true; input.setAttribute("aria-expanded", "false"); active = -1; };
    const draw = () => {
      if (!index) index = E.search.buildIndex();
      results = E.search.search(index, input.value, 12);
      if (!input.value.trim()) { close(); return; }
      list.innerHTML = results.length
        ? results.map((r, i) => `<li role="option" id="sr-${i}" data-i="${i}" class="${i === active ? "on" : ""}"><span class="sr-type">${esc(r.type)}</span><span class="sr-title">${esc(r.title)}</span></li>`).join("")
        : `<li class="sr-empty">No matches for “${esc(input.value)}”</li>`;
      list.hidden = false;
      input.setAttribute("aria-expanded", "true");
    };
    const pick = (i) => { const r = results[i]; if (!r) return; input.value = ""; close(); input.blur(); go(r.route, r.anchor); };
    input.addEventListener("input", () => { active = -1; draw(); });
    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") { active = Math.min(results.length - 1, active + 1); draw(); e.preventDefault(); }
      else if (e.key === "ArrowUp") { active = Math.max(0, active - 1); draw(); e.preventDefault(); }
      else if (e.key === "Enter") { pick(active >= 0 ? active : 0); e.preventDefault(); }
      else if (e.key === "Escape") { input.value = ""; close(); input.blur(); }
    });
    list.addEventListener("mousedown", (e) => { const li = e.target.closest("[data-i]"); if (li) { e.preventDefault(); pick(Number(li.dataset.i)); } });
    input.addEventListener("blur", () => setTimeout(close, 120));
    document.addEventListener("keydown", (e) => {
      const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName);
      if ((e.key === "k" && (e.ctrlKey || e.metaKey)) || (e.key === "/" && !typing)) { e.preventDefault(); input.focus(); input.select(); }
    });
  }

  function setupTheme() {
    const btn = document.getElementById("theme-btn");
    const apply = (t) => {
      if (t === "system") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", t);
      btn.textContent = { system: "Theme: auto", light: "Theme: light", dark: "Theme: dark" }[t];
    };
    let t = "system";
    try { t = localStorage.getItem("eris-lab:theme") || "system"; } catch (_) {}
    apply(t);
    btn.addEventListener("click", () => {
      t = { system: "light", light: "dark", dark: "system" }[t];
      try { localStorage.setItem("eris-lab:theme", t); } catch (_) {}
      apply(t);
    });
  }

  function boot() {
    document.getElementById("nav").innerHTML = navHTML();
    document.getElementById("menu-btn").addEventListener("click", () => document.body.classList.toggle("nav-open"));
    document.addEventListener("click", (e) => {
      const a = e.target.closest("a[data-anchor]");
      if (a) { e.preventDefault(); go(a.getAttribute("href").slice(1), a.dataset.anchor); }
    });
    setupSearch();
    setupTheme();
    window.addEventListener("hashchange", render);
    render();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
