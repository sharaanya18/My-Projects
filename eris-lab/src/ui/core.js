/* UI core: escaping, provenance and severity chips, copy/download helpers,
 * persistent store, a small Markdown renderer and the toast. */
(function () {
  const E = (globalThis.Eris = globalThis.Eris || {});
  const UI = (E.ui = E.ui || {});

  UI.esc = (s) => String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const esc = UI.esc;

  const PROV = {
    rule: ["Platform rule", "Stated in your platform documents"],
    verified: ["Verified in your files", "Observed directly in your own files"],
    hypothesis: ["Hypothesis", "General practice, not yet validated on your challenges"],
    verify: ["Check first", "Ambiguous or conflicting; confirm before relying on it"],
  };
  UI.prov = (p, cite) => {
    const [label, title] = PROV[p] || PROV.hypothesis;
    return `<span class="tag tag-${esc(p)}" title="${esc(title)}">${esc(label)}</span>${cite ? UI.cite(cite) : ""}`;
  };
  UI.cite = (c) => (c ? `<span class="cite" title="Source">${esc(c)}</span>` : "");
  const SEV_LABEL = { blocker: "Blocker", high: "High", medium: "Medium", low: "Low", info: "Info", grey: "Grey area", ok: "OK" };
  UI.sev = (s) => `<span class="sev sev-${esc(s)}">${esc(SEV_LABEL[s] || s)}</span>`;

  /* ---------- persistence (localStorage; wrapped because it can throw) ----------
   * v2 layout: one workspace per challenge plus global learning progress.
   *   { version: 2, active, workspaces: [{ id, name, color, created, data }], global }
   * load() returns the ACTIVE workspace's data, so views read and write the
   * same shape as before. v1 data (a single workspace) is migrated on first load. */
  const KEY = "eris-lab:v2";
  const KEY_V1 = "eris-lab:v1";
  const WS_COLORS = ["teal", "coral", "lilac", "sun", "sky", "mint"];
  const freshData = (name) => ({ experiments: [], settings: { direction: "max", budgetMin: 75, challenge: name || "" }, checklist: {}, workflow: {}, leakage: {}, analysis: null, audit: null, promptCtx: {}, analyzerText: null });
  const freshGlobal = () => ({ xp: 0, awarded: {}, badges: {}, lessons: {}, quiz: {}, game: { best: null, plays: 0 }, tourDone: false });
  let root = null;

  function readRoot() {
    let r = null;
    try { r = JSON.parse(localStorage.getItem(KEY) || "null"); } catch (_) { r = null; }
    if (!r || r.version !== 2 || !Array.isArray(r.workspaces) || !r.workspaces.length) {
      let v1 = null;
      try { v1 = JSON.parse(localStorage.getItem(KEY_V1) || "null"); } catch (_) { v1 = null; }
      const id = "ws" + Date.now().toString(36);
      const name = (v1 && v1.settings && v1.settings.challenge) || "My first challenge";
      r = { version: 2, active: id, workspaces: [{ id, name, color: "teal", created: Date.now(), data: Object.assign(freshData(name), v1 || {}) }], global: freshGlobal() };
    }
    r.global = Object.assign(freshGlobal(), r.global || {});
    r.workspaces.forEach((w) => { w.data = Object.assign(freshData(w.name), w.data || {}); });
    if (!r.workspaces.some((w) => w.id === r.active)) r.active = r.workspaces[0].id;
    return r;
  }
  const activeWs = () => root.workspaces.find((w) => w.id === root.active);

  UI.store = {
    root() { if (!root) root = readRoot(); return root; },
    load() { UI.store.root(); return activeWs().data; },
    global() { UI.store.root(); return root.global; },
    active() { UI.store.root(); return activeWs(); },
    list() { UI.store.root(); return root.workspaces; },
    save() {
      try { localStorage.setItem(KEY, JSON.stringify(root)); return true; } catch (_) { return false; }
    },
    update(fn) { const s = UI.store.load(); fn(s); UI.store.save(); return s; },
    updateGlobal(fn) { const g = UI.store.global(); fn(g); UI.store.save(); return g; },
    create(name) {
      UI.store.root();
      const id = "ws" + Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
      const nm = String(name || "").trim() || `Challenge ${root.workspaces.length + 1}`;
      root.workspaces.push({ id, name: nm, color: WS_COLORS[root.workspaces.length % WS_COLORS.length], created: Date.now(), data: freshData(nm) });
      root.active = id;
      UI.store.save();
      return id;
    },
    switchTo(id) { UI.store.root(); if (root.workspaces.some((w) => w.id === id)) { root.active = id; UI.store.save(); } },
    rename(id, name) { const w = UI.store.list().find((x) => x.id === id); if (w && String(name).trim()) { w.name = String(name).trim(); w.data.settings.challenge = w.name; UI.store.save(); } },
    remove(id) {
      UI.store.root();
      if (root.workspaces.length <= 1) return false;
      root.workspaces = root.workspaces.filter((w) => w.id !== id);
      if (root.active === id) root.active = root.workspaces[0].id;
      UI.store.save();
      return true;
    },
    reset() { root = null; try { localStorage.removeItem(KEY); localStorage.removeItem(KEY_V1); } catch (_) {} },
  };

  /* ---------- XP, levels and badges ----------
   * Each award key pays out once, so repeating an action can't farm XP. */
  UI.level = (xp) => {
    const L = E.LEVELS || [{ xp: 0, name: "Seed" }];
    let i = 0;
    while (i + 1 < L.length && xp >= L[i + 1].xp) i++;
    const next = L[i + 1];
    return { index: i, name: L[i].name, xp, floor: L[i].xp, next: next ? next.xp : null, nextName: next ? next.name : null, pct: next ? (xp - L[i].xp) / (next.xp - L[i].xp) : 1 };
  };
  UI.award = (key, xp, label) => {
    const g = UI.store.global();
    if (g.awarded[key]) return false;
    const before = UI.level(g.xp).index;
    UI.store.updateGlobal((gg) => { gg.awarded[key] = Date.now(); gg.xp += xp; });
    const after = UI.level(g.xp);
    UI.toast(`+${xp} XP · ${label}`);
    if (after.index > before) setTimeout(() => { UI.toast(`Level up: you're now a ${after.name}!`); UI.confetti(); }, 900);
    document.dispatchEvent(new CustomEvent("eris:xp"));
    return true;
  };
  UI.badge = (id) => {
    const g = UI.store.global();
    if (g.badges[id] || !E.BADGES || !E.BADGES[id]) return false;
    UI.store.updateGlobal((gg) => { gg.badges[id] = Date.now(); });
    setTimeout(() => { UI.toast(`Badge unlocked: ${E.BADGES[id].name}`); UI.confetti(); }, 400);
    document.dispatchEvent(new CustomEvent("eris:xp"));
    return true;
  };

  /* ---------- confetti (skipped when the viewer prefers reduced motion) ---------- */
  UI.confetti = () => {
    if (typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const c = document.createElement("canvas");
    c.className = "confetti";
    c.width = innerWidth; c.height = innerHeight;
    document.body.appendChild(c);
    const ctx = c.getContext("2d");
    if (!ctx) { c.remove(); return; }
    const cs = getComputedStyle(document.documentElement);
    const cols = ["--accent", "--coral", "--lilac", "--sun", "--sky", "--mint"].map((v) => cs.getPropertyValue(v).trim() || "#f5b700");
    const P = Array.from({ length: 90 }, () => ({ x: innerWidth / 2 + (Math.random() - 0.5) * 160, y: innerHeight * 0.35, vx: (Math.random() - 0.5) * 11, vy: -Math.random() * 11 - 4, r: Math.random() * 6 + 4, a: Math.random() * 6, va: (Math.random() - 0.5) * 0.4, c: cols[Math.floor(Math.random() * cols.length)] }));
    const t0 = performance.now();
    const frame = (t) => {
      const dt = t - t0;
      ctx.clearRect(0, 0, c.width, c.height);
      P.forEach((p) => { p.vy += 0.32; p.x += p.vx; p.y += p.vy; p.a += p.va; ctx.save(); ctx.translate(p.x, p.y); ctx.rotate(p.a); ctx.fillStyle = p.c; ctx.globalAlpha = Math.max(0, 1 - dt / 1700); ctx.fillRect(-p.r / 2, -p.r / 4, p.r, p.r / 2); ctx.restore(); });
      if (dt < 1700) requestAnimationFrame(frame); else c.remove();
    };
    requestAnimationFrame(frame);
  };

  /* ---------- Pip, the golden apple (Eris's apple of discord) ----------
   * mood: happy | wow | sad | think | wink. Character colours are fixed in both themes. */
  let pipSeq = 0;
  UI.pip = (mood = "happy", size = 56, label = "Pip the golden apple") => {
    const g = "pipg" + ++pipSeq;
    const mouths = {
      happy: '<path d="M25 41 Q32 48 39 41" stroke="#5a3200" stroke-width="2.4" fill="none" stroke-linecap="round"/>',
      wow: '<ellipse cx="32" cy="43" rx="3.6" ry="4.4" fill="#5a3200"/>',
      sad: '<path d="M26 45 Q32 39.5 38 45" stroke="#5a3200" stroke-width="2.4" fill="none" stroke-linecap="round"/>',
      think: '<path d="M27 43 L37 42" stroke="#5a3200" stroke-width="2.4" stroke-linecap="round"/>',
      wink: '<path d="M25 41 Q32 48 39 41" stroke="#5a3200" stroke-width="2.4" fill="none" stroke-linecap="round"/>',
    };
    const eyes = mood === "wink"
      ? '<circle cx="25" cy="33" r="3.3" fill="#3b2200"/><circle cx="26.2" cy="31.8" r="1.1" fill="#fff"/><path d="M36 33 Q39.5 30.5 43 33" stroke="#3b2200" stroke-width="2.2" fill="none" stroke-linecap="round"/>'
      : mood === "sad"
        ? '<circle cx="25" cy="34" r="3" fill="#3b2200"/><circle cx="39" cy="34" r="3" fill="#3b2200"/><path d="M21 29 L28 30.5 M43 29 L36 30.5" stroke="#3b2200" stroke-width="1.8" stroke-linecap="round"/>'
        : '<circle cx="25" cy="33" r="3.3" fill="#3b2200"/><circle cx="39" cy="33" r="3.3" fill="#3b2200"/><circle cx="26.2" cy="31.8" r="1.1" fill="#fff"/><circle cx="40.2" cy="31.8" r="1.1" fill="#fff"/>';
    return `<svg class="pip pip-${esc(mood)}" width="${size}" height="${size}" viewBox="0 0 64 64" role="img" aria-label="${esc(label)}">
      <defs><radialGradient id="${g}" cx="38%" cy="32%" r="70%"><stop offset="0" stop-color="#ffe680"/><stop offset="0.55" stop-color="#ffc533"/><stop offset="1" stop-color="#e89a00"/></radialGradient></defs>
      <path d="M32 15 C22 9 8 15 9 33 C10 50 22 59 32 56 C42 59 54 50 55 33 C56 15 42 9 32 15 Z" fill="url(#${g})" stroke="#c98200" stroke-width="1.5"/>
      <path d="M32 16 C31 11 32 7 35 4" stroke="#7a4a12" stroke-width="2.6" fill="none" stroke-linecap="round"/>
      <path d="M35 8 C41 2 50 4 52 8 C46 12 40 12 35 8 Z" fill="#4cc38a" stroke="#2f9a68" stroke-width="1.2"/>
      <ellipse cx="18" cy="24" rx="4" ry="6" fill="#fff6c9" opacity="0.65" transform="rotate(-25 18 24)"/>
      ${eyes}
      <ellipse cx="19.5" cy="40" rx="3.6" ry="2.3" fill="#ff8a8a" opacity="0.55"/><ellipse cx="44.5" cy="40" rx="3.6" ry="2.3" fill="#ff8a8a" opacity="0.55"/>
      ${mouths[mood] || mouths.happy}
    </svg>`;
  };
  UI.speech = (mood, html) => `<div class="pip-say">${UI.pip(mood, 52)}<div class="bubble">${html}</div></div>`;

  /* ---------- copy / download ---------- */
  const copyTexts = new Map();
  let copySeq = 0;
  UI.copyBtn = (text, label = "Copy") => {
    const id = "c" + ++copySeq;
    copyTexts.set(id, text);
    return `<button type="button" class="btn btn-small" data-copy="${id}">${esc(label)}</button>`;
  };
  UI.codeBlock = (text, lang = "", opts = {}) => {
    const id = "c" + ++copySeq;
    copyTexts.set(id, text);
    return `<figure class="code${opts.tall ? " code-tall" : ""}">
      <figcaption><span>${esc(lang)}</span><span class="code-actions">${opts.download ? `<button type="button" class="btn btn-small btn-ghost" data-download="${id}" data-filename="${esc(opts.download)}">Download</button>` : ""}<button type="button" class="btn btn-small" data-copy="${id}">Copy</button></span></figcaption>
      <pre id="${id}" tabindex="0"><code>${esc(text)}</code></pre></figure>`;
  };
  UI.setCopy = (id, text) => copyTexts.set(id, text);

  async function copyText(text, srcEl) {
    try {
      await navigator.clipboard.writeText(text);
      UI.toast("Copied to clipboard");
    } catch (_) {
      const pre = srcEl && srcEl.closest("figure") && srcEl.closest("figure").querySelector("pre");
      if (pre) {
        const r = document.createRange(); r.selectNodeContents(pre);
        const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
        UI.toast("Clipboard blocked here. The text is selected; press Ctrl+C (⌘C).");
      } else UI.toast("Clipboard blocked here. Select the text and copy it manually.");
    }
  }
  /* Inside the claude.ai artifact viewer, files go through its "downloads"
   * capability (the viewer confirms each save). Opened locally, a plain link works. */
  let downloadsNs;
  const getDownloads = () => {
    if (downloadsNs !== undefined) return Promise.resolve(downloadsNs);
    const c = globalThis.claude;
    if (!c || typeof c.use !== "function") return Promise.resolve((downloadsNs = null));
    return c.use("downloads").then((n) => (downloadsNs = n || null), () => (downloadsNs = null));
  };
  UI.download = async (filename, text, type = "text/plain") => {
    const ns = await getDownloads();
    if (ns) {
      try { await ns.save({ filename, data: text }); UI.toast(`Saved ${filename}`); }
      catch (e) {
        const code = e && e.code;
        UI.toast(code === "declined" ? "Download cancelled"
          : code === "rejected_extension" ? `.${filename.split(".").pop()} files can't be saved from this view. Use Copy instead.`
          : code === "rate_limited" ? "A save prompt is already open."
          : "Downloads aren't available here. Use Copy instead.");
      }
      return;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = filename;
    document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
    UI.toast(`Downloading ${filename}. If nothing happens, use Copy instead.`);
  };
  document.addEventListener("click", (ev) => {
    const c = ev.target.closest("[data-copy]");
    if (c) { copyText(copyTexts.get(c.dataset.copy) || "", c); return; }
    const d = ev.target.closest("[data-download]");
    if (d) UI.download(d.dataset.filename || "file.txt", copyTexts.get(d.dataset.download) || "");
  });

  /* ---------- toast ---------- */
  let toastTimer = null;
  UI.toast = (msg) => {
    let t = document.getElementById("toast");
    if (!t) { t = document.createElement("div"); t.id = "toast"; t.setAttribute("role", "status"); document.body.appendChild(t); }
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 2600);
  };

  UI.readFile = (file) => new Promise((res, rej) => { const r = new FileReader(); r.onload = () => res(String(r.result)); r.onerror = rej; r.readAsText(file); });

  /* ---------- minimal Markdown → HTML (for the report preview) ---------- */
  function inline(s) {
    return esc(s)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[^\w])_([^_]+)_(?=[^\w]|$)/g, "$1<em>$2</em>");
  }
  UI.md = (src) => {
    const lines = String(src).split("\n");
    const out = [];
    let i = 0;
    while (i < lines.length) {
      const l = lines[i];
      if (/^```/.test(l)) {
        const buf = []; i++;
        while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
        i++; out.push(`<pre class="md-pre"><code>${esc(buf.join("\n"))}</code></pre>`); continue;
      }
      if (/^\|/.test(l)) {
        const rows = [];
        while (i < lines.length && /^\|/.test(lines[i])) rows.push(lines[i++]);
        const cells = (r) => r.replace(/^\||\|$/g, "").split("|").map((c) => c.trim());
        const head = cells(rows[0]);
        const body = rows.slice(2).map(cells);
        out.push(`<div class="table-wrap"><table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join("")}</tr></thead><tbody>${body.map((r) => `<tr>${r.map((c) => `<td>${inline(c)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      const h = l.match(/^(#{1,4})\s+(.*)/);
      if (h) { out.push(`<h${h[1].length + 1}>${inline(h[2])}</h${h[1].length + 1}>`); i++; continue; }
      if (/^\s*([-*]|\d+\.)\s+/.test(l)) {
        const ordered = /^\s*\d+\./.test(l);
        const items = [];
        while (i < lines.length && /^\s*([-*]|\d+\.)\s+/.test(lines[i])) items.push(lines[i++].replace(/^\s*([-*]|\d+\.)\s+/, ""));
        out.push(`<${ordered ? "ol" : "ul"}>${items.map((it) => {
          const cb = it.match(/^\[( |x)\]\s+(.*)/);
          return cb ? `<li class="md-check ${cb[1] === "x" ? "done" : ""}"><span aria-hidden="true">${cb[1] === "x" ? "☑" : "☐"}</span> ${inline(cb[2])}</li>` : `<li>${inline(it)}</li>`;
        }).join("")}</${ordered ? "ol" : "ul"}>`);
        continue;
      }
      if (l.trim()) out.push(`<p>${inline(l)}</p>`);
      i++;
    }
    return out.join("\n");
  };

  UI.fmt = (x, d = 4) => (x === null || x === undefined || Number.isNaN(Number(x)) || x === "" ? "–" : Number(x).toFixed(d));
  UI.uid = () => Math.random().toString(36).slice(2, 9) + Date.now().toString(36).slice(-4);
})();
