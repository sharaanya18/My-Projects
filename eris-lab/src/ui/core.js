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

  /* ---------- persistence (localStorage; wrapped because it can throw) ---------- */
  const KEY = "eris-lab:v1";
  let memory = null;
  UI.store = {
    load() {
      if (memory) return memory;
      let data = {};
      try { data = JSON.parse(localStorage.getItem(KEY) || "{}") || {}; } catch (_) { data = {}; }
      memory = Object.assign({ experiments: [], settings: { direction: "max", budgetMin: 75 }, checklist: {}, workflow: {}, leakage: {}, analysis: null, audit: null, promptCtx: {}, analyzerText: null }, data);
      return memory;
    },
    save() {
      try { localStorage.setItem(KEY, JSON.stringify(memory)); return true; } catch (_) { return false; }
    },
    update(fn) { const s = UI.store.load(); fn(s); UI.store.save(); return s; },
    reset() { memory = null; try { localStorage.removeItem(KEY); } catch (_) {} },
  };

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
  UI.download = (filename, text, type = "text/plain") => {
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
