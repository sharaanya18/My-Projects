/* Account sync for the published artifact, through the viewer's private
 * per-person database subtree (data/users/<id>/…). Push is automatic after a
 * change once sync is on; pulling a newer account copy always asks first.
 * Opened locally (no claude runtime), sync reports itself unavailable. */
(function () {
  const E = globalThis.Eris;
  const UI = E.ui;
  const S = (UI.sync = { state: "checking", message: "", lastSync: null, remoteUpdated: null });
  let db = null, uid = null, timer = null, ready = null;
  const listeners = new Set();
  const emit = () => listeners.forEach((fn) => fn(S));
  S.onChange = (fn) => { listeners.add(fn); return () => listeners.delete(fn); };
  const base = () => "data/users/" + uid;

  // keep documents well under the 256 KiB cap: the audited source is local-only
  const slim = (w) => {
    const d = Object.assign({}, w.data);
    if (d.audit) { d.audit = Object.assign({}, d.audit); delete d.audit.source; delete d.audit.lineSev; }
    if (d.prevAudit) { d.prevAudit = Object.assign({}, d.prevAudit); delete d.prevAudit.source; delete d.prevAudit.lineSev; }
    return { id: w.id, name: w.name, color: w.color, created: w.created, caseSlug: w.caseSlug || null, data: JSON.parse(JSON.stringify(d)) };
  };

  S.init = () => {
    if (ready) return ready;
    ready = (async () => {
      const c = globalThis.claude;
      if (!c || typeof c.use !== "function") { S.state = "unavailable"; S.message = "Account sync works in the published version of this page."; emit(); return; }
      const [d, user] = await Promise.all([c.use("db"), c.use("user")]);
      if (!d || !user) { S.state = "unavailable"; S.message = "Sync isn't available in this view."; emit(); return; }
      uid = await user.id();
      if (!uid) { S.state = "unavailable"; S.message = "Sign in to sync."; emit(); return; }
      db = d;
      try {
        const meta = await db.doc(base() + "/meta").get();
        S.remoteUpdated = meta.exists ? meta.data().updated : null;
      } catch (e) { S.remoteUpdated = null; }
      const g = UI.store.global();
      S.state = g.syncOn ? "idle" : "off";
      S.message = "";
      emit();
    })();
    return ready;
  };

  S.remoteIsNewer = () => S.remoteUpdated && S.remoteUpdated > (UI.store.root().updated || 0) + 1000;

  S.push = async () => {
    await S.init();
    if (!db) return false;
    S.state = "syncing"; emit();
    try {
      const r = UI.store.root();
      const ids = r.workspaces.map((w) => w.id);
      for (const w of r.workspaces) await db.doc(base() + "/ws_" + w.id).set(slim(w));
      await db.doc(base() + "/global").set(JSON.parse(JSON.stringify(r.global)));
      const snap = await db.collection(base()).get();
      for (const doc of snap.docs) if (/^ws_/.test(doc.id) && !ids.includes(doc.id.slice(3))) await db.doc(base() + "/" + doc.id).delete();
      const updated = r.updated || Date.now();
      await db.doc(base() + "/meta").set({ updated, active: r.active, ids, savedAt: Date.now() });
      S.remoteUpdated = updated; S.lastSync = Date.now(); S.state = "idle"; S.message = ""; emit();
      return true;
    } catch (e) {
      S.state = "error";
      S.message = e && e.code === "quota_exceeded" ? "Your account storage for this page is full." : e && e.code === "invalid_argument" ? "A workspace is too large to sync (over 256 KiB)." : "Sync failed. Your data is safe in this browser.";
      emit(); return false;
    }
  };

  S.pull = async () => {
    await S.init();
    if (!db) return false;
    S.state = "syncing"; emit();
    try {
      const meta = await db.doc(base() + "/meta").get();
      if (!meta.exists) { S.state = "idle"; S.message = "Nothing saved to your account yet."; emit(); return false; }
      const m = meta.data();
      const workspaces = [];
      for (const id of m.ids || []) { const d = await db.doc(base() + "/ws_" + id).get(); if (d.exists) workspaces.push(JSON.parse(JSON.stringify(d.data()))); }
      if (!workspaces.length) throw new Error("empty");
      const g = await db.doc(base() + "/global").get();
      UI.store.replaceRoot({ version: 2, active: m.active, updated: m.updated, workspaces, global: g.exists ? JSON.parse(JSON.stringify(g.data())) : {} });
      S.remoteUpdated = m.updated; S.lastSync = Date.now(); S.state = "idle"; S.message = ""; emit();
      return true;
    } catch (e) { S.state = "error"; S.message = "Couldn't read your account copy."; emit(); return false; }
  };

  S.enable = async (on) => {
    UI.store.updateGlobal((g) => { g.syncOn = !!on; });
    await S.init();
    if (!db) return;
    S.state = on ? "idle" : "off"; emit();
    if (on && !S.remoteIsNewer()) await S.push();
  };

  // automatic push a moment after the last change, once sync is on
  document.addEventListener("eris:saved", () => {
    if (!UI.store.global().syncOn || !db || S.state === "syncing") return;
    clearTimeout(timer);
    timer = setTimeout(() => { if (!S.remoteIsNewer()) S.push(); }, 2500);
  });
})();
