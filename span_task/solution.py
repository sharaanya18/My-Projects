import csv, json, math, os, random, re, sys, time
from collections import Counter, defaultdict
import numpy as np

T_START = time.time()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
csv.field_size_limit(10 ** 9)
WORD = re.compile(r"\w+", re.UNICODE)


def log(*a):
    print(f"[{time.time()-T_START:6.0f}s]", *a, flush=True)


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", newline="")))
    for r in rows:
        L = len(r["source_text"])
        if r.get("normalized_spans"):
            r["spans"] = [(int(math.floor(a * L + .5)), int(math.floor(b * L + .5)))
                          for a, b in json.loads(r["normalized_spans"])]
    return rows


def grams(s, n):
    s = f"^{s}$"
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def dice(a, b):
    return 2 * len(a & b) / (len(a) + len(b)) if a and b else 0.0


def cprefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def note_info(note):
    quoted = set()
    for m in re.finditer(r"[„“”‚‘’'\"»«›‹]([^„“”‚‘’'\"»«›‹]{1,60})[„“”‚‘’'\"»«›‹]", note):
        quoted.update(w.lower() for w in WORD.findall(m.group(1)))
    info = []
    for m in WORD.finditer(note):
        w = m.group()
        lw = w.lower()
        if len(lw) < 3 and not w[0].isupper():
            continue
        info.append(dict(w=w, lw=lw, g2=grams(lw, 2), g3=grams(lw, 3), cap=w[0].isupper(), q=lw in quoted))
    return info, quoted


CUES = ["frz", "span", "lat", "engl", "ital", "russ", "griech", "vermutlich", "vgl", "gemeint", "name",
        "bezeichn", "heute", "hier", "d. h", "bedeutet", "zu ", "auch", "bzw", "evtl", "möglich", "messung",
        "temperatur", "höhe", "grad", "fuß", "pflanze", "gattung", "art", "mineral", "s.", "übers", "wört"]


def build_idf(rows):
    df = Counter()
    for s in {r["source_text"] for r in rows}:
        df.update({m.group().lower() for m in WORD.finditer(s)})
    N = len({r["source_text"] for r in rows})
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    idf["__unk__"] = math.log(N + 1)
    return idf


def featurize(r, idf):
    src, note = r["source_text"], r["note_text"]
    toks = [(m.start(), m.end(), m.group()) for m in WORD.finditer(src)]
    n = len(toks)
    if n == 0:
        return None
    ninfo, quoted = note_info(note)
    lws = [t[2].lower() for t in toks]
    cnt = Counter(lws)
    cols = defaultdict(lambda: np.zeros(n, dtype=np.float32))
    nl = note.lower()
    for i, (s, e, w) in enumerate(toks):
        lw = lws[i]
        g2, g3 = grams(lw, 2), grams(lw, 3)
        best3 = best2 = bestp = exact = exactcs = bw = bq = bcap = sub = 0.0
        for ni in ninfo:
            if ni["lw"] == lw:
                exact = 1.0
                exactcs = max(exactcs, float(ni["w"] == w))
                bq = max(bq, float(ni["q"]))
            d3, d2 = dice(g3, ni["g3"]), dice(g2, ni["g2"])
            p = cprefix(lw, ni["lw"])
            pp = p / max(len(lw), len(ni["lw"]))
            if len(lw) >= 4 and len(ni["lw"]) >= 4 and (lw in ni["lw"] or ni["lw"] in lw):
                sub = 1.0
            sc = 0.5 * d3 + 0.3 * d2 + 0.2 * pp
            if sc > bw:
                bw, bcap = sc, float(ni["cap"])
            best3, best2 = max(best3, d3), max(best2, d2)
            bestp = max(bestp, pp if p >= 3 else 0.0)
            bq = max(bq, float(ni["q"]) * (d3 > .5))
        for k, v in (("m_exact", exact), ("m_exactcs", exactcs), ("m_d3", best3), ("m_d2", best2), ("m_pref", bestp),
                     ("m_sub", sub), ("m_best", bw), ("m_q", bq), ("m_bcap", bcap)):
            cols[k][i] = v
        cols["t_len"][i] = len(w)
        cols["t_cap"][i] = w[0].isupper()
        cols["t_upper"][i] = w.isupper() and len(w) > 1
        cols["t_digit"][i] = w.isdigit()
        cols["t_hasdigit"][i] = any(c.isdigit() for c in w)
        cols["t_alpha"][i] = w.isalpha()
        cols["t_nonascii"][i] = sum(ord(c) > 127 for c in w)
        cols["t_cnt"][i] = math.log1p(cnt[lw])
        cols["t_pos"][i] = i / n
        cols["t_sentstart"][i] = float(i == 0 or src[max(0, s - 3):s].strip().endswith((".", ":", "!", "?", ";")))
        cols["t_prevpunct"][i] = float(i > 0 and src[toks[i - 1][1]:s].strip() != "")
        cols["t_nextpunct"][i] = float(i + 1 < n and src[e:toks[i + 1][0]].strip() != "")
        cols["t_paren"][i] = float(s > 0 and src[s - 1] == "(")
        cols["t_idf"][i] = idf.get(lw, idf["__unk__"])
    key = cols["m_best"]
    ex = np.maximum(cols["m_exact"], (cols["m_d3"] > 0.6).astype(np.float32))
    for wd in (1, 2, 3, 5, 8, 15):
        lo = np.maximum(np.arange(n) - wd, 0)
        hi = np.minimum(np.arange(n) + wd + 1, n)
        cs = np.concatenate([[0], np.cumsum(ex)])
        cols[f"w_cnt{wd}"] = (cs[hi] - cs[lo]).astype(np.float32)
        cols[f"w_max{wd}"] = np.array([key[a:b].max() for a, b in zip(lo, hi)], dtype=np.float32)
    strong = ex > 0
    d_l = np.full(n, 99, dtype=np.float32)
    d_r = np.full(n, 99, dtype=np.float32)
    last = -999
    for i in range(n):
        last = i if strong[i] else last
        d_l[i] = min(99, i - last)
    last = 999999
    for i in range(n - 1, -1, -1):
        last = i if strong[i] else last
        d_r[i] = min(99, last - i)
    cols["d_l"], cols["d_r"] = d_l, d_r
    cols["m_rank"] = (-key).argsort().argsort().astype(np.float32)
    cols["m_docmax"] = np.full(n, key.max(), dtype=np.float32)
    cols["m_nstrong"] = np.full(n, strong.sum(), dtype=np.float32)
    for nm in ("t_cap", "t_len", "t_digit", "t_nonascii"):
        v = cols[nm]
        cols[nm + "_p1"] = np.concatenate([[0], v[:-1]]).astype(np.float32)
        cols[nm + "_n1"] = np.concatenate([v[1:], [0]]).astype(np.float32)
    cols["n_ntok"] = np.full(n, len(ninfo), dtype=np.float32)
    cols["n_nquote"] = np.full(n, len(quoted), dtype=np.float32)
    for ci, c in enumerate(CUES):
        cols[f"c_{ci}"] = np.full(n, float(c in nl), dtype=np.float32)
    names = sorted(cols)
    return toks, np.stack([cols[k] for k in names], 1).astype(np.float32), names


def token_labels(toks, spans):
    y = np.zeros(len(toks), dtype=np.float32)
    for i, (s, e, _) in enumerate(toks):
        if any(s < b and e > a for a, b in spans):
            y[i] = 1
    return y


def f1(src, pred, gold):
    def nz(spans):
        return {k for a, b in spans for k in range(a, b) if not src[k].isspace()}
    P, G = nz(pred), nz(gold)
    return 2 * len(P & G) / (len(P) + len(G)) if P and G else 0.0


def decode(toks, p, scale=1.0, maxlen=60):
    n = len(toks)
    w = np.array([len(t[2]) for t in toks], dtype=np.float64)
    cw = np.concatenate([[0], np.cumsum(w)])
    cp = np.concatenate([[0], np.cumsum(w * np.clip(p, 1e-6, 1))])
    EG = cp[-1] * scale
    best, bi, bj = -1.0, 0, 0
    for i in range(n):
        j = np.arange(i, min(n, i + maxlen))
        g = 2 * (cp[j + 1] - cp[i]) / ((cw[j + 1] - cw[i]) + EG)
        k = int(g.argmax())
        if g[k] > best:
            best, bi, bj = g[k], i, int(j[k])
    return [(toks[bi][0], toks[bj][1])]


ENC_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WSIZES = (1, 3, 6)


class BiEnc:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        torch.set_num_threads(min(10, os.cpu_count() or 4))
        self.tok = AutoTokenizer.from_pretrained(ENC_NAME)
        self.m = AutoModel.from_pretrained(ENC_NAME)
        self.init_state = {k: v.clone() for k, v in self.m.state_dict().items()}
        for p in self.m.embeddings.word_embeddings.parameters():
            p.requires_grad = False
        self.cache = {}

    def reset(self):
        self.m.load_state_dict(self.init_state)
        self.cache = {}

    def _enc(self, texts, maxlen):
        torch = self.torch
        x = self.tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = self.m(**x).last_hidden_state
        msk = x["attention_mask"].unsqueeze(-1).float()
        return torch.nn.functional.normalize((h * msk).sum(1) / msk.sum(1).clamp(min=1), dim=-1)

    def embed(self, texts, maxlen=48, bs=256):
        torch = self.torch
        self.m.eval()
        need = sorted({t for t in texts if t not in self.cache}, key=len)
        with torch.no_grad():
            for b in range(0, len(need), bs):
                chunk = need[b:b + bs]
                for t, v in zip(chunk, self._enc(chunk, maxlen).numpy()):
                    self.cache[t] = v

    def fit(self, rows, budget_sec, bs=32, nneg=5, lr=8e-5, tau=0.05, seed=0):
        torch = self.torch
        torch.manual_seed(seed)
        rng = random.Random(seed)
        self.reset()
        params = [p for p in self.m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        items = []
        for r in rows:
            src = r["source_text"]
            words = [(m.start(), m.end()) for m in WORD.finditer(src)]
            if words and r.get("spans"):
                a, b = max(r["spans"], key=lambda s: s[1] - s[0])
                items.append((r["note_text"], src[a:b], src, words, (a, b)))
        t0, step, order, ema = time.time(), 0, [], None
        self.m.train()
        while time.time() - t0 < budget_sec:
            if len(order) < bs:
                order = list(range(len(items)))
                rng.shuffle(order)
            batch = [items[order.pop()] for _ in range(bs)]
            neg = []
            for note, sp, src, words, (a, b) in batch:
                got = tries = 0
                while got < nneg and tries < 30:
                    tries += 1
                    i = rng.randrange(len(words))
                    j = min(len(words) - 1, i + rng.choice(WSIZES) - 1)
                    s, e = words[i][0], words[j][1]
                    if s < b and e > a:
                        continue
                    neg.append(src[s:e])
                    got += 1
                neg += ["x"] * (nneg - got)
            frac = (time.time() - t0) / budget_sec
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 15) * max(0.05, 1 - frac)
            q = self._enc([it[0] for it in batch], 64)
            cand = self._enc([it[1] for it in batch] + neg, 48)
            loss = torch.nn.functional.cross_entropy(q @ cand.T / tau, torch.arange(len(batch)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
        log(f"   encoder fit: {step} steps, loss {ema:.3f}, {len(items)} pairs")
        self.m.eval()
        self.cache = {}

    def score(self, rows):
        srcs = {}
        for r in rows:
            s = r["source_text"]
            if s not in srcs:
                ws = [m.group() for m in WORD.finditer(s)]
                d = {}
                for w in WSIZES:
                    d[w] = []
                    for i in range(len(ws)):
                        hi = min(len(ws), max(0, i - w // 2) + w)
                        d[w].append(" ".join(ws[max(0, hi - w):hi]))
                srcs[s] = d
        txt = {r["note_text"] for r in rows}
        for d in srcs.values():
            for v in d.values():
                txt.update(v)
        self.embed(sorted(txt))
        out = []
        for r in rows:
            q = self.cache[r["note_text"]]
            d = srcs[r["source_text"]]
            out.append({w: (np.stack([self.cache[t] for t in d[w]]) @ q if d[w] else np.zeros(0)) for w in WSIZES})
        self.cache = {}
        return out


def bi_columns(sc):
    n = len(sc[WSIZES[0]])
    cols, names = [], []

    def z(v):
        return (v - v.mean()) / (v.std() + 1e-6)

    zs = []
    for w in WSIZES:
        v = sc[w].astype(np.float64)
        zv = z(v)
        zs.append(zv)
        rk = (-v).argsort().argsort() / max(1, n - 1)
        cols += [v, zv, rk]
        names += [f"b{w}", f"b{w}_z", f"b{w}_rk"]
        for wd in (2, 5):
            lo, hi = np.maximum(np.arange(n) - wd, 0), np.minimum(np.arange(n) + wd + 1, n)
            cs = np.concatenate([[0], np.cumsum(zv)])
            cols.append((cs[hi] - cs[lo]) / (hi - lo))
            names.append(f"b{w}_z_sm{wd}")
    zm = np.max(zs, 0)
    top = int(zm.argmax())
    srt = np.sort(zm)[::-1]
    cols += [zm, np.abs(np.arange(n) - top).astype(np.float64), np.full(n, srt[0]), np.full(n, srt[0] - (srt[1] if n > 1 else 0))]
    names += ["bz_max", "b_dist_top", "b_doc_top", "b_doc_gap"]
    return np.stack(cols, 1).astype(np.float32), names


def hand_matrix(rows, idf):
    Xs, metas, names = [], [], None
    for r in rows:
        out = featurize(r, idf)
        if out is None:
            metas.append(None)
            continue
        toks, X, names = out
        metas.append(toks)
        Xs.append(X)
    return Xs, metas, names


def group_folds(groups, k):
    ug = sorted(set(groups), key=lambda g: -sum(1 for x in groups if x == g))
    fold_of = {}
    load_ = [0] * k
    for g in ug:
        j = int(np.argmin(load_))
        fold_of[g] = j
        load_[j] += sum(1 for x in groups if x == g)
    return np.array([fold_of[g] for g in groups])


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dev = "--dev" in sys.argv
    if len(args) != 2:
        raise SystemExit("usage: python3 solution.py <public_dir> <submission_out> [--dev]")
    public, out_path = args
    train = load(os.path.join(public, "train.csv"))
    if dev:
        target = load(os.path.join(public, "validation.csv"))
    else:
        train = train + load(os.path.join(public, "validation.csv"))
        target = load(os.path.join(public, "test.csv"))
    log(f"train rows {len(train)}, target rows {len(target)}, dev={dev}")

    idf = build_idf(train)
    Xtr_l, mtr, names = hand_matrix(train, idf)
    Xte_l, mte, _ = hand_matrix(target, idf)
    log("hand features", sum(len(x) for x in Xtr_l), "train tokens")
    ytr_l = [token_labels(m, r["spans"]) for r, m in zip(train, mtr) if m is not None]
    tr_ok = [i for i, m in enumerate(mtr) if m is not None]
    te_ok = [i for i, m in enumerate(mte) if m is not None]


    bi_names = []
    Btr = [None] * len(train)
    Bte = [None] * len(target)
    ENC_BUDGET = float(os.environ.get("ENC_BUDGET", 1500))
    try:
        enc = BiEnc()
        groups = [r["group_id"] for r in train]
        K = min(5, len(set(groups)))
        folds = group_folds(groups, K)
        per_fold = ENC_BUDGET / (K + 1.5)
        for f in range(K):
            if time.time() - T_START > 3000:
                raise RuntimeError("time guard: skipping remaining encoder folds")
            tri = [r for r, fo in zip(train, folds) if fo != f]
            hold = [i for i, fo in enumerate(folds) if fo == f]
            enc.fit(tri, per_fold, seed=f)
            sc = enc.score([train[i] for i in hold])
            for i, s in zip(hold, sc):
                Btr[i], bi_names = bi_columns(s)
            log(f"encoder fold {f+1}/{K} done")
        if time.time() - T_START > 3400:
            raise RuntimeError("time guard: skipping final encoder")
        enc.fit(train, per_fold * 1.5, seed=99)
        sc = enc.score(target)
        for i, s in enumerate(sc):
            Bte[i], bi_names = bi_columns(s)
        log("encoder features done")
    except Exception as e:
        log("encoder stage unavailable, falling back to tabular features only:", repr(e)[:200])
        Btr = [None] * len(train)
        Bte = [None] * len(target)
        bi_names = []

    def assemble(Xl, B, ok):
        parts = []
        for X, i in zip(Xl, ok):
            parts.append(np.hstack([X, B[i]]) if bi_names else X)
        return parts

    Xtr_parts = assemble(Xtr_l, Btr, tr_ok)
    Xte_parts = assemble(Xte_l, Bte, te_ok)
    Xtr = np.concatenate(Xtr_parts)
    ytr = np.concatenate(ytr_l)
    Xte = np.concatenate(Xte_parts)
    row_of = np.concatenate([[i] * len(x) for i, x in zip(tr_ok, Xtr_parts)])
    all_names = list(names) + list(bi_names)
    drop = {"n_len", "s_len", "t_abspos"}
    keep = [i for i, n in enumerate(all_names) if n not in drop]
    log("feature matrix", Xtr.shape, "positives", int(ytr.sum()))

    from sklearn.ensemble import HistGradientBoostingClassifier

    def make(iters):
        return HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.05, max_leaf_nodes=15,
                                              min_samples_leaf=100, l2_regularization=5.0, max_bins=64,
                                              early_stopping=False, random_state=SEED)


    ITERS = (100, 200, 300, 450)
    SCALES = (0.4, 0.7, 1.0, 1.5)
    tr_groups = np.array([train[i]["group_id"] for i in row_of])
    kf = group_folds([r["group_id"] for r in train], min(5, len(set(r["group_id"] for r in train))))
    fold_row = kf[row_of]
    oof = {it: np.zeros(len(ytr)) for it in ITERS}
    for f in sorted(set(fold_row)):
        tr_m = fold_row != f
        m = make(max(ITERS)).fit(Xtr[tr_m][:, keep], ytr[tr_m])
        for i, pr in enumerate(m.staged_predict_proba(Xtr[~tr_m][:, keep])):
            if (i + 1) in ITERS:
                oof[i + 1][~tr_m] = pr[:, 1]
        log(f"tabular CV fold {f} done")
    best = (-1, ITERS[-1], 1.0)
    starts = np.concatenate([[0], np.cumsum([len(x) for x in Xtr_parts])])
    for it in ITERS:
        for sc_ in SCALES:
            fs = []
            for k, i in enumerate(tr_ok):
                p = oof[it][starts[k]:starts[k + 1]]
                fs.append(f1(train[i]["source_text"], decode(mtr[i], p, sc_), train[i]["spans"]))
            v = float(np.mean(fs))
            if v > best[0]:
                best = (v, it, sc_)
        log(f"  CV iters={it}: best so far {best[0]:.4f} (iters {best[1]}, scale {best[2]})")
    _, iters, scale = best
    log(f"selected iters={iters} scale={scale} grouped-CV F1={best[0]:.4f}")

    final = make(iters).fit(Xtr[:, keep], ytr)
    pte = final.predict_proba(Xte[:, keep])[:, 1]


    L_of = [len(r["source_text"]) for r in target]
    pred = [[(0, max(1, L_of[i]))] for i in range(len(target))]
    pos = 0
    for k, i in enumerate(te_ok):
        n = len(Xte_parts[k])
        pred[i] = decode(mte[i], pte[pos:pos + n], scale)
        pos += n
    if dev:
        s = np.mean([f1(r["source_text"], p, r["spans"]) for r, p in zip(target, pred)])
        log(f"DEV validation F1 = {s:.4f}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "normalized_spans"])
        for r, p in zip(target, pred):
            L = len(r["source_text"])
            enc_sp = []
            for a, b in sorted(p):
                assert 0 <= a < b <= L
                assert math.floor(a / L * L + .5) == a and math.floor(b / L * L + .5) == b
                enc_sp.append([a / L, b / L])
            w.writerow([r["id"], json.dumps(enc_sp)])
    chk = list(csv.DictReader(open(out_path, encoding="utf-8")))
    assert len(chk) == len(target) and len({c["id"] for c in chk}) == len(target)
    log(f"wrote {out_path} rows={len(chk)}")


if __name__ == "__main__":
    main()
