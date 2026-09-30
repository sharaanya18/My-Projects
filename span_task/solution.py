#!/usr/bin/env python3
"""Explanatory-note grounding: token-level LightGBM tagger + expected-F1 span decoder.

Usage: python solution.py [--data DIR] [--out PATH] [--final]
  default: train on train.csv, report validation F1.
  --final: train on train+validation, write submission for test.csv.
"""
import argparse, csv, json, math, os, re, sys, time
from collections import Counter, defaultdict
import numpy as np
import lightgbm as lgb

csv.field_size_limit(10**9)
WORD = re.compile(r"\w+", re.UNICODE)


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
    if not a or not b:
        return 0.0
    return 2 * len(a & b) / (len(a) + len(b))


def cprefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# ----------------------------------------------------------------- features
def note_info(note):
    toks = [(m.group(), m.start()) for m in WORD.finditer(note)]
    quoted = set()
    for m in re.finditer(r"[„“”‚‘’'\"»«›‹]([^„“”‚‘’'\"»«›‹]{1,60})[„“”‚‘’'\"»«›‹]", note):
        for w in WORD.findall(m.group(1)):
            quoted.add(w.lower())
    info = []
    for w, pos in toks:
        lw = w.lower()
        if len(lw) < 3 and not w[0].isupper():
            continue
        info.append(dict(w=w, lw=lw, g2=grams(lw, 2), g3=grams(lw, 3), cap=w[0].isupper(),
                         q=lw in quoted, pos=pos / max(1, len(note)), alpha=w.isalpha()))
    return info, quoted


CUES = ["frz", "span", "lat", "engl", "ital", "russ", "griech", "vermutlich", "vgl", "gemeint", "name",
        "bezeichn", "heute", "hier", "d. h", "bedeutet", "zu ", "auch", "bzw", "evtl", "möglich", "messung",
        "temperatur", "höhe", "grad", "fuß", "pflanze", "gattung", "art", "mineral", "s.", "übers", "wört"]


def featurize(r, idf=None):
    src, note = r["source_text"], r["note_text"]
    toks = [(m.start(), m.end(), m.group()) for m in WORD.finditer(src)]
    n = len(toks)
    if n == 0:
        return None
    ninfo, quoted = note_info(note)
    lws = [t[2].lower() for t in toks]
    cnt = Counter(lws)
    F = np.zeros((n, 0)).tolist()
    cols = defaultdict(lambda: np.zeros(n, dtype=np.float32))
    nl = note.lower()
    # per-token match to note words
    for i, (s, e, w) in enumerate(toks):
        lw = lws[i]
        g2, g3 = grams(lw, 2), grams(lw, 3)
        best3 = best2 = bestp = 0.0
        exact = exactcs = 0.0
        bw = 0.0
        bq = 0.0
        bcap = 0.0
        sub = 0.0
        for ni in ninfo:
            if ni["lw"] == lw:
                exact = 1.0
                if ni["w"] == w:
                    exactcs = 1.0
                bq = max(bq, float(ni["q"]))
            d3 = dice(g3, ni["g3"])
            d2 = dice(g2, ni["g2"])
            p = cprefix(lw, ni["lw"])
            pp = p / max(len(lw), len(ni["lw"]))
            if len(lw) >= 4 and len(ni["lw"]) >= 4 and (lw in ni["lw"] or ni["lw"] in lw):
                sub = 1.0
            sc = 0.5 * d3 + 0.3 * d2 + 0.2 * pp
            if sc > bw:
                bw = sc
                bcap = float(ni["cap"])
            best3 = max(best3, d3)
            best2 = max(best2, d2)
            bestp = max(bestp, pp if p >= 3 else 0.0)
            bq = max(bq, float(ni["q"]) * (d3 > .5))
        cols["m_exact"][i] = exact
        cols["m_exactcs"][i] = exactcs
        cols["m_d3"][i] = best3
        cols["m_d2"][i] = best2
        cols["m_pref"][i] = bestp
        cols["m_sub"][i] = sub
        cols["m_best"][i] = bw
        cols["m_q"][i] = bq
        cols["m_bcap"][i] = bcap
        cols["t_len"][i] = len(w)
        cols["t_cap"][i] = w[0].isupper()
        cols["t_upper"][i] = w.isupper() and len(w) > 1
        cols["t_digit"][i] = w.isdigit()
        cols["t_hasdigit"][i] = any(c.isdigit() for c in w)
        cols["t_alpha"][i] = w.isalpha()
        cols["t_nonascii"][i] = sum(ord(c) > 127 for c in w)
        cols["t_cnt"][i] = math.log1p(cnt[lw])
        cols["t_pos"][i] = i / n
        cols["t_abspos"][i] = i
        cols["t_sentstart"][i] = float(i == 0 or src[max(0, s - 3):s].strip().endswith((".", ":", "!", "?", ";")))
        cols["t_prevpunct"][i] = float(i > 0 and src[toks[i - 1][1]:s].strip() != "")
        cols["t_nextpunct"][i] = float(i + 1 < n and src[e:toks[i + 1][0]].strip() != "")
        cols["t_paren"][i] = float(s > 0 and src[s - 1] in "(")
        if idf is not None:
            cols["t_idf"][i] = idf.get(lw, idf["__unk__"])
    # neighborhood aggregates of match score
    key = cols["m_best"]
    ex = np.maximum(cols["m_exact"], (cols["m_d3"] > 0.6).astype(np.float32))
    for wdw in (1, 2, 3, 5, 8, 15):
        ma = np.zeros(n, dtype=np.float32)
        ea = np.zeros(n, dtype=np.float32)
        for i in range(n):
            lo, hi = max(0, i - wdw), min(n, i + wdw + 1)
            ma[i] = key[lo:hi].max()
            ea[i] = ex[lo:hi].sum()
        cols[f"w_max{wdw}"] = ma
        cols[f"w_cnt{wdw}"] = ea
    # distance to nearest strong match left / right
    strong = ex > 0
    d_l = np.full(n, 99, dtype=np.float32)
    d_r = np.full(n, 99, dtype=np.float32)
    last = -999
    for i in range(n):
        if strong[i]:
            last = i
        d_l[i] = min(99, i - last)
    last = 999999
    for i in range(n - 1, -1, -1):
        if strong[i]:
            last = i
        d_r[i] = min(99, last - i)
    cols["d_l"], cols["d_r"] = d_l, d_r
    # relative match rank within doc
    cols["m_rank"] = (-key).argsort().argsort().astype(np.float32)
    cols["m_docmax"] = np.full(n, key.max(), dtype=np.float32)
    cols["m_nstrong"] = np.full(n, strong.sum(), dtype=np.float32)
    # neighbours' token features
    for nm in ("t_cap", "t_len", "t_digit", "t_nonascii"):
        v = cols[nm]
        cols[nm + "_p1"] = np.concatenate([[0], v[:-1]]).astype(np.float32)
        cols[nm + "_n1"] = np.concatenate([v[1:], [0]]).astype(np.float32)
    # note-level
    cols["n_len"] = np.full(n, len(note), dtype=np.float32)
    cols["n_ntok"] = np.full(n, len(ninfo), dtype=np.float32)
    cols["n_nquote"] = np.full(n, len(quoted), dtype=np.float32)
    cols["s_len"] = np.full(n, len(src), dtype=np.float32)
    for ci, c in enumerate(CUES):
        cols[f"c_{ci}"] = np.full(n, float(c in nl), dtype=np.float32)
    names = sorted(cols)
    X = np.stack([cols[k] for k in names], 1).astype(np.float32)
    return toks, X, names


def token_labels(toks, spans):
    y = np.zeros(len(toks), dtype=np.float32)
    for i, (s, e, _) in enumerate(toks):
        for a, b in spans:
            if s < b and e > a:
                y[i] = 1
    return y


# ----------------------------------------------------------------- metric / decode
def nonws_set(src, spans):
    S = set()
    for a, b in spans:
        for k in range(a, b):
            if not src[k].isspace():
                S.add(k)
    return S


def f1(src, pred, gold):
    P, G = nonws_set(src, pred), nonws_set(src, gold)
    if not P or not G:
        return 0.0
    return 2 * len(P & G) / (len(P) + len(G))


def decode(toks, src, p, scale=1.0, power=1.0, maxlen=60):
    """Pick contiguous token range maximising approx. expected F1."""
    n = len(toks)
    w = np.array([sum(not c.isspace() for c in t[2]) for t in toks], dtype=np.float64)
    q = np.clip(p, 1e-6, 1) ** power
    q = q / max(q.sum(), 1e-9) * min(1.0, 1.0) if False else q
    cw = np.concatenate([[0], np.cumsum(w)])
    cp = np.concatenate([[0], np.cumsum(w * q)])
    EG = cp[-1] * scale
    best, bi, bj = -1, 0, 0
    for i in range(n):
        for j in range(i, min(n, i + maxlen)):
            g = 2 * (cp[j + 1] - cp[i]) / ((cw[j + 1] - cw[i]) + EG)
            if g > best:
                best, bi, bj = g, i, j
    return [(toks[bi][0], toks[bj][1])]


def build_idf(rows):
    df = Counter()
    for r in rows:
        df.update({m.group().lower() for m in WORD.finditer(r["source_text"])})
    N = len(rows)
    idf = {w: math.log((N + 1) / (c + 1)) for w, c in df.items()}
    idf["__unk__"] = math.log(N + 1)
    return idf


def make_matrix(rows, idf, with_labels=True):
    Xs, ys, meta = [], [], []
    for qi, r in enumerate(rows):
        out = featurize(r, idf)
        if out is None:
            meta.append(None)
            continue
        toks, X, names = out
        meta.append((toks, X.shape[0]))
        Xs.append(X)
        if with_labels:
            ys.append(token_labels(toks, r["spans"]))
    return np.concatenate(Xs), (np.concatenate(ys) if with_labels else None), meta, names


def predict_spans(rows, meta, prob, **kw):
    out, k = [], 0
    for r, m in zip(rows, meta):
        if m is None:
            out.append([(0, max(1, len(r["source_text"])))])
            continue
        toks, n = m
        p = prob[k:k + n]
        k += n
        out.append(decode(toks, r["source_text"], p, **kw))
    return out


PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_data_in_leaf=50, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/data")
    ap.add_argument("--out", default="/working/submission.csv")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    a = ap.parse_args()
    t0 = time.time()
    tr, va = load(f"{a.data}/train.csv"), load(f"{a.data}/validation.csv")
    idf = build_idf(tr)
    Xtr, ytr, mtr, names = make_matrix(tr, idf)
    Xva, yva, mva, _ = make_matrix(va, idf)
    print("features", Xtr.shape, "pos rate", ytr.mean(), f"{time.time()-t0:.0f}s", flush=True)
    d = lgb.Dataset(Xtr, ytr)
    dv = lgb.Dataset(Xva, yva, reference=d)
    m = lgb.train(PARAMS, d, a.rounds, valid_sets=[dv], callbacks=[lgb.log_evaluation(50)])
    pv = m.predict(Xva)
    for sc in (0.5, 1.0, 1.5, 2.0):
        preds = predict_spans(va, mva, pv, scale=sc)
        print("scale", sc, "val F1", np.mean([f1(r["source_text"], p, r["spans"]) for r, p in zip(va, preds)]))
    imp = sorted(zip(m.feature_importance("gain"), names), reverse=True)[:20]
    print(imp)


if __name__ == "__main__":
    main()
