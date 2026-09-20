"""Ledger witness selection + question recovery: trained transformer seq2seq."""
import itertools
import json
import math
import re
import unicodedata
from collections import Counter

import numpy as np

import ad
from ad import T

PAD, BOS, EOS, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4, 5, 6
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>", "<cls>", "<sep>", "<mask>"]

WORD_RE = re.compile(r"\d+(?:\.\d+)+|[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\sa-z0-9]", re.UNICODE)
NO_SPACE_BEFORE = set(list(".,;:!?)]}%") + ["'s", "'t", "'re", "'ve", "'ll", "'d", "'m", "n't"])
NO_SPACE_AFTER = set(list("([{$"))


def words(text):
    return WORD_RE.findall(unicodedata.normalize("NFKC", str(text)).casefold())


def detok(toks):
    out = []
    for i, w in enumerate(toks):
        if i == 0 or w in NO_SPACE_BEFORE or toks[i - 1] in NO_SPACE_AFTER:
            out.append(w)
        else:
            out.append(" " + w)
    return re.sub(r"\s+", " ", "".join(out)).strip()


class Vocab:
    def __init__(self, counter, min_freq, cap):
        items = [(w, c) for w, c in counter.items() if c >= min_freq]
        items.sort(key=lambda kv: (-kv[1], kv[0]))
        self.itos = list(SPECIALS) + [w for w, _ in items[:cap]]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def enc(self, toks):
        s = self.stoi
        return [s.get(w, UNK) for w in toks]


def parse_rows(df):
    rows = []
    for r in df.itertuples(index=False):
        recs = json.loads(r.records_json)
        rows.append({"id": str(r.id), "group": str(r.group), "R": len(recs),
                     "ans": [words(o["answer"]) for o in recs],
                     "evi": [words(o["evidence"]) for o in recs]})
    return rows


def build_examples(rows, cfg, svoc, tvoc, labels=None):
    exs = []
    for row in rows:
        recs, spans = [], []
        for i in range(row["R"]):
            a = svoc.enc(row["ans"][i][:cfg["ans_cap"]])
            e = svoc.enc(row["evi"][i][:cfg["evi_cap"]])
            recs.append(np.array([CLS] + a + [SEP] + e, dtype=np.int32))
            spans.append((len(a), len(e)))
        ex = {"id": row["id"], "group": row["group"], "R": row["R"],
              "recs": recs, "spans": spans}
        if labels is not None:
            lab = labels[row["id"]]
            wm = np.zeros(row["R"], dtype=np.float32)
            for tok in lab["witnesses"].split():
                wm[int(tok[1:])] = 1.0
            ex["wmask"] = wm
            q = words(lab["question"])[:cfg["max_tgt"] - 2]
            ex["tgt"] = np.array([BOS] + tvoc.enc(q) + [EOS], dtype=np.int32)
            ex["question"] = lab["question"]
        exs.append(ex)
    return exs


def pairs_for(R):
    return list(itertools.combinations(range(R), 2))


_SUBSET_CACHE = {}


def subsets_for(R):
    """Candidate witness subsets and the pair->subset averaging matrix."""
    if R not in _SUBSET_CACHE:
        prs = pairs_for(R)
        subs = list(itertools.combinations(range(R), R - 2))
        M = np.zeros((len(prs), len(subs)), dtype=np.float32)
        for si, sub in enumerate(subs):
            ss = set(sub)
            inside = [pi for pi, (i, j) in enumerate(prs) if i in ss and j in ss]
            for pi in inside:
                M[pi, si] = 1.0 / len(inside)
        _SUBSET_CACHE[R] = (subs, M)
    return _SUBSET_CACHE[R]


def make_batch(exs, with_target=True, wmask_override=None):
    B = len(exs)
    R = exs[0]["R"]
    assert all(e["R"] == R for e in exs)
    Lr = max(len(r) for e in exs for r in e["recs"])
    rec = np.zeros((B, R, Lr), dtype=np.int32)
    rmask = np.zeros((B, R, Lr), dtype=np.float32)
    flag = np.zeros((B, R, Lr, 1), dtype=np.float32)
    w_ans = np.zeros((B * R, 1, Lr), dtype=np.float32)
    w_evi = np.zeros((B * R, 1, Lr), dtype=np.float32)
    for b, e in enumerate(exs):
        for i, r in enumerate(e["recs"]):
            rec[b, i, :len(r)] = r
            rmask[b, i, :len(r)] = 1.0
            na, ne = e["spans"][i]
            n = b * R + i
            if na > 0:
                w_ans[n, 0, 1:1 + na] = 1.0 / na
            else:
                w_ans[n, 0, 0] = 1.0
            if ne > 0:
                w_evi[n, 0, 2 + na:2 + na + ne] = 1.0 / ne
            else:
                w_evi[n, 0, 0] = 1.0
        wm = wmask_override[b] if wmask_override is not None else e.get("wmask")
        if wm is not None:
            for i in range(R):
                if wm[i] > 0.5:
                    flag[b, i, :, 0] = 1.0
    A = max(1, max(e["spans"][i][0] for e in exs for i in range(R)))
    a_idx = np.zeros((B * R, A), dtype=np.int64)
    a_msk = np.zeros((B * R, A), dtype=np.float32)
    for b, e in enumerate(exs):
        for i in range(R):
            na = e["spans"][i][0]
            n = b * R + i
            for t in range(min(na, A)):
                a_idx[n, t] = n * Lr + 1 + t
                a_msk[n, t] = 1.0
            if na == 0:
                a_idx[n, 0] = n * Lr
                a_msk[n, 0] = 1.0
    prs = pairs_for(R)
    idx_i = np.array([b * R + i for b in range(B) for (i, j) in prs], dtype=np.int64)
    idx_j = np.array([b * R + j for b in range(B) for (i, j) in prs], dtype=np.int64)
    out = {"rec": rec.reshape(B * R, Lr), "rmask": rmask, "B": B, "R": R, "Lr": Lr,
           "kmask": rmask.reshape(B, R * Lr), "flag": flag.reshape(B, R * Lr, 1),
           "cls_idx": np.array([n * Lr for n in range(B * R)], dtype=np.int64),
           "idx_i": idx_i, "idx_j": idx_j, "pairs": prs,
           "pos": np.tile(np.arange(Lr, dtype=np.int32), (B * R, 1)),
           "w_ans": w_ans, "w_evi": w_evi, "a_idx": a_idx, "a_msk": a_msk, "A": A}
    if wmask_override is not None or (exs[0].get("wmask") is not None):
        pt, ut = [], []
        for b, e in enumerate(exs):
            wm = wmask_override[b] if wmask_override is not None else e["wmask"]
            pt.extend([1.0 if (wm[i] > .5 and wm[j] > .5) else 0.0 for (i, j) in prs])
            ut.extend(wm.tolist())
        out["ptgt"] = np.array(pt, dtype=np.float32)
        out["utgt"] = np.array(ut, dtype=np.float32)
        subs, M = subsets_for(R)
        sidx = {sub: k for k, sub in enumerate(subs)}
        tgt = []
        for b, e in enumerate(exs):
            wm = wmask_override[b] if wmask_override is not None else e["wmask"]
            tgt.append(sidx[tuple(i for i in range(R) if wm[i] > 0.5)])
        out["stgt"] = np.array(tgt, dtype=np.int64)
        out["smat"] = M
    if with_target:
        Lt = max(len(e["tgt"]) for e in exs)
        tin = np.zeros((B, Lt - 1), dtype=np.int32)
        tout = np.zeros((B, Lt - 1), dtype=np.int32)
        twt = np.zeros((B, Lt - 1), dtype=np.float32)
        for b, e in enumerate(exs):
            t = e["tgt"]; m = len(t) - 1
            tin[b, :m] = t[:-1]; tout[b, :m] = t[1:]; twt[b, :m] = 1.0
        out.update({"tin": tin, "tout": tout, "twt": twt})
    return out


class Model:
    def __init__(self, cfg, nsrc, ntgt, seed):
        self.cfg = cfg
        d = cfg["d_model"]
        rng = np.random.default_rng(seed)
        self.p = {}

        def par(name, shape, std):
            self.p[name] = rng.normal(0.0, std, shape).astype(np.float32)

        def zer(name, shape):
            self.p[name] = np.zeros(shape, dtype=np.float32)

        self.par, self.zer = par, zer
        par("Esrc", (nsrc, d), d ** -0.5)
        par("Etgt", (ntgt, d), d ** -0.5)
        par("Epos", (cfg["ans_cap"] + cfg["evi_cap"] + 8, d), 0.02)
        par("Edpos", (cfg["max_tgt"] + 2, d), 0.02)
        par("wflag", (d,), 0.02)
        for L in range(cfg["enc_layers"]):
            self._block(f"e{L}", d, cfg["d_ff"], cross=False)
        for L in range(cfg["xrec_layers"]):
            self._block(f"x{L}", d, cfg["d_ff"], cross=False)
        for L in range(cfg["rec_layers"]):
            self._block(f"r{L}", d, cfg["d_ff"], cross=False)
        for L in range(cfg["dec_layers"]):
            self._block(f"d{L}", d, cfg["d_ff"], cross=True)
        for n in ("enc_ln", "rec_ln", "dec_ln", "un_ln", "xrec_ln"):
            self._ln(n, d)
        self._ln("pair_ln", 3 * d)
        par("rvec", (3 * d, d), (3 * d) ** -0.5); zer("rvecb", d)
        self._ln("rvec_ln", 3 * d)
        par("pair_1", (3 * d, d), (3 * d) ** -0.5); zer("pair_1b", d)
        par("pair_2", (d, 1), d ** -0.5); zer("pair_2b", 1)
        par("un_1", (d, d), d ** -0.5); zer("un_1b", d)
        par("un_2", (d, 1), d ** -0.5); zer("un_2b", 1)
        zer("logit_b", ntgt)
        zer("mlm_b", nsrc)
        self.training = True
        self.drng = np.random.default_rng(seed + 977)

    def _ln(self, name, d):
        self.zer(name + "_g", d); self.p[name + "_g"] += 1.0
        self.zer(name + "_b", d)

    def _block(self, pre, d, dff, cross):
        parts = ["sa"] + (["ca"] if cross else []) + ["ff"]
        for part in parts:
            self._ln(f"{pre}_{part}_ln", d)
        for part in parts[:-1]:
            for w in "qkvo":
                self.par(f"{pre}_{part}_{w}", (d, d), d ** -0.5)
                self.zer(f"{pre}_{part}_{w}b", d)
        self.par(f"{pre}_ff_1", (d, dff), d ** -0.5); self.zer(f"{pre}_ff_1b", dff)
        self.par(f"{pre}_ff_2", (dff, d), dff ** -0.5); self.zer(f"{pre}_ff_2b", d)

    def start(self):
        self.t = {k: T(v) for k, v in self.p.items()}

    def P(self, n):
        return self.t[n]

    def lin(self, x, w, b):
        return ad.add(ad.matmul(x, self.P(w)), self.P(b))

    def ln(self, x, n):
        return ad.layernorm(x, self.P(n + "_g"), self.P(n + "_b"))

    def drop(self, x, p):
        if not self.training or p <= 0:
            return x
        m = (self.drng.random(x.d.shape) >= p).astype(np.float32) / (1.0 - p)
        return ad.mul_arr(x, m)

    def mha(self, q_in, kv_in, pre, part, bias):
        d = self.cfg["d_model"]; H = self.cfg["heads"]; dh = d // H
        B, Lq = q_in.d.shape[0], q_in.d.shape[1]
        Lk = kv_in.d.shape[1]
        q = ad.transpose(ad.reshape(self.lin(q_in, f"{pre}_{part}_q", f"{pre}_{part}_qb"),
                                    (B, Lq, H, dh)), (0, 2, 1, 3))
        k = ad.transpose(ad.reshape(self.lin(kv_in, f"{pre}_{part}_k", f"{pre}_{part}_kb"),
                                    (B, Lk, H, dh)), (0, 2, 1, 3))
        v = ad.transpose(ad.reshape(self.lin(kv_in, f"{pre}_{part}_v", f"{pre}_{part}_vb"),
                                    (B, Lk, H, dh)), (0, 2, 1, 3))
        sc = ad.mul_const(ad.matmul(q, ad.transpose(k, (0, 1, 3, 2))), 1.0 / math.sqrt(dh))
        if bias is not None:
            sc = ad.add_const(sc, bias)
        att = self.drop(ad.softmax(sc, axis=-1), self.cfg["attn_dropout"])
        o = ad.reshape(ad.transpose(ad.matmul(att, v), (0, 2, 1, 3)), (B, Lq, d))
        return self.lin(o, f"{pre}_{part}_o", f"{pre}_{part}_ob")

    def ffn(self, x, pre):
        h = ad.gelu(self.lin(x, f"{pre}_ff_1", f"{pre}_ff_1b"))
        return self.lin(self.drop(h, self.cfg["dropout"]), f"{pre}_ff_2", f"{pre}_ff_2b")

    def _stack(self, x, pre, n, bias, memory=None, mbias=None):
        for L in range(n):
            p = f"{pre}{L}"
            xl = self.ln(x, f"{p}_sa_ln")
            x = ad.add(x, self.drop(self.mha(xl, xl, p, "sa", bias), self.cfg["dropout"]))
            if memory is not None:
                x = ad.add(x, self.drop(self.mha(self.ln(x, f"{p}_ca_ln"), memory, p, "ca", mbias),
                                        self.cfg["dropout"]))
            x = ad.add(x, self.drop(self.ffn(self.ln(x, f"{p}_ff_ln"), p), self.cfg["dropout"]))
        return x

    def encode(self, bt):
        cfg = self.cfg; d = cfg["d_model"]
        B, R, Lr = bt["B"], bt["R"], bt["Lr"]
        x = ad.add(ad.mul_const(ad.embed(self.P("Esrc"), bt["rec"]), math.sqrt(d)),
                   ad.embed(self.P("Epos"), bt["pos"]))
        x = self.drop(x, cfg["dropout"])
        kb = ((1.0 - bt["rmask"].reshape(B * R, Lr)) * -1e9).astype(np.float32)[:, None, None, :]
        x = self.ln(self._stack(x, "e", cfg["enc_layers"], kb), "enc_ln")
        tok = ad.reshape(x, (B, R * Lr, d))
        cls = ad.reshape(ad.gather_rows(ad.reshape(x, (B * R * Lr, d)), bt["cls_idx"]), (B * R, d))
        ev = ad.reshape(ad.matmul(T(bt["w_evi"]), x), (B * R, d))
        A = bt["A"]
        xa = ad.reshape(ad.gather_rows(ad.reshape(x, (B * R * Lr, d)), bt["a_idx"].reshape(-1)),
                        (B, R * A, d))
        if cfg["xrec_layers"] > 0:
            ab = ((1.0 - bt["a_msk"].reshape(B, R * A)) * -1e9).astype(np.float32)[:, None, None, :]
            xa = self.ln(self._stack(xa, "x", cfg["xrec_layers"], ab), "xrec_ln")
        wpool = bt["a_msk"].reshape(B * R, 1, A)
        wpool = wpool / np.maximum(wpool.sum(axis=-1, keepdims=True), 1e-6)
        av = ad.reshape(ad.matmul(T(wpool), ad.reshape(xa, (B * R, A, d))), (B * R, d))
        if not cfg.get("rvec_use_evi", 1):
            ev = ad.mul_const(ev, 0.0)     # witness path sees the answer only
        rv = self.lin(self.ln(ad.concat([cls, av, ev], -1), "rvec_ln"), "rvec", "rvecb")
        ctx = self.ln(self._stack(ad.reshape(rv, (B, R, d)), "r", cfg["rec_layers"], None), "rec_ln")
        return tok, ctx

    def encode_tokens(self, ids, mask):
        """Token-level encoder forward only (used for MLM pretraining)."""
        cfg = self.cfg
        d = cfg["d_model"]
        n, Lr = ids.shape
        pos = np.tile(np.arange(Lr, dtype=np.int32), (n, 1))
        x = ad.add(ad.mul_const(ad.embed(self.P("Esrc"), ids), math.sqrt(d)),
                   ad.embed(self.P("Epos"), pos))
        x = self.drop(x, cfg["dropout"])
        kb = ((1.0 - mask) * -1e9).astype(np.float32)[:, None, None, :]
        return self.ln(self._stack(x, "e", cfg["enc_layers"], kb), "enc_ln")

    def mlm_loss(self, ids, mask, flat_pos, targets):
        self.start()
        d = self.cfg["d_model"]
        h = self.encode_tokens(ids, mask)
        n, Lr = ids.shape
        sel = ad.gather_rows(ad.reshape(h, (n * Lr, d)), flat_pos)
        logits = ad.add(ad.matmul(sel, ad.transpose(self.P("Esrc"), (1, 0))), self.P("mlm_b"))
        return ad.ce_loss(logits, targets, np.ones(len(targets), dtype=np.float32), 0.0)

    def witness_scores(self, ctx, bt):
        d = self.cfg["d_model"]
        flat = ad.reshape(ctx, (bt["B"] * bt["R"], d))
        hi = ad.gather_rows(flat, bt["idx_i"])
        hj = ad.gather_rows(flat, bt["idx_j"])
        f = ad.concat([ad.add(hi, hj), ad.abs_(ad.sub(hi, hj)), ad.mul(hi, hj)], -1)
        h = self.drop(ad.gelu(self.lin(self.ln(f, "pair_ln"), "pair_1", "pair_1b")),
                      self.cfg["dropout"])
        pair = ad.reshape(self.lin(h, "pair_2", "pair_2b"), (h.d.shape[0],))
        hu = self.drop(ad.gelu(self.lin(self.ln(flat, "un_ln"), "un_1", "un_1b")),
                       self.cfg["dropout"])
        un = ad.reshape(self.lin(hu, "un_2", "un_2b"), (flat.d.shape[0],))
        return pair, un

    def condition(self, tok, flag):
        if not self.cfg["witness_cond"]:
            return tok
        d = self.cfg["d_model"]
        return ad.add(tok, ad.mul(T(flag), ad.reshape(self.P("wflag"), (1, 1, d))))

    def memory(self, tok, ctx, bt, flag_tok, flag_rec):
        """Decoder memory: per-token states plus one summary vector per record."""
        mem = ad.concat([self.condition(tok, flag_tok), self.condition(ctx, flag_rec)], 1)
        km = np.concatenate([bt["kmask"], np.ones((bt["B"], bt["R"]), dtype=np.float32)], 1)
        return mem, km

    def decode(self, mem, kmask, tin):
        cfg = self.cfg; d = cfg["d_model"]
        B, Lt = tin.shape
        dpos = np.tile(np.arange(Lt, dtype=np.int32), (B, 1))
        y = ad.add(ad.mul_const(ad.embed(self.P("Etgt"), tin), math.sqrt(d)),
                   ad.embed(self.P("Edpos"), dpos))
        y = self.drop(y, cfg["dropout"])
        causal = np.triu(np.full((Lt, Lt), -1e9, dtype=np.float32), 1)[None, None, :, :]
        mb = ((1.0 - kmask) * -1e9).astype(np.float32)[:, None, None, :]
        y = self.ln(self._stack(y, "d", cfg["dec_layers"], causal, memory=mem, mbias=mb), "dec_ln")
        flat = ad.reshape(y, (B * Lt, d))
        return ad.add(ad.matmul(flat, ad.transpose(self.P("Etgt"), (1, 0))), self.P("logit_b")), Lt

    def loss(self, bt):
        self.start()
        tok, ctx = self.encode(bt)
        pair, un = self.witness_scores(ctx, bt)
        lp = ad.bce_loss(pair, bt["ptgt"], np.ones_like(bt["ptgt"]))
        lu = ad.bce_loss(un, bt["utgt"], np.ones_like(bt["utgt"]))
        P = len(bt["pairs"])
        slog = ad.matmul(ad.reshape(pair, (bt["B"], P)), T(bt["smat"]))
        ls = ad.ce_loss(slog, bt["stgt"], np.ones(bt["B"], dtype=np.float32), 0.0)
        lp = ad.add(ad.mul_const(lp, 1.0 - self.cfg["sub_mix"]),
                    ad.mul_const(ls, self.cfg["sub_mix"]))
        if self.cfg.get("witness_only"):
            frec = None
        else:
            frec = bt["flag"].reshape(bt["B"], bt["R"], bt["Lr"], 1)[:, :, 0, :]
        mem, km = (None, None) if frec is None else self.memory(tok, ctx, bt, bt["flag"], frec)
        if self.cfg.get("witness_only"):
            return ad.add(ad.mul_const(lp, self.cfg["lambda_pair"]),
                          ad.mul_const(lu, self.cfg["lambda_un"])), 0.0, float(lp.d), float(lu.d)
        logits, Lt = self.decode(mem, km, bt["tin"])
        lq = ad.ce_loss(logits, bt["tout"].reshape(-1), bt["twt"].reshape(-1),
                        self.cfg["label_smooth"])
        tot = ad.add(lq, ad.add(ad.mul_const(lp, self.cfg["lambda_pair"]),
                                ad.mul_const(lu, self.cfg["lambda_un"])))
        return tot, float(lq.d), float(lp.d), float(lu.d)


# ---------------------------------------------------------------- training

class Adam:
    def __init__(self, params, lr, b1=0.9, b2=0.98, eps=1e-8, wd=0.01):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps; self.wd = wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads, lr_scale, clip):
        tot = 0.0
        for g in grads.values():
            if g is not None:
                tot += float(np.sum(g.astype(np.float64) ** 2))
        gs = min(1.0, clip / (math.sqrt(tot) + 1e-6)) if clip > 0 else 1.0
        self.t += 1
        lr = self.lr * lr_scale
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, w in self.p.items():
            g = grads.get(k)
            if g is None:
                continue
            g = g * gs
            m, v = self.m[k], self.v[k]
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            upd = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.wd > 0 and w.ndim > 1:
                upd = upd + self.wd * w
            w -= lr * upd


def batches_for(exs, batch_size, rng=None):
    by_r = {}
    for i, e in enumerate(exs):
        by_r.setdefault(e["R"], []).append(i)
    out = []
    for R in sorted(by_r):
        idx = sorted(by_r[R], key=lambda i: sum(len(r) for r in exs[i]["recs"]))
        for s in range(0, len(idx), batch_size):
            out.append(idx[s:s + batch_size])
    if rng is not None:
        out = [out[i] for i in rng.permutation(len(out))]
    return out


def train_model(exs, cfg, nsrc, ntgt, seed, log=None, init=None):
    model = init if init is not None else Model(cfg, nsrc, ntgt, seed)
    model.training = True
    opt = Adam(model.p, cfg["lr"], wd=cfg["weight_decay"])
    rng = np.random.default_rng(seed + 13)
    nb = len(batches_for(exs, cfg["batch"]))
    total = nb * cfg["epochs"]
    step = 0
    for ep in range(cfg["epochs"]):
        acc = [0.0, 0.0, 0.0]
        for bidx in batches_for(exs, cfg["batch"], rng):
            bt = make_batch([exs[i] for i in bidx], with_target=not cfg.get("witness_only"))
            loss, lq, lp, lu = model.loss(bt)
            loss.backward()
            warm = cfg["warmup"] * total
            if step < warm:
                sc = (step + 1) / max(1.0, warm)
            else:
                pr = (step - warm) / max(1.0, total - warm)
                sc = cfg["lr_floor"] + (1 - cfg["lr_floor"]) * 0.5 * (1 + math.cos(math.pi * pr))
            opt.step({k: t.g for k, t in model.t.items()}, sc, cfg["clip"])
            step += 1
            acc[0] += lq; acc[1] += lp; acc[2] += lu
        if log is not None:
            log(f"  epoch {ep + 1:3d}/{cfg['epochs']}  q={acc[0] / nb:.4f} "
                f"pair={acc[1] / nb:.4f} un={acc[2] / nb:.4f}")
    model.training = False
    return model


# ---------------------------------------------------------------- inference

def choose_subset(pair_lp, un_lp, R, prs, w_un, w_out):
    pm = {}
    for k, (i, j) in enumerate(prs):
        pm[(i, j)] = float(pair_lp[k])
    best, best_s = None, None
    for sub in itertools.combinations(range(R), R - 2):
        ss = set(sub)
        ins = [pm[(i, j)] for (i, j) in prs if i in ss and j in ss]
        out = [pm[(i, j)] for (i, j) in prs if not (i in ss and j in ss)]
        s = (sum(ins) / len(ins)) if ins else 0.0
        if w_out:
            s -= w_out * (sum(out) / len(out) if out else 0.0)
        if w_un:
            s += w_un * sum(float(un_lp[i]) for i in sub) / len(sub)
        if best_s is None or s > best_s:
            best_s, best = s, sub
    return tuple(best)


def select_for(model, exs, cfg, force_wmask=None):
    """Witness subset per row, from the trained pairwise head."""
    model.training = False
    bt = make_batch(exs, with_target=False)
    model.start()
    tok, ctx = model.encode(bt)
    pair, un = model.witness_scores(ctx, bt)
    B, R, prs = bt["B"], bt["R"], bt["pairs"]
    pl = pair.d.reshape(B, len(prs))
    ul = un.d.reshape(B, R)
    out = []
    for b in range(B):
        if force_wmask is not None:
            out.append(tuple(i for i in range(R) if force_wmask[b][i] > 0.5))
        else:
            out.append(choose_subset(pl[b], ul[b], R, prs, cfg["sel_w_un"], cfg["sel_w_out"]))
    return out


def generate_for(model, exs, cfg, tvoc, chosen):
    """Question candidates per row, conditioned on the given witness subsets."""
    model.training = False
    bt = make_batch(exs, with_target=False)
    model.start()
    tok, ctx = model.encode(bt)
    B, R = bt["B"], bt["R"]
    flag = np.zeros((B, R, bt["Lr"], 1), dtype=np.float32)
    for b in range(B):
        for i in chosen[b]:
            flag[b, i, :, 0] = 1.0
    mem, km = model.memory(tok, ctx, bt, flag.reshape(B, R * bt["Lr"], 1), flag[:, :, 0, :])
    return beam_candidates(model, mem.d, km, cfg, tvoc)


def predict_two(wmodel, qmodel, exs, wcfg, qcfg, tvoc, force_wmask=None):
    chosen = select_for(wmodel, exs, wcfg, force_wmask)
    cands = generate_for(qmodel, exs, qcfg, tvoc, chosen)
    return chosen, [mbr_select(c, qcfg) for c in cands], cands


def predict_batch(model, exs, cfg, tvoc, force_wmask=None):
    model.training = False
    bt = make_batch(exs, with_target=False)
    model.start()
    tok, ctx = model.encode(bt)
    pair, un = model.witness_scores(ctx, bt)
    B, R, prs = bt["B"], bt["R"], bt["pairs"]
    P = len(prs)
    pl = pair.d.reshape(B, P)
    ul = un.d.reshape(B, R)
    chosen = []
    for b in range(B):
        if force_wmask is not None:
            chosen.append(tuple(i for i in range(R) if force_wmask[b][i] > 0.5))
        else:
            chosen.append(choose_subset(pl[b], ul[b], R, prs, cfg["sel_w_un"], cfg["sel_w_out"]))
    flag = np.zeros((B, R, bt["Lr"], 1), dtype=np.float32)
    for b in range(B):
        for i in chosen[b]:
            flag[b, i, :, 0] = 1.0
    mem, km = model.memory(tok, ctx, bt, flag.reshape(B, R * bt["Lr"], 1), flag[:, :, 0, :])
    cands = beam_candidates(model, mem.d, km, cfg, tvoc)
    questions = [mbr_select(c, cfg) for c in cands]
    return chosen, questions, cands


def beam_candidates(model, mem_np, kmask, cfg, tvoc):
    """Deterministic diverse beam search. Each row is decoded independently."""
    G = max(1, cfg["dbs_groups"])
    per = max(1, cfg["beam"] // G)
    beam = per * G
    maxlen, minlen = cfg["max_tgt"], cfg["min_tgt"]
    lam = cfg["dbs_lambda"]
    nrn = cfg["no_repeat"]
    ntgt = model.p["Etgt"].shape[0]
    block = np.zeros(ntgt, dtype=np.float64)
    for t in (PAD, BOS, UNK, CLS, SEP):
        block[t] = -1e9
    allc = []
    for b in range(mem_np.shape[0]):
        m1 = np.repeat(mem_np[b:b + 1], beam, axis=0)
        km = np.repeat(kmask[b:b + 1], beam, axis=0)
        seqs = [[BOS] for _ in range(beam)]
        scores = np.zeros(beam)
        done, seen_seq = [], set()
        for step in range(maxlen):
            nb = len(seqs)
            model.start()
            logits, Lt = model.decode(T(m1[:nb]), km[:nb], np.array(seqs, dtype=np.int32))
            lg = logits.d.reshape(nb, Lt, ntgt)[:, -1, :].astype(np.float64)
            lg = lg / max(1e-6, cfg["gen_temp"])
            lg -= lg.max(axis=-1, keepdims=True)
            lp = lg - np.log(np.exp(lg).sum(axis=-1, keepdims=True)) + block
            if step + 1 < minlen:
                lp[:, EOS] = -1e9
            for i in range(nb):
                if len(seqs[i]) > 1:
                    lp[i, seqs[i][-1]] = -1e9          # never repeat a token twice in a row
            if nrn > 0:
                for i in range(nb):
                    sq = seqs[i]
                    if len(sq) >= nrn:
                        pre = tuple(sq[-(nrn - 1):])
                        for k in range(len(sq) - nrn + 1):
                            if tuple(sq[k:k + nrn - 1]) == pre:
                                lp[i, sq[k + nrn - 1]] = -1e9
            used = np.zeros(ntgt, dtype=np.float64)
            nseq, nsc = [], []
            for g in range(G):
                rows = [i for i in range(nb) if i // per == g]
                if not rows:
                    continue
                cand = []
                for i in rows:
                    pen = lp[i] - lam * used
                    top = np.argpartition(-pen, min(per + 2, ntgt - 1))[:per + 2]
                    for tk in top:
                        cand.append((float(pen[tk]), float(scores[i] + lp[i, tk]), i, int(tk)))
                cand.sort(key=lambda x: (-x[0], x[2], x[3]))
                taken = 0
                for _, sc, i, tk in cand:
                    if taken >= per:
                        break
                    nxt = seqs[i] + [tk]
                    key = tuple(nxt)
                    if tk == EOS:
                        if len(seqs[i]) > 1 and tuple(seqs[i]) not in seen_seq:
                            seen_seq.add(tuple(seqs[i]))
                            done.append((sc, list(seqs[i][1:])))
                        continue
                    if key in seen_seq:
                        continue
                    seen_seq.add(key)
                    nseq.append(nxt)
                    nsc.append(sc)
                    used[tk] += 1.0
                    taken += 1
            if not nseq:
                break
            seqs, scores = nseq, np.array(nsc)
        for sc, sq in zip(scores, seqs):
            if len(sq) > 1:
                done.append((float(sc), list(sq[1:])))
        done.sort(key=lambda x: -x[0] / max(1, len(x[1])) ** cfg["len_norm"])
        pool, seen = [], set()

        def offer(sc, toks):
            txt = detok([tvoc.itos[i] for i in toks])
            if not txt or txt in seen:
                return False
            seen.add(txt)
            pool.append((sc, len(toks), txt))
            return True

        # keep the best hypothesis at each output length first, so the pool spans
        # short and long phrasings, then fill the rest by score
        best_at_len = {}
        for sc, toks in done:
            n = len(toks)
            if n not in best_at_len:
                best_at_len[n] = (sc, toks)
        for n in sorted(best_at_len):
            if len(pool) >= cfg["n_cand"]:
                break
            offer(*best_at_len[n])
        for sc, toks in done:
            if len(pool) >= cfg["n_cand"]:
                break
            offer(sc, toks)
        if not pool:
            pool = [(0.0, 4, "what datasets are used")]
        allc.append(pool)
    return allc


def _ngrams(s):
    return [Counter(s[i:i + n] for i in range(max(0, len(s) - n + 1))) for n in range(1, 7)]


def _chrf_from(hc, hn, rc, rn):
    vals = 0.0
    for k in range(6):
        ov = sum((hc[k] & rc[k]).values())
        if ov == 0:
            continue
        P = ov / hn[k] if hn[k] else 0.0
        R = ov / rn[k] if rn[k] else 0.0
        d = 4.0 * P + R
        if d:
            vals += 5.0 * P * R / d
    return vals / 6.0


def mbr_select(pool, cfg):
    if len(pool) == 1:
        return pool[0][2]
    texts = [p[2] for p in pool]
    norm = [_nrm(t) for t in texts]
    cs = [_ngrams(s) for s in norm]
    ns = [[sum(c.values()) for c in cc] for cc in cs]
    lp = np.array([p[0] / max(1, p[1]) ** cfg["len_norm"] for p in pool], dtype=np.float64)
    w = np.exp((lp - lp.max()) / max(1e-6, cfg["mbr_tau"]))
    w /= w.sum()
    best, bs = 0, None
    for i in range(len(pool)):
        u = 0.0
        for j in range(len(pool)):
            if w[j] <= 1e-6:
                continue
            u += w[j] * _chrf_from(cs[i], ns[i], cs[j], ns[j])
        if bs is None or u > bs:
            bs, best = u, i
    return texts[best]


# ---------------------------------------------------------------- scoring

BASELINE_Q = "what question do these records answer?"


def _nrm(t):
    return " ".join(unicodedata.normalize("NFKC", str(t)).casefold().split())


def chrf(h, r):
    h, r = _nrm(h), _nrm(r)
    vals = []
    for n in range(1, 7):
        hc = Counter(h[i:i + n] for i in range(max(0, len(h) - n + 1)))
        rc = Counter(r[i:i + n] for i in range(max(0, len(r) - n + 1)))
        ov = sum((hc & rc).values())
        P = ov / sum(hc.values()) if hc else 0.0
        R = ov / sum(rc.values()) if rc else 0.0
        d = 4.0 * P + R
        vals.append(5.0 * P * R / d if d else 0.0)
    return sum(vals) / 6.0


def qscore(h, r):
    c = chrf(h, r); b = chrf(BASELINE_Q, r)
    if b == 1.0:
        return float(c == 1.0)
    return min(1.0, max(0.0, (c - b) / (1.0 - b)))


def witness_forward(model, exs):
    """Pair/unary logits per example, batched by record count."""
    model.training = False
    out = {}
    by_r = {}
    for i, e in enumerate(exs):
        by_r.setdefault(e["R"], []).append(i)
    for R, idxs in by_r.items():
        for s in range(0, len(idxs), 16):
            sel = idxs[s:s + 16]
            bt = make_batch([exs[i] for i in sel], with_target=False)
            model.start()
            tok, ctx = model.encode(bt)
            pair, un = model.witness_scores(ctx, bt)
            P = len(bt["pairs"])
            pl = pair.d.reshape(len(sel), P)
            ul = un.d.reshape(len(sel), R)
            for k, i in enumerate(sel):
                out[i] = (pl[k], ul[k], R, bt["pairs"])
    return out


def generate_all(model, exs, cfg, tvoc, wmasks, bs=16):
    """Candidate pools for each example given fixed witness masks."""
    pools = {}
    by_r = {}
    for i, e in enumerate(exs):
        by_r.setdefault(e["R"], []).append(i)
    for R, idxs in by_r.items():
        for s in range(0, len(idxs), bs):
            sel = idxs[s:s + bs]
            _, _, cands = predict_batch(model, [exs[i] for i in sel], cfg, tvoc,
                                        force_wmask=[wmasks[i] for i in sel])
            for k, i in enumerate(sel):
                pools[i] = cands[k]
    return pools


def pretrain_encoder(model, exs, cfg, seed, log=None):
    """Masked-language-model pretraining of the record encoder on supplied training text."""
    seqs = [r for e in exs for r in e["recs"]]
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    rng = np.random.default_rng(seed)
    nsrc = model.p["Esrc"].shape[0]
    keys = ["Esrc", "Epos", "mlm_b", "enc_ln_g", "enc_ln_b"]
    keys += [k for k in model.p if k.startswith("e") and k[1].isdigit()]
    sub = {k: model.p[k] for k in keys}
    opt = Adam(sub, cfg["mlm_lr"], wd=cfg["weight_decay"])
    bs = cfg["mlm_batch"]
    groups = [order[i:i + bs] for i in range(0, len(order), bs)]
    total = len(groups) * cfg["mlm_epochs"]
    step = 0
    model.training = True
    for ep in range(cfg["mlm_epochs"]):
        tot, nb = 0.0, 0
        for gi in rng.permutation(len(groups)):
            idxs = groups[gi]
            Lr = max(len(seqs[i]) for i in idxs)
            ids = np.zeros((len(idxs), Lr), dtype=np.int32)
            msk = np.zeros((len(idxs), Lr), dtype=np.float32)
            for r, i in enumerate(idxs):
                s = seqs[i]
                ids[r, :len(s)] = s
                msk[r, :len(s)] = 1.0
            cand = np.argwhere((ids > MASK) & (msk > 0))
            if len(cand) < 2:
                continue
            k = max(1, int(round(cfg["mlm_rate"] * len(cand))))
            pick = cand[rng.choice(len(cand), size=k, replace=False)]
            flat = (pick[:, 0] * Lr + pick[:, 1]).astype(np.int64)
            targets = ids[pick[:, 0], pick[:, 1]].astype(np.int64)
            roll = rng.random(k)
            inp = ids.copy()
            for t in range(k):
                r, c = pick[t]
                if roll[t] < 0.8:
                    inp[r, c] = MASK
                elif roll[t] < 0.9:
                    inp[r, c] = int(rng.integers(MASK + 1, nsrc))
            loss = model.mlm_loss(inp, msk, flat, targets)
            loss.backward()
            warm = 0.06 * total
            sc = ((step + 1) / max(1.0, warm) if step < warm
                  else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1.0, total - warm))))
            opt.step({k2: model.t[k2].g for k2 in sub}, sc, cfg["clip"])
            step += 1
            tot += float(loss.d); nb += 1
        if log is not None and nb:
            log(f"  mlm epoch {ep + 1:2d}/{cfg['mlm_epochs']}  loss={tot / nb:.4f}")
    return model
