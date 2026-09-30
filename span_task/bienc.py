"""Contrastive fine-tuning of a multilingual sentence encoder: note  <->  gold source span."""
import os, random, re, time
import numpy as np
import torch
import torch.nn.functional as Fn
from transformers import AutoModel, AutoTokenizer

NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WORD = re.compile(r"\w+", re.UNICODE)
WSIZES = (1, 3, 6)


class BiEnc:
    def __init__(self, name=NAME):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name)
        for p in self.m.embeddings.word_embeddings.parameters():
            p.requires_grad = False
        self.cache = {}

    def _enc(self, texts, maxlen):
        x = self.tok(texts, padding=True, truncation=True, max_length=maxlen, return_tensors="pt")
        h = self.m(**x).last_hidden_state
        msk = x["attention_mask"].unsqueeze(-1).float()
        return Fn.normalize((h * msk).sum(1) / msk.sum(1).clamp(min=1), dim=-1)

    @torch.no_grad()
    def embed(self, texts, maxlen=48, bs=256, log=None):
        self.m.eval()
        need = sorted({t for t in texts if t not in self.cache}, key=len)
        t0 = time.time()
        for b in range(0, len(need), bs):
            chunk = need[b:b + bs]
            e = self._enc(chunk, maxlen).numpy()
            for t, v in zip(chunk, e):
                self.cache[t] = v
            if log and (b // bs) % 300 == 0:
                log(f"  embedded {b}/{len(need)} {time.time()-t0:.0f}s")
        return np.stack([self.cache[t] for t in texts])

    def fit(self, rows, budget_sec, bs=32, nneg=6, lr=3e-5, tau=0.05, seed=0, log=print):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        self.cache = {}
        params = [p for p in self.m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        items = []
        for r in rows:
            src = r["source_text"]
            words = [(m.start(), m.end()) for m in WORD.finditer(src)]
            if not words or not r.get("spans"):
                continue
            a, b = max(r["spans"], key=lambda s: s[1] - s[0])
            items.append((r["note_text"], src[a:b], src, words, (a, b)))
        t0 = time.time()
        step = 0
        order = []
        self.m.train()
        ema = None
        while time.time() - t0 < budget_sec:
            if len(order) < bs:
                order = list(range(len(items)))
                rng.shuffle(order)
            batch = [items[order.pop()] for _ in range(bs)]
            notes = [it[0] for it in batch]
            pos = [it[1] for it in batch]
            neg = []
            for note, sp, src, words, (a, b) in batch:
                got = 0
                tries = 0
                while got < nneg and tries < 30:
                    tries += 1
                    w = rng.choice(WSIZES)
                    i = rng.randrange(len(words))
                    j = min(len(words) - 1, i + w - 1)
                    s, e = words[i][0], words[j][1]
                    if s < b and e > a:  # overlaps gold
                        continue
                    neg.append(src[s:e])
                    got += 1
                while got < nneg:
                    neg.append(sp[::-1][:20] or "x")
                    got += 1
            frac = (time.time() - t0) / budget_sec
            for g in opt.param_groups:
                g["lr"] = lr * min(1.0, (step + 1) / 20) * max(0.05, 1 - frac)
            q = self._enc(notes, 64)
            cand = self._enc(pos + neg, 48)
            logits = q @ cand.T / tau
            loss = Fn.cross_entropy(logits, torch.arange(len(batch)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad()
            step += 1
            ema = loss.item() if ema is None else 0.95 * ema + 0.05 * loss.item()
            if step % 20 == 0:
                log(f"  biencoder step {step} loss {ema:.3f} {time.time()-t0:.0f}s")
        self.m.eval()
        self.cache = {}
        return self


def windows(words, w):
    n = len(words)
    out = []
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, lo + w)
        lo = max(0, hi - w)
        out.append(" ".join(words[lo:hi]))
    return out


def score_rows(be, rows, log=print):
    """Per row: dict of arrays over \\w tokens: cosine(note, centered window of w words), w in WSIZES."""
    srcs = {}
    for r in rows:
        s = r["source_text"]
        if s not in srcs:
            ws = [m.group() for m in WORD.finditer(s)]
            srcs[s] = {w: windows(ws, w) for w in WSIZES}
    txt = set(r["note_text"] for r in rows)
    for d in srcs.values():
        for v in d.values():
            txt.update(v)
    log(f"scoring {len(txt)} unique texts")
    be.embed(sorted(txt), log=log)
    out = []
    for r in rows:
        q = be.cache[r["note_text"]]
        d = srcs[r["source_text"]]
        f = {}
        for w in WSIZES:
            E = np.stack([be.cache[t] for t in d[w]]) if d[w] else np.zeros((0, q.shape[0]), np.float32)
            f[f"b{w}"] = E @ q
        out.append(f)
    return out
