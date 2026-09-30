"""Semantic note<->source-window similarity features from a pretrained multilingual sentence encoder."""
import re, time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WORD = re.compile(r"\w+", re.UNICODE)
SENT = re.compile(r"(?<=[.!?;:])\s+")


class Embedder:
    def __init__(self, name=NAME):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name).eval()
        self.cache = {}

    @torch.no_grad()
    def embed(self, texts, bs=256, log=None):
        need = sorted({t for t in texts if t not in self.cache}, key=len)
        t0 = time.time()
        for b in range(0, len(need), bs):
            chunk = need[b:b + bs]
            x = self.tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
            h = self.m(**x).last_hidden_state
            msk = x["attention_mask"].unsqueeze(-1).float()
            e = (h * msk).sum(1) / msk.sum(1).clamp(min=1)
            e = torch.nn.functional.normalize(e, dim=-1).numpy()
            for t, v in zip(chunk, e):
                self.cache[t] = v
            if log and (b // bs) % 200 == 0:
                log(f"  embedded {b}/{len(need)} {time.time()-t0:.0f}s")
        return np.stack([self.cache[t] for t in texts])


def src_windows(words, w):
    n = len(words)
    out = []
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, lo + w)
        lo = max(0, hi - w)
        out.append(" ".join(words[lo:hi]))
    return out


def note_units(note):
    ws = [m.group() for m in WORD.finditer(note)]
    sents = [s for s in SENT.split(note.strip()) if s]
    return dict(whole=[note], sent=sents[:12] or [note], word=[w for w in ws if len(w) >= 3][:60] or ws[:60] or [note],
                tri=[" ".join(ws[i:i + 3]) for i in range(0, max(1, len(ws) - 2))][:80] or [note])


def emb_features(emb, rows, log=print):
    """Returns per-row dict name -> array over the row's \\w tokens."""
    # collect all texts
    srcs = {}
    for r in rows:
        s = r["source_text"]
        if s not in srcs:
            words = [m.group() for m in WORD.finditer(s)]
            srcs[s] = {w: src_windows(words, w) for w in (1, 3, 5)}
    alltxt = set()
    for d in srcs.values():
        for v in d.values():
            alltxt.update(v)
    notes = {}
    for r in rows:
        u = note_units(r["note_text"])
        notes[r["note_text"]] = u
        for v in u.values():
            alltxt.update(v)
    log(f"embedding {len(alltxt)} unique texts")
    emb.embed(sorted(alltxt), log=log)
    out = []
    for r in rows:
        d = srcs[r["source_text"]]
        u = notes[r["note_text"]]
        U = {k: emb.embed(v) for k, v in u.items()}
        feats = {}
        for w in (1, 3, 5):
            E = emb.embed(d[w])
            for k, M in U.items():
                if w == 1 and k == "sent":
                    pass
                sim = E @ M.T
                mx = sim.max(1)
                feats[f"e{w}_{k}_max"] = mx
                feats[f"e{w}_{k}_z"] = (mx - mx.mean()) / (mx.std() + 1e-6)
                if k in ("word", "tri"):
                    feats[f"e{w}_{k}_mean"] = sim.mean(1)
        # neighbourhood smoothing of the strongest signals
        for key in ("e1_word_max", "e3_tri_max", "e3_whole_max", "e3_sent_max"):
            v = feats[key]
            for wd in (2, 5):
                k = np.convolve(v, np.ones(2 * wd + 1) / (2 * wd + 1), mode="same")
                feats[f"{key}_sm{wd}"] = k
        out.append(feats)
    return out
