"""Cross-encoder tagger: [CLS] note [SEP] source-window [SEP] -> per-subword P(in gold span)."""
import math, os, random, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

MODEL = os.environ.get("NN_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
MAXLEN = 384
MAXNOTE = 96


class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL)
        self.head = nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, ids, att):
        h = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        return self.head(h).squeeze(-1)


def get_tok():
    return AutoTokenizer.from_pretrained(MODEL)


def encode_row(tok, r):
    src = r["source_text"]
    e = tok(src, add_special_tokens=False, return_offsets_mapping=True)
    note = tok(r["note_text"], add_special_tokens=False)["input_ids"][:MAXNOTE]
    ids = e["input_ids"]
    offs = e["offset_mapping"]
    y = None
    if r.get("spans"):
        y = np.zeros(len(ids), dtype=np.float32)
        for i, (s, t) in enumerate(offs):
            for a, b in r["spans"]:
                if s < b and t > a and src[max(s, a):min(t, b)].strip():
                    y[i] = 1
    return dict(ids=ids, offs=offs, note=note, y=y)


def make_window(tok, enc, start, W):
    ids = [tok.cls_token_id] + enc["note"] + [tok.sep_token_id] + enc["ids"][start:start + W] + [tok.sep_token_id]
    return ids, 2 + len(enc["note"])


def train(rows, budget_sec, seed=0, bs=8, lr=6e-5, log=print):
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_num_threads(min(10, os.cpu_count() or 4))
    tok = get_tok()
    model = Tagger()
    for p in model.enc.embeddings.word_embeddings.parameters():
        p.requires_grad = False
    encs = [encode_row(tok, r) for r in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head")], "lr": lr},
        {"params": model.head.parameters(), "lr": lr * 10}], weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    t0 = time.time()
    step = 0
    order = []
    model.train()
    ema = None
    while True:
        elapsed = time.time() - t0
        if elapsed > budget_sec:
            break
        if not order:
            order = list(range(len(rows)))
            random.shuffle(order)
        batch_idx = [order.pop() for _ in range(min(bs, len(order)))]
        seqs, labs = [], []
        for qi in batch_idx:
            enc = encs[qi]
            n = len(enc["ids"])
            W = MAXLEN - len(enc["note"]) - 3
            pos = np.nonzero(enc["y"])[0]
            if n <= W:
                starts = [0]
            else:
                if random.random() < 0.75 and len(pos):
                    c = int(random.choice(pos))
                    st = c - random.randint(8, W - 8)
                else:
                    st = random.randint(0, n - W)
                starts = [min(max(0, st), n - W)]
            for st in starts:
                ids, off = make_window(tok, enc, st, W)
                y = np.zeros(len(ids), dtype=np.float32)
                m = np.zeros(len(ids), dtype=np.float32)
                seg = enc["y"][st:st + W]
                y[off:off + len(seg)] = seg
                m[off:off + len(seg)] = 1
                seqs.append(ids)
                labs.append((y, m))
        ml = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), ml), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), ml), dtype=torch.long)
        Y = torch.zeros((len(seqs), ml))
        M = torch.zeros((len(seqs), ml))
        for i, (s, (y, m)) in enumerate(zip(seqs, labs)):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
            Y[i, :len(s)] = torch.tensor(y)
            M[i, :len(s)] = torch.tensor(m)
        frac = elapsed / budget_sec
        mult = min(1.0, (step + 1) / 30) * max(0.02, 1 - frac)
        for g, b in zip(opt.param_groups, base):
            g["lr"] = b * mult
        logits = model(ids, att)
        l = lossf(logits, Y)
        w = torch.where(Y > 0, torch.tensor(3.0), torch.tensor(1.0)) * M
        loss = (l * w).sum() / M.sum().clamp(min=1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
        if step % 25 == 0:
            log(f"nn step {step} loss {ema:.4f} elapsed {elapsed:.0f}s seqs {step*bs}")
    model.eval()
    return model, tok


@torch.no_grad()
def predict(model, tok, rows, bs=16):
    """Return per-row float32 char-level probability arrays."""
    torch.set_num_threads(min(10, os.cpu_count() or 4))
    model.eval()
    jobs = []
    encs = []
    for qi, r in enumerate(rows):
        enc = encode_row(tok, r)
        encs.append(enc)
        n = len(enc["ids"])
        W = MAXLEN - len(enc["note"]) - 3
        stride = max(1, W // 2)
        st = 0
        while True:
            jobs.append((qi, st, W))
            if st + W >= n:
                break
            st += stride
    acc = [np.zeros(len(e["ids"]), dtype=np.float64) for e in encs]
    cnt = [np.zeros(len(e["ids"]), dtype=np.float64) for e in encs]
    jobs.sort(key=lambda j: len(encs[j[0]]["note"]))
    for b in range(0, len(jobs), bs):
        chunk = jobs[b:b + bs]
        seqs, offs = [], []
        for qi, st, W in chunk:
            ids, off = make_window(tok, encs[qi], st, W)
            seqs.append(ids)
            offs.append(off)
        ml = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), ml), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), ml), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            att[i, :len(s)] = 1
        pr = torch.sigmoid(model(ids, att)).numpy()
        for i, (qi, st, W) in enumerate(chunk):
            k = min(W, len(encs[qi]["ids"]) - st)
            acc[qi][st:st + k] += pr[i, offs[i]:offs[i] + k]
            cnt[qi][st:st + k] += 1
    out = []
    for r, enc, a, c in zip(rows, encs, acc, cnt):
        p = a / np.maximum(c, 1)
        ch = np.zeros(len(r["source_text"]), dtype=np.float32)
        for (s, t), v in zip(enc["offs"], p):
            ch[s:t] = np.maximum(ch[s:t], v)
        out.append(ch)
    return out
