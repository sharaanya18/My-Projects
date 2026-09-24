import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import math
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from transformers import AutoModel, AutoTokenizer

SEED = 42
BACKBONES = [
    {"name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
     "revision": "a2a36cb6d490fd8362f47e6c29a66e1345151b65", "max_epochs": 8, "lr": 3e-5},
    {"name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
     "revision": "d66eff4d8a8598f264f166af8db67f7797164651", "max_epochs": 10, "lr": 5e-5},
]
USE_BF16 = False
D = 256
COG_DIM = 64
MAX_LEN = 64
BATCH = 64
EVAL_BATCH = 256
HEAD_LR = 1e-3
LAYER_DECAY = 0.9
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
WORD_DROPOUT = 0.10
LABEL_SMOOTH = 0.05
TAG_L2 = 1e-4
N_POOL = 80
N_GOLD = 3
DUTCH_WORDS = {"het", "een", "gezicht", "portret", "op", "met", "uit", "bij",
               "naar", "voor", "aan", "zijn", "door", "tussen", "boven"}

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def average_precision(ranked, gold):
    hits, precs = 0, []
    for i, c in enumerate(ranked, 1):
        if c in gold:
            hits += 1
            precs.append(hits / i)
    return sum(precs) / len(gold) if precs else 0.0


def rank_order(scores, cand):
    return np.lexsort((np.asarray(cand), -np.asarray(scores)))


def map_from_logits(logits, frame):
    aps = []
    for i, (cand, gold) in enumerate(zip(frame["candidate_tags"], frame["assigned_tags"])):
        cand = cand.split()
        order = rank_order(logits[i], cand)
        aps.append(average_precision([cand[j] for j in order], set(gold.split())))
    return float(np.mean(aps))


def pool_ids(frame, tag2id):
    return torch.tensor([[tag2id[t] for t in s.split()] for s in frame["candidate_tags"]],
                        dtype=torch.long)


def gold_mask(frame):
    m = np.zeros((len(frame), N_POOL), dtype=np.float32)
    for i, (cand, gold) in enumerate(zip(frame["candidate_tags"], frame["assigned_tags"])):
        g = set(gold.split())
        for j, c in enumerate(cand.split()):
            if c in g:
                m[i, j] = 1.0
    return torch.tensor(m)


def dutch_holdout_mask(titles):
    return np.array([bool(DUTCH_WORDS & set(str(t).lower().split())) for t in titles])


def word_dropout(titles, rng):
    out = []
    for t in titles:
        w = str(t).split()
        if len(w) > 1:
            keep = [x for x in w if rng.random() >= WORD_DROPOUT]
            w = keep if keep else [w[int(rng.integers(len(w)))]]
        out.append(" ".join(w))
    return out


class Ranker(nn.Module):
    def __init__(self, backbone, hidden, n_tags):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(nn.Linear(hidden, D), nn.LayerNorm(D), nn.Dropout(0.1))
        self.tag_emb = nn.Embedding(n_tags, D)
        self.register_buffer("tag_init", torch.zeros(n_tags, D))
        self.register_buffer("cog", torch.zeros(n_tags, COG_DIM))
        self.cog_proj = nn.Linear(COG_DIM, D)
        nn.init.normal_(self.cog_proj.weight, std=0.01)
        nn.init.zeros_(self.cog_proj.bias)
        self.type_emb = nn.Embedding(2, D)
        layer = nn.TransformerEncoderLayer(D, 4, 4 * D, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.ctx = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.ctx_head = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, 1))
        nn.init.zeros_(self.ctx_head[-1].weight)
        nn.init.zeros_(self.ctx_head[-1].bias)
        self.log_scale = nn.Parameter(torch.tensor(math.log(20.0)))

    def pooled(self, ids, mask):
        h = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).to(h.dtype)
        return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)

    def forward(self, ids, mask, pool):
        t = self.proj(self.pooled(ids, mask))
        v = self.tag_emb(pool) + self.cog_proj(self.cog[pool])
        cos = (F.normalize(t, dim=-1).unsqueeze(1) * F.normalize(v, dim=-1)).sum(-1)
        seq = torch.cat([(t + self.type_emb.weight[0]).unsqueeze(1),
                         v + self.type_emb.weight[1]], dim=1)
        ctx = self.ctx(seq)[:, 1:]
        return cos * self.log_scale.exp() + self.ctx_head(ctx).squeeze(-1)


def cogold_embeddings(fit_frame, tag2id):
    T = len(tag2id)
    gc = np.zeros(T)
    rows, cols = [], []
    for g in fit_frame["assigned_tags"]:
        ids = [tag2id[t] for t in g.split()]
        for a in ids:
            gc[a] += 1
            for b in ids:
                if a != b:
                    rows.append(a)
                    cols.append(b)
    C = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(T, T)).tocsr().tocoo()
    n = len(fit_frame)
    vals = np.maximum(np.log(C.data * n / (gc[C.row] * gc[C.col] + 1e-9)), 0.0)
    P = coo_matrix((vals, (C.row, C.col)), shape=(T, T)).tocsr()
    v0 = np.full(T, 1.0 / math.sqrt(T))
    U, S, _ = svds(P, k=COG_DIM, v0=v0)
    order = np.argsort(-S)
    E = U[:, order] * np.sqrt(S[order])
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-9
    return torch.tensor(E, dtype=torch.float32)


def build_optimizer(model, top_lr):
    n_layers = model.backbone.config.num_hidden_layers
    groups = {}
    for name, p in model.backbone.named_parameters():
        if name.startswith("embeddings."):
            p.requires_grad = False
            continue
        if ".layer." in name or name.startswith("layer."):
            idx = int(name.split("layer.")[1].split(".")[0])
            lr = top_lr * (LAYER_DECAY ** (n_layers - 1 - idx))
        else:
            lr = top_lr
        groups.setdefault(lr, []).append(p)
    param_groups = [{"params": ps, "lr": lr} for lr, ps in sorted(groups.items())]
    head = [p for n, p in model.named_parameters() if not n.startswith("backbone.")]
    param_groups.append({"params": head, "lr": HEAD_LR})
    return torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)


def train_run(cfg, fit_frame, tag2id, device, label, sched_epochs, train_epochs,
              eval_frame=None, score_frame=None):
    seed_everything(SEED)
    use_amp = device.type == "cuda" and USE_BF16
    tok = AutoTokenizer.from_pretrained(cfg["name"], revision=cfg.get("revision"))
    backbone = AutoModel.from_pretrained(cfg["name"], revision=cfg.get("revision"),
                                         attn_implementation="eager")
    model = Ranker(backbone, backbone.config.hidden_size, len(tag2id)).to(device)

    def enc(titles):
        e = tok([str(t) for t in titles], padding=True, truncation=True,
                max_length=MAX_LEN, return_tensors="pt")
        return e["input_ids"].to(device), e["attention_mask"].to(device)

    def score(frame):
        model.eval()
        titles = frame["record_title"].tolist()
        pools = pool_ids(frame, tag2id)
        out = []
        with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            for s in range(0, len(titles), EVAL_BATCH):
                ids, m = enc(titles[s:s + EVAL_BATCH])
                out.append(model(ids, m, pools[s:s + EVAL_BATCH].to(device)).float().cpu())
        return torch.cat(out).numpy()

    model.eval()
    fit_titles = fit_frame["record_title"].tolist()
    pooled = []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
        for s in range(0, len(fit_titles), EVAL_BATCH):
            ids, m = enc(fit_titles[s:s + EVAL_BATCH])
            pooled.append(model.pooled(ids, m).float().cpu())
    pooled = torch.cat(pooled)
    proto_sum = torch.zeros(len(tag2id), pooled.shape[1])
    proto_cnt = torch.zeros(len(tag2id))
    for i, g in enumerate(fit_frame["assigned_tags"]):
        for t in g.split():
            proto_sum[tag2id[t]] += pooled[i]
            proto_cnt[tag2id[t]] += 1
    has = proto_cnt > 0
    proto = torch.empty_like(proto_sum)
    proto[has] = proto_sum[has] / proto_cnt[has].unsqueeze(1)
    proto[~has] = proto[has].mean(0)
    with torch.no_grad():
        init = model.proj(proto.to(device)).float()
        model.tag_emb.weight.copy_(init)
        model.tag_init.copy_(init)
        model.cog.copy_(cogold_embeddings(fit_frame, tag2id).to(device))
    log(f"[{label}] tag init: {int(has.sum())}/{len(tag2id)} tags have a gold prototype")

    optimizer = build_optimizer(model, cfg["lr"])
    fit_pool = pool_ids(fit_frame, tag2id)
    fit_gold = gold_mask(fit_frame)
    n = len(fit_frame)
    steps_per_epoch = n // BATCH
    total = steps_per_epoch * sched_epochs
    warm = max(1, int(WARMUP_FRAC * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warm if s < warm
        else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm)))))
    gen = torch.Generator().manual_seed(SEED)
    np_rng = np.random.default_rng(SEED)
    epoch_maps = []

    for epoch in range(1, train_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for s in range(steps_per_epoch):
            idx = perm[s * BATCH:(s + 1) * BATCH]
            titles = word_dropout(fit_frame["record_title"].iloc[idx.numpy()].tolist(), np_rng)
            ids, m = enc(titles)
            shuf = torch.argsort(torch.rand(len(idx), N_POOL, generator=gen), dim=1)
            pool = fit_pool[idx].gather(1, shuf).to(device)
            gold = fit_gold[idx].gather(1, shuf).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits = model(ids, m, pool).float()
            target = (1 - LABEL_SMOOTH) * gold / N_GOLD + LABEL_SMOOTH / N_POOL
            loss = -(target * F.log_softmax(logits, dim=1)).sum(1).mean()
            drift = (model.tag_emb(pool) - model.tag_init[pool]).pow(2).sum(-1).mean()
            loss = loss + TAG_L2 * drift
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sched.step()
            tot += loss.item()
        msg = f"[{label}] epoch {epoch}/{train_epochs} loss={tot / max(1, steps_per_epoch):.4f}"
        if eval_frame is not None:
            epoch_maps.append(map_from_logits(score(eval_frame), eval_frame))
            msg += f"  holdout MAP={epoch_maps[-1]:.4f}"
        log(msg)

    logits = None
    if score_frame is not None:
        logits = score(score_frame)
        k = min(64, len(score_frame))
        with torch.no_grad():
            ids, m = enc(score_frame["record_title"].tolist()[:k])
            p = pool_ids(score_frame.iloc[:k], tag2id).to(device)
            a = model(ids, m, p).float()
            b = model(ids, m, p.flip(1)).float().flip(1)
        diff = (a - b).abs().max().item()
        if diff > 1e-3:
            raise RuntimeError(f"permutation test failed (max diff {diff:.2e})")
        log(f"[{label}] permutation test passed (max diff {diff:.1e})")

    del model, backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return epoch_maps, logits


def zscore_rows(x):
    return (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)


def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 solution.py <public_dir> <submission_out>")
    public_dir, submission_out = Path(sys.argv[1]), Path(sys.argv[2])
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    train = pd.read_csv(public_dir / "train.csv")
    labels = pd.read_csv(public_dir / "train_labels.csv")
    test = pd.read_csv(public_dir / "test.csv")
    df = train.merge(labels, on="case_id", how="inner")
    assert len(df) == len(train), "every training case needs its labels"
    df = df.sort_values(["record_title", "candidate_tags", "assigned_tags"],
                        kind="mergesort").reset_index(drop=True)
    tags = sorted({t for s in pd.concat([train["candidate_tags"], test["candidate_tags"]])
                   for t in s.split()})
    tag2id = {t: i for i, t in enumerate(tags)}
    log(f"train {len(df)}, test {len(test)}, tag vocabulary {len(tag2id)}")

    hold = dutch_holdout_mask(df["record_title"])
    tr_fold = df[~hold].reset_index(drop=True)
    va_fold = df[hold].reset_index(drop=True)
    assert len(va_fold) > 0, "holdout is empty"
    log(f"Dutch holdout: {len(va_fold)} rows (target MAP >= 0.37 here ~ 0.40 on unseen institutions)")

    test_logits = []
    for cfg in BACKBONES:
        short = cfg["name"].split("/")[-1]
        maps, _ = train_run(cfg, tr_fold, tag2id, device, f"select {short}",
                            sched_epochs=cfg["max_epochs"], train_epochs=cfg["max_epochs"],
                            eval_frame=va_fold)
        best_epoch = int(np.argmax(maps)) + 1
        log(f"[select {short}] best holdout MAP {max(maps):.4f} at epoch {best_epoch}")
        _, lg = train_run(cfg, df, tag2id, device, f"final {short}",
                          sched_epochs=cfg["max_epochs"], train_epochs=best_epoch,
                          score_frame=test)
        test_logits.append(zscore_rows(lg))
    ens = np.mean(test_logits, 0)

    cand = [s.split() for s in test["candidate_tags"]]
    ranked = [" ".join(cand[i][j] for j in rank_order(ens[i], cand[i]))
              for i in range(len(test))]
    sub = pd.DataFrame({"case_id": test["case_id"], "ranked_tags": ranked})

    assert len(sub) == len(test) and sub["case_id"].is_unique
    for r, c in zip(sub["ranked_tags"], test["candidate_tags"]):
        assert sorted(r.split()) == sorted(c.split()), "ranking must be a permutation of the pool"

    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)
    log(f"wrote {submission_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
