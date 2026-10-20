"""Restoring the order of speeches in a scene (CPU solution).

    python3 solution.py <public_dir> <submission_out>
"""
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

SEED = 42
DEVICE = torch.device("cpu")
NUM_THREADS = 4
FULL_HEAD, FULL_TAIL = 72, 54      # long speeches: keep start and end
EDGE_TOKENS = 40                   # separate "opening" / "closing" views
EMBED_BATCH = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(NUM_THREADS)


def log(*a):
    print(*a, flush=True)


@torch.inference_mode()
def encode_ids(model, id_lists, pad_id):
    """Mean-pooled, L2-normalised e5 vectors; batches built from length-sorted ids."""
    order = np.argsort([len(x) for x in id_lists], kind="stable")
    out = np.zeros((len(id_lists), model.config.hidden_size), dtype=np.float32)
    for s in range(0, len(order), EMBED_BATCH):
        idx = order[s:s + EMBED_BATCH]
        L = max(len(id_lists[k]) for k in idx)
        x = torch.full((len(idx), L), pad_id, dtype=torch.long)
        a = torch.zeros((len(idx), L), dtype=torch.long)
        for r, k in enumerate(idx):
            x[r, :len(id_lists[k])] = torch.tensor(id_lists[k])
            a[r, :len(id_lists[k])] = 1
        h = model(input_ids=x, attention_mask=a).last_hidden_state
        m = a.unsqueeze(-1).float()
        v = (h * m).sum(1) / m.sum(1)
        out[idx] = F.normalize(v, dim=-1).numpy()
    return out


def embed_speeches(ckpt_dir, speeches):
    """Three views per speech: full (head+tail), opening, closing -> [n, 3, d]."""
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    model = AutoModel.from_pretrained(ckpt_dir).eval()
    raw = tok(["query: " + s for s in speeches], add_special_tokens=False)["input_ids"]
    cls_id, sep_id = tok.cls_token_id, tok.sep_token_id
    full, edge_idx, heads, tails = [], [], [], []
    for i, t in enumerate(raw):
        f = t if len(t) <= FULL_HEAD + FULL_TAIL else t[:FULL_HEAD] + t[-FULL_TAIL:]
        full.append([cls_id] + f + [sep_id])
        if len(t) > EDGE_TOKENS:         # short speeches: all views are the full text
            edge_idx.append(i)
            heads.append([cls_id] + t[:EDGE_TOKENS] + [sep_id])
            # closing view keeps the "query: " prefix so every view is in-format
            tails.append([cls_id] + t[:3] + t[-(EDGE_TOKENS - 3):] + [sep_id])
    v_full = encode_ids(model, full, tok.pad_token_id)
    v_head, v_tail = v_full.copy(), v_full.copy()
    if edge_idx:
        v_head[edge_idx] = encode_ids(model, heads, tok.pad_token_id)
        v_tail[edge_idx] = encode_ids(model, tails, tok.pad_token_id)
    return np.stack([v_full, v_head, v_tail], 1)


# ── surface features (generic punctuation / length cues, per speech) ──────────
def surface_features(s):
    t = s.strip()
    end, start = t[-1:] if t else "", t[:1] if t else ""
    return [
        np.log1p(len(t)) / 8.0,
        np.log1p(t.count("\n")) / 3.0,
        float(end == "?"),
        float(end == "!"),
        float(end in "–-—" or t.endswith("...") or end == "…"),
        float(start in "–-—" or t.startswith("...") or start == "…"),
        float(start.islower()),
        float("?" in t),
    ]


N_SURF = 8


def scene_inputs(scenes, emb, offset):
    """List of [n, 3*d + N_SURF] float32 arrays, one per scene."""
    out = []
    for sc in scenes:
        n = len(sc)
        e = emb[offset:offset + n].reshape(n, -1)
        f = np.array([surface_features(s) for s in sc], dtype=np.float32)
        out.append(np.concatenate([e, f], 1).astype(np.float32))
        offset += n
    return out, offset


# ── model ─────────────────────────────────────────────────────────────────────
class SceneOrderer(nn.Module):
    def __init__(self, in_dim, d, layers, dropout, noise):
        super().__init__()
        self.noise = noise
        self.inp = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, d), nn.GELU(), nn.LayerNorm(d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.pair = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, 1))
        self.pos = nn.Linear(d, 1)
        self.nxt_q, self.nxt_k = nn.Linear(d, d), nn.Linear(d, d)

    def forward(self, x, pad):
        if self.training and self.noise > 0:
            x = x + self.noise * torch.randn_like(x)
        h = self.context(self.inp(x), src_key_padding_mask=pad)
        B, N, D = h.shape
        hi = h.unsqueeze(2).expand(B, N, N, D)
        hj = h.unsqueeze(1).expand(B, N, N, D)
        f = self.pair(torch.cat([hi, hj, hi * hj, hi - hj], -1)).squeeze(-1)
        P = f - f.transpose(1, 2)                       # antisymmetric precedence logit
        pos = torch.sigmoid(self.pos(h).squeeze(-1))
        A = self.nxt_q(h) @ self.nxt_k(h).transpose(1, 2) / D ** 0.5   # i -> next j
        return P, pos, A


def make_batch(feats, ranks, idx, rng=None, crop_p=0.0, min_len=8):
    xs, rs = [], []
    for k in idx:
        x, r = feats[k], ranks[k]
        n = len(r)
        if rng is not None and n > min_len and rng.rand() < crop_p:
            L = rng.randint(min_len, n)                 # contiguous sub-scene
            a = rng.randint(0, n - L + 1)
            keep = np.where((r >= a) & (r < a + L))[0]
            x, r = x[keep], r[keep] - a
        xs.append(x)
        rs.append(r)
    N = max(len(r) for r in rs)
    X = np.zeros((len(idx), N, xs[0].shape[1]), dtype=np.float32)
    R = np.full((len(idx), N), -1, dtype=np.int64)
    for b, (x, r) in enumerate(zip(xs, rs)):
        X[b, :len(r)], R[b, :len(r)] = x, r
    R = torch.from_numpy(R)
    return torch.from_numpy(X), R, R < 0


def loss_fn(P, pos, A, R, pad, w_pos, w_next):
    valid = ~pad
    n = valid.sum(1).float()
    eye = torch.eye(P.shape[1], dtype=torch.bool).unsqueeze(0)
    pm = valid.unsqueeze(2) & valid.unsqueeze(1) & ~eye
    y = (R.unsqueeze(2) < R.unsqueeze(1)).float()
    lp = F.binary_cross_entropy_with_logits(P, y, reduction="none") * pm
    l_pair = (lp.sum((1, 2)) / pm.sum((1, 2))).mean()              # each scene weighs equally
    tgt = R.float() / (n.unsqueeze(1) - 1)
    l_pos = (((pos - tgt) ** 2) * valid).sum(1).div(n).mean()
    # next-speech: for every speech except the last, softmax over the other speeches
    An = A.masked_fill(~(valid.unsqueeze(1) & ~eye), -1e4)
    nxt = R.unsqueeze(2) + 1 == R.unsqueeze(1)
    has = nxt.any(2) & valid
    tgt_j = nxt.float().argmax(2)
    l_next = F.cross_entropy(An[has], tgt_j[has])
    return l_pair + w_pos * l_pos + w_next * l_next


@torch.no_grad()
def predict(model, feats, batch=64):
    model.eval()
    out = []
    for s in range(0, len(feats), batch):
        idx = list(range(s, min(len(feats), s + batch)))
        X, _, pad = make_batch([feats[k] for k in idx], [np.arange(len(feats[k])) for k in idx], range(len(idx)))
        P, pos, _ = model(X, pad)
        for b, k in enumerate(idx):
            n = len(feats[k])
            out.append((P[b, :n, :n].numpy().astype(np.float64), pos[b, :n].numpy().astype(np.float64)))
    return out


def decode(P, pos, blend, swap):
    n = len(pos)
    prob = 1.0 / (1.0 + np.exp(-P))
    np.fill_diagonal(prob, 0.0)
    early = (1.0 - blend) * prob.sum(1) / (n - 1) + blend * (1.0 - pos)
    order = list(np.lexsort((np.arange(n), -early)))
    if swap:
        for _ in range(n):
            changed = False
            for k in range(n - 1):
                a, b = order[k], order[k + 1]
                if P[b, a] > 0:
                    order[k], order[k + 1] = b, a
                    changed = True
            if not changed:
                break
    return [int(x) for x in order]


def kendall_tau(true_order, pred_order):
    n = len(true_order)
    rt, rp = np.empty(n, np.int64), np.empty(n, np.int64)
    rt[np.asarray(true_order)] = np.arange(n)
    rp[np.asarray(pred_order)] = np.arange(n)
    dt = np.sign(rt[:, None] - rt[None, :])
    dp = np.sign(rp[:, None] - rp[None, :])
    return (int(((dt * dp) > 0).sum() // 2) - int(((dt * dp) < 0).sum() // 2)) / (n * (n - 1) / 2)


def mean_tau(orders_true, preds, blend=0.0, swap=False):
    return float(np.mean([kendall_tau(t, decode(P, p, blend, swap)) for t, (P, p) in zip(orders_true, preds)]))


def train_model(cfg, tr_feats, tr_ranks, va_feats, va_orders, seed):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = SceneOrderer(tr_feats[0].shape[1], cfg["d"], cfg["layers"], cfg["dropout"], cfg["noise"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = cfg["epochs"] * int(np.ceil(len(tr_feats) / cfg["batch"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    best, best_state, bad = -9.0, None, 0
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(len(tr_feats))
        for s in range(0, len(perm), cfg["batch"]):
            X, R, pad = make_batch(tr_feats, tr_ranks, perm[s:s + cfg["batch"]], rng, cfg["crop_p"])
            P, pos, A = model(X, pad)
            loss = loss_fn(P, pos, A, R, pad, cfg["w_pos"], cfg["w_next"])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        tau = mean_tau(va_orders, predict(model, va_feats))
        if tau > best:
            best, best_state, bad = tau, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= cfg["patience"]:                       # early stopping on validation score
            break
    model.load_state_dict(best_state)
    return model, best, ep + 1


# ── configuration search space (fixed list, fixed order) ─────────────────────
BASE = dict(d=256, layers=2, dropout=0.15, noise=0.02, lr=1e-3, wd=0.05, batch=32,
            epochs=40, patience=6, crop_p=0.5, w_pos=0.5, w_next=0.3)
CONFIGS = [
    dict(BASE),
    dict(BASE, d=384, layers=3, dropout=0.2),
    dict(BASE, w_next=0.0, crop_p=0.0),
]
N_FOLDS = 5
HPO_FOLDS = 2                       # configs compared on the first 2 author folds
BLEND_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
SWAP_GRID = [False, True]


def author_folds(groups, n_folds):
    """Balanced folds of whole author groups (largest first, seeded tie order)."""
    rng = np.random.RandomState(SEED)
    sizes = groups.value_counts()
    names = np.array(sorted(sizes.index))
    names = names[rng.permutation(len(names))]
    names = sorted(names, key=lambda g: -sizes[g])          # stable: keeps seeded order
    load, fold_of = np.zeros(n_folds), {}
    for g in names:
        f = int(np.argmin(load))
        fold_of[g] = f
        load[f] += sizes[g]
    return groups.map(fold_of).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir", nargs="?", default="./dataset/public")
    ap.add_argument("submission_out", nargs="?", default="./working/submission.csv")
    args, _unknown = ap.parse_known_args()
    public_dir, out_path = Path(args.public_dir), Path(args.submission_out)
    ckpt_dir = str(public_dir / "pretrained" / "multilingual-e5-small")
    for name in ["train.csv", "test.csv", "train_groups.csv", "sample_submission.csv",
                 "pretrained/multilingual-e5-small/config.json"]:
        if not (public_dir / name).exists():
            raise FileNotFoundError(f"required file missing: {public_dir / name}")

    train = pd.read_csv(public_dir / "train.csv", keep_default_na=False)
    test = pd.read_csv(public_dir / "test.csv", keep_default_na=False)
    groups = pd.read_csv(public_dir / "train_groups.csv", keep_default_na=False)
    sample = pd.read_csv(public_dir / "sample_submission.csv", keep_default_na=False)
    train = train.merge(groups, on="id", how="left", validate="one_to_one")
    tr_sc = train["speeches"].map(json.loads).tolist()
    te_sc = test["speeches"].map(json.loads).tolist()
    tr_orders = [[int(x) for x in s.split()] for s in train["order"]]
    tr_ranks = []
    for sc, od in zip(tr_sc, tr_orders):
        if sorted(od) != list(range(len(sc))):
            raise ValueError("train order is not a permutation")
        r = np.empty(len(od), np.int64)
        r[np.asarray(od)] = np.arange(len(od))
        tr_ranks.append(r)
    log(f"train {len(tr_sc)} scenes / {train['group'].nunique()} authors, test {len(te_sc)} scenes")

    log("embedding every speech with the frozen e5-small encoder (3 views) ...")
    emb = embed_speeches(ckpt_dir, [s for sc in tr_sc + te_sc for s in sc])
    tr_feats, off = scene_inputs(tr_sc, emb, 0)
    te_feats, _ = scene_inputs(te_sc, emb, off)

    fold = author_folds(train["group"], N_FOLDS)

    def run_fold(cfg, f, seed):
        tr_i, va_i = np.where(fold != f)[0], np.where(fold == f)[0]
        model, tau, ep = train_model(cfg, [tr_feats[i] for i in tr_i], [tr_ranks[i] for i in tr_i],
                                     [tr_feats[i] for i in va_i], [tr_orders[i] for i in va_i], seed)
        return model, va_i, tau, ep

    log("hyperparameter search on held-out author folds")
    hpo = []
    for c, cfg in enumerate(CONFIGS):
        taus = [run_fold(cfg, f, SEED + f)[2] for f in range(HPO_FOLDS)]
        hpo.append(float(np.mean(taus)))
        log(f"  config {c}: {cfg}  held-out tau {np.round(taus, 4).tolist()} -> {hpo[-1]:.4f}")
    cfg = CONFIGS[int(np.argmax(hpo))]
    log(f"selected config {int(np.argmax(hpo))}")

    log(f"{N_FOLDS}-fold author-grouped training of the selected config")
    oof = [None] * len(tr_sc)
    te_P = [np.zeros((len(x), len(x))) for x in te_feats]
    te_pos = [np.zeros(len(x)) for x in te_feats]
    for f in range(N_FOLDS):
        model, va_i, tau, ep = run_fold(cfg, f, SEED + f)
        for i, p in zip(va_i, predict(model, [tr_feats[i] for i in va_i])):
            oof[i] = p
        for k, (P, pos) in enumerate(predict(model, te_feats)):
            te_P[k] += P / N_FOLDS
            te_pos[k] += pos / N_FOLDS
        log(f"  fold {f}: held-out tau {tau:.4f} after {ep} epochs")

    best = None
    for blend in BLEND_GRID:
        for swap in SWAP_GRID:
            t = mean_tau(tr_orders, oof, blend, swap)
            if best is None or t > best[0]:
                best = (t, blend, swap)
    _, blend, swap = best
    log(f"out-of-fold (unseen authors) Kendall tau {best[0]:.4f}  [blend={blend}, swap={swap}]")

    orders = [decode(P, pos, blend, swap) for P, pos in zip(te_P, te_pos)]
    sub = pd.DataFrame({"id": test["id"], "order": [" ".join(map(str, o)) for o in orders]})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()

    n_by_id = dict(zip(test["id"], map(len, te_sc)))
    if list(sub.columns) != list(sample.columns) or len(sub) != len(sample) or sub["id"].duplicated().any():
        raise ValueError("submission does not match sample_submission.csv")
    for i, o in zip(sub["id"], sub["order"]):
        if sorted(int(x) for x in o.split()) != list(range(n_by_id[i])):
            raise ValueError(f"row {i}: order is not a permutation")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    back = pd.read_csv(out_path, keep_default_na=False)
    if back.shape != sample.shape or (back["order"].astype(str).str.strip() == "").any():
        raise ValueError("re-read submission is malformed")
    log(f"wrote {out_path}: {len(sub)} rows")


if __name__ == "__main__":
    main()
