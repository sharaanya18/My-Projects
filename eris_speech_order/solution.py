"""Restoring the order of speeches in a scene.

    python3 solution.py <public_dir> <submission_out>

Approach
--------
Every row is one scene whose speeches are shuffled. The metric is Kendall's
tau, i.e. the fraction of speech PAIRS put in the right relative order, so the
model is trained directly on pairwise precedence.

1. Speech encoder: the provided multilingual-e5-small checkpoint, fine-tuned
   end to end (word-embedding matrix frozen to save memory and to limit
   overfitting on 5k scenes). Long speeches keep their head and their tail,
   because the start of a turn answers the previous one and the end of a turn
   (often a question) sets up the next one.
2. Scene context: a 2-layer transformer over the speech vectors of one scene,
   with NO positional encoding (the presented order carries no information),
   so every speech is represented in the context of the other speeches of its
   own scene - the row itself is the unit of inference, nothing is pooled
   across test rows.
3. Two heads trained jointly:
   - pairwise precedence logit P[i, j] (antisymmetric: P[j, i] = -P[i, j]),
     binary cross-entropy against "i was spoken before j";
   - per-speech relative position in [0, 1], squared error.
4. Decoding: each speech gets an "earliness" score = blend of the fraction of
   other speeches it is predicted to precede and its predicted position, then
   an optional adjacent-swap pass makes neighbours agree with P.

Model selection happens inside this script on author-grouped validation (the
test split is by author): train on 80% of the author groups, pick the epoch
count and the decode settings by validation Kendall tau, then retrain on all
training rows with the selected epoch count. Everything is fixed in advance:
seeds, epochs, batch composition, candidate grids - no clock and no
machine-dependent branches.
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
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

# ── fixed configuration ───────────────────────────────────────────────────────
SEED = 42
DEVICE = torch.device("cuda")          # the test environment is one A10G
AMP_DTYPE = torch.bfloat16
NUM_THREADS = 4
HEAD_TOKENS = 72                       # tokens kept from the start of a speech
TAIL_TOKENS = 54                       # tokens kept from the end of a speech
SCENES_PER_BATCH = 8
EPOCHS_MAX = 4                         # fixed training length, selected below
LR_ENCODER = 3e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
POS_LOSS_WEIGHT = 0.5
VAL_GROUP_FRAC = 0.2                   # share of author groups held out
CTX_LAYERS = 2
CTX_HEADS = 6
PAIR_HIDDEN = 256
DROPOUT = 0.1
INFER_SCENES_PER_BATCH = 16
BLEND_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]   # weight of the position head
SWAP_GRID = [False, True]                  # adjacent-swap refinement on/off

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(NUM_THREADS)


def log(*a):
    print(*a, flush=True)


# ── metric (identical to the challenge's evaluate) ────────────────────────────
def kendall_tau(true_order, pred_order):
    n = len(true_order)
    rank_true = np.empty(n, dtype=np.int64)
    rank_pred = np.empty(n, dtype=np.int64)
    rank_true[np.asarray(true_order)] = np.arange(n)
    rank_pred[np.asarray(pred_order)] = np.arange(n)
    dt = np.sign(rank_true[:, None] - rank_true[None, :])
    dp = np.sign(rank_pred[:, None] - rank_pred[None, :])
    concordant = int(((dt * dp) > 0).sum() // 2)
    discordant = int(((dt * dp) < 0).sum() // 2)
    return (concordant - discordant) / (n * (n - 1) / 2)


def mean_tau(true_orders, pred_orders):
    return float(np.mean([kendall_tau(t, p) for t, p in zip(true_orders, pred_orders)]))


# ── data ──────────────────────────────────────────────────────────────────────
def load_data(public_dir):
    def need(name):
        p = public_dir / name
        if not p.exists():
            raise FileNotFoundError(f"required file missing: {p}")
        return p

    train = pd.read_csv(need("train.csv"), keep_default_na=False)
    test = pd.read_csv(need("test.csv"), keep_default_na=False)
    groups = pd.read_csv(need("train_groups.csv"), keep_default_na=False)
    sample = pd.read_csv(need("sample_submission.csv"), keep_default_na=False)
    need("pretrained/multilingual-e5-small/config.json")
    train = train.merge(groups, on="id", how="left", validate="one_to_one")
    if train["group"].eq("").any() or train["group"].isna().any():
        raise ValueError("train rows without an author group")
    train["speeches"] = train["speeches"].map(json.loads)
    test["speeches"] = test["speeches"].map(json.loads)
    train["order"] = train["order"].map(lambda s: [int(x) for x in s.split()])
    for sp, od in zip(train["speeches"], train["order"]):
        if sorted(od) != list(range(len(sp))):
            raise ValueError("train order is not a permutation of its speeches")
    return train, test, sample


class Tokenized:
    """Every speech tokenized once: head + tail tokens wrapped in <s> ... </s>."""

    def __init__(self, tokenizer, scenes):
        flat = ["query: " + s for sc in scenes for s in sc]
        enc = tokenizer(flat, add_special_tokens=False, truncation=False)["input_ids"]
        cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.pad_id = tokenizer.pad_token_id
        ids = []
        for t in enc:
            if len(t) > HEAD_TOKENS + TAIL_TOKENS:
                t = t[:HEAD_TOKENS] + t[-TAIL_TOKENS:]
            ids.append([cls_id] + t + [sep_id])
        self.offsets = np.concatenate([[0], np.cumsum([len(sc) for sc in scenes])])
        self.ids = ids

    def batch(self, scene_idx):
        rows = [r for s in scene_idx for r in range(self.offsets[s], self.offsets[s + 1])]
        L = max(len(self.ids[r]) for r in rows)
        inp = np.full((len(rows), L), self.pad_id, dtype=np.int64)
        att = np.zeros((len(rows), L), dtype=np.int64)
        for k, r in enumerate(rows):
            inp[k, : len(self.ids[r])] = self.ids[r]
            att[k, : len(self.ids[r])] = 1
        sizes = [int(self.offsets[s + 1] - self.offsets[s]) for s in scene_idx]
        return torch.from_numpy(inp), torch.from_numpy(att), sizes


def rank_targets(orders, scene_idx, n_max):
    """rank[b, i] = original position of presented speech i (-1 for padding)."""
    rank = np.full((len(scene_idx), n_max), -1, dtype=np.int64)
    for b, s in enumerate(scene_idx):
        od = orders[s]
        rank[b, np.asarray(od)] = np.arange(len(od))
    return torch.from_numpy(rank)


# ── model ─────────────────────────────────────────────────────────────────────
class SceneOrderer(nn.Module):
    def __init__(self, ckpt_dir):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(ckpt_dir)
        for p in self.encoder.embeddings.word_embeddings.parameters():
            p.requires_grad = False
        d = self.encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(d, CTX_HEADS, 2 * d, DROPOUT,
                                           batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, CTX_LAYERS)
        self.pair = nn.Sequential(nn.Linear(4 * d, PAIR_HIDDEN), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(PAIR_HIDDEN, 1))
        self.pos = nn.Linear(d, 1)

    def forward(self, input_ids, attention_mask, sizes):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        vec = (out * m).sum(1) / m.sum(1).clamp(min=1.0)          # mean pooling (e5)
        B, n_max = len(sizes), max(sizes)
        h = vec.new_zeros(B, n_max, vec.shape[-1])
        pad = torch.ones(B, n_max, dtype=torch.bool, device=vec.device)
        start = 0
        for b, n in enumerate(sizes):
            h[b, :n] = vec[start:start + n]
            pad[b, :n] = False
            start += n
        h = self.context(h, src_key_padding_mask=pad)
        hi = h.unsqueeze(2).expand(B, n_max, n_max, h.shape[-1])
        hj = h.unsqueeze(1).expand(B, n_max, n_max, h.shape[-1])
        f = self.pair(torch.cat([hi, hj, hi * hj, hi - hj], -1)).squeeze(-1)
        P = f - f.transpose(1, 2)                                   # antisymmetric
        pos = torch.sigmoid(self.pos(h).squeeze(-1))
        return P, pos, pad


def scene_loss(P, pos, pad, rank):
    valid = ~pad
    pair_mask = valid.unsqueeze(2) & valid.unsqueeze(1)
    pair_mask &= ~torch.eye(P.shape[1], dtype=torch.bool, device=P.device).unsqueeze(0)
    y = (rank.unsqueeze(2) < rank.unsqueeze(1)).float()          # i spoken before j
    l_pair = F.binary_cross_entropy_with_logits(P.float()[pair_mask], y[pair_mask])
    n = valid.sum(1, keepdim=True).float()
    target = rank.float() / (n - 1)
    l_pos = F.mse_loss(pos.float()[valid], target[valid])
    return l_pair + POS_LOSS_WEIGHT * l_pos


# ── training / inference ──────────────────────────────────────────────────────
def build_optimizer(model, total_steps):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER},
                             {"params": head, "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
    warm = max(1, int(WARMUP_FRAC * total_steps))

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def epoch_batches(n_scenes, epoch):
    rng = np.random.RandomState(SEED + epoch)
    perm = rng.permutation(n_scenes)
    return [perm[i:i + SCENES_PER_BATCH] for i in range(0, n_scenes, SCENES_PER_BATCH)]


def train_epoch(model, opt, sched, tok, orders, epoch):
    model.train()
    losses = []
    for scene_idx in epoch_batches(len(orders), epoch):
        inp, att, sizes = tok.batch(scene_idx)
        rank = rank_targets(orders, scene_idx, max(sizes)).to(DEVICE)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            P, pos, pad = model(inp.to(DEVICE), att.to(DEVICE), sizes)
        loss = scene_loss(P, pos, pad, rank)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model, tok, n_scenes):
    """Per scene: (P as float numpy [n, n], predicted position [n])."""
    model.eval()
    out = []
    for s in range(0, n_scenes, INFER_SCENES_PER_BATCH):
        scene_idx = list(range(s, min(n_scenes, s + INFER_SCENES_PER_BATCH)))
        inp, att, sizes = tok.batch(scene_idx)
        with torch.autocast(device_type=DEVICE.type, dtype=AMP_DTYPE):
            P, pos, _ = model(inp.to(DEVICE), att.to(DEVICE), sizes)
        P, pos = P.float().cpu().numpy(), pos.float().cpu().numpy()
        for b, n in enumerate(sizes):
            out.append((P[b, :n, :n], pos[b, :n]))
    return out


def decode(P, pos, blend, swap):
    """Order of presented indices, earliest first."""
    n = len(pos)
    prob = 1.0 / (1.0 + np.exp(-P))
    np.fill_diagonal(prob, 0.0)
    wins = prob.sum(1) / (n - 1)                       # share of others it precedes
    early = (1.0 - blend) * wins + blend * (1.0 - pos)
    order = list(np.lexsort((np.arange(n), -early)))   # stable, index tie-break
    if swap:
        for _ in range(n):                             # fixed number of passes
            changed = False
            for k in range(n - 1):
                a, b = order[k], order[k + 1]
                if P[b, a] > 0:                        # model prefers b before a
                    order[k], order[k + 1] = b, a
                    changed = True
            if not changed:
                break
    return [int(x) for x in order]


def fit(ckpt_dir, tok, orders, n_epochs, eval_fn=None):
    torch.manual_seed(SEED)
    model = SceneOrderer(ckpt_dir).to(DEVICE)
    steps_per_epoch = len(epoch_batches(len(orders), 0))
    opt, sched = build_optimizer(model, steps_per_epoch * EPOCHS_MAX)
    history = []
    for ep in range(n_epochs):
        loss = train_epoch(model, opt, sched, tok, orders, ep)
        score = eval_fn(model) if eval_fn is not None else None
        history.append(score)
        log(f"  epoch {ep + 1}/{n_epochs}  train loss {loss:.4f}"
            + ("" if score is None else f"  val tau {score[0]:.4f} (blend={score[1]}, swap={score[2]})"))
    return model, history


def best_decode(preds, true_orders):
    best = None
    for blend in BLEND_GRID:
        for swap in SWAP_GRID:
            pred = [decode(P, pos, blend, swap) for P, pos in preds]
            tau = mean_tau(true_orders, pred)
            if best is None or tau > best[0]:
                best = (tau, blend, swap)
    return best


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir", nargs="?", default="./dataset/public")
    ap.add_argument("submission_out", nargs="?", default="./working/submission.csv")
    args, _unknown = ap.parse_known_args()
    public_dir, out_path = Path(args.public_dir), Path(args.submission_out)
    ckpt_dir = str(public_dir / "pretrained" / "multilingual-e5-small")

    train, test, sample = load_data(public_dir)
    log(f"train {len(train)} scenes / {train['group'].nunique()} author groups, test {len(test)} scenes")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)

    # author-grouped holdout, mirroring the by-author test split
    rng = np.random.RandomState(SEED)
    grp = np.array(sorted(train["group"].unique()))
    grp = grp[rng.permutation(len(grp))]
    counts = train["group"].value_counts()
    cum = np.cumsum([counts[g] for g in grp])
    val_groups = set(grp[: int(np.searchsorted(cum, VAL_GROUP_FRAC * len(train))) + 1])
    is_val = train["group"].isin(val_groups).to_numpy()
    tr_part, va_part = train[~is_val].reset_index(drop=True), train[is_val].reset_index(drop=True)
    log(f"holdout: {len(val_groups)} author groups, {len(va_part)} scenes")

    tok_tr = Tokenized(tokenizer, tr_part["speeches"].tolist())
    tok_va = Tokenized(tokenizer, va_part["speeches"].tolist())
    va_true = va_part["order"].tolist()

    def eval_val(model):
        return best_decode(predict(model, tok_va, len(va_part)), va_true)

    log("stage 1: train on 80% of authors, select epochs + decoding on held-out authors")
    _, history = fit(ckpt_dir, tok_tr, tr_part["order"].tolist(), EPOCHS_MAX, eval_val)
    best_ep = int(np.argmax([h[0] for h in history])) + 1
    _, blend, swap = history[best_ep - 1]
    log(f"selected: epochs={best_ep}, blend={blend}, swap={swap}, held-out-author tau={history[best_ep - 1][0]:.4f}")

    log("stage 2: retrain on all training scenes with the selected settings")
    tok_all = Tokenized(tokenizer, train["speeches"].tolist())
    model, _ = fit(ckpt_dir, tok_all, train["order"].tolist(), best_ep)

    tok_te = Tokenized(tokenizer, test["speeches"].tolist())
    preds = predict(model, tok_te, len(test))
    orders = [decode(P, pos, blend, swap) for P, pos in preds]

    sub = pd.DataFrame({"id": test["id"], "order": [" ".join(map(str, o)) for o in orders]})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()     # sample's id order

    # validate before writing: a malformed file is ranked below a score of 0
    n_by_id = dict(zip(test["id"], test["speeches"].map(len)))
    if list(sub.columns) != list(sample.columns):
        raise ValueError(f"columns {list(sub.columns)} != {list(sample.columns)}")
    if len(sub) != len(sample) or sub["id"].duplicated().any():
        raise ValueError("row count or ids do not match sample_submission.csv")
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
