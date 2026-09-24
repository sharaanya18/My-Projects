"""Batching, joint training and decoding for the plan+realize model."""
import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import MAX_SRC, MAX_STAGE_TGT, MAX_STAGES, encode_source, encode_stage_target
from tokenizer import BOS, EOS, PAD, UNK

DEFAULT_CFG = dict(
    d_model=192, nhead=4, enc_layers=2, plan_layers=2, dec_layers=2, ff=384,
    dropout=0.25, max_src=MAX_SRC, max_tgt=MAX_STAGE_TGT, max_stages=MAX_STAGES,
    n_poscls=6, lr=3e-4, warmup=300, batch_size=48, label_smooth=0.1,
    prev_head_drop=0.3, plan_weight=1.0, epochs=40, grad_clip=1.0,
    # inverse-frequency weighting of the plan loss over pipeline length.
    # 75% of training commands are a single stage, so an unweighted plan loss
    # collapses onto "emit one head".  alpha=0 disables it; it is chosen on the
    # compositional folds, which are built only from public training rows.
    len_balance_alpha=0.0,
    bag_weight=0.0,          # auxiliary unordered head-set loss
)


def pos_class(stage_idx, n_stages):
    """Where this stage sits in the pipeline (first/middle/last, of how many)."""
    return min(stage_idx, 2) * 2 + (1 if n_stages > 1 else 0)


def prepare(ex, src_vocab, tgt_vocab, head_vocab, with_labels=True):
    """Attach encoder ids, pointer bookkeeping and (optionally) targets."""
    enc_ids, ext_ids, oov = encode_source(ex["src_tok"], src_vocab, tgt_vocab)
    ex["enc_ids"], ex["ext_ids"], ex["oov"] = enc_ids, ext_ids, oov
    if with_labels:
        ex["head_ids"] = [head_vocab.get(h) for h in ex["heads"]]
        ex["stages_enc"] = [encode_stage_target(s, tgt_vocab, oov) for s in ex["stage_tok"]]
    return ex


def _pad(seqs, device, value=PAD):
    m = max((len(s) for s in seqs), default=1) or 1
    return torch.tensor([list(s) + [value] * (m - len(s)) for s in seqs],
                        dtype=torch.long, device=device)


def collate(batch, n_tgt, device):
    src = _pad([b["enc_ids"] for b in batch], device)
    ext = _pad([b["ext_ids"] for b in batch], device)
    src_pad = src.eq(PAD)
    n_ext = max((len(b["oov"]) for b in batch), default=0)
    out = {"src": src, "src_ext": ext, "src_pad": src_pad, "n_ext": n_ext}

    if "head_ids" in batch[0]:
        out["plan_in"] = _pad([[BOS] + b["head_ids"] for b in batch], device)
        out["plan_tgt"] = _pad([b["head_ids"] + [EOS] for b in batch], device)
        out["plan_w"] = torch.tensor([b.get("len_w", 1.0) for b in batch],
                                     dtype=torch.float, device=device)
        bag = torch.zeros(len(batch), batch[0]["n_head_vocab"], device=device)
        for i, b in enumerate(batch):
            for h in set(b["head_ids"]):
                bag[i, h] = 1.0
        out["bag_tgt"] = bag
        rows, heads, poss, dins, tgts = [], [], [], [], []
        for i, b in enumerate(batch):
            n = len(b["stages_enc"])
            for k, (din, tgt) in enumerate(b["stages_enc"]):
                rows.append(i)
                heads.append(b["head_ids"][k])
                poss.append(pos_class(k, n))
                dins.append(din)
                tgts.append(tgt)
        out["rz_row"] = torch.tensor(rows, dtype=torch.long, device=device)
        out["rz_head"] = torch.tensor(heads, dtype=torch.long, device=device)
        out["rz_pos"] = torch.tensor(poss, dtype=torch.long, device=device)
        out["rz_in"] = _pad(dins, device)
        out["rz_tgt"] = _pad(tgts, device)
    return out


def joint_loss(model, b, cfg):
    mem = model.encoder(b["src"], b["src_pad"])

    logits = model.plan(b["plan_in"], mem, b["src_pad"], prev_drop=cfg["prev_head_drop"])
    tok_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), b["plan_tgt"].reshape(-1),
        ignore_index=PAD, label_smoothing=cfg["label_smooth"], reduction="none",
    ).view_as(b["plan_tgt"])
    pm = b["plan_tgt"].ne(PAD).float()
    w = b["plan_w"].unsqueeze(1)
    plan_loss = (tok_loss * pm * w).sum() / (pm * w).sum().clamp(min=1e-6)

    if cfg["bag_weight"] > 0:
        pooled = (mem * (~b["src_pad"]).unsqueeze(-1)).sum(1) / \
                 (~b["src_pad"]).sum(1, keepdim=True).clamp(min=1)
        bag_loss = F.binary_cross_entropy_with_logits(model.bag(pooled), b["bag_tgt"])
    else:
        bag_loss = torch.zeros((), device=mem.device)

    r = b["rz_row"]
    mem_r, pad_r, ext_r = mem[r], b["src_pad"][r], b["src_ext"][r]
    h = model.realize.hidden(b["rz_in"], mem_r, pad_r, b["rz_head"], b["rz_pos"])
    logp, _ = model.realize.dist(h, mem_r, pad_r, ext_r, b["n_ext"])
    tgt = b["rz_tgt"]
    mask = tgt.ne(PAD)
    nll = -logp.gather(2, tgt.unsqueeze(2)).squeeze(2)
    # label smoothing over the fixed vocabulary part only (copy targets are
    # per-example, so smoothing them is not well defined)
    if cfg["label_smooth"] > 0:
        smooth = -logp[:, :, : model.realize.n_tgt].mean(dim=-1)
        nll = (1 - cfg["label_smooth"]) * nll + cfg["label_smooth"] * smooth
    rz_loss = (nll * mask).sum() / mask.sum().clamp(min=1)
    total = cfg["plan_weight"] * plan_loss + rz_loss + cfg["bag_weight"] * bag_loss
    return total, plan_loss.item(), rz_loss.item()


def lr_at(step, cfg):
    w = cfg["warmup"]
    if step < w:
        return cfg["lr"] * (step + 1) / w
    # cosine decay over the nominal schedule
    total = max(cfg.get("total_steps", 4000), w + 1)
    t = min(1.0, (step - w) / (total - w))
    return cfg["lr"] * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))


def apply_length_weights(examples, alpha, n_head_vocab):
    """Inverse-frequency weight per pipeline length, normalised to mean 1.

    Standard imbalance correction on the plan decoder's length decision.  It is
    derived from the training data alone; nothing about the test set is used.
    """
    from collections import Counter
    cnt = Counter(len(e["heads"]) for e in examples)
    n = len(examples)
    raw = {k: (n / (len(cnt) * c)) ** alpha for k, c in cnt.items()}
    mean = sum(raw[len(e["heads"])] for e in examples) / max(n, 1)
    for e in examples:
        e["len_w"] = raw[len(e["heads"])] / mean
        e["n_head_vocab"] = n_head_vocab
    return examples


def train(model, examples, cfg, deadline=None, log=None, eval_fn=None, eval_every=0):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    n_tgt = model.realize.n_tgt
    dev = next(model.parameters()).device
    rng = random.Random(cfg.get("seed", 0))
    order = list(range(len(examples)))
    bs = cfg["batch_size"]
    cfg["total_steps"] = cfg["epochs"] * math.ceil(len(examples) / bs)
    step = 0
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        rng.shuffle(order)
        tot = tp = tr = nb = 0.0
        for i in range(0, len(order), bs):
            idx = order[i: i + bs]
            b = collate([examples[j] for j in idx], n_tgt, dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            loss, pl, rl = joint_loss(model, b, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1
            tot += loss.item(); tp += pl; tr += rl; nb += 1
        rec = {"epoch": ep, "loss": tot / nb, "plan": tp / nb, "realize": tr / nb}
        if eval_fn and eval_every and (ep + 1) % eval_every == 0:
            rec.update(eval_fn(model, ep))
        history.append(rec)
        if log:
            log(rec)
        if deadline and time.time() > deadline:
            if log:
                log({"epoch": ep, "note": "time budget reached, stopping training"})
            break
    return history
