"""Inference: plan the heads, realize each stage, splice the pipeline.

Every test row is decoded on its own -- the encoder sees exactly one
description at a time and nothing is shared, pooled, calibrated or counted
across the test set (Guidebook 4.2.5).
"""
import torch

from engine import collate, pos_class, prepare
from tokenizer import BOS, EOS, PAD, UNK, tokenize


@torch.no_grad()
def encode_one(model, ex, device):
    src = torch.tensor([ex["enc_ids"]], dtype=torch.long, device=device)
    ext = torch.tensor([ex["ext_ids"]], dtype=torch.long, device=device)
    pad = src.eq(PAD)
    mem = model.encoder(src, pad)
    return mem, pad, ext


@torch.no_grad()
def bag_logodds(model, mem, pad):
    """Per-head log-odds from the unordered head-set classifier."""
    keep = (~pad).unsqueeze(-1).float()
    pooled = (mem * keep).sum(1) / keep.sum(1).clamp(min=1)
    return model.bag(pooled)[0].float().clamp(-8, 8)


@torch.no_grad()
def plan_beam(model, mem, pad, head_vocab, beam=4, max_stages=3, min_stages=1,
              bag_lambda=0.0):
    """Beam search over the head sequence.

    With bag_lambda > 0 each emitted head also earns the unordered head-set
    classifier's log-odds for it.  That classifier never sees head order, so it
    can vouch for a pair of heads whose adjacency the autoregressive plan
    decoder has never observed.
    """
    device = mem.device
    bag = bag_logodds(model, mem, pad) * bag_lambda if bag_lambda > 0 else None
    live = [([], 0.0)]
    done = []
    for step in range(max_stages):
        cand = []
        dec_in = torch.tensor([[BOS] + h for h, _ in live], dtype=torch.long, device=device)
        m = mem.expand(len(live), -1, -1)
        p = pad.expand(len(live), -1)
        logits = model.plan(dec_in, m, p)[:, -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        # never emit the structural placeholders as a command head
        logp[:, PAD] = -1e9
        logp[:, BOS] = -1e9
        logp[:, UNK] = -1e9
        if step + 1 < min_stages:
            logp[:, EOS] = -1e9
        if bag is not None:
            logp = logp + torch.cat([bag[:EOS], bag.new_zeros(1), bag[EOS + 1:]]).unsqueeze(0) \
                if EOS < bag.numel() else logp
            # a head already in this plan gets no second bonus
            for i, (hs, _) in enumerate(live):
                for h in hs:
                    logp[i, h] -= bag[h]
        top = torch.topk(logp, min(beam + 1, logp.size(-1)), dim=-1)
        for i, (hs, sc) in enumerate(live):
            for j in range(top.indices.size(1)):
                tid = int(top.indices[i, j])
                s = sc + float(top.values[i, j])
                if tid == EOS:
                    if len(hs) >= min_stages:
                        done.append((hs, s / max(len(hs), 1)))
                else:
                    cand.append((hs + [tid], s))
        if not cand:
            break
        cand.sort(key=lambda x: -x[1] / max(len(x[0]), 1))
        live = cand[:beam]
    for hs, sc in live:
        done.append((hs, sc / max(len(hs), 1)))
    done.sort(key=lambda x: -x[1])
    return done[:beam] if done else [([head_vocab.get("find")], 0.0)]


@torch.no_grad()
def _blocks_repeat(seq, tid, n=3):
    """True if appending tid would repeat an n-gram already in seq."""
    if len(seq) < n - 1:
        return False
    cand = tuple(seq[-(n - 1):]) + (tid,)
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n]) == cand:
            return True
    return False


def realize_beam(model, mem, pad, ext, head_id, pos_id, tgt_vocab, oov,
                 beam=4, max_len=40, len_norm=1.2, prefix=None, no_repeat=3):
    """Beam search for one stage's token sequence over the extended vocabulary.

    `prefix` holds the grader tokens of the planned command head.  They are
    force-decoded first so the stage is guaranteed to start with the head the
    plan chose -- without this the decoder can drift back to a head it saw
    more often in training, which undoes the plan entirely.
    """
    device = mem.device
    n_tgt = model.realize.n_tgt
    n_ext = len(oov)
    pre_ext, pre_din = [], []
    oov_idx = {t: i for i, t in enumerate(oov)}
    for t in prefix or []:
        vid = tgt_vocab.stoi.get(t)
        if vid is not None:
            pre_ext.append(vid); pre_din.append(vid)
        elif t in oov_idx:
            pre_ext.append(n_tgt + oov_idx[t]); pre_din.append(UNK)
        else:
            break                        # unrepresentable head: let the model decode freely
    live = [(pre_ext, pre_din, 0.0)]  # (extended ids, decoder-input ids, logprob)
    finished = []
    for _ in range(max_len):
        if not live:
            break
        B = len(live)
        dec_in = torch.tensor([[BOS] + d for _, d, _ in live], dtype=torch.long, device=device)
        m = mem.expand(B, -1, -1)
        p = pad.expand(B, -1)
        e = ext.expand(B, -1)
        hid = torch.tensor([head_id] * B, dtype=torch.long, device=device)
        pid = torch.tensor([pos_id] * B, dtype=torch.long, device=device)
        h = model.realize.hidden(dec_in, m, p, hid, pid)
        logp, att = model.realize.dist(h[:, -1:], m, p, e, n_ext)
        logp = logp[:, 0].float()
        logp[:, PAD] = -1e9
        logp[:, BOS] = -1e9
        top = torch.topk(logp, min(beam + 2, logp.size(-1)), dim=-1)
        cand = []
        for i, (seq, din, sc) in enumerate(live):
            for j in range(top.indices.size(1)):
                tid = int(top.indices[i, j])
                s = sc + float(top.values[i, j])
                if tid == EOS:
                    if len(seq) > len(pre_ext) or seq:
                        finished.append((seq, s / (max(len(seq), 1) ** len_norm)))
                    continue
                if tid == UNK:
                    # unk-replacement: emit the source token the copy head is
                    # attending to most, rather than a literal placeholder
                    src_pos = int(att[i, -1].argmax())
                    tid = int(ext[0, src_pos])
                if no_repeat and _blocks_repeat(seq, tid, no_repeat):
                    continue
                cand.append((seq + [tid], din + [tid if tid < n_tgt else UNK], s))
        if not cand:
            break
        cand.sort(key=lambda x: -x[2] / (len(x[0]) ** len_norm))
        live = cand[:beam]
        if len(finished) >= beam * 2:
            break
    if not finished:
        finished = [(s, sc / max(len(s), 1) ** len_norm) for s, _, sc in live if s]
    if not finished:
        return [], 0.0
    finished.sort(key=lambda x: -x[1])
    return finished[0]


def ids_to_tokens(ids, tgt_vocab, oov):
    n = len(tgt_vocab)
    out = []
    for i in ids:
        if i < n:
            t = tgt_vocab.itos[i]
            if t not in ("<pad>", "<bos>", "<eos>", "<unk>"):
                out.append(t)
        elif i - n < len(oov):
            out.append(oov[i - n])
    return out


@torch.no_grad()
def predict_one(models, ex, src_vocab, tgt_vocab, head_vocab, device,
                plan_beam_size=4, rz_beam_size=4, min_stages=1, max_stages=3,
                force_heads=None, bag_lambda=0.0):
    """Decode a full pipeline for a single row.

    `models` is a list; a multi-seed ensemble averages the *plan* distribution
    (it shares one small output space) and picks the realization with the best
    mean log-probability.  Everything stays within this one row.
    """
    m0 = models[0]
    encs = [encode_one(m, ex, device) for m in models]

    if force_heads is not None:
        # diagnostic path only (dev harness): isolates realize quality from
        # plan quality by supplying the gold head sequence.
        heads = [head_vocab.get(h) for h in force_heads]
        return _realize_all(models, encs, heads, ex, tgt_vocab, head_vocab,
                            rz_beam_size), list(force_heads)

    # --- plan: average the head distributions of the ensemble members
    plans = plan_beam(m0, encs[0][0], encs[0][1], head_vocab,
                      beam=plan_beam_size, max_stages=max_stages, min_stages=min_stages,
                      bag_lambda=bag_lambda)
    if len(models) > 1:
        rescored = []
        for hs, _ in plans:
            if not hs:
                continue
            tot = 0.0
            for m, (mem, pad, _) in zip(models, encs):
                dec = torch.tensor([[BOS] + hs], dtype=torch.long, device=device)
                lg = torch.log_softmax(m.plan(dec, mem, pad).float(), dim=-1)[0]
                tot += sum(float(lg[i, t]) for i, t in enumerate(hs + [EOS]))
            rescored.append((hs, tot / (len(models) * (len(hs) + 1))))
        if rescored:
            rescored.sort(key=lambda x: -x[1])
            plans = rescored
    heads = plans[0][0] if plans else [head_vocab.get("find")]

    toks = _realize_all(models, encs, heads, ex, tgt_vocab, head_vocab, rz_beam_size)
    return toks, [head_vocab.itos[h] for h in heads]


@torch.no_grad()
def _realize_all(models, encs, heads, ex, tgt_vocab, head_vocab, rz_beam_size):
    """Realize each planned stage and splice them with a pipe."""
    n = len(heads)
    all_tokens = []
    for k, hid in enumerate(heads):
        best, best_s = None, -1e18
        for m, (mem, pad, ext) in zip(models, encs):
            t, s = realize_beam(m, mem, pad, ext, hid, pos_class(k, n),
                                tgt_vocab, ex["oov"], beam=rz_beam_size,
                                prefix=tokenize(head_vocab.itos[hid]))
            if s > best_s:
                best, best_s = t, s
        stage = ids_to_tokens(best or [], tgt_vocab, ex["oov"])
        if not stage:
            stage = [head_vocab.itos[hid]] if hid < len(head_vocab) else ["ls"]
        if k:
            all_tokens.append("|")
        all_tokens.extend(stage)
    return all_tokens
