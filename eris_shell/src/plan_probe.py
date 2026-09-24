"""Isolate the planning bottleneck (development only).

The plan decoder has the tiny output space, so it trains in seconds, which
makes it the right place to search for what actually unlocks the compositional
split.  Reports the predicted stage-count distribution as well as accuracy,
because predicting the wrong pipeline *length* and predicting the wrong heads
are different failures needing different fixes.
"""
import argparse
import collections
import json
import time

import torch
import torch.nn.functional as F

from dataset import build_examples, read_csv
from decode import plan_beam
from engine import DEFAULT_CFG, _pad, prepare
from model import ShellSynth
from splits import make_compositional_folds, random_fold
from tokenizer import BOS, EOS, PAD, build_vocabs


def run(cfg, tr_rows, val_sets, seed=0, epochs=25, log=True):
    torch.manual_seed(seed)
    ex = build_examples(tr_rows)
    src_v, tgt_v, head_v = build_vocabs(tr_rows, [e["heads"] for e in ex],
                                        src_max=cfg["src_vocab"], tgt_max=cfg["tgt_vocab"])
    for e in ex:
        prepare(e, src_v, tgt_v, head_v)
    from engine import apply_length_weights
    apply_length_weights(ex, cfg["len_balance_alpha"], len(head_v))
    model = ShellSynth(len(src_v), len(tgt_v), len(head_v), cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    dev = torch.device("cpu")
    import random as _r
    rng = _r.Random(seed)
    order = list(range(len(ex)))
    bs = cfg["batch_size"]
    step = 0
    cfg["total_steps"] = epochs * (len(ex) // bs + 1)
    from engine import lr_at
    for epch in range(epochs):
        model.train()
        rng.shuffle(order)
        tot = nb = 0
        for i in range(0, len(order), bs):
            bb = [ex[j] for j in order[i:i + bs]]
            src = _pad([b["enc_ids"] for b in bb], dev)
            pad = src.eq(PAD)
            pin = _pad([[BOS] + b["head_ids"] for b in bb], dev)
            ptg = _pad([b["head_ids"] + [EOS] for b in bb], dev)
            lw = torch.tensor([b["len_w"] for b in bb], dtype=torch.float)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg)
            mem = model.encoder(src, pad)
            lg = model.plan(pin, mem, pad, prev_drop=cfg["prev_head_drop"])
            tl = F.cross_entropy(lg.reshape(-1, lg.size(-1)), ptg.reshape(-1),
                                 ignore_index=PAD, label_smoothing=cfg["label_smooth"],
                                 reduction="none").view_as(ptg)
            pm = ptg.ne(PAD).float()
            loss = (tl * pm * lw.unsqueeze(1)).sum() / (pm * lw.unsqueeze(1)).sum().clamp(min=1e-6)
            if cfg["bag_weight"] > 0:
                pooled = (mem * (~pad).unsqueeze(-1)).sum(1) / (~pad).sum(1, keepdim=True).clamp(min=1)
                bt = torch.zeros(len(bb), len(head_v))
                for bi, b in enumerate(bb):
                    for h in set(b["head_ids"]):
                        bt[bi, h] = 1.0
                loss = loss + cfg["bag_weight"] * F.binary_cross_entropy_with_logits(
                    model.bag(pooled), bt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            tot += loss.item(); nb += 1
    model.eval()
    out = {}
    for name, rows in val_sets.items():
        vex = build_examples(rows)
        ok = 0
        lens = collections.Counter()
        gold_lens = collections.Counter()
        head_recall = 0
        for e in vex:
            prepare(e, src_v, tgt_v, head_v)
            mem, pad, _ = None, None, None
            src = torch.tensor([e["enc_ids"]], dtype=torch.long)
            pad = src.eq(PAD)
            mem = model.encoder(src, pad)
            plans = plan_beam(model, mem, pad, head_v, beam=cfg.get("plan_beam", 4),
                              max_stages=3, min_stages=cfg.get("min_stages", 1))
            pred = [head_v.itos[h] for h in plans[0][0]]
            lens[len(pred)] += 1
            gold_lens[len(e["heads"])] += 1
            if pred == e["heads"]:
                ok += 1
            head_recall += len(set(pred) & set(e["heads"])) / max(len(e["heads"]), 1)
        out[name] = {"plan_acc": round(ok / len(vex), 4),
                     "head_recall": round(head_recall / len(vex), 4),
                     "pred_len": dict(sorted(lens.items())),
                     "gold_len": dict(sorted(gold_lens.items()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--sweep", default="prev_head_drop")
    ap.add_argument("--values", default="0.0,0.3,0.6")
    ap.add_argument("--set", action="append", default=[])
    args = ap.parse_args()
    torch.set_num_threads(4)

    rows = read_csv("data/train.csv")
    comp = make_compositional_folds(rows, 6, 180, 0)[args.fold]
    tr = [rows[i] for i in comp["train"]]
    cval = [rows[i] for i in comp["val"]][:150]
    rnd = random_fold(rows, 0.08, seed=7)
    tset = set(comp["train"])
    rval = [rows[i] for i in rnd["val"] if i in tset][:150]
    rval_multi = [r for r in rval if "|" in r["output"]]
    tr = [r for r in tr if r not in rval]

    base = dict(DEFAULT_CFG)
    base.update(src_vocab=6000, tgt_vocab=1500)
    for kv in args.set:
        k, v = kv.split("=", 1)
        base[k] = type(base[k])(v) if k in base else float(v)

    for v in args.values.split(","):
        cfg = dict(base)
        cfg[args.sweep] = type(base.get(args.sweep, 0.0))(v)
        t = time.time()
        res = run(cfg, tr, {"comp": cval, "rand": rval, "rand_multi": rval_multi},
                  epochs=args.epochs)
        print(f"{args.sweep}={v} ({time.time()-t:.0f}s)", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
