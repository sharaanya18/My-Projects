"""Development harness -- NOT part of the graded submission.

Trains on a compositional-holdout fold and reports the grader metric on both
that fold and a random split, which is the over/underfitting instrument the
build spec asks for.
"""
import argparse
import copy
import json
import sys
import time

import torch

from dataset import build_examples, read_csv
from decode import predict_one
from detok import SpacingModel, finalize
from engine import DEFAULT_CFG, prepare, train
from metric import score_breakdown
from model import ShellSynth
from splits import make_compositional_folds, make_frequent_pair_folds, random_fold
from tokenizer import build_vocabs


def build(rows_train, cfg, seed):
    torch.manual_seed(seed)
    ex = build_examples(rows_train)
    src_v, tgt_v, head_v = build_vocabs(rows_train, [e["heads"] for e in ex],
                                        src_max=cfg["src_vocab"], tgt_max=cfg["tgt_vocab"])
    for e in ex:
        prepare(e, src_v, tgt_v, head_v)
    from engine import apply_length_weights
    apply_length_weights(ex, cfg["len_balance_alpha"], len(head_v))
    model = ShellSynth(len(src_v), len(tgt_v), len(head_v), cfg)
    return model, ex, src_v, tgt_v, head_v


def evaluate(model, rows_val, src_v, tgt_v, head_v, spacing, device, limit=None,
             plan_beam_size=4, rz_beam_size=4, min_stages=1, oracle_plan=False,
             bag_lambda=0.0):
    model.eval()
    rows = rows_val[:limit] if limit else rows_val
    ex = build_examples(rows)
    preds, truths, plan_ok = [], [], 0
    for e, r in zip(ex, rows):
        prepare(e, src_v, tgt_v, head_v, with_labels=True)
        toks, heads = predict_one([model], e, src_v, tgt_v, head_v, device,
                                  plan_beam_size=plan_beam_size,
                                  rz_beam_size=rz_beam_size, min_stages=min_stages,
                                  force_heads=e["heads"] if oracle_plan else None,
                                  bag_lambda=bag_lambda)
        preds.append(finalize(toks, spacing))
        truths.append(r["output"])
        if heads == e["heads"]:
            plan_ok += 1
    final, sim, exact = score_breakdown(preds, truths)
    return {"score": final, "sim": sim, "exact": exact,
            "plan_acc": plan_ok / max(len(rows), 1)}, preds, truths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--eval-limit", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=25.0)
    ap.add_argument("--set", action="append", default=[], help="cfg override k=v")
    ap.add_argument("--tag", default="dev")
    ap.add_argument("--kind", default="freq", choices=["freq", "rare"])
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = dict(DEFAULT_CFG)
    cfg.update(src_vocab=6000, tgt_vocab=1500, epochs=args.epochs, seed=args.seed)
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = type(cfg[k])(v) if k in cfg and not isinstance(cfg[k], bool) else float(v)
    print("CFG", json.dumps({k: v for k, v in cfg.items() if k != "total_steps"}), flush=True)

    rows = read_csv("data/train.csv")
    comp = (make_frequent_pair_folds(rows, 5, 6)[args.fold] if args.kind == "freq"
            else make_compositional_folds(rows, 6, 180, 0)[args.fold])
    rnd = random_fold(rows, 0.08, seed=7)

    tr_rows = [rows[i] for i in comp["train"]]
    comp_val = [rows[i] for i in comp["val"]]
    # random-split check uses the same training subset minus its own val rows
    rnd_val = [rows[i] for i in rnd["val"] if i in set(comp["train"])][: args.eval_limit]
    tr_rows2 = [r for r in tr_rows if r not in rnd_val]

    spacing = SpacingModel().fit([r["output"] for r in tr_rows2])
    device = torch.device("cpu")
    model, ex, src_v, tgt_v, head_v = build(tr_rows2, cfg, args.seed)
    model.to(device)
    print(f"train rows {len(tr_rows2)} | comp val {len(comp_val)} | rand val {len(rnd_val)} | "
          f"src_v {len(src_v)} tgt_v {len(tgt_v)} head_v {len(head_v)} | "
          f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    t0 = time.time()
    deadline = t0 + args.minutes * 60

    def ev(m, epoch):
        c, _, _ = evaluate(m, comp_val, src_v, tgt_v, head_v, spacing, device, args.eval_limit)
        r, _, _ = evaluate(m, rnd_val, src_v, tgt_v, head_v, spacing, device, args.eval_limit)
        o, _, _ = evaluate(m, comp_val, src_v, tgt_v, head_v, spacing, device,
                           args.eval_limit, oracle_plan=True)
        return {"comp": {k: round(v, 4) for k, v in c.items()},
                "rand": {k: round(v, 4) for k, v in r.items()},
                "comp_oracle_plan": {k: round(v, 4) for k, v in o.items()}}

    def log(rec):
        rec["t"] = round(time.time() - t0, 1)
        print(json.dumps(rec), flush=True)

    hist = train(model, ex, cfg, deadline=deadline, log=log, eval_fn=ev,
                 eval_every=args.eval_every)
    final = ev(model, -1)
    m2, _, _ = evaluate(model, comp_val, src_v, tgt_v, head_v, spacing, device,
                        args.eval_limit, min_stages=2)
    final["comp_min2"] = {k: round(v, 4) for k, v in m2.items()}
    if args.kind == "freq":
        final["held_pairs"] = comp["held_pairs"]
    print("FINAL", args.tag, json.dumps(final), flush=True)
    import os
    os.makedirs("ckpt", exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": cfg}, f"ckpt/{args.tag}.pt")
    _, preds, truths = evaluate(model, comp_val, src_v, tgt_v, head_v, spacing, device, 12)
    for p, t in zip(preds, truths):
        print("  PRED:", p, "\n  GOLD:", t)


if __name__ == "__main__":
    main()
