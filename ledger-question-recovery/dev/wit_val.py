"""Fast witness-only experiment: does MLM pretraining close the gap?"""
import argparse, time
import numpy as np
from common import *

def wacc(model, exs, labels, w_un=0.0, w_out=0.0):
    wf = L.witness_forward(model, exs)
    hit = 0
    per = {}
    for i, e in enumerate(exs):
        pl, ul, R, prs = wf[i]
        ch = L.choose_subset(pl, ul, R, prs, w_un, w_out)
        gold = tuple(sorted(int(x[1:]) for x in labels[e["id"]]["witnesses"].split()))
        ok = (ch == gold); hit += ok
        per.setdefault(R, [0, 0]); per[R][0] += ok; per[R][1] += 1
    return hit / len(exs), per

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=str, default="0")
    ap.add_argument("--set", type=str, default="")
    a = ap.parse_args()
    cfg = dict(CFG); cfg["witness_only"] = True; cfg["epochs"] = 20
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        cfg[k] = bool(int(v)) if isinstance(cfg.get(k), bool) else type(cfg.get(k, 0.0))(float(v))
    rows, labels, terows, tedf = load()
    assign = folds_by_group(rows, 4)
    accs = []
    for fold in [int(x) for x in a.folds.split(",")]:
        tr = [r for r in rows if assign[r["group"]] != fold]
        va = [r for r in rows if assign[r["group"]] == fold]
        svoc, tvoc = build_vocabs(tr, labels, cfg)
        trex = L.build_examples(tr, cfg, svoc, tvoc, labels)
        vaex = L.build_examples(va, cfg, svoc, tvoc, labels)
        t0 = time.time()
        model = L.Model(cfg, len(svoc), len(tvoc), 20260920)
        if cfg["pretrain"]:
            L.pretrain_encoder(model, trex, cfg, 4242,
                               log=lambda s: print(s, flush=True))
            print("  mlm %.1f min" % ((time.time() - t0) / 60), flush=True)
        opt_model = L.train_model(trex, cfg, len(svoc), len(tvoc), 20260920,
                                  log=lambda s: print(s, flush=True) if int(s.split('/')[0].split()[-1]) % 5 == 0 else None,
                                  init=model)
        acc, per = wacc(opt_model, vaex, labels)
        accs.append(acc)
        print(f"fold {fold}: witness acc = {acc:.4f}  ({time.time()-t0:.0f}s)  "
              + " ".join(f"R{R}:{v[0]}/{v[1]}" for R, v in sorted(per.items())), flush=True)
    print(f"\nMEAN witness acc over folds {a.folds} = {np.mean(accs):.4f}")
