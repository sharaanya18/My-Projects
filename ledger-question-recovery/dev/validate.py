import argparse, time, itertools, pickle
import numpy as np
from common import *

def witness_acc(wf, exs, labels, w_un, w_out):
    hits, preds = 0, {}
    for i, e in enumerate(exs):
        pl, ul, R, prs = wf[i]
        ch = L.choose_subset(pl, ul, R, prs, w_un, w_out)
        preds[i] = ch
        gold = tuple(sorted(int(x[1:]) for x in labels[e["id"]]["witnesses"].split()))
        hits += (ch == gold)
    return hits / len(exs), preds

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--set", type=str, default="")
    ap.add_argument("--tag", type=str, default="run")
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in filter(None, a.set.split(",")):
        key, val = kv.split("=")
        cfg[key] = bool(int(val)) if isinstance(CFG[key], bool) else type(CFG[key])(float(val))
    rows, labels, terows, tedf = load()
    assign = folds_by_group(rows, a.k)
    trrows = [r for r in rows if assign[r["group"]] != a.fold]
    varows = [r for r in rows if assign[r["group"]] == a.fold]
    print(f"fold {a.fold}: train={len(trrows)} val={len(varows)} rows", flush=True)
    svoc, tvoc = build_vocabs(trrows, labels, cfg)
    trex = L.build_examples(trrows, cfg, svoc, tvoc, labels)
    vaex = L.build_examples(varows, cfg, svoc, tvoc, labels)
    t0 = time.time()
    init = L.Model(cfg, len(svoc), len(tvoc), 20260920)
    if cfg.get("pretrain"):
        L.pretrain_encoder(init, trex, cfg, 20260920 + 4242,
                           log=lambda s: print(s, flush=True) if int(s.split('/')[0].split()[-1]) % 4 == 0 else None)
        print("  mlm done %.1f min" % ((time.time() - t0) / 60), flush=True)
    model = L.train_model(trex, cfg, len(svoc), len(tvoc), 20260920, init=init,
                          log=lambda s: print(s, flush=True) if int(s.split('/')[0].split()[-1]) % 6 == 0 else None)
    print("train %.1f min" % ((time.time() - t0) / 60), flush=True)

    wf = L.witness_forward(model, vaex)
    print("\n=== witness selection sweep ===", flush=True)
    best = None
    for w_un in (0.0, 0.25, 0.5, 1.0):
        for w_out in (0.0, 0.25, 0.5, 1.0):
            acc, _ = witness_acc(wf, vaex, labels, w_un, w_out)
            print(f"  w_un={w_un} w_out={w_out}: acc={acc:.4f}", flush=True)
            if best is None or acc > best[0]:
                best = (acc, w_un, w_out)
    print(f"  BEST witness acc={best[0]:.4f} (w_un={best[1]}, w_out={best[2]})", flush=True)
    cfg["sel_w_un"], cfg["sel_w_out"] = best[1], best[2]
    acc, preds = witness_acc(wf, vaex, labels, best[1], best[2])
    for nw in (2, 3, 4):
        sub = [i for i, e in enumerate(vaex) if e["R"] - 2 == nw]
        if sub:
            h = sum(preds[i] == tuple(sorted(int(x[1:]) for x in labels[vaex[i]["id"]]["witnesses"].split())) for i in sub)
            print(f"    nw={nw}: n={len(sub)} acc={h/len(sub):.3f}")

    wmasks = {}
    for i, e in enumerate(vaex):
        m = np.zeros(e["R"], dtype=np.float32)
        for j in preds[i]: m[j] = 1.0
        wmasks[i] = m
    t0 = time.time()
    pools = L.generate_all(model, vaex, cfg, tvoc, wmasks)
    print("generate %.1fs" % (time.time() - t0), flush=True)
    pickle.dump({"pools": pools, "preds": preds, "ids": [e["id"] for e in vaex],
                 "acc": acc, "sel": (best[1], best[2])}, open(f"pools_{a.tag}_{a.fold}.pkl", "wb"))

    print("\n=== MBR sweep ===", flush=True)
    golds = [labels[e["id"]]["question"] for e in vaex]
    Wv = np.array([1.0 if preds[i] == tuple(sorted(int(x[1:]) for x in labels[vaex[i]["id"]]["witnesses"].split())) else 0.0
                   for i in range(len(vaex))])
    rowsout = []
    for ln in (0.0, 0.5, 0.8, 1.1):
        for tau in (0.5, 1.0, 3.0, 10.0, 1e6):
            c2 = dict(cfg); c2["len_norm"] = ln; c2["mbr_tau"] = tau
            qs = [L.mbr_select(pools[i], c2) for i in range(len(vaex))]
            Q = np.array([L.qscore(q, g) for q, g in zip(qs, golds)])
            rowsout.append((float((Wv*Q).mean()), ln, tau, float(Q.mean()), np.mean([len(q.split()) for q in qs])))
            print(f"  len_norm={ln} tau={tau}: Q(all)={Q.mean():.4f} FINAL={(Wv*Q).mean():.4f} "
                  f"words={np.mean([len(q.split()) for q in qs]):.1f}", flush=True)
    # top-1 beam (no MBR) reference
    qs1 = [pools[i][0][2] for i in range(len(vaex))]
    Q1 = np.array([L.qscore(q, g) for q, g in zip(qs1, golds)])
    print(f"  [beam top-1 no MBR]: Q(all)={Q1.mean():.4f} FINAL={(Wv*Q1).mean():.4f} "
          f"words={np.mean([len(q.split()) for q in qs1]):.1f}")
    rowsout.sort(reverse=True)
    print(f"\n>>> fold {a.fold} BEST FINAL={rowsout[0][0]:.4f} (len_norm={rowsout[0][1]}, tau={rowsout[0][2]}, "
          f"Q={rowsout[0][3]:.4f}, wAcc={acc:.4f})")
    c2 = dict(cfg); c2["len_norm"] = rowsout[0][1]; c2["mbr_tau"] = rowsout[0][2]
    print("\nsamples:")
    for i in range(min(8, len(vaex))):
        q = L.mbr_select(pools[i], c2)
        print(f"  W={Wv[i]:.0f} Q={L.qscore(q, golds[i]):.3f} pred={q!r}\n           gold={golds[i]!r}")
