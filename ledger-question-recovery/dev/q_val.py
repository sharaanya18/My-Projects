"""Question-model experiment: score held-out Q with gold witnesses (isolates generation)."""
import argparse, time, pickle
import numpy as np
from common import *

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--set", type=str, default="")
    ap.add_argument("--tag", type=str, default="q")
    a = ap.parse_args()
    cfg = dict(CFG)
    for kv in filter(None, a.set.split(",")):
        k, v = kv.split("=")
        cfg[k] = bool(int(v)) if isinstance(cfg.get(k), bool) else type(cfg.get(k, 0.0))(float(v))
    rows, labels, terows, tedf = load()
    assign = folds_by_group(rows, 4)
    tr = [r for r in rows if assign[r["group"]] != a.fold]
    va = [r for r in rows if assign[r["group"]] == a.fold]
    svoc, tvoc = build_vocabs(tr, labels, cfg)
    trex = L.build_examples(tr, cfg, svoc, tvoc, labels)
    vaex = L.build_examples(va, cfg, svoc, tvoc, labels)
    t0 = time.time()
    init = L.Model(cfg, len(svoc), len(tvoc), 20260920)
    if cfg.get("pretrain"):
        L.pretrain_encoder(init, trex, cfg, 20260920 + 4242)
    model = L.train_model(trex, cfg, len(svoc), len(tvoc), 20260920, init=init,
                          log=lambda s: print(s, flush=True) if int(s.split('/')[0].split()[-1]) % 6 == 0 else None)
    print("train %.1f min" % ((time.time() - t0) / 60), flush=True)
    gold_w = [e["wmask"] for e in vaex]
    golds = [labels[e["id"]]["question"] for e in vaex]
    print("\n=== gen_temp x tau sweep (GOLD witnesses, so this is pure question quality) ===", flush=True)
    best = None
    for temp in (1.0, 1.5, 2.0, 3.0):
        c = dict(cfg); c["gen_temp"] = temp
        pools = {}
        for R in sorted({e["R"] for e in vaex}):
            idxs = [i for i, e in enumerate(vaex) if e["R"] == R]
            for s in range(0, len(idxs), 16):
                sel = idxs[s:s + 16]
                ch = [tuple(i for i in range(R) if gold_w[j][i] > 0.5) for j in sel]
                cds = L.generate_for(model, [vaex[j] for j in sel], c, tvoc, ch)
                for k, j in enumerate(sel):
                    pools[j] = cds[k]
        pickle.dump(pools, open(f"qpools_{a.tag}_{a.fold}_t{temp}.pkl", "wb"))
        for ln in (0.0, 0.6, 1.0):
            for tau in (0.5, 2.0, 10.0, 1e6):
                c2 = dict(c); c2["len_norm"] = ln; c2["mbr_tau"] = tau
                qs = [L.mbr_select(pools[i], c2) for i in range(len(vaex))]
                Q = float(np.mean([L.qscore(q, g) for q, g in zip(qs, golds)]))
                w = float(np.mean([len(q.split()) for q in qs]))
                if best is None or Q > best[0]:
                    best = (Q, temp, ln, tau)
                print(f"  temp={temp} len_norm={ln} tau={tau:g}: Q={Q:.4f} words={w:.1f}", flush=True)
    print(f"\n>>> fold {a.fold} BEST Q={best[0]:.4f} (temp={best[1]}, len_norm={best[2]}, tau={best[3]:g})")
    print(f"    reference: greedy hedge held-out Q = 0.1439")
    c2 = dict(cfg); c2["gen_temp"] = best[1]; c2["len_norm"] = best[2]; c2["mbr_tau"] = best[3]
    pools = pickle.load(open(f"qpools_{a.tag}_{a.fold}_t{best[1]}.pkl", "rb"))
    print("\nsamples:")
    for i in range(min(6, len(vaex))):
        q = L.mbr_select(pools[i], c2)
        print(f"  Q={L.qscore(q, golds[i]):.3f} pred={q!r}\n          gold={golds[i]!r}")
