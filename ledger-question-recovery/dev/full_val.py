"""Group-held-out validation of the combined two-model system (the shipped configuration)."""
import argparse, json, time
import numpy as np
from common import *

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--k", type=int, default=4)
    a = ap.parse_args()
    cfg = json.load(open("final_config.json"))

    def sub(which):
        o = {k: v for k, v in cfg.items() if k not in ("wit", "qst")}
        o.update(cfg[which]); o.setdefault("gen_temp", 1.0); o.setdefault("len_norm", 0.8)
        o.setdefault("mbr_tau", 1.0); o.setdefault("witness_only", False)
        return o

    wcfg, qcfg = sub("wit"), sub("qst")
    rows, labels, terows, tedf = load()
    assign = folds_by_group(rows, a.k)
    tr = [r for r in rows if assign[r["group"]] != a.fold]
    va = [r for r in rows if assign[r["group"]] == a.fold]
    svoc, tvoc = build_vocabs(tr, labels, qcfg)
    trex = L.build_examples(tr, qcfg, svoc, tvoc, labels)
    vaex = L.build_examples(va, qcfg, svoc, tvoc, labels)
    t0 = time.time()
    wm = L.Model(wcfg, len(svoc), len(tvoc), 20260920)
    L.pretrain_encoder(wm, trex, wcfg, 20260920 + 4242)
    wm = L.train_model(trex, wcfg, len(svoc), len(tvoc), 20260920, init=wm)
    qm = L.Model(qcfg, len(svoc), len(tvoc), 20260920 + 101)
    L.pretrain_encoder(qm, trex, qcfg, 20260920 + 101 + 4242)
    qm = L.train_model(trex, qcfg, len(svoc), len(tvoc), 20260920 + 101, init=qm)
    print(f"fold {a.fold}: trained in {(time.time()-t0)/60:.1f} min", flush=True)

    rec = []
    byR = {}
    for i, e in enumerate(vaex):
        byR.setdefault(e["R"], []).append(i)
    for R in sorted(byR):
        idxs = byR[R]
        for s in range(0, len(idxs), 16):
            sel = idxs[s:s + 16]
            ch, qs, _ = L.predict_two(wm, qm, [vaex[i] for i in sel], wcfg, qcfg, tvoc)
            for k, i in enumerate(sel):
                lab = labels[vaex[i]["id"]]
                gold = tuple(sorted(int(x[1:]) for x in lab["witnesses"].split()))
                W = 1.0 if tuple(sorted(ch[k])) == gold else 0.0
                Q = L.qscore(qs[k], lab["question"])
                rec.append({"nw": len(gold), "W": W, "Q": Q, "S": W * Q,
                            "pred": qs[k], "gold": lab["question"]})
    n = len(rec)
    W = sum(r["W"] for r in rec) / n
    Q = sum(r["Q"] for r in rec) / n
    S = sum(r["S"] for r in rec) / n
    corr = [r for r in rec if r["W"] > 0]
    Qc = sum(r["Q"] for r in corr) / len(corr) if corr else 0.0
    print(f"FOLD {a.fold}  n={n}  witnessAcc={W:.4f}  Q(all)={Q:.4f}  Q(|correct)={Qc:.4f}  FINAL={S:.4f}", flush=True)
    for nw in (2, 3, 4):
        s2 = [r for r in rec if r["nw"] == nw]
        if s2:
            print(f"   nw={nw}: n={len(s2):3d} wAcc={sum(r['W'] for r in s2)/len(s2):.3f} "
                  f"Q={sum(r['Q'] for r in s2)/len(s2):.3f} S={sum(r['S'] for r in s2)/len(s2):.3f}")
    json.dump({"fold": a.fold, "n": n, "W": W, "Q": Q, "Qc": Qc, "S": S},
              open(f"result_fold{a.fold}.json", "w"))

if __name__ == "__main__":
    main()
