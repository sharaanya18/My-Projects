"""
Dev-only validation harness (NOT part of the submission). Held-out-CLUSTER cross-validation.

Why not plain GroupKFold on gallery_id: galleries are already disjoint (no paragraph is shared), so
that only guards literal leakage. The private set is a later legislative term (new agenda, ministers,
partly new speakers), so folds here hold out whole clusters of topically similar galleries
(competitor_patterns.md B12). Cluster features come from the SAME frozen, pinned backbone and use
train paragraphs only. Clusters are packed into folds greedily by size.

Usage: python validate.py <public_dir> <out_json> [--cfg '{"epochs":2}'] [--folds 5] [--k 12]
                          [--cluster-seeds 0] [--frozen-only]
Prints one compact SUMMARY table so a GPU round trip is a single paste-back.
"""
import argparse
import gc
import json
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans

import solution as S


def cluster_folds(gal_ids, gal_vecs, nfolds, k, seed):
    lab = KMeans(k, n_init=10, random_state=seed).fit_predict(gal_vecs)
    sizes = np.bincount(lab, minlength=k)
    load, fold_of = [0] * nfolds, {}
    for c in np.argsort(-sizes, kind="stable"):
        f = int(np.argmin(load))
        load[f] += sizes[c]
        fold_of[int(c)] = f
    return {g: fold_of[int(l)] for g, l in zip(gal_ids, lab)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("out_json")
    ap.add_argument("--cfg", default="{}")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--cluster-seeds", default="0")
    ap.add_argument("--frozen-only", action="store_true")
    ap.add_argument("--only-folds", default="", help="comma list of fold indices to run (default: all)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = {**S.CFG, **json.loads(a.cfg)}
    device = torch.device(a.device)
    if device.type == "cuda":
        S.set_determinism(cfg["seed"], cfg["threads"])
    texts, train, _ = S.load_tables(a.public_dir)
    gals = S.build_galleries(train, with_answers=True)
    gid = sorted(gals)

    tok, base = S.load_model(device, cfg)
    ids = sorted({p for g in gals.values() for p in g["cands"]} | {q[1] for g in gals.values() for q in g["queries"]})
    frozen = dict(zip(ids, S.embed_many(base, tok, [texts[i] for i in ids], device, cfg)))
    gv = []
    for g in gid:  # gallery vector = mean Spanish query + mean Basque candidate (frozen), normalised
        v = torch.stack([frozen[q[1]] for q in gals[g]["queries"]]).mean(0) + torch.stack([frozen[c] for c in gals[g]["cands"]]).mean(0)
        gv.append((v / v.norm()).numpy())
    gv = np.array(gv)
    del base

    rows = []
    for seed in [int(s) for s in a.cluster_seeds.split(",")]:
        fold_of = cluster_folds(gid, gv, a.folds, a.k, seed)
        for f in range(a.folds):
            if a.only_folds and f not in [int(x) for x in a.only_folds.split(",")]:
                continue
            val_g = {g: gals[g] for g in gid if fold_of[g] == f}
            tr_g = {g: gals[g] for g in gid if fold_of[g] != f}
            assert not set(val_g) & set(tr_g)
            tok, model = S.load_model(device, cfg)
            ans = {q[0]: q[2] for g in val_g.values() for q in g["queries"]}
            def score(m):
                r = {}
                for g in sorted(val_g):
                    r.update(S.rank_gallery_queries(m, tok, val_g[g], texts, device, cfg))
                return S.mrr(r, ans)
            m0 = score(model)
            m1 = m0
            if not a.frozen_only:
                S.set_determinism(cfg["seed"], cfg["threads"]) if device.type == "cuda" else None
                model = S.fine_tune(model, tok, tr_g, texts, device, cfg, log=lambda s: print(f"  [seed {seed} fold {f}] {s}", flush=True))
                m1 = score(model)
            rows.append(dict(cluster_seed=seed, fold=f, n_val_galleries=len(val_g), n_val_queries=len(ans), frozen=m0, tuned=m1))
            print(f"seed {seed} fold {f}: val galleries {len(val_g)} queries {len(ans)} frozen {m0:.4f} tuned {m1:.4f}", flush=True)
            del model
            gc.collect()  # optimizer/scheduler form reference cycles; free them or the next fold OOMs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fr, tu = np.array([r["frozen"] for r in rows]), np.array([r["tuned"] for r in rows])
    print("\nSUMMARY  cfg=" + json.dumps({k: cfg[k] for k in ("model", "epochs", "lr", "max_len", "queries_per_step", "temperature")}))
    print("seed fold  nq   frozen   tuned   delta")
    for r in rows:
        print(f"{r['cluster_seed']:>4} {r['fold']:>4} {r['n_val_queries']:>4}  {r['frozen']:.4f}  {r['tuned']:.4f}  {r['tuned'] - r['frozen']:+.4f}")
    print(f"MEAN frozen {fr.mean():.4f} (sd {fr.std():.4f}) | tuned {tu.mean():.4f} (sd {tu.std():.4f}) | delta {np.mean(tu - fr):+.4f}")
    json.dump(dict(cfg=cfg, rows=rows), open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
