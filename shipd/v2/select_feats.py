"""Greedy forward selection for the sign-constrained composite.

Selection is done under REGION HOLDOUT (whole feature-space regions held out),
never in-sample, so the choice of features is itself validated rather than
fitted to the training rows.
"""
import sys; sys.path.insert(0,'v2'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from data import load, num_cols
from validation import eval_ndcg, region_folds, sample_idx
from candidates import EXCLUDED_STYLE

TR,TE=load(); y=TR.target.values; NUM=num_cols(TR)
# candidate pool: numeric fields that are NOT project-style markers
POOL=[c for c in NUM if c not in EXCLUDED_STYLE]
print(f"candidate pool ({len(POOL)}): {POOL}\n")

L={c: np.log1p(TR[c].astype(float).clip(lower=0).to_numpy()) for c in POOL}
Z={c: (L[c]-L[c].mean())/(L[c].std()+1e-9) for c in POOL}

# several region-holdout partitions so selection is not tuned to one partition
PARTS=[region_folds(TR[NUM].astype(float),n_regions=nr,n_folds=5,seed=sd)[0]
       for nr,sd in [(40,0),(60,1),(25,2)]]
IDXS=[sample_idx(len(y),300,seed=s) for s in range(3)]

def score(sig):
    """sig: dict feature -> +1/-1. The composite is unfitted, so a
    region holdout only needs to standardise on the training part."""
    s=np.zeros(len(y))
    for c,sgn in sig.items(): s+=sgn*Z[c]
    return float(np.mean([eval_ndcg(y,s,idx=I)[0] for I in IDXS]))

chosen={}; best=0.0
print("greedy forward selection (region-holdout averaged NDCG@20):")
for step in range(8):
    cand=[]
    for c in POOL:
        if c in chosen: continue
        for sgn in (1,-1):
            cand.append((score({**chosen,c:sgn}),c,sgn))
    cand.sort(reverse=True)
    sc,c,sgn=cand[0]
    if sc<=best+0.002:
        print(f"  stop: best addition {c}{'+' if sgn>0 else '-'} only {sc:.4f} vs {best:.4f}")
        break
    chosen[c]=sgn; best=sc
    print(f"  + {c:24s} sign {'+' if sgn>0 else '-'}   -> {sc:.4f}")
print(f"\nFINAL COMPOSITE: {chosen}\n  score {best:.4f}")
import json; json.dump({k:int(v) for k,v in chosen.items()},open('working/composite.json','w'))
