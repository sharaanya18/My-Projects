"""Selection with signs LOCKED to causal priors.

The unconstrained greedy search reached 0.858 by assigning implausible signs
(file_count negative, body_question_count negative). That is the 0.247 failure
mode one level up: the search fits project structure and no validation built
from training data can detect it.

So the sign of every feature is fixed a priori by a mechanism argument, and the
search may only choose WHICH causal signals to include -- never their direction.
"""
import sys; sys.path.insert(0,'v2'); sys.path.insert(0,'src')
import numpy as np, pandas as pd, json
from data import load, num_cols
from validation import eval_ndcg, region_folds, sample_idx

TR,TE=load(); y=TR.target.values; NUM=num_cols(TR)

# sign fixed by mechanism, NOT by the data
CAUSAL_SIGNS = {
    'test_file_count':        +1,   # touching tests invites review of the tests
    'code_file_count':        +1,   # real code change
    'changed_lines':          +1,   # more to read
    'additions':              +1,
    'deletions':              +1,
    'max_file_changes':       +1,   # one large file is hard to review
    'max_file_additions':     +1,
    'max_file_deletions':     +1,
    'file_count':             +1,   # more files, more surface
    'body_code_block_count':  +1,   # code in the description invites discussion
    'body_question_count':    +1,   # the author is asking for input
    'docs_file_count':        -1,   # docs-only backports get rubber-stamped
}
POOL=list(CAUSAL_SIGNS)
L={c: np.log1p(TR[c].astype(float).clip(lower=0).to_numpy()) for c in POOL}
Z={c:(L[c]-L[c].mean())/(L[c].std()+1e-9) for c in POOL}
IDXS=[sample_idx(len(y),300,seed=s) for s in range(3)]

def score(cols):
    s=np.zeros(len(y))
    for c in cols: s+=CAUSAL_SIGNS[c]*Z[c]
    return float(np.mean([eval_ndcg(y,s,idx=I)[0] for I in IDXS]))

print("single-feature scores (sign locked to the causal prior):")
for c in sorted(POOL,key=lambda c:-score([c])):
    print(f"  {c:24s} {'+' if CAUSAL_SIGNS[c]>0 else '-'}  {score([c]):.4f}")

chosen=[]; best=0.0
print("\ngreedy forward selection, signs locked:")
for _ in range(8):
    cand=sorted(((score(chosen+[c]),c) for c in POOL if c not in chosen),reverse=True)
    sc,c=cand[0]
    if sc<=best+0.003:
        print(f"  stop ({c} would give {sc:.4f} vs {best:.4f})"); break
    chosen.append(c); best=sc
    print(f"  + {c:24s} -> {best:.4f}")
print(f"\nCAUSAL COMPOSITE ({len(chosen)} feats): {chosen}\n  score {best:.4f}")
json.dump({c:CAUSAL_SIGNS[c] for c in chosen},open('working/causal_composite.json','w'))
