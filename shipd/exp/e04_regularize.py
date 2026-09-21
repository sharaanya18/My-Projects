"""Phase 10/13: regularisation strength under shift + project-disjointness.

KB (validation/distribution_shift.md): 'Under shift the CV-optimal complexity
is past the test-optimal complexity. Regularize harder than CV alone suggests.'
Test that claim here instead of assuming it.
"""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from data import load
from features import build
from models import GainRegressor
from validate import (template_clusters, cluster_shift_folds, oof_prefolded,
                      paired_ndcg, testlike_cluster_holdout, resampled_ndcg)
TR,TE=load(); y=TR.target.values; p_test=np.load('working/p_test.npy')
X,XT=build(TR,TE,drop=['body_mention_count','body_mention_band'])

CFG={
 'A_default   (leaves15,mcs20,ne400)': dict(),
 'B_shallow   (leaves7, mcs40)':       dict(num_leaves=7, min_child_samples=40),
 'C_veryshallow(leaves4,mcs60)':       dict(num_leaves=4, min_child_samples=60),
 'D_stumps    (depth2, mcs60)':        dict(num_leaves=3, max_depth=2, min_child_samples=60),
 'E_shallow+L2(leaves7,mcs40,l2=10)':  dict(num_leaves=7, min_child_samples=40, reg_lambda=10.0),
 'F_shallow+fewtrees(leaves7,ne150)':  dict(num_leaves=7, min_child_samples=40, n_estimators=150),
 'G_deep      (leaves31,mcs10)':       dict(num_leaves=31, min_child_samples=10),
}
SEEDS=[0,1,2]; NCS=[60,100,160]
cv_res={k:[] for k in CFG}; ho_res={k:[] for k in CFG}
for seed in SEEDS:
    for nc in NCS:
        g=template_clusters(TR,nc,seed=seed); f=cluster_shift_folds(g,p_test,5)
        oofs={k:oof_prefolded(GainRegressor(**kw),X,y,f) for k,kw in CFG.items()}
        holds={}
        for k,kw in CFG.items():
            va,pr=testlike_cluster_holdout(GainRegressor(**kw),X,y,g,p_test,frac=0.30)
            holds[k]=(va,pr)
        ref='A_default   (leaves15,mcs20,ne400)'
        for k in CFG:
            cv_res[k].append(paired_ndcg(y,oofs[k],oofs[ref],draws=250,seed=seed)['delta'])
            va=holds[ref][0]; assert (holds[k][0]==va).all()
            ho_res[k].append(paired_ndcg(y[va],holds[k][1],holds[ref][1],
                                         n=min(586,int(len(va)*.85)),draws=250,seed=seed)['delta'])
print(f"{'config':38s} {'CV delta':>20s} {'HOLDOUT delta':>22s}")
rows=[]
for k in CFG:
    c=np.array(cv_res[k]); h=np.array(ho_res[k])
    print(f"  {k:38s} {c.mean():+.4f}+/-{c.std():.4f} {int((c>0).sum())}/{len(c)}"
          f"   {h.mean():+.4f}+/-{h.std():.4f} {int((h>0).sum())}/{len(h)}")
    rows.append(dict(cfg=k,cv=c.mean(),cv_std=c.std(),cv_win=int((c>0).sum()),
                     ho=h.mean(),ho_std=h.std(),ho_win=int((h>0).sum())))
pd.DataFrame(rows).to_csv('working/reg_sweep.csv',index=False)
print("\nsaved working/reg_sweep.csv")
