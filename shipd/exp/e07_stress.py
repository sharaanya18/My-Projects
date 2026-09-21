"""Stress test: how low can an HONEST estimate of this model go?

The question is not "what does CV say" but "what is the pessimistic floor on a
project-disjoint, heavily-shifted holdout". Each design below is deliberately
harsher than the real test is likely to be. If the floor across all of them
clears the target, the target is safe; if not, say so.
"""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from data import load
from features import build
from models import GainRegressor
from validate import (template_clusters, cluster_shift_folds, oof_prefolded, sibling_groups,
                      resampled_ndcg, testlike_cluster_holdout, TEST_N)
from metric import ndcg_at_k

TR,TE=load(); y=TR.target.values; p_test=np.load('working/p_test.npy')
X,XT=build(TR,TE,drop=['body_mention_count','body_mention_band'])
SH=dict(num_leaves=7,min_child_samples=40)
mk=lambda: GainRegressor(**SH)
r01=lambda v: rankdata(v)/len(v)
rows=[]

def rec(design,score,std,note=""):
    rows.append(dict(design=design,ndcg=score,std=std,note=note))
    print(f"  {design:46s} {score:.4f} +/- {std:.4f}   {note}")

print("=== FLOORS ===")
rng=np.random.default_rng(0)
m,s=resampled_ndcg(y,rng.random(len(y))); rec("random predictions",m,s,"absolute floor")

print("\n=== A. Coarse pseudo-project GroupKFold (harsher as clusters coarsen) ===")
for nc in [200,100,50,25,12,6]:
    g=template_clusters(TR,nc,seed=0)
    f=cluster_shift_folds(g,p_test,5)
    o=oof_prefolded(mk(),X,y,f)
    m,s=resampled_ndcg(y,o); rec(f"GroupKFold, {nc} template clusters",m,s,
                                 f"largest cluster={pd.Series(g).value_counts().max()}")

print("\n=== B. Leave-one-BIG-cluster-out (each fold = one whole pseudo-project) ===")
g=template_clusters(TR,25,seed=0)
sizes=pd.Series(g).value_counts()
big=sizes[sizes>=60].index[:8]
sc=[]
for c in big:
    va=np.where(g==c)[0]; tr=np.where(g!=c)[0]
    mm=mk().fit(X.iloc[tr],y[tr]); pr=mm.predict(X.iloc[va])
    if len(va)>=25 and len(set(y[va]))>1:
        sc.append(ndcg_at_k(y[va],pr))
sc=np.array(sc); rec("leave-one-big-cluster-out (mean over 8)",sc.mean(),sc.std(),
                     f"per-cluster: {np.round(sc,3).tolist()}")

print("\n=== C. Shift-extreme: train on least test-like half, score most test-like half ===")
for frac in [0.5,0.4]:
    g=template_clusters(TR,60,seed=0)
    va,pr=testlike_cluster_holdout(mk(),X,y,g,p_test,frac=frac)
    n=min(TEST_N,int(len(va)*0.85))
    m,s=resampled_ndcg(y[va],pr,n=n); rec(f"test-like cluster holdout frac={frac}",m,s,f"n_va={len(va)}")

print("\n=== D. Adversarially reweighted scoring (up-weight the most test-like rows) ===")
g=template_clusters(TR,100,seed=0); f=cluster_shift_folds(g,p_test,5)
o=oof_prefolded(mk(),X,y,f)
w=p_test/p_test.sum()
rng=np.random.default_rng(3); sc=[]
for _ in range(400):
    i=rng.choice(len(y),TEST_N,replace=False,p=w)   # sample test-like rows preferentially
    sc.append(ndcg_at_k(y[i],o[i]))
sc=np.array(sc); rec("p_test-weighted resampling of OOF",sc.mean(),sc.std())

print("\n=== E. Worst observed single draw across every design above ===")
g=template_clusters(TR,25,seed=0); f=cluster_shift_folds(g,p_test,5)
o=oof_prefolded(mk(),X,y,f)
rng=np.random.default_rng(5)
draws=np.array([ndcg_at_k(y[i],o[i]) for i in (rng.choice(len(y),TEST_N,replace=False) for _ in range(2000))])
print(f"  coarse(25)-cluster OOF single-draw distribution:")
for q in [1,5,10,25,50]:
    print(f"     p{q:<3d} = {np.percentile(draws,q):.4f}")
print(f"     P(draw < 0.60) = {(draws<0.60).mean():.4%}")
rec("coarse-25 OOF, 5th percentile draw",float(np.percentile(draws,5)),0.0,"pessimistic single-draw")

pd.DataFrame(rows).to_csv('working/stress_test.csv',index=False)
print("\nsaved working/stress_test.csv")
allsc=[r['ndcg'] for r in rows if r['design']!='random predictions']
print(f"\n  MINIMUM across all honest designs: {min(allsc):.4f}")
