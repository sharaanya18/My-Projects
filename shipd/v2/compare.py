"""Compare candidates across every view, and expose the memorisation premium."""
import sys; sys.path.insert(0,'v2'); sys.path.insert(0,'src')
import numpy as np, pandas as pd
from data import load, num_cols
from ndcg import ndcg_at_k
from validation import eval_ndcg, extrapolation_split, region_folds, sample_idx
from candidates import UnfittedComposite, MonotoneGBM, PlainGBM, EXCLUDED_STYLE

TR,TE=load(); y=TR.target.values; NUM=num_cols(TR)

CANDS={
 'A unfitted_composite'      : lambda: UnfittedComposite(),
 'B monotone_gbm(causal,13)' : lambda: MonotoneGBM(),
 'C monotone_gbm(tiny)'      : lambda: MonotoneGBM(num_leaves=3,min_child_samples=120,n_estimators=200),
 'D plain_gbm(no style)'     : lambda: PlainGBM(exclude_style=True),
 'E plain_gbm(ALL numeric)'  : lambda: PlainGBM(exclude_style=False),
}

def fitpred(mk,tr_idx,va_idx):
    m=mk().fit(TR.iloc[tr_idx],y[tr_idx]); return m.predict(TR.iloc[va_idx])

print("="*78); print("VIEW 1 -- REGION HOLDOUT (whole KMeans regions of feature space held out)")
print("="*78)
folds,_=region_folds(TR[NUM].astype(float),n_regions=40,n_folds=5,seed=0)
res={}
for k,mk in CANDS.items():
    oof=np.zeros(len(y))
    for f in np.unique(folds):
        va=np.where(folds==f)[0]; tr=np.where(folds!=f)[0]
        oof[va]=fitpred(mk,tr,va)
    m,s=eval_ndcg(y,oof,draws=500)
    ins=mk().fit(TR,y).predict(TR); im,_=eval_ndcg(y,ins,draws=500)
    res[k]=dict(region=m,region_std=s,insample=im,gap=im-m)
    print(f"  {k:28s} OOF {m:.4f}+/-{s:.4f}   in-sample {im:.4f}   GAP {im-m:+.4f}")

print("\n"+"="*78); print("VIEW 2 -- EXTRAPOLATION (train on simple half, score complex half)")
print("="*78)
tr,va=extrapolation_split(TR,frac=0.35)
idx=sample_idx(len(va),400,seed=1)
for k,mk in CANDS.items():
    p=fitpred(mk,tr,va); m,s=eval_ndcg(y[va],p,idx)
    res[k]['extrap']=m
    print(f"  {k:28s} {m:.4f} +/- {s:.4f}")

print("\n"+"="*78); print("VIEW 3 -- MEMORISATION PREMIUM (fitted score minus the unfitted rule)")
print("="*78)
base=res['A unfitted_composite']['region']
print("  Premium = how much of a model's score comes from FITTING rather than from")
print("  the causal relationship. The 71-feature model's premium did not transfer.")
for k in CANDS:
    print(f"  {k:28s} region {res[k]['region']:.4f}   premium {res[k]['region']-base:+.4f}   gap {res[k]['gap']:+.4f}")

print("\n"+"="*78); print("VIEW 4 -- SEED STABILITY on the real test set (determinism + variance)")
print("="*78)
for k,mk in CANDS.items():
    ps=[]
    for s in range(3):
        m=mk(); m.fit(TR,y); ps.append(m.predict(TE))
    ps=np.array(ps)
    top=[set(np.argsort(-p,kind='mergesort')[:20]) for p in ps]
    ov=len(top[0]&top[1]&top[2])/20
    print(f"  {k:28s} top-20 overlap across refits: {ov:.0%}")
pd.DataFrame(res).T.to_csv('working/v2_compare.csv')
