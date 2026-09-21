"""Final feature-set decision, multi-seed, both views."""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from data import load
from features import build
from models import GainRegressor
from validate import (template_clusters, cluster_shift_folds, oof_prefolded,
                      paired_ndcg, testlike_cluster_holdout)
TR,TE=load(); y=TR.target.values; p_test=np.load('working/p_test.npy')

MENTION=['body_mention_count','body_mention_band']
DIGITS =['title_digit_count','body_digit_count','title_digit_rate','body_digit_rate']
VARIANTS={
 'V1_noMention':            dict(drop=MENTION),
 'V2_noMention_noDigits':   dict(drop=MENTION+DIGITS),
 'V3_noMention_noRawDigit': dict(drop=MENTION+['title_digit_count','body_digit_count']),
 'V4_all':                  dict(drop=[]),
}
Xs={k:build(TR,TE,**kw)[0] for k,kw in VARIANTS.items()}
for k,X in Xs.items(): print(f"  {k:26s} {X.shape[1]} feats")
REF='V1_noMention'
cv={k:[] for k in VARIANTS}; ho={k:[] for k in VARIANTS}
for seed in [0,1,2]:
    for nc in [60,100,160]:
        g=template_clusters(TR,nc,seed=seed); f=cluster_shift_folds(g,p_test,5)
        o={k:oof_prefolded(GainRegressor(num_leaves=7,min_child_samples=40),Xs[k],y,f) for k in VARIANTS}
        h={}
        for k in VARIANTS:
            va,pr=testlike_cluster_holdout(GainRegressor(num_leaves=7,min_child_samples=40),
                                           Xs[k],y,g,p_test,frac=0.30)
            h[k]=(va,pr)
        for k in VARIANTS:
            cv[k].append(paired_ndcg(y,o[k],o[REF],draws=250,seed=seed)['delta'])
            va=h[REF][0]
            ho[k].append(paired_ndcg(y[va],h[k][1],h[REF][1],n=min(586,int(len(va)*.85)),
                                     draws=250,seed=seed)['delta'])
print(f"\n{'variant':26s} {'CV delta vs V1':>22s} {'HOLDOUT delta vs V1':>24s}")
rows=[]
for k in VARIANTS:
    c=np.array(cv[k]); hh=np.array(ho[k])
    print(f"  {k:26s} {c.mean():+.4f}+/-{c.std():.4f} {int((c>0).sum())}/9"
          f"    {hh.mean():+.4f}+/-{hh.std():.4f} {int((hh>0).sum())}/9")
    rows.append(dict(v=k,cv=c.mean(),cv_std=c.std(),cv_win=int((c>0).sum()),
                     ho=hh.mean(),ho_std=hh.std(),ho_win=int((hh>0).sum())))
pd.DataFrame(rows).to_csv('working/feature_final.csv',index=False)
