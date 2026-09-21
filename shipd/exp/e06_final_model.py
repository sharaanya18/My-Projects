"""Final model selection: seed-averaging and blending, on both views."""
import sys; sys.path.insert(0,'src')
import numpy as np, pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from data import load
from features import build
from models import GainRegressor, GlobalRanker
from validate import (template_clusters, cluster_shift_folds, oof_prefolded,
                      paired_ndcg, testlike_cluster_holdout)
TR,TE=load(); y=TR.target.values; p_test=np.load('working/p_test.npy')
X,XT=build(TR,TE,drop=['body_mention_count','body_mention_band'])
r01=lambda v: rankdata(v)/len(v)

class SeedAvg(BaseEstimator):
    """Average several seeds. Pure variance reduction; the metric is noisy enough
    that this is worth more here than it usually is."""
    def __init__(self, n_seeds=10, **kw): self.n_seeds=n_seeds; self.kw=kw
    def get_params(self, deep=True): return dict(self.kw, n_seeds=self.n_seeds)
    def set_params(self, **kw):
        self.n_seeds=kw.pop('n_seeds',self.n_seeds); self.kw.update(kw); return self
    def fit(self,X,y):
        self.ms_=[GainRegressor(random_state=s,**self.kw).fit(X,y) for s in range(self.n_seeds)]
        return self
    def predict(self,X):
        return np.mean([r01(m.predict(X)) for m in self.ms_],axis=0)

class Blend(BaseEstimator):
    def __init__(self,w=0.5,n_seeds=5,**kw): self.w=w; self.n_seeds=n_seeds; self.kw=kw
    def get_params(self,deep=True): return dict(self.kw,w=self.w,n_seeds=self.n_seeds)
    def set_params(self,**kw):
        self.w=kw.pop('w',self.w); self.n_seeds=kw.pop('n_seeds',self.n_seeds)
        self.kw.update(kw); return self
    def fit(self,X,y):
        self.a_=SeedAvg(n_seeds=self.n_seeds,**self.kw).fit(X,y)
        self.b_=[GlobalRanker(truncation=4000,random_state=s,
                              num_leaves=self.kw.get('num_leaves',15),
                              min_child_samples=self.kw.get('min_child_samples',20)).fit(X,y)
                 for s in range(self.n_seeds)]
        return self
    def predict(self,X):
        rb=np.mean([r01(m.predict(X)) for m in self.b_],axis=0)
        return self.w*r01(self.a_.predict(X))+(1-self.w)*rb

SH=dict(num_leaves=7,min_child_samples=40)
CAND={
 'M1_single_leaves15':  GainRegressor(),
 'M2_single_shallow':   GainRegressor(**SH),
 'M3_seedavg10_shallow':SeedAvg(n_seeds=10,**SH),
 'M4_seedavg10_l15':    SeedAvg(n_seeds=10),
 'M5_blend70_shallow':  Blend(w=0.7,n_seeds=5,**SH),
}
REF='M2_single_shallow'
cv={k:[] for k in CAND}; ho={k:[] for k in CAND}
for seed in [0,1,2,3]:
    for nc in [60,100,160]:
        g=template_clusters(TR,nc,seed=seed); f=cluster_shift_folds(g,p_test,5)
        o={k:oof_prefolded(m,X,y,f) for k,m in CAND.items()}
        h={k:testlike_cluster_holdout(m,X,y,g,p_test,frac=0.30) for k,m in CAND.items()}
        for k in CAND:
            cv[k].append(paired_ndcg(y,o[k],o[REF],draws=250,seed=seed)['delta'])
            va=h[REF][0]
            ho[k].append(paired_ndcg(y[va],h[k][1],h[REF][1],n=min(586,int(len(va)*.85)),
                                     draws=250,seed=seed)['delta'])
print(f"\n{'candidate':24s} {'CV delta vs M2':>22s} {'HOLDOUT delta vs M2':>24s}")
rows=[]
for k in CAND:
    c=np.array(cv[k]); hh=np.array(ho[k]); n=len(c)
    print(f"  {k:24s} {c.mean():+.4f}+/-{c.std():.4f} {int((c>0).sum())}/{n}"
          f"    {hh.mean():+.4f}+/-{hh.std():.4f} {int((hh>0).sum())}/{n}")
    rows.append(dict(cand=k,cv=c.mean(),cv_std=c.std(),cv_win=int((c>0).sum()),
                     ho=hh.mean(),ho_std=hh.std(),ho_win=int((hh>0).sum()),n=n))
pd.DataFrame(rows).to_csv('working/final_model.csv',index=False)
