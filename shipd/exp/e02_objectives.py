"""Phase 8: which surrogate objective best serves NDCG@20?

The metric is one global ranked list with gain 2^r-1 and cutoff 20. Candidate
surrogates differ in what they spend capacity on.
"""
import sys; sys.path.insert(0,'src'); sys.path.insert(0,'../nlp_knowledge/code')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.base import BaseEstimator, clone
from scipy.stats import spearmanr
from data import load
from features import build, FINGERPRINT, SHIFTED
from validate import (oof_prefolded, resampled_ndcg, report_paired,
                      testlike_cluster_holdout, _score_of)

TR,TE=load(); y=TR.target.values
groups=np.load('working/groups.npy'); p_test=np.load('working/p_test.npy')
folds=np.load('working/folds_cs.npy')
X,XT=build(TR,TE,drop=['body_mention_count','body_mention_band'])   # adopted from ablation
print(f"features: {X.shape[1]}")

BASE=dict(n_estimators=400,learning_rate=0.05,num_leaves=15,min_child_samples=20,
          subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,random_state=0)

class GainRegressor(BaseEstimator):
    """Regress the metric's own gain, 2^r - 1, instead of the raw label."""
    def __init__(self,**kw): self.kw=kw or BASE
    def fit(self,X,y):
        self.m_=lgb.LGBMRegressor(**self.kw).fit(X,2.0**np.asarray(y,float)-1.0); return self
    def predict(self,X): return self.m_.predict(X)

class P2Classifier(BaseEstimator):
    """P(active). NDCG@20 is dominated by getting 2s to the top."""
    def __init__(self,**kw): self.kw=kw or BASE
    def fit(self,X,y):
        self.m_=lgb.LGBMClassifier(**self.kw).fit(X,(np.asarray(y)==2).astype(int)); return self
    def predict(self,X): return self.m_.predict_proba(X)[:,1]

class ExpGainClassifier(BaseEstimator):
    """3-class softmax -> E[2^r - 1] = P1*1 + P2*3. Matches the gain exactly."""
    def __init__(self,**kw): self.kw=kw or BASE
    def fit(self,X,y):
        self.m_=lgb.LGBMClassifier(objective='multiclass',num_class=3,**self.kw).fit(X,y)
        return self
    def predict(self,X):
        P=self.m_.predict_proba(X); return P[:,1]*1.0+P[:,2]*3.0

class GlobalRanker(BaseEstimator):
    """LGBMRanker lambdarank, ONE group = the whole list, eval_at=20.
    This is the only objective that optimises the actual metric shape."""
    def __init__(self,**kw): self.kw=kw or BASE
    def fit(self,X,y):
        kw=dict(self.kw); kw.pop('subsample',None); kw.pop('subsample_freq',None)
        self.m_=lgb.LGBMRanker(objective='lambdarank',metric='ndcg',eval_at=[20],
                               label_gain=[0,1,3],**kw)
        self.m_.fit(X,y,group=[len(y)]); return self
    def predict(self,X): return self.m_.predict(X)

MODELS={
 'reg_label':        lgb.LGBMRegressor(**BASE),
 'reg_gain(2^r-1)':  GainRegressor(),
 'clf_P(r=2)':       P2Classifier(),
 'clf_E[gain]':      ExpGainClassifier(),
 'ranker_lambdarank':GlobalRanker(),
}
print("\n=== objective comparison ===")
print(f"{'model':22s} {'cluster-CV':>20s}   {'test-like-cluster holdout':>26s}")
oofs={}; hold={}
for k,mdl in MODELS.items():
    oofs[k]=oof_prefolded(mdl,X,y,folds)
    gm,gs=resampled_ndcg(y,oofs[k])
    va,pr=testlike_cluster_holdout(mdl,X,y,groups,p_test); hold[k]=(va,pr)
    hm,hs=resampled_ndcg(y[va],pr,n=min(586,int(len(va)*0.8)))
    print(f"  {k:22s} {gm:.4f}+/-{gs:.4f}      {hm:.4f}+/-{hs:.4f}  (n_va={len(va)}, sp {spearmanr(pr,y[va]).statistic:+.3f})")

print("\n  paired vs reg_label (cluster-CV):")
for k in MODELS:
    if k!='reg_label': report_paired(y,oofs[k],oofs['reg_label'],f"{k:18s}","reg_label")
va0=hold['reg_label'][0]
print("\n  paired vs reg_label (test-like-cluster holdout):")
for k in MODELS:
    if k=='reg_label': continue
    assert (hold[k][0]==va0).all()
    report_paired(y[va0],hold[k][1],hold['reg_label'][1],f"{k:18s}","reg_label",n=min(586,int(len(va0)*0.8)))
np.savez('working/oofs_objectives.npz',**oofs)
np.savez('working/hold_objectives.npz',va=va0,**{k:hold[k][1] for k in hold})
