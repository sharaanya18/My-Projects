"""Final head-to-head. Every candidate has causally-locked directions."""
import sys; sys.path.insert(0,'v2'); sys.path.insert(0,'src')
import numpy as np, pandas as pd, lightgbm as lgb
from data import load, num_cols
from validation import eval_ndcg, region_folds, extrapolation_split, sample_idx
from select_causal import CAUSAL_SIGNS

TR,TE=load(); y=TR.target.values; NUM=num_cols(TR)

def logz(df, c, stats):
    v=np.log1p(df[c].astype(float).clip(lower=0).to_numpy())
    mu,sd=stats[c]; return (v-mu)/sd

def make_stats(df,cols):
    out={}
    for c in cols:
        v=np.log1p(df[c].astype(float).clip(lower=0).to_numpy())
        out[c]=(v.mean(),v.std()+1e-9)
    return out

class Composite:
    def __init__(self,cols): self.cols=list(cols)
    def fit(self,df,y=None): self.st_=make_stats(df,self.cols); return self
    def predict(self,df):
        return sum(CAUSAL_SIGNS[c]*logz(df,c,self.st_) for c in self.cols)

class MonoGBM:
    def __init__(self,cols,num_leaves=4,mcs=60,n_est=300,seeds=5):
        self.cols=list(cols); self.kw=dict(num_leaves=num_leaves,min_child_samples=mcs,
            n_estimators=n_est,learning_rate=0.05,verbose=-1); self.seeds=seeds
    def _X(self,df): return np.column_stack([np.log1p(df[c].astype(float).clip(lower=0)) for c in self.cols])
    def fit(self,df,yy):
        mono=[CAUSAL_SIGNS[c] for c in self.cols]
        g=2.0**np.asarray(yy,float)-1.0
        self.ms_=[lgb.LGBMRegressor(random_state=s,monotone_constraints=mono,**self.kw).fit(self._X(df),g)
                  for s in range(self.seeds)]; return self
    def predict(self,df): return np.mean([m.predict(self._X(df)) for m in self.ms_],axis=0)

class PlainGBM:
    def __init__(self,cols,seeds=5):
        self.cols=list(cols); self.seeds=seeds
    def fit(self,df,yy):
        g=2.0**np.asarray(yy,float)-1.0
        self.ms_=[lgb.LGBMRegressor(random_state=s,n_estimators=400,num_leaves=7,
                    min_child_samples=40,learning_rate=0.05,colsample_bytree=0.8,
                    subsample=0.8,subsample_freq=1,verbose=-1).fit(df[self.cols].astype(float),g)
                  for s in range(self.seeds)]; return self
    def predict(self,df): return np.mean([m.predict(df[self.cols].astype(float)) for m in self.ms_],axis=0)

ALL12=list(CAUSAL_SIGNS); TOP3=['test_file_count','max_file_deletions','docs_file_count']
CANDS={
 'C1 composite(3, greedy)'   : lambda: Composite(TOP3),
 'C2 composite(12, NO selection)': lambda: Composite(ALL12),
 'C3 monotone_gbm(12 causal)': lambda: MonoGBM(ALL12),
 'C4 monotone_gbm(3 causal)' : lambda: MonoGBM(TOP3),
 'C5 plain_gbm(12 causal)'   : lambda: PlainGBM(ALL12),
 'C6 plain_gbm(ALL numeric)' : lambda: PlainGBM(NUM),     # ~ the model that scored 0.247
}
rows={}
print("="*86)
print(f"{'candidate':32s} {'region-OOF':>12s} {'extrapol':>10s} {'in-sample':>10s} {'GAP':>8s} {'premium':>9s}")
print("="*86)
folds,_=region_folds(TR[NUM].astype(float),n_regions=40,n_folds=5,seed=0)
tr_e,va_e=extrapolation_split(TR,frac=0.35); idx_e=sample_idx(len(va_e),400,seed=1)
for k,mk in CANDS.items():
    oof=np.zeros(len(y))
    for f in np.unique(folds):
        va=np.where(folds==f)[0]; tr=np.where(folds!=f)[0]
        oof[va]=mk().fit(TR.iloc[tr],y[tr]).predict(TR.iloc[va])
    r,rs=eval_ndcg(y,oof,draws=500)
    ins,_=eval_ndcg(y,mk().fit(TR,y).predict(TR),draws=500)
    ex,_=eval_ndcg(y[va_e],mk().fit(TR.iloc[tr_e],y[tr_e]).predict(TR.iloc[va_e]),idx_e)
    rows[k]=dict(region=r,region_std=rs,extrap=ex,insample=ins,gap=ins-r)
    print(f"  {k:32s} {r:.4f}+/-{rs:.3f} {ex:9.4f} {ins:10.4f} {ins-r:+8.4f}")
base=rows['C2 composite(12, NO selection)']['region']
for k in rows: rows[k]['premium']=rows[k]['region']-base
print("\npremium over the zero-selection composite (the part at risk of not transferring):")
for k in rows: print(f"  {k:32s} {rows[k]['premium']:+.4f}")
pd.DataFrame(rows).T.to_csv('working/v2_final_compare.csv')

print("\n=== agreement of the TEST top-20 between candidates ===")
preds={k:mk().fit(TR,y).predict(TE) for k,mk in CANDS.items()}
ks=list(CANDS)
top={k:set(np.argsort(-preds[k],kind='mergesort')[:20]) for k in ks}
print(f"  {'':32s}"+"".join(f"{k.split()[0]:>6s}" for k in ks))
for a in ks:
    print(f"  {a:32s}"+"".join(f"{len(top[a]&top[b])/20:6.0%}" for b in ks))
