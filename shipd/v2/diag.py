"""Diagnose the 0.247: is within-project ranking fine but cross-project broken?"""
import sys; sys.path.insert(0,'v2'); sys.path.insert(0,'src')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from ndcg import ndcg_at_k
from data import load, CAT_COLS, num_cols
from features import build

TR,TE=load(); y=TR.target.values
X,XT=build(TR,TE,drop=['body_mention_count','body_mention_band'])
PARAMS=dict(n_estimators=400,learning_rate=0.05,num_leaves=7,min_child_samples=40,
            colsample_bytree=0.8,subsample=0.8,subsample_freq=1,verbose=-1)

def fit_pred(Xtr,ytr,Xva,seeds=5):
    g=2.0**ytr-1.0
    return np.mean([lgb.LGBMRegressor(random_state=s,**PARAMS).fit(Xtr,g).predict(Xva)
                    for s in range(seeds)],axis=0)

# groups over the FULL feature space: an unseen project is an unseen REGION of
# feature space, so holding out whole regions is the faithful simulation.
Z=StandardScaler().fit_transform(X[[c for c in X.columns if X[c].dtype!='category']].astype(float))
for NC in [40]:
    km=KMeans(NC,n_init=10,random_state=0).fit_predict(Z)
    sizes=pd.Series(km).value_counts()
    print(f"=== {NC} feature-space clusters, sizes {sizes.min()}..{sizes.max()} ===")
    rng=np.random.default_rng(0)
    glob,perproj=[],[]
    for rep in range(40):
        # hold out a SET of clusters totalling ~586 rows -- exactly like the test set
        order=rng.permutation(NC); held=[]; n=0
        for c in order:
            if n>=586: break
            held.append(c); n+=int((km==c).sum())
        va=np.where(np.isin(km,held))[0]; tr=np.where(~np.isin(km,held))[0]
        if len(set(y[va]))<2 or (2.0**y[va]-1).sum()==0: continue
        p=fit_pred(X.iloc[tr],y[tr],X.iloc[va],seeds=3)
        glob.append(ndcg_at_k(y[va],p))
        # per-held-out-cluster ranking quality, same predictions
        pp=[]
        for c in held:
            m=km[va]==c
            if m.sum()>=25 and (2.0**y[va][m]-1).sum()>0:
                pp.append(ndcg_at_k(y[va][m],p[m]))
        if pp: perproj.append(np.mean(pp))
    glob=np.array(glob); perproj=np.array(perproj)
    print(f"  GLOBAL pooled NDCG@20 : {glob.mean():.4f} +/- {glob.std():.4f}   (n={len(glob)})")
    print(f"  mean WITHIN-cluster   : {perproj.mean():.4f} +/- {perproj.std():.4f}")
    print(f"  -> gap {perproj.mean()-glob.mean():+.4f}")
