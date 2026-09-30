import pickle
from embfeat import smooth
from solution import *
S="/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/"
Xtr,ytr,mtr,names,Xva,yva,mva=pickle.load(open(S+'feat.pkl','rb'))
ftr,fva=pickle.load(open(S+'embfeat.pkl','rb'))
va=load(S+"data/validation.csv")
def fix(fs):
    for f in fs:
        for key in ("e1_word_max","e3_tri_max","e3_whole_max","e3_sent_max"):
            for wd in (2,5): f[f"{key}_sm{wd}"]=smooth(f[key],wd)
fix(ftr); fix(fva)
def stack(X,fs):
    ks=sorted(fs[0]); E=np.concatenate([np.stack([f[k] for k in ks],1) for f in fs]).astype(np.float32)
    return np.hstack([X,E]),ks
X2tr,ks=stack(Xtr,ftr); X2va,_=stack(Xva,fva)
allnames=names+ks
drop={'s_len','t_abspos','n_len'}
keep=[i for i,n in enumerate(allnames) if n not in drop]
P=dict(PARAMS); P.update(num_leaves=15,min_data_in_leaf=200,learning_rate=0.03,feature_fraction=0.5,num_threads=4)
for tag,cols in (("base",[i for i in keep if i<len(names)]),("base+emb",keep)):
    m=lgb.train(P,lgb.Dataset(X2tr[:,cols],ytr),400)
    for nr in (150,250,400):
        pv=m.predict(X2va[:,cols],num_iteration=nr)
        preds=predict_spans(va,mva,pv,scale=1.0)
        print(tag,nr,np.mean([f1(r["source_text"],p,r["spans"]) for r,p in zip(va,preds)]),flush=True)
imp=sorted(zip(m.feature_importance('gain'),[allnames[i] for i in keep]),reverse=True)[:12]; print(imp)
