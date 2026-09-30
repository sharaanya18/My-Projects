import pickle
from solution import *
from embfeat import smooth
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
keep=[i for i,n in enumerate(allnames) if n not in {'s_len','t_abspos','n_len'}]
P=dict(PARAMS); P.update(num_leaves=15,min_data_in_leaf=200,learning_rate=0.03,feature_fraction=0.5,num_threads=4)
m=lgb.train(P,lgb.Dataset(X2tr[:,keep],ytr),250)
pv=m.predict(X2va[:,keep])
pickle.dump(pv,open(S+'lgb_val.pkl','wb'))
k=0; hit=0; res={}
sc=collections=None
rows=[]
for r,mt in zip(va,mva):
    toks,n=mt; p=pv[k:k+n]; y=yva[k:k+n]; k+=n
    top=int(p.argmax()); hit+=y[top]
    # F1 of best token only
    sp=[(toks[top][0],toks[top][1])]
    rows.append((f1(r['source_text'],sp,r['spans']), f1(r['source_text'],decode(toks,r['source_text'],p),r['spans']), y[top], len(sp)))
rows=np.array(rows)
print('top1 token in gold',hit/len(va),'F1 top token only',rows[:,0].mean(),'decoder',rows[:,1].mean())
# decoded length vs gold length
k=0; L=[];G=[]
for r,mt in zip(va,mva):
    toks,n=mt; p=pv[k:k+n]; k+=n
    d=decode(toks,r['source_text'],p)[0]; L.append(d[1]-d[0]); G.append(sum(b-a for a,b in r['spans']))
print('pred len mean/median',np.mean(L),np.median(L),'gold',np.mean(G),np.median(G))
