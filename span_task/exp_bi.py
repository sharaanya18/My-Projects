import pickle, sys, time
from solution import *
import bienc, torch
torch.set_num_threads(4)
S="/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/"
tr,va=load(S+"data/train.csv"),load(S+"data/validation.csv")
be=bienc.BiEnc()
t=time.time()
be.fit(tr,float(sys.argv[1]),log=lambda s:print(s,flush=True)); print('fit',time.time()-t,flush=True)
fva=bienc.score_rows(be,va,log=lambda s:print(s,flush=True)); print('score va',time.time()-t,flush=True)
pickle.dump(fva,open(S+"bi_val.pkl","wb"))
# quick eval: token top1 hit using max over sizes
_,_,mva=pickle.load(open(S+'feat.pkl','rb'))[4:7]
hit=0;n=0
for r,f in zip(va,fva):
    toks=[(m.start(),m.end(),m.group()) for m in WORD.finditer(r['source_text'])]
    y=token_labels(toks,r['spans'])
    for k in f:
        pass
    s=np.max([ (f[k]-f[k].mean())/(f[k].std()+1e-6) for k in f],0)
    hit+=y[int(s.argmax())]; n+=1
print('bi top1 token in gold',hit/n)
