import pickle, time
from solution import *
from embfeat import *
torch.set_num_threads(4)
S="/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/"
tr,va=load(S+"data/train.csv"),load(S+"data/validation.csv")
emb=Embedder()
t=time.time()
ftr=emb_features(emb,tr,log=lambda s:print(s,flush=True)); print('tr',time.time()-t,flush=True)
fva=emb_features(emb,va,log=lambda s:print(s,flush=True)); print('va',time.time()-t,flush=True)
pickle.dump((ftr,fva),open(S+"embfeat.pkl","wb"))
