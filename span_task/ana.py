from solution import *
import pickle
D="/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/data"
tr,va=load(D+"/train.csv"),load(D+"/validation.csv")
idf=build_idf(tr)
Xtr,ytr,mtr,names=make_matrix(tr,idf); Xva,yva,mva,_=make_matrix(va,idf)
pickle.dump((Xtr,ytr,mtr,names,Xva,yva,mva),open('/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/feat.pkl','wb'))
