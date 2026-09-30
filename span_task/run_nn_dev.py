import sys, pickle, time
from solution import load
import nnmodel
S = "/tmp/claude-0/-home-user-My-Projects/f611bc0d-2eba-51c2-85de-4bfb4cdf8b24/scratchpad/"
tr, va = load(S + "data/train.csv"), load(S + "data/validation.csv")
budget = float(sys.argv[1])
tag = sys.argv[2]
t = time.time()
model, tok = nnmodel.train(tr, budget, log=lambda s: print(s, flush=True))
print("train done", time.time() - t, flush=True)
t = time.time()
pv = nnmodel.predict(model, tok, va)
print("predict val", time.time() - t, flush=True)
pickle.dump(pv, open(S + f"nn_val_{tag}.pkl", "wb"))
import numpy as np
from solution import f1
# quick eval: best contiguous chars range by simple threshold decode
preds = []
for r, p in zip(va, pv):
    idx = np.nonzero(p > 0.5)[0]
    if len(idx) == 0:
        idx = [int(p.argmax())]
    preds.append([(int(idx[0]), int(idx[-1]) + 1)])
print("thr-range F1 (crude)", np.mean([f1(r["source_text"], q, r["spans"]) for r, q in zip(va, preds)]))
