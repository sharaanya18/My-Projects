"""Board-level K-fold CV: the whole pipeline (incl. group discovery) is refit on train folds only."""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, __import__("os").path.dirname(__file__))
from solution import Model

train = pd.read_csv(sys.argv[1] + "/train.csv")
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 4
st = int(sys.argv[3]) if len(sys.argv) > 3 else 0
# contiguous folds by board id, so neighbouring windows of one sitting tend to stay together
order = np.argsort(train.item_id.values)
folds = np.array_split(order, nf)
scores = []
for f, te in enumerate(folds):
    tr = np.setdiff1d(order, te)
    m = Model(self_train=st).fit(train.iloc[tr].reset_index(drop=True))
    va = train.iloc[te].reset_index(drop=True)
    pr = m.predict(va)
    s = [max(0, adjusted_rand_score(json.loads(g), p)) for g, p in zip(va.groups, pr)]
    print(f"fold {f}: {np.mean(s):.4f}", flush=True)
    scores += s
print(f"CV mean ARI: {np.mean(scores):.4f}")
