import sys, json, time
import numpy as np, pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics import adjusted_rand_score
sys.path.insert(0, __import__("os").path.dirname(__file__))
import solution
from solution import Model

train = pd.read_csv(sys.argv[1] + "/train.csv")
nf = int(sys.argv[2]) if len(sys.argv) > 2 else 4

docs = [" ".join(json.loads(s)) for s in train.speeches]
X = normalize(TfidfVectorizer(min_df=3, max_df=0.5, sublinear_tf=True).fit_transform(docs))
S = (X @ X.T).toarray()
np.fill_diagonal(S, 0)
_, comp = connected_components(csr_matrix(S > 0.2))
rng = np.random.RandomState(0)
perm = rng.permutation(comp.max() + 1)
fold_of_comp = np.empty_like(perm)
load = np.zeros(nf)
for c in perm:
    f = load.argmin()
    fold_of_comp[c] = f
    load[f] += (comp == c).sum()
fold = fold_of_comp[comp]

scores = []
for f in range(nf):
    te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
    solution.T0 = time.time()
    m = Model(verbose=False).fit(train.iloc[tr].reset_index(drop=True))
    va = train.iloc[te].reset_index(drop=True)
    pr = m.predict(va)
    s = [max(0, adjusted_rand_score(json.loads(g), p)) for g, p in zip(va.groups, pr)]
    print(f"fold {f}: {np.mean(s):.4f} ({len(te)} boards)", flush=True)
    scores += s
print(f"CV mean ARI: {np.mean(scores):.4f} +- {np.std(scores) / np.sqrt(len(scores)):.4f}")
