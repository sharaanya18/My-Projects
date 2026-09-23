"""Parliamentary group partitioning of shuffled debate boards.

Pipeline (uses only the supplied speech texts and anonymous training partitions):
  1. Discover the recurring groups: every (board, group) in train becomes a
     "group document"; these are clustered into K global clusters with the
     constraint that groups from the same board map to distinct clusters
     (Hungarian assignment per board, iterated like constrained k-means,
     then refined with a cross-fitted classifier).
  2. Train a TF-IDF + logistic-regression speech classifier on the discovered
     global labels.
  3. For each test board, pick exactly n_groups global clusters and assign
     every speech to one of them (each chosen cluster used at least once),
     maximising the summed log-probability (exact, via Hungarian).
Usage: python3 solution.py <public_dir> <submission_out>
"""
import sys, json, re, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.sparse import vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed

SEED = 0
K = 6            # number of recurring groups (max n_groups on any board)
N_REFINE = 20    # classifier-based refinement rounds of the global labels
MOMENTUM = 0.5   # averaging of cross-fitted probabilities across rounds
SELF_TRAIN = 0   # rounds of transductive self-training on test texts
N_FOLDS = 4
N_JOBS = 4


def clean(t):
    t = t.replace("[PARTEI]", " PARTEITOKEN ").replace("[NAME]", " NAMETOKEN ")
    return t


def board_assign(score):
    """score: (g, K) similarity of each board-group to each cluster -> distinct clusters."""
    r, c = linear_sum_assignment(-score)
    out = np.empty(score.shape[0], int)
    out[r] = c
    return out


def discover_global_labels(boards, X):
    """boards: list of (speech_idx_array, local_labels). X: speech feature matrix (normalized).
    Returns a global label per training speech."""
    # group documents = mean of member speech vectors
    gdocs, gboard, gmembers = [], [], []
    for b, (idx, lab) in enumerate(boards):
        for l in np.unique(lab):
            m = idx[lab == l]
            gdocs.append(np.asarray(X[m].mean(axis=0)).ravel())
            gboard.append(b)
            gmembers.append(m)
    G = normalize(np.vstack(gdocs))
    gboard = np.array(gboard)
    bidx = [np.where(gboard == b)[0] for b in range(len(boards))]

    best, best_obj = None, -np.inf
    rng = np.random.RandomState(SEED)
    for restart in range(10):
        # init from a random board with K groups
        full = [b for b in range(len(boards)) if len(bidx[b]) == K]
        b0 = full[rng.randint(len(full))]
        C = G[bidx[b0]].copy()
        assign = np.zeros(len(G), int)
        for it in range(50):
            S = G @ C.T
            new = assign.copy()
            for gi in bidx:
                new[gi] = board_assign(S[gi])
            if it > 0 and np.array_equal(new, assign):
                break
            assign = new
            C = normalize(np.vstack([G[assign == k].mean(0) if (assign == k).any() else C[k]
                                     for k in range(K)]))
        obj = (G * C[assign]).sum()
        if obj > best_obj:
            best_obj, best = obj, assign.copy()
    return best, gmembers, bidx


def fit_clf(Xs, y):
    clf = LogisticRegression(C=10.0, max_iter=300, tol=1e-3)
    clf.fit(Xs, y)
    return clf


def _fit_predict(X, y, tr, te):
    return fit_clf(X[tr], y[tr]).predict_log_proba(X[te])


def solve_board(logp, k):
    """logp: (n, K) log-probs. choose k clusters, each used >=1, maximise total."""
    n, Kc = logp.shape
    k = min(k, n, Kc)
    best, best_val = None, -np.inf
    for S in itertools.combinations(range(Kc), k):
        S = list(S)
        L = logp[:, S]
        # columns: k mandatory slots + (n-k) free slots (best of S)
        free = L.max(1, keepdims=True)
        M = np.hstack([L, np.repeat(free, n - k, axis=1)])
        r, c = linear_sum_assignment(-M)
        val = M[r, c].sum()
        if val > best_val:
            lab = np.empty(n, int)
            for ri, ci in zip(r, c):
                lab[ri] = S[ci] if ci < k else S[int(L[ri].argmax())]
            best_val, best = val, lab
    return best


class Model:
    """Fit on training boards; predict partitions of unseen boards."""

    def __init__(self, n_refine=N_REFINE, momentum=MOMENTUM, self_train=SELF_TRAIN):
        self.n_refine, self.momentum, self.self_train = n_refine, momentum, self_train

    def _vec(self):
        return TfidfVectorizer(preprocessor=lambda s: clean(s).lower(), min_df=5, max_df=0.5,
                               sublinear_tf=True, max_features=40000)

    def fit(self, train):
        speeches, boards, off = [], [], 0
        for s, g in zip(train.speeches, train.groups):
            sp, gl = json.loads(s), np.array(json.loads(g))
            speeches += sp
            boards.append((np.arange(off, off + len(sp)), gl))
            off += len(sp)
        self.vec = self._vec()
        X = self.vec.fit_transform(speeches)
        # 1. initial global groups: constrained clustering of group documents in LSA space
        Z = normalize(TruncatedSVD(300, random_state=SEED).fit_transform(X))
        gassign, gmembers, bidx = discover_global_labels(boards, Z)

        def labels():
            y = np.empty(len(speeches), int)
            for gi, m in enumerate(gmembers):
                y[m] = gassign[gi]
            return y

        # 2. refinement: cross-fitted classifier log-probs (averaged with momentum)
        #    re-assign each board's groups to distinct global groups
        nb = len(boards)
        fold = np.arange(nb) % N_FOLDS
        sfold = np.empty(len(speeches), int)
        for b in range(nb):
            sfold[boards[b][0]] = fold[b]
        Pa = None
        for r in range(self.n_refine):
            y = labels()
            res = Parallel(N_JOBS)(delayed(_fit_predict)(X, y, sfold != f, sfold == f)
                                   for f in range(N_FOLDS))
            P = np.zeros((len(speeches), K))
            for f, p in enumerate(res):
                P[sfold == f] = p
            Pa = P if Pa is None else self.momentum * Pa + (1 - self.momentum) * P
            for b in range(nb):
                gi = bidx[b]
                gassign[gi] = board_assign(np.vstack([Pa[gmembers[g]].sum(0) for g in gi]))
        self.X, self.y = X, labels()
        self.clf = fit_clf(X, self.y)
        return self

    def predict(self, test):
        sps = [json.loads(s) for s in test.speeches]
        ks = [int(k) for k in test.n_groups]
        Xt = [self.vec.transform(sp) for sp in sps]
        clf = self.clf
        for it in range(self.self_train + 1):
            labs = [solve_board(clf.predict_log_proba(x), k) for x, k in zip(Xt, ks)]
            if it < self.self_train:
                # transductive self-training: add pseudo-labelled test speeches (texts only)
                clf = fit_clf(vstack([self.X] + Xt), np.concatenate([self.y] + labs))
        return [np.unique(l, return_inverse=True)[1].tolist() for l in labs]


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    model = Model().fit(train)
    preds = model.predict(test)
    sub = pd.DataFrame({"item_id": test.item_id, "groups": [json.dumps(p) for p in preds]})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(submission_out, index=False)


if __name__ == "__main__":
    main()
