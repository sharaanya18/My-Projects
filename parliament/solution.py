"""Parliamentary group partitioning of shuffled debate boards.

Everything is learned inside this script from train.csv (no external data, no
test-set adaptation; each test board is predicted on its own).
  1. Discover the recurring groups: every (board, group) of train becomes a group
     document; these are clustered into K global groups with the constraint that the
     groups of one board go to distinct global groups (constrained k-means with a
     Hungarian assignment per board).
  2. Refine the global labels by self-consistency: a cross-fitted speech classifier
     scores every board group and the board's groups are re-assigned (Hungarian),
     iterated with momentum on the probabilities.
  3. Hyperparameter search (regularisation, n-gram range) by cross-fitted board ARI.
  4. Final classifier; each test board picks exactly n_groups global groups and
     assigns every speech (each chosen group used >= 1 time) by max log-probability.
Usage: python3 solution.py <public_dir> <submission_out>
"""
import sys, json, time, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from joblib import Parallel, delayed

SEED = 0
K = 6            # number of recurring groups (max n_groups on any board)
N_REFINE = 20    # classifier-based refinement rounds of the global labels
MOMENTUM = 0.5   # averaging of cross-fitted probabilities across rounds
C_GRID = (3.0, 10.0, 30.0)
NGRAM_GRID = ((1, 1), (1, 2))
TIME_BUDGET = 3000  # seconds; stop refinement/search and predict after this
T0 = time.time()
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
    """boards: list of (speech_idx_array, local_labels); X: speech vectors (normalized).
    Returns (global label per board group, member speech indices per group, group ids per board)."""
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


def fit_clf(Xs, y, C=10.0):
    clf = LogisticRegression(C=C, max_iter=300, tol=1e-3)
    clf.fit(Xs, y)
    return clf


def _fit_predict(X, y, tr, te, C):
    return fit_clf(X[tr], y[tr], C).predict_log_proba(X[te])


def ari(a, b):
    from sklearn.metrics import adjusted_rand_score
    return adjusted_rand_score(a, b)


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

    def __init__(self, n_refine=N_REFINE, momentum=MOMENTUM, verbose=True):
        self.n_refine, self.momentum, self.verbose = n_refine, momentum, verbose

    def log(self, *a):
        if self.verbose:
            print(f"[{time.time() - T0:6.0f}s]", *a, file=sys.stderr, flush=True)

    @staticmethod
    def _vec(ngram):
        return TfidfVectorizer(preprocessor=lambda s: clean(s).lower(), ngram_range=ngram,
                               min_df=5, max_df=0.5, sublinear_tf=True,
                               max_features=40000 * ngram[1])

    def _cross_fit(self, X, y, sfold, C):
        res = Parallel(N_JOBS)(delayed(_fit_predict)(X, y, sfold != f, sfold == f, C)
                               for f in range(N_FOLDS))
        P = np.zeros((X.shape[0], K))
        for f, p in enumerate(res):
            P[sfold == f] = p
        return P

    def fit(self, train):
        speeches, boards, off = [], [], 0
        for s, g in zip(train.speeches, train.groups):
            sp, gl = json.loads(s), np.array(json.loads(g))
            speeches += sp
            boards.append((np.arange(off, off + len(sp)), gl))
            off += len(sp)
        vecs = {ng: self._vec(ng) for ng in NGRAM_GRID}
        Xs = {ng: v.fit_transform(speeches) for ng, v in vecs.items()}
        X = Xs[NGRAM_GRID[-1]]
        # 1. initial global groups: constrained clustering of group documents in LSA space
        Z = normalize(TruncatedSVD(300, random_state=SEED).fit_transform(Xs[NGRAM_GRID[0]]))
        gassign, gmembers, bidx = discover_global_labels(boards, Z)

        def labels():
            y = np.empty(len(speeches), int)
            for gi, m in enumerate(gmembers):
                y[m] = gassign[gi]
            return y

        nb = len(boards)
        fold = np.arange(nb) % N_FOLDS
        sfold = np.empty(len(speeches), int)
        for b in range(nb):
            sfold[boards[b][0]] = fold[b]
        # 2. refinement of the global labels
        Pa = None
        for r in range(self.n_refine):
            if time.time() - T0 > TIME_BUDGET * 0.6:
                self.log("time budget: stopping refinement")
                break
            P = self._cross_fit(X, labels(), sfold, 10.0)
            Pa = P if Pa is None else self.momentum * Pa + (1 - self.momentum) * P
            new = gassign.copy()
            for b in range(nb):
                gi = bidx[b]
                new[gi] = board_assign(np.vstack([Pa[gmembers[g]].sum(0) for g in gi]))
            self.log(f"refine {r}: {(new != gassign).sum()} group labels changed")
            gassign = new
        y = labels()
        # 3. hyperparameter search by cross-fitted board ARI
        best, best_score = (NGRAM_GRID[-1], 10.0), -1
        for ng in NGRAM_GRID:
            for C in C_GRID:
                if time.time() - T0 > TIME_BUDGET * 0.85:
                    break
                P = self._cross_fit(Xs[ng], y, sfold, C)
                score = np.mean([ari(lab, solve_board(P[idx], len(np.unique(lab))))
                                 for idx, lab in boards])
                self.log(f"search ngram={ng} C={C}: board ARI {score:.4f}")
                if score > best_score:
                    best, best_score = (ng, C), score
        self.log(f"selected ngram={best[0]} C={best[1]}")
        self.vec = vecs[best[0]]
        self.clf = fit_clf(Xs[best[0]], y, best[1])
        return self

    def predict(self, test):
        preds = []
        for s, k in zip(test.speeches, test.n_groups):
            # each board is predicted on its own (no information shared across test boards)
            lp = self.clf.predict_log_proba(self.vec.transform(json.loads(s)))
            lab = solve_board(lp, int(k))
            preds.append(np.unique(lab, return_inverse=True)[1].tolist())
        return preds


def main():
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    model = Model().fit(train)
    preds = model.predict(test)
    sub = pd.DataFrame({"item_id": test.item_id, "groups": [json.dumps(p) for p in preds]})
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    assert len(sub) == len(test) and all(
        len(p) == len(json.loads(s)) for p, s in zip(preds, test.speeches))
    sub.to_csv(submission_out, index=False)


if __name__ == "__main__":
    main()
