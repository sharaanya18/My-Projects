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

SEED = 0
K = 6            # number of recurring groups (max n_groups on any board)
N_REFINE = 3     # classifier-based refinement rounds of the global labels


def clean(t):
    t = t.replace("[PARTEI]", " PARTEITOKEN ").replace("[NAME]", " NAMETOKEN ")
    return t


def make_vectorizers():
    word = TfidfVectorizer(preprocessor=lambda s: clean(s).lower(), ngram_range=(1, 2),
                           min_df=3, max_df=0.5, sublinear_tf=True, max_features=300000)
    char = TfidfVectorizer(preprocessor=lambda s: clean(s).lower(), analyzer="char_wb",
                           ngram_range=(3, 5), min_df=5, max_df=0.5, sublinear_tf=True,
                           max_features=300000)
    return word, char


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
    clf = LogisticRegression(C=10.0, max_iter=2000)
    clf.fit(Xs, y)
    return clf


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
    def fit(self, train):
        speeches, boards, off = [], [], 0
        for s, g in zip(train.speeches, train.groups):
            sp, gl = json.loads(s), np.array(json.loads(g))
            speeches += sp
            boards.append((np.arange(off, off + len(sp)), gl))
            off += len(sp)
        self.wv, self.cv = make_vectorizers()
        Xw = self.wv.fit_transform(speeches)
        Xc = self.cv.fit_transform(speeches)
        X = normalize(self._stack(Xw, Xc))
        # dense LSA space for discovery
        self.svd = TruncatedSVD(300, random_state=SEED)
        Z = normalize(self.svd.fit_transform(X))
        gassign, gmembers, bidx = discover_global_labels(boards, Z)
        y = np.empty(len(speeches), int)
        for gi, m in enumerate(gmembers):
            y[m] = gassign[gi]
        # refinement: cross-fitted classifier scores -> re-assign board groups
        nb = len(boards)
        fold = np.arange(nb) % 5
        for r in range(N_REFINE):
            P = np.zeros((len(speeches), K))
            for f in range(5):
                trb = np.concatenate([boards[b][0] for b in range(nb) if fold[b] != f])
                teb = np.concatenate([boards[b][0] for b in range(nb) if fold[b] == f])
                clf = fit_clf(X[trb], y[trb])
                P[teb] = clf.predict_log_proba(X[teb])
            changed = 0
            for b in range(nb):
                gi = bidx[b]
                S = np.vstack([P[gmembers[g]].sum(0) for g in gi])
                na = board_assign(S)
                changed += (na != gassign[gi]).sum()
                gassign[gi] = na
            for gi, m in enumerate(gmembers):
                y[m] = gassign[gi]
            print(f"refine {r}: changed {changed} group labels", file=sys.stderr)
        self.clf = fit_clf(X, y)
        return self

    def _stack(self, Xw, Xc):
        from scipy.sparse import hstack
        return hstack([normalize(Xw), normalize(Xc)]).tocsr()

    def predict(self, test):
        preds = []
        for s, k in zip(test.speeches, test.n_groups):
            sp = json.loads(s)
            X = normalize(self._stack(self.wv.transform(sp), self.cv.transform(sp)))
            lp = self.clf.predict_log_proba(X)
            lab = solve_board(lp, int(k))
            _, lab = np.unique(lab, return_inverse=True)
            preds.append(lab.tolist())
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
    sub.to_csv(submission_out, index=False)


if __name__ == "__main__":
    main()
