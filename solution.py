"""
Medical-device failure report -> structured triage record.

Run:  python3 solution.py <public_dir> <submission_out>

COMPLIANCE SUMMARY (each item maps to a challenge / platform rule)
- Hardware: CPU only. No GPU, no CUDA.
- Pretrained weights / external models: none. Every model is trained from scratch in this script.
- Hosted / closed-source LLM APIs: none.
- Data sources: only train.csv and test.csv from <public_dir>. No external data, no network access,
  no synthetic data, no record lookup or matching against outside corpora.
- Training data: train.csv only. test.csv is used only to produce predictions: the hashing featurizer is
  stateless, and the TF-IDF weighting, all models, thresholds and the priority table are fit on train alone.
  No pseudo-labelling, no test-time adaptation, no calibration on test statistics.
- Learned model: linear classifiers (SGD, modified-huber loss) trained on hashed word 1-3-gram TF-IDF text
  features. Four learned heads: harm-reported-vs-unknown, harm severity, breadth, delay.
- Decoding is also learned inside the script from grouped out-of-fold train predictions: ordinal decision
  thresholds (QWK-tuned), the abstention threshold (macro-F1-tuned) and an upward "shade" for harm
  (asymmetric-cost hedge) are fit on out-of-fold data. The priority lookup table is estimated from train labels.
- Validation: 5-fold GroupKFold over proxy "reporting organisation" clusters (KMeans over the opening text of
  each report), so validation reports come from organisations unseen by that fold's training data.
- Determinism: fixed seeds, single-threaded solvers, fixed epoch counts and fixed fold counts. Elapsed time
  is never read back into any decision. Hashing is stateless so the parallel worker count cannot change output.
"""
import sys
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

SEED = 42
N_FOLDS = 5
N_GROUPS = 40
HASH_DIM = 2 ** 20
ALPHA = 1e-5          # L2 strength of the linear heads
EPOCHS = 12           # fixed epoch count for every SGD fit
SHADES = [0.0, 0.1, 0.2, 0.3]   # candidate upward shifts on expected harm (under-triage costs more)
UNDER_PENALTY = 2.0   # weight on under-triage when scoring priority in out-of-fold selection

random.seed(SEED)
np.random.seed(SEED)

REC = r"priority=(\w+)\|breadth=(\d)\|delay=(\d)\|harm=(\w+)"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir", nargs="?", default="./dataset/public")
    ap.add_argument("submission_out", nargs="?", default="./working/submission.csv")
    ap.add_argument("--data-dir", dest="data_dir")
    ap.add_argument("--output", dest="output")
    a, _ = ap.parse_known_args()
    return Path(a.data_dir or a.public_dir), Path(a.output or a.submission_out)


# ---------------------------------------------------------------- features
_HV = HashingVectorizer(n_features=HASH_DIM, ngram_range=(1, 3), alternate_sign=False, norm=None,
                        lowercase=True, dtype=np.float32)


def hash_texts(texts):
    """Stateless word 1-3-gram counts; chunks are independent so worker count cannot change the result."""
    chunks = np.array_split(np.arange(len(texts)), 16)
    parts = Parallel(n_jobs=4)(delayed(_HV.transform)(texts[c[0]:c[-1] + 1]) for c in chunks if len(c))
    return sparse.vstack(parts).tocsr()


def proxy_groups(texts):
    """Cluster the opening text of each report: boilerplate / house phrasing acts as an organisation proxy."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True, min_df=5, dtype=np.float32)
    x = v.fit_transform([t[:400] for t in texts])
    z = TruncatedSVD(60, random_state=SEED).fit_transform(x)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    return MiniBatchKMeans(N_GROUPS, random_state=SEED, n_init=3, batch_size=4096).fit_predict(z)


# ---------------------------------------------------------------- models
def fit_heads(Xc, y_rep, y_h, y_b, y_d):
    """Fit TF-IDF weighting + four SGD heads on the given rows; returns a predict function."""
    tf = TfidfTransformer(sublinear_tf=True).fit(Xc)
    X = tf.transform(Xc)
    mk = lambda: SGDClassifier(loss="modified_huber", alpha=ALPHA, max_iter=EPOCHS, tol=None, random_state=SEED)
    m_rep = mk().fit(X, y_rep)
    m_b = mk().fit(X, y_b)
    m_d = mk().fit(X, y_d)
    r = y_h > 0
    m_h = mk().fit(X[r], y_h[r])

    def predict(Xnew):
        Z = tf.transform(Xnew)
        return dict(rep=m_rep.decision_function(Z), b=m_b.decision_function(Z),
                    d=m_d.decision_function(Z), h=m_h.decision_function(Z))
    return predict


def expected(scores):
    """Turn modified-huber margins into a class-probability-like vector and take its expected class index."""
    p = np.clip((scores + 1) / 2, 1e-6, 1)
    p /= p.sum(1, keepdims=True)
    return p @ np.arange(scores.shape[1])


def tune_thresholds(v, y, init):
    """Coordinate search of ordinal cut points maximising quadratic-weighted kappa."""
    th = np.array(init, float)
    grid = np.linspace(v.min() + 1e-3, v.max() - 1e-3, 60)
    for _ in range(3):
        for i in range(len(th)):
            best_q, best_c = -9, th[i]
            for c in grid:
                t2 = th.copy()
                t2[i] = c
                q = cohen_kappa_score(y, np.digitize(v, np.sort(t2)), weights="quadratic")
                if q > best_q:
                    best_q, best_c = q, c
            th[i] = best_c
    return np.sort(th)


def awk(y, p, K=10, under=UNDER_PENALTY):
    """Asymmetric weighted kappa: quadratic disagreement, under-prediction weighted `under` times more."""
    i = np.arange(K)[:, None]
    j = np.arange(K)[None, :]
    W = ((i - j) / (K - 1)) ** 2 * np.where(j < i, under, 1.0)
    O = np.zeros((K, K))
    np.add.at(O, (y, p), 1)
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    return 1 - (W * O).sum() / (W * E).sum()


class Decoder:
    """Learned mapping from head scores to the four record fields (fit on out-of-fold train predictions)."""

    def fit(self, S, y_rep, y_h, y_b, y_d, prio_tab):
        self.tab = prio_tab
        self.rep_thr = max(((f1_score(y_rep, S["rep"] > t, average="macro"), t)
                            for t in np.linspace(-0.6, 0.6, 25)))[1]
        self.tb = tune_thresholds(expected(S["b"]), y_b, [0.5, 1.5][:2])
        self.td = tune_thresholds(expected(S["d"]), y_d, [0.5, 1.5, 2.5])
        r = y_h > 0
        vh = expected(S["h"])
        self.th = tune_thresholds(vh[r], y_h[r] - 1, [0.5, 1.5, 2.5])
        # choose the upward shade that maximises out-of-fold asymmetric priority agreement
        best = (-9, 0.0)
        for sh in SHADES:
            self.shade = sh
            out = self.decode(S)
            m = r
            pr = np.where(out["priority"][m] < 0, 0, out["priority"][m])
            true_p = np.array([prio_tab[(int(h), int(b), int(d))] for h, b, d in zip(y_h[m], y_b[m], y_d[m])])
            best = max(best, (awk(true_p, pr), sh))
        self.shade = best[1]
        return self

    def decode(self, S):
        b = np.digitize(expected(S["b"]), self.tb)
        d = np.digitize(expected(S["d"]), self.td)
        h = np.digitize(expected(S["h"]) + self.shade, self.th) + 1
        h = np.where(S["rep"] > self.rep_thr, h, 0)
        pr = np.full(len(h), -1)
        m = h > 0
        pr[m] = [self.tab[(int(x), int(y), int(z))] for x, y, z in zip(h[m], b[m], d[m])]
        return dict(priority=pr, breadth=b, delay=d, harm=h)


def format_records(o):
    fmt = lambda p, b, d, h: "priority=%s|breadth=%d|delay=%d|harm=%s" % (
        "unknown" if p < 0 else p, b, d, "unknown" if h == 0 else h)
    return [fmt(*x) for x in zip(o["priority"], o["breadth"], o["delay"], o["harm"])]


def main():
    public_dir, out_path = parse_args()
    train = pd.read_csv(public_dir / "train.csv")
    test = pd.read_csv(public_dir / "test.csv")
    sample = pd.read_csv(public_dir / "sample_submission.csv")
    print("train", train.shape, "test", test.shape, flush=True)

    f = train["triage_record"].str.extract(REC)
    f.columns = ["p", "b", "d", "h"]
    y_rep = (f["h"] != "unknown").values.astype(int)
    y_h = np.where(y_rep == 1, f["h"].replace("unknown", "0").astype(int), 0)
    y_b = f["b"].astype(int).values
    y_d = f["d"].astype(int).values
    y_p = np.where(y_rep == 1, f["p"].replace("unknown", "-1").astype(int), -1)

    # priority is estimated from the training labels as the modal tier of each (harm, breadth, delay) cell
    cells = pd.DataFrame({"h": y_h, "b": y_b, "d": y_d, "p": y_p})[y_rep == 1]
    prio_tab = {k: int(g["p"].mode()[0]) for k, g in cells.groupby(["h", "b", "d"])}

    texts = train["report_text"].fillna("").tolist()
    Xc = hash_texts(np.array(texts, dtype=object))
    groups = proxy_groups(texts)

    # grouped out-of-fold predictions (train only) used to fit the decoder
    S = dict(rep=np.zeros(len(texts), np.float32), b=np.zeros((len(texts), 3), np.float32),
             d=np.zeros((len(texts), 4), np.float32), h=np.zeros((len(texts), 4), np.float32))
    for k, (tri, tei) in enumerate(GroupKFold(N_FOLDS).split(Xc, groups=groups)):
        pred = fit_heads(Xc[tri], y_rep[tri], y_h[tri], y_b[tri], y_d[tri])(Xc[tei])
        for key in S:
            S[key][tei] = pred[key]
        print("oof fold", k, flush=True)
    dec = Decoder().fit(S, y_rep, y_h, y_b, y_d, prio_tab)
    o = dec.decode(S)
    r = y_rep == 1
    print("OOF macroF1(reported) %.4f | QWK harm %.4f breadth %.4f delay %.4f | AWK(priority) %.4f | shade %.2f" % (
        f1_score(y_rep, o["harm"] > 0, average="macro"),
        cohen_kappa_score(y_h[r], np.where(o["harm"][r] > 0, o["harm"][r], 1), weights="quadratic"),
        cohen_kappa_score(y_b, o["breadth"], weights="quadratic"),
        cohen_kappa_score(y_d, o["delay"], weights="quadratic"),
        awk(y_p[r], np.where(o["priority"][r] < 0, 0, o["priority"][r])), dec.shade), flush=True)

    # final fit on all of train, then predict test
    Xt = hash_texts(np.array(test["report_text"].fillna("").tolist(), dtype=object))
    St = fit_heads(Xc, y_rep, y_h, y_b, y_d)(Xt)
    rec = format_records(dec.decode(St))

    sub = pd.DataFrame({"id": test["id"], "triage_record": rec})
    sub = sub.set_index("id").loc[sample["id"]].reset_index()
    assert len(sub) == len(test) and sub["id"].is_unique and sub["triage_record"].str.len().min() > 0
    assert sub["triage_record"].str.match(r"^priority=(\d|unknown)\|breadth=[0-2]\|delay=[0-3]\|harm=([1-4]|unknown)$").all()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print("wrote", out_path, sub.shape, flush=True)
    print(sub["triage_record"].str.extract(r"harm=(\w+)")[0].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
