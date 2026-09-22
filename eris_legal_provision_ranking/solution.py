"""Cross-lingual legal-provision ranking.

Reads ./dataset/public/, writes ./working/submission.csv.
Each test row is scored on its own; the evaluator does the slate grouping.
"""
import os, time, itertools, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
T0 = time.time()
SEED = 20240917
N_BUCKETS = 384
SLATE = 6
np.random.seed(SEED)

PUB, OUT = "./dataset/public", "./working"
os.makedirs(OUT, exist_ok=True)

elapsed = lambda: time.time() - T0   # log timestamps only, never control flow
def log(*a):
    print("[%7.1fs]" % elapsed(), *a, flush=True)


train = pd.read_csv(f"{PUB}/train.csv").merge(pd.read_csv(f"{PUB}/train_targets.csv"), on="id")
test = pd.read_csv(f"{PUB}/test.csv")
log(f"train={train.shape}  test={test.shape}")

y = train["target"].to_numpy(np.float64)
BAGS = ["query_features", "parallel_query_features", "anchor_features", "candidate_features"]


# ---------------------------------------------------------------- features
# Bucket ids are permuted per query, so an id means nothing across rows. What
# does carry over is how many buckets land in each membership pattern across
# the four bags - the 16 Venn counts. Every Jaccard/Dice/cosine is a function
# of those, so they are the whole usable basis.

def to_bits(col):
    M = np.zeros((len(col), N_BUCKETS), dtype=np.float32)
    for i, s in enumerate(col.to_numpy()):
        idx = np.fromstring(str(s), dtype=int, sep=" ")
        if idx.size:
            M[i, idx] = 1.0
    return M


def build_features(df):
    Q, P, A, C = (to_bits(df[c]) for c in BAGS)
    pat = (Q + 2 * P + 4 * A + 8 * C).astype(np.int8)
    V = np.stack([(pat == k).sum(1) for k in range(16)], 1).astype(np.float64)

    nq, npp, na, nc = Q.sum(1), P.sum(1), A.sum(1), C.sum(1)
    cp = (C * P).sum(1).astype(np.float64)
    cq = (C * Q).sum(1).astype(np.float64)
    ca = (C * A).sum(1).astype(np.float64)

    # overlap you'd expect by chance given the bag sizes, and its spread
    exp_cp = nc * npp / N_BUCKETS
    var_cp = np.maximum(exp_cp * (1 - nc / N_BUCKETS) * (N_BUCKETS - npp) / (N_BUCKETS - 1), 1e-6)

    feats = {
        "cp": cp, "cq": cq, "ca": ca,
        "cp_over_nc": cp / np.maximum(nc, 1),
        "cp_ovl": cp / np.maximum(np.minimum(nc, npp), 1),
        "cp_z": (cp - exp_cp) / np.sqrt(var_cp),
        "cq_over_nc": cq / np.maximum(nc, 1),
        "len_c": nc,
    }
    # languages stay as separate marginals - joining them into one category
    # would leave rus>eng with no representation at all
    ctx = np.column_stack([
        (df["source_lang"].to_numpy() == "kaz").astype(np.float64),
        (df["source_lang"].to_numpy() == "rus").astype(np.float64),
        (df["target_lang"].to_numpy() == "kaz").astype(np.float64),
        (df["target_lang"].to_numpy() == "rus").astype(np.float64),
        (df["target_lang"].to_numpy() == "eng").astype(np.float64),
    ])
    return dict(
        V_free=V[:, 1:],
        X_small=np.column_stack([cp, feats["cp_over_nc"], nc]),
        X_cp=np.column_stack([cp, feats["cp_over_nc"], feats["cp_z"]]),
        ctx=ctx, feats=feats, C=C,
    )


log("building features ...")
FTR, FTE = build_features(train), build_features(test)
log(f"venn={FTR['V_free'].shape}  ctx={FTR['ctx'].shape}")


# ------------------------------------------------- train slates and metric
# Slates are recoverable from the columns that stay constant across a slate.
# Used for the offline metric and the listwise loss only - never a feature,
# never applied to test.
key = (train["source_lang"] + "|" + train["target_lang"] + "|" + train["query_features"]
       + "|" + train["parallel_query_features"] + "|" + train["anchor_features"])
grp = pd.factorize(key)[0]
assert (pd.Series(grp).value_counts() == SLATE).all(), "unexpected slate sizes"
assert (pd.Series(y).groupby(grp).sum() == 1).all(), "expected one positive per slate"
log(f"reconstructed {grp.max() + 1} train slates of {SLATE}, one positive each")

# Reorder so each slate is a contiguous block of 6, as the listwise loss needs.
# Derived from the grouping, not from row order - the block layout in the CSV
# is a prep artefact and nothing here leans on it.
ORD = np.lexsort((np.arange(len(train)), grp))
G, Y = grp[ORD], y[ORD]
IDS = train["id"].to_numpy()[ORD]
DIR = (train["source_lang"] + ">" + train["target_lang"]).to_numpy()[ORD]
TR = {k: v[ORD] for k, v in FTR.items() if k != "feats"}
TRF = {k: v[ORD] for k, v in FTR["feats"].items()}
NG = len(np.unique(G))
IDRANK = np.empty(len(IDS), int)
IDRANK[np.argsort(IDS, kind="stable")] = np.arange(len(IDS))


def utility(scores, g=G, yy=Y, ir=IDRANK):
    """0.65/r + 0.35*[r<=3] for the true candidate, ties broken by ascending id."""
    order = np.lexsort((ir, -scores, g))
    r = np.empty(len(scores), int)
    r[order] = np.tile(np.arange(1, SLATE + 1), len(scores) // SLATE)
    rp = r[yy == 1]
    return float(np.mean(0.65 / rp + 0.35 * (rp <= 3)))


def rank01(x):
    return pd.Series(x).rank().to_numpy() / len(x)


# ------------------------------------------------------------------ models
class SetRanker:
    """DeepSets ranker over bucket membership patterns.

    Each bucket is described by its 4-bit pattern (in query? parallel? anchor?
    candidate?). The model owns an embedding per pattern type, sums it over the
    row's buckets and runs the pooled vector plus language flags through an MLP.
    Summing embeddings is just (pattern counts) @ E, so it's a single matmul.

    Embedding patterns rather than bucket ids is the whole point: with a
    per-query permutation an id-indexed table can only fit noise.

    Trained listwise (softmax over a training slate). Plain BCE collapses to a
    near-constant score at this base rate, which is why the loss is listwise.
    """

    def __init__(self, d_emb=10, hidden=(20,), lr=5e-3, epochs=60, l2=1e-2,
                 seed=0, batch_slates=256):
        self.hp = dict(d_emb=d_emb, hidden=hidden, lr=lr, epochs=epochs,
                       l2=l2, seed=seed, batch_slates=batch_slates)

    def _init(self, n_pat, n_ctx):
        r = np.random.RandomState(self.hp["seed"])
        d = self.hp["d_emb"]
        self.E = r.normal(0, 0.5, (n_pat, d))
        dims = [d + n_ctx] + list(self.hp["hidden"]) + [1]
        self.W = [r.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
                  for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self._m, self._v, self.t = {}, {}, 0

    def _adam(self, key, p, gr, lr):
        if key not in self._m:
            self._m[key] = np.zeros_like(p)
            self._v[key] = np.zeros_like(p)
        m, v = self._m[key], self._v[key]
        m *= 0.9
        m += 0.1 * gr
        v *= 0.999
        v += 0.001 * gr ** 2
        p -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)

    def _forward(self, Vc, Ctx):
        h = np.hstack([Vc @ self.E, Ctx])
        acts = [h]
        for i in range(len(self.W) - 1):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
            acts.append(h)
        return (h @ self.W[-1] + self.b[-1]).ravel(), acts

    def _backward(self, Vc, acts, dz, lr):
        dh = dz.reshape(-1, 1)
        gW = [None] * len(self.W)
        gb = [None] * len(self.b)
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = acts[i].T @ dh + self.hp["l2"] * self.W[i]
            gb[i] = dh.sum(0)
            dh = dh @ self.W[i].T
            if i > 0:
                dh = dh * (acts[i] > 0)
        gE = Vc.T @ dh[:, :self.hp["d_emb"]] + self.hp["l2"] * self.E
        self.t += 1
        for i in range(len(self.W)):
            self._adam(("W", i), self.W[i], gW[i], lr)
            self._adam(("b", i), self.b[i], gb[i], lr)
        self._adam(("E", 0), self.E, gE, lr)

    def fit(self, Vc, Ctx, yy):
        """Rows must arrive ordered so every block of 6 is one slate."""
        self._init(Vc.shape[1], Ctx.shape[1])
        rs = np.random.RandomState(self.hp["seed"] + 1)
        ng = len(yy) // SLATE
        tgt = yy.reshape(ng, SLATE).argmax(1)
        gidx = np.arange(ng)
        bs = self.hp["batch_slates"]
        for _ in range(self.hp["epochs"]):
            rs.shuffle(gidx)
            for s in range(0, ng, bs):
                sl = gidx[s:s + bs]
                rows = (sl[:, None] * SLATE + np.arange(SLATE)).ravel()
                z, acts = self._forward(Vc[rows], Ctx[rows])
                z3 = z.reshape(len(sl), SLATE)
                e = np.exp(z3 - z3.max(1, keepdims=True))
                pr = e / e.sum(1, keepdims=True)
                d3 = pr.copy()
                d3[np.arange(len(sl)), tgt[sl]] -= 1.0
                d3 /= len(sl)
                self._backward(Vc[rows], acts, d3.ravel(), self.hp["lr"])
        return self

    def decision_function(self, Vc, Ctx):
        return self._forward(Vc, Ctx)[0]


class CondLogit:
    """Linear listwise model over the candidate/parallel overlap family.

    Learns how to weigh the raw count against its length-normalised and
    chance-corrected forms. Low capacity, so it's the piece that survives an
    unseen language direction best.
    """

    def __init__(self, l2=1.0, iters=600, lr=0.5):
        self.l2, self.iters, self.lr = l2, iters, lr

    def fit(self, X, yy):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        ng = len(yy) // SLATE
        Z3 = Z.reshape(ng, SLATE, -1)
        t = yy.reshape(ng, SLATE).argmax(1)
        w = np.zeros(Z.shape[1])
        for _ in range(self.iters):
            s = Z3 @ w
            s -= s.max(1, keepdims=True)
            e = np.exp(s)
            pr = e / e.sum(1, keepdims=True)
            gr = -(Z3[np.arange(ng), t] - np.einsum("gk,gkf->gf", pr, Z3)).mean(0) + self.l2 * w / ng
            w -= self.lr * gr
        self.w = w
        return self

    def decision_function(self, X):
        return ((X - self.mu) / self.sd) @ self.w


GB_SEEDS = (0, 1, 2)


def gb_scores(Xtr, ytr, Xva, **kw):
    """Trees on a compact feature set - picks up curvature the linear models can't."""
    out = np.zeros(len(Xva))
    for s in GB_SEEDS:
        m = HistGradientBoostingClassifier(random_state=s, **kw).fit(Xtr, ytr)
        out += rank01(m.predict_proba(Xva)[:, 1])
    return out / len(GB_SEEDS)


# -------------------------------------------------------------- validation
# With 4800 slates the standard error on a utility estimate is about +-0.005,
# so anything smaller than that is noise. Selection uses repeated grouped CV
# plus leave-one-direction-out, since rus>eng has no labels to learn from.
SEED_LIST = (0, 1, 2)


def component(name, seeds=SEED_LIST, **hp):
    """fn(train_idx, val_idx) -> rank-normalised scores on val_idx."""
    if name == "set":
        def f(tri, vai):
            mu, sd = TR["V_free"][tri].mean(0), TR["V_free"][tri].std(0) + 1e-9
            Zt, Zv = (TR["V_free"][tri] - mu) / sd, (TR["V_free"][vai] - mu) / sd
            out = np.zeros(len(vai))
            for s in seeds:
                out += rank01(SetRanker(seed=s, **hp)
                              .fit(Zt, TR["ctx"][tri], Y[tri])
                              .decision_function(Zv, TR["ctx"][vai]))
            return out / len(seeds)
    elif name == "clg":
        def f(tri, vai):
            return rank01(CondLogit(**hp).fit(TR["X_cp"][tri], Y[tri])
                          .decision_function(TR["X_cp"][vai]))
    elif name == "gbdt":
        def f(tri, vai):
            return rank01(gb_scores(TR["X_small"][tri], Y[tri], TR["X_small"][vai], **hp))
    else:
        raise ValueError(name)
    return f


def cv_oof(fn, reps=2):
    outs = []
    for rep in range(reps):
        gp = np.random.RandomState(1000 + rep).permutation(NG)[G]
        oof = np.zeros(len(Y))
        for tri, vai in GroupKFold(5).split(np.zeros(len(Y)), Y, gp):
            oof[vai] = fn(tri, vai)
        outs.append(rank01(oof))
    return outs


def lodo_oof(fn):
    """Each direction scored by a model that never saw it."""
    oof = np.zeros(len(Y))
    for d in np.unique(DIR):
        vai = np.where(DIR == d)[0]
        tri = np.where(DIR != d)[0]
        oof[vai] = rank01(fn(tri, vai))
    return oof


def summarize(outs_cv, out_lodo):
    cv = float(np.mean([utility(o) for o in outs_cv]))
    ld = float(np.mean([utility(out_lodo[DIR == d], G[DIR == d], Y[DIR == d], IDRANK[DIR == d])
                        for d in np.unique(DIR)]))
    return cv, ld, (cv + ld) / 2


log("=" * 78)
log("BASELINES (no learned parameters)")
rng = np.random.RandomState(SEED)
rand_u = float(np.mean([utility(rng.rand(len(Y))) for _ in range(5)]))
log(f"  random scoring                         {rand_u:.4f}")
for nm, v in [("max candidate<->parallel overlap", TRF["cp_ovl"]),
              ("max candidate<->query overlap", TRF["cq_over_nc"]),
              ("max candidate<->anchor overlap", TRF["ca"]),
              ("unweighted sum of the three",
               rank01(TRF["cp_ovl"]) + rank01(TRF["cq_over_nc"]) + rank01(TRF["ca"]))]:
    log(f"  {nm:<38s} {utility(np.asarray(v, float)):.4f}")
HEUR = utility(TRF["cp_ovl"])
log("  -> no single rule clears random by more than ~0.03")

log("=" * 78)
log("ABLATION: raw bucket ids as columns (expect noise)")
ab = np.zeros(len(Y))
for tri, vai in GroupKFold(5).split(np.zeros(len(Y)), Y, G):
    ab[vai] = HistGradientBoostingClassifier(random_state=0, max_depth=4, max_iter=150) \
        .fit(TR["C"][tri], Y[tri]).predict_proba(TR["C"][vai])[:, 1]
log(f"  GBDT on the 384 raw candidate bits     {utility(ab):.4f}   (random {rand_u:.4f})")
log("  -> bucket identity carries nothing, so every feature stays permutation-invariant")


# ------------------------------------------------------------ search + blend
log("=" * 78)
log("HYPERPARAMETER SEARCH")
SPACE = {
    "set": [dict(d_emb=4, hidden=(8,), epochs=60, lr=5e-3, l2=1e-2),
            dict(d_emb=6, hidden=(12,), epochs=60, lr=5e-3, l2=1e-2),
            dict(d_emb=10, hidden=(20,), epochs=60, lr=5e-3, l2=1e-2),
            dict(d_emb=16, hidden=(24,), epochs=60, lr=5e-3, l2=1e-1)],
    "clg": [dict(l2=0.1), dict(l2=1.0), dict(l2=10.0)],
    "gbdt": [dict(max_depth=2, learning_rate=0.05, max_iter=150,
                  l2_regularization=10, min_samples_leaf=200),
             dict(max_depth=3, learning_rate=0.05, max_iter=200,
                  l2_regularization=10, min_samples_leaf=100)],
}
BEST, PRED = {}, {}
for fam, cfgs in SPACE.items():
    rows = []
    for hp in cfgs:
        fn = component(fam, **hp)
        cvs, ld = cv_oof(fn), lodo_oof(fn)
        cv, l, comb = summarize(cvs, ld)
        rows.append((comb, cv, l, hp, cvs, ld))
        log(f"  {fam:<5s} {str(hp):<66s} CV {cv:.4f}  LODO {l:.4f}  comb {comb:.4f}")
    rows.sort(key=lambda r: -r[0])
    comb, cv, l, hp, cvs, ld = rows[0]
    BEST[fam], PRED[fam] = hp, (cvs, ld)
    log(f"  -> best {fam}: {hp}  (CV {cv:.4f}, LODO {l:.4f})")

log("=" * 78)
log("BLEND SELECTION")
FAMS = ["set", "clg", "gbdt"]
n_reps = min(len(PRED[f][0]) for f in FAMS)
cand = []
for w in itertools.product([0, 1, 2, 3], repeat=3):
    if sum(w) == 0:
        continue
    W = np.array(w, float) / sum(w)
    cvs = [sum(W[i] * PRED[f][0][r] for i, f in enumerate(FAMS)) for r in range(n_reps)]
    ldb = sum(W[i] * PRED[f][1] for i, f in enumerate(FAMS))
    cv, l, comb = summarize(cvs, ldb)
    cand.append((comb, cv, l, w))
cand.sort(key=lambda r: -r[0])
for comb, cv, l, w in cand[:8]:
    log(f"  weights(set,clg,gbdt)={w}  CV {cv:.4f}  LODO {l:.4f}  comb {comb:.4f}")
BEST_W = np.array(cand[0][3], float)
BEST_W /= BEST_W.sum()
log(f"  -> weights (set, clg, gbdt) = {np.round(BEST_W, 3).tolist()}")
log(f"     blended CV {cand[0][1]:.4f}  LODO {cand[0][2]:.4f}  "
    f"(heuristic {HEUR:.4f}, random {rand_u:.4f})")

ldb = sum(BEST_W[i] * PRED[f][1] for i, f in enumerate(FAMS))
log("  held-out-direction utility (the stand-in for unseen rus>eng):")
for d in np.unique(DIR):
    k = DIR == d
    log(f"    {d:<9s} {utility(ldb[k], G[k], Y[k], IDRANK[k]):.4f}")


# ------------------------------------------------------------- final + write
# Refit on all of train. Scalers are fitted on train and only applied to test,
# so no test row's score depends on any other test row.
log("=" * 78)
log("FINAL FIT on 100% of train")
mu, sd = TR["V_free"].mean(0), TR["V_free"].std(0) + 1e-9
Zt_full = (TR["V_free"] - mu) / sd
Zte = (FTE["V_free"] - mu) / sd

set_te = np.zeros(len(test))
for s in SEED_LIST:
    set_te += rank01(SetRanker(seed=s, **BEST["set"]).fit(Zt_full, TR["ctx"], Y)
                     .decision_function(Zte, FTE["ctx"]))
set_te = rank01(set_te / len(SEED_LIST))
clg_te = rank01(CondLogit(**BEST["clg"]).fit(TR["X_cp"], Y).decision_function(FTE["X_cp"]))
gbdt_te = rank01(gb_scores(TR["X_small"], Y, FTE["X_small"], **BEST["gbdt"]))
log("  three components fitted")

# Blending rank-normalised components makes scores collide, which would leave
# those rows to an arbitrary id-order tie-break. A tiny continuous term off the
# conditional logit separates them on the signal itself instead.
tiebreak = CondLogit(**BEST["clg"]).fit(TR["X_cp"], Y).decision_function(FTE["X_cp"])
tiebreak = (tiebreak - tiebreak.mean()) / (tiebreak.std() + 1e-12)
pred = rank01(BEST_W[0] * set_te + BEST_W[1] * clg_te + BEST_W[2] * gbdt_te
              + 1e-6 * tiebreak)

# staged files so the progression is reproducible from this run, not claimed
STAGES = {
    "submission_v1_condlogit.csv": clg_te,
    "submission_v2_setranker.csv": rank01(0.5 * set_te + 0.5 * clg_te),
    "submission.csv": pred,
}


def validate(p):
    sub = pd.DataFrame({"id": test["id"].to_numpy(), "prediction": np.asarray(p, float)})
    errs = []
    if len(sub) != len(test):
        errs.append(f"row count {len(sub)} != test {len(test)}")
    if sub["id"].duplicated().any():
        errs.append("duplicate ids")
    if set(sub["id"]) != set(test["id"]):
        errs.append("id set differs from test.csv")
    if not np.isfinite(sub["prediction"]).all():
        errs.append("non-finite predictions")
    if sub["prediction"].nunique() <= 1:
        errs.append("constant prediction")
    if errs:
        raise SystemExit("SUBMISSION VALIDATION FAILED: " + "; ".join(errs))
    return sub


for fname, p in STAGES.items():
    validate(p).to_csv(f"{OUT}/{fname}", index=False)
    log(f"  wrote {OUT}/{fname}")

sub = validate(pred)
log("=" * 78)
log(f"submission.csv: {len(sub)} rows, {sub['prediction'].nunique()} distinct scores, "
    f"range [{sub['prediction'].min():.4f}, {sub['prediction'].max():.4f}]")
log(f"expected score -> grouped CV {cand[0][1]:.4f} | unseen direction {cand[0][2]:.4f}")
log(f"runtime {elapsed():.1f}s")
