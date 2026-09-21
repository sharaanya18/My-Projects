"""Leakage-resistant validation for a project-disjoint, heavily-shifted holdout.

Three views, because no single one is trustworthy here:
  1. GroupKFold on template clusters  -> project-disjointness proxy (PRIMARY)
  2. Test-like holdout via adversarial p(test) -> distribution-shift proxy
  3. Repeated 586-row resampling of the OOF vector -> beats down NDCG@20 noise

Clusters are built WITHOUT the target and are used only to split. They are
never features.
"""
from __future__ import annotations
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.base import clone
from scipy.stats import spearmanr
from metric import ndcg_at_k
from data import CAT_COLS, num_cols

TEST_N = 586

# Features that identify a project's PR template (structure, not content size).
TEMPLATE_FEATS = ['base_channel', 'body_structure', 'body_heading_count',
                  'body_checklist_item_count', 'body_line_count',
                  'body_code_block_count', 'title_word_count',
                  'title_colon_count', 'title_digit_count',
                  'body_question_count', 'body_link_count']


def template_clusters(df: pd.DataFrame, n_clusters: int = 100, seed: int = 0) -> np.ndarray:
    """Pseudo-project ids from template fingerprints. Target is never used."""
    num = num_cols(df)
    Z = df[[c for c in TEMPLATE_FEATS if c in num]].astype(float).copy()
    for c in TEMPLATE_FEATS:
        if c in CAT_COLS:
            for v in sorted(df[c].unique()):
                Z[f'{c}={v}'] = (df[c] == v).astype(float)
    Zs = StandardScaler().fit_transform(Z)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(Zs)


def adversarial_p_test(TRX: pd.DataFrame, TEX: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """p(row looks like test) for each TRAIN row, out-of-fold."""
    X = pd.concat([TRX, TEX], ignore_index=True)
    y = np.r_[np.zeros(len(TRX)), np.ones(len(TEX))]
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                           verbose=-1, random_state=seed)
    p = cross_val_predict(m, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                          method='predict_proba')[:, 1]
    return p[:len(TRX)]


def resampled_ndcg(y, pred, n=TEST_N, draws=400, seed=1):
    """Mean +/- std of NDCG@20 over `draws` random subsets sized like the test set."""
    y = np.asarray(y); pred = np.asarray(pred)
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    sc = np.array([ndcg_at_k(y[i], pred[i])
                   for i in (rng.choice(len(y), n, replace=False) for _ in range(draws))])
    return float(sc.mean()), float(sc.std())


def oof_grouped(model, X, y, groups, n_splits=5):
    """OOF predictions under GroupKFold. Works for regressors and classifiers."""
    oof = np.zeros(len(y), dtype=float)
    for tr, va in GroupKFold(n_splits).split(X, y, groups=groups):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        oof[va] = _score_of(m, X.iloc[va])
    return oof


def _score_of(m, X):
    """A single ranking score, whatever the estimator type."""
    if hasattr(m, "predict_proba"):
        P = m.predict_proba(X)
        classes = list(getattr(m, "classes_", range(P.shape[1])))
        # expected gain E[2^r - 1] -- matches the metric's gain mapping
        gains = np.array([2.0 ** float(c) - 1.0 for c in classes])
        return P @ gains
    return np.asarray(m.predict(X), dtype=float)


def shift_holdout(model, X, y, p_test, frac=0.30):
    """Train on the LEAST test-like rows, validate on the MOST test-like.

    Simulates being scored on a distribution you did not train on.
    """
    order = np.argsort(-p_test)
    n_va = int(len(y) * frac)
    va, tr = order[:n_va], order[n_va:]
    m = clone(model)
    m.fit(X.iloc[tr], y[tr])
    return va, _score_of(m, X.iloc[va])


def evaluate(model, X, y, groups, p_test, name="", draws=400, verbose=True):
    """Run all three views. Returns a dict; this is the only comparison I trust."""
    oof = oof_grouped(model, X, y, groups)
    g_mean, g_std = resampled_ndcg(y, oof, draws=draws)
    g_sp = spearmanr(oof, y).statistic

    va, pred = shift_holdout(model, X, y, p_test)
    s_mean, s_std = resampled_ndcg(y[va], pred, n=min(TEST_N, len(va)), draws=draws)
    s_sp = spearmanr(pred, y[va]).statistic

    res = dict(name=name, grp_ndcg=g_mean, grp_std=g_std, grp_spearman=g_sp,
               shift_ndcg=s_mean, shift_std=s_std, shift_spearman=s_sp, oof=oof)
    if verbose:
        print(f"  {name:38s} GRP {g_mean:.4f}+/-{g_std:.4f} (sp {g_sp:+.3f})   "
              f"SHIFT {s_mean:.4f}+/-{s_std:.4f} (sp {s_sp:+.3f})")
    return res


def paired_ndcg(y, pred_a, pred_b, n=TEST_N, draws=800, seed=7):
    """Compare two prediction vectors on the SAME resampled subsets.

    Independent mean+/-std comparisons are far too weak here: the NDCG@20
    sampling noise (std ~0.05-0.10) is shared between the two models and
    cancels when paired. This is the only comparison with real power.
    """
    y = np.asarray(y); a = np.asarray(pred_a); b = np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    n = min(n, len(y))
    da, db = [], []
    for _ in range(draws):
        i = rng.choice(len(y), n, replace=False)
        da.append(ndcg_at_k(y[i], a[i])); db.append(ndcg_at_k(y[i], b[i]))
    da, db = np.array(da), np.array(db)
    d = da - db
    se = d.std(ddof=1) / np.sqrt(draws)
    return dict(a=da.mean(), b=db.mean(), delta=d.mean(), delta_std=d.std(ddof=1),
                se=se, win_rate=float((d > 0).mean()),
                ci_lo=d.mean() - 1.96 * se, ci_hi=d.mean() + 1.96 * se)


def report_paired(y, pred_a, pred_b, name_a, name_b, **kw):
    r = paired_ndcg(y, pred_a, pred_b, **kw)
    verdict = ("A>B" if r['ci_lo'] > 0 else "B>A" if r['ci_hi'] < 0 else "TIE")
    print(f"    {name_a} vs {name_b}: dNDCG={r['delta']:+.4f} "
          f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] win={r['win_rate']:.0%}  -> {verdict}")
    return r


def cluster_shift_folds(groups, p_test, n_folds=5):
    """Project-disjoint AND shift-aware folds.

    Rank pseudo-projects by how test-like their members are, then deal whole
    clusters into folds so that each fold is BOTH unseen (no shared cluster)
    and drawn from a different part of the shift spectrum. This is the closest
    available simulation of the real holdout: project-disjoint + shifted.
    """
    groups = np.asarray(groups); p_test = np.asarray(p_test)
    gdf = pd.DataFrame({'g': groups, 'p': p_test}).groupby('g').agg(p=('p', 'mean'), n=('p', 'size'))
    gdf = gdf.sort_values('p', ascending=False)
    # snake-deal clusters by test-likeness so folds are balanced in size and shift
    order = list(gdf.index)
    assign, sizes = {}, np.zeros(n_folds)
    for i, g in enumerate(order):
        f = int(np.argmin(sizes))
        assign[g] = f
        sizes[f] += gdf.loc[g, 'n']
    return np.array([assign[g] for g in groups])


def oof_prefolded(model, X, y, fold_ids):
    oof = np.zeros(len(y), dtype=float)
    for f in np.unique(fold_ids):
        va = np.where(fold_ids == f)[0]; tr = np.where(fold_ids != f)[0]
        m = clone(model); m.fit(X.iloc[tr], y[tr])
        oof[va] = _score_of(m, X.iloc[va])
    return oof


def testlike_cluster_holdout(model, X, y, groups, p_test, frac=0.30):
    """Train on the least test-like CLUSTERS, validate on the most test-like ones.

    Both leakage-resistant (whole clusters held out) and shift-aware.
    """
    groups = np.asarray(groups)
    gdf = pd.DataFrame({'g': groups, 'p': p_test}).groupby('g').p.mean().sort_values(ascending=False)
    va_groups, n = [], 0
    for g in gdf.index:
        if n >= frac * len(y):
            break
        va_groups.append(g); n += int((groups == g).sum())
    va = np.where(np.isin(groups, va_groups))[0]
    tr = np.where(~np.isin(groups, va_groups))[0]
    m = clone(model); m.fit(X.iloc[tr], y[tr])
    return va, _score_of(m, X.iloc[va])


def sibling_groups(TR, n_clusters=100, seed=0, near_dup_thresh=0.97):
    """Strictest anti-sibling grouping.

    A 'sibling' here is any row that shares provenance with another: an exact
    duplicate profile, a near-duplicate profile, or the same project template
    cluster. All three relations are unioned into connected components, so no
    two related rows can ever land on opposite sides of a split.

    This is the split that makes memorising a project worthless.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(TR)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # 1. exact duplicate profiles
    key = TR.drop(columns=[c for c in ('id', 'target') if c in TR.columns]) \
            .astype(str).agg('|'.join, axis=1)
    for _, idx in key.groupby(key).groups.items():
        idx = list(idx)
        for j in idx[1:]:
            union(TR.index.get_loc(idx[0]), TR.index.get_loc(j))

    # 2. near-duplicate profiles (char n-gram cosine on the serialised record)
    txt = key.values
    V = TfidfVectorizer(analyzer='char_wb', ngram_range=(4, 4), min_df=2)
    M = V.fit_transform(txt)
    S = (M @ M.T).tocoo()
    for i, j, v in zip(S.row, S.col, S.data):
        if i < j and v >= near_dup_thresh:
            union(int(i), int(j))

    # 3. template cluster (same project PR template)
    tc = template_clusters(TR, n_clusters, seed=seed)
    first = {}
    for i, c in enumerate(tc):
        if c in first:
            union(first[c], i)
        else:
            first[c] = i

    roots, out = {}, np.empty(n, dtype=np.int64)
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        out[i] = roots[r]
    return out
