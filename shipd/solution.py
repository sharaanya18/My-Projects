from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
START_TIME = time.time()
TIME_BUDGET_S = 3000.0
SEED = 20240921
TOP_K = 20

def ndcg_at_k(truth, prediction, k: int=TOP_K) -> float:
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if truth.size == 0:
        raise ValueError('The answer set cannot be empty.')
    cutoff = min(k, truth.size)
    gains = 2.0 ** truth - 1.0
    order = np.argsort(-prediction, kind='mergesort')[:cutoff]
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=float))
    dcg = float(np.sum(gains[order] * discounts))
    ideal_dcg = float(np.sum(np.sort(gains)[::-1][:cutoff] * discounts))
    return 0.0 if ideal_dcg <= 0.0 else dcg / ideal_dcg

def _self_check_metric() -> None:
    d = 1.0 / np.log2(np.arange(2, 5))
    assert abs(ndcg_at_k([2, 1, 0], [3, 2, 1], 3) - 1.0) < 1e-12
    rev = (3 * d[2] + 1 * d[1]) / (3 * d[0] + 1 * d[1])
    assert abs(ndcg_at_k([2, 1, 0], [1, 2, 3], 3) - rev) < 1e-12
    assert ndcg_at_k([0, 0, 0], [3, 2, 1], 3) == 0.0
    assert abs(ndcg_at_k([2, 0, 0], [1, 1, 1], 3) - 1.0) < 1e-12

def parse_profile(df: pd.DataFrame) -> pd.DataFrame:
    if 'profile_text' not in df.columns or 'id' not in df.columns:
        raise ValueError("expected 'id' and 'profile_text' columns")
    records = []
    for text in df['profile_text'].astype(str):
        rec = {}
        for token in text.split():
            key, sep, value = token.partition('=')
            if sep:
                rec[key] = value
        records.append(rec)
    out = pd.DataFrame.from_records(records)
    out.insert(0, 'id', df['id'].to_numpy())
    return out

def build_feature_frame(parsed: pd.DataFrame) -> pd.DataFrame:
    n = len(parsed)
    feats: dict[str, np.ndarray] = {}
    numeric_cols, categorical_cols = ([], [])
    for col in parsed.columns:
        if col == 'id':
            continue
        values = pd.to_numeric(parsed[col], errors='coerce')
        if values.notna().mean() > 0.9:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    for col in numeric_cols:
        v = pd.to_numeric(parsed[col], errors='coerce').to_numpy(dtype=float)
        v = np.clip(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
        feats[f'log_{col}'] = np.log1p(v)
        feats[f'nz_{col}'] = (v > 0).astype(float)
    for col in categorical_cols:
        for level in sorted(parsed[col].dropna().astype(str).unique()):
            feats[f'{col}={level}'] = (parsed[col].astype(str) == level).astype(float).to_numpy()

    def g(name):
        return feats.get(f'log_{name}', np.zeros(n))
    eps = 0.001
    for num, den in [('test_file_count', 'file_count'), ('docs_file_count', 'file_count'), ('code_file_count', 'file_count'), ('changed_lines', 'file_count'), ('max_file_changes', 'changed_lines'), ('additions', 'changed_lines'), ('body_code_block_count', 'body_word_count')]:
        feats[f'ratio_{num}__{den}'] = g(num) / (g(den) + eps)
    return pd.DataFrame(feats, index=parsed.index)

def align_columns(train_X: pd.DataFrame, test_X: pd.DataFrame) -> pd.DataFrame:
    return test_X.reindex(columns=train_X.columns, fill_value=0.0)

def build_group_folds(X: np.ndarray, n_folds: int, n_regions: int, seed: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    Z = (X - X.mean(0)) / (X.std(0) + 1e-09)
    n_regions = int(min(n_regions, max(2, len(X) // 12)))
    labels = KMeans(n_clusters=n_regions, n_init=5, random_state=seed).fit_predict(Z)
    sizes = pd.Series(labels).value_counts()
    assign, totals = ({}, np.zeros(n_folds))
    for cluster in sizes.index:
        f = int(np.argmin(totals))
        assign[cluster] = f
        totals[f] += sizes[cluster]
    return np.array([assign[c] for c in labels], dtype=int)

def learn_feature_directions(X: np.ndarray, y: np.ndarray, n_subsets: int, seed: int) -> np.ndarray:
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    n, d = X.shape
    votes = np.zeros((n_subsets, d))
    for s in range(n_subsets):
        folds = build_group_folds(X, n_folds=4, n_regions=24 + 8 * s, seed=seed + s)
        for f in np.unique(folds):
            m = folds == f
            if m.sum() < 30 or len(np.unique(y[m])) < 2:
                continue
            for j in range(d):
                col = X[m, j]
                if col.std() < 1e-12:
                    continue
                rho = spearmanr(col, y[m]).statistic
                if np.isfinite(rho):
                    votes[s, j] += np.sign(rho)
    total = np.abs(votes).sum(0)
    signed = votes.sum(0) / np.maximum(total, 1.0)
    return signed

def lookup_excess(v: np.ndarray, y: np.ndarray, n_folds: int=4, min_n: int=6, seed: int=SEED) -> float:
    from sklearn.isotonic import IsotonicRegression
    v = np.asarray(v, dtype=float)
    yy = np.asarray(y, dtype=float)
    if v.std() < 1e-12:
        return 0.0
    rng = np.random.default_rng(seed)
    folds = rng.permutation(len(yy)) % n_folds
    pred_mono, pred_lookup = (np.zeros(len(yy)), np.zeros(len(yy)))
    for f in range(n_folds):
        va, tr = (folds == f, folds != f)
        if tr.sum() < 50 or va.sum() < 10:
            continue
        best, best_pred = (-np.inf, None)
        for sgn in (1.0, -1.0):
            iso = IsotonicRegression(out_of_bounds='clip').fit(sgn * v[tr], yy[tr])
            p = iso.predict(sgn * v[va])
            s = -float(((yy[va] - p) ** 2).mean())
            if s > best:
                best, best_pred = (s, p)
        pred_mono[va] = best_pred
        d = pd.DataFrame({'v': v[tr], 'y': yy[tr]})
        stats = d.groupby('v').y.agg(['mean', 'size'])
        mapped = pd.Series(v[va]).map(stats['mean'])
        count = pd.Series(v[va]).map(stats['size']).fillna(0)
        pred_lookup[va] = np.where(count >= min_n, mapped.fillna(yy[tr].mean()), yy[tr].mean())
    var = yy.var()
    if var < 1e-12:
        return 0.0
    r2_mono = 1 - ((yy - pred_mono) ** 2).mean() / var
    r2_lookup = 1 - ((yy - pred_lookup) ** 2).mean() / var
    return float(r2_lookup - r2_mono)

def extrapolation_split(X: np.ndarray, frac: float=0.35):
    axis = X.mean(axis=1)
    order = np.argsort(-axis)
    n_va = int(len(X) * frac)
    return (order[n_va:], order[:n_va])

class NeuralRanker:

    def __init__(self, hidden=64, depth=2, dropout=0.1, lr=0.003, epochs=120, list_size=256, weight_decay=0.0001, mono_weight=0.0, n_seeds=3, directions=None, seed=SEED):
        self.p = dict(hidden=hidden, depth=depth, dropout=dropout, lr=lr, epochs=epochs, list_size=list_size, weight_decay=weight_decay, mono_weight=mono_weight)
        self.directions = directions
        self.seed = seed
        self.n_seeds = n_seeds

    def fit(self, X, y):
        import torch
        self.nets_, self.out_stats_ = ([], [])
        for k in range(self.n_seeds):
            net, mono = self._fit_one(X, y, self.seed + 1000 * k)
            self.nets_.append((net, mono))
            Xn = (X - self.mu_) / self.sd_
            with torch.no_grad():
                o = self._apply(net, mono, torch.tensor(Xn, dtype=torch.float32)).squeeze(-1).numpy().astype(float)
            self.out_stats_.append((float(o.mean()), float(o.std()) + 1e-09))
        return self

    def predict(self, X):
        import torch
        Xn = (X - self.mu_) / self.sd_
        xt = torch.tensor(Xn, dtype=torch.float32)
        outs = []
        for (net, mono), (mu, sd) in zip(self.nets_, self.out_stats_):
            with torch.no_grad():
                o = self._apply(net, mono, xt).squeeze(-1).numpy().astype(float)
            outs.append((o - mu) / sd)
        return np.mean(outs, axis=0)

    def _apply(self, net, mono, x):
        out = net(x)
        if mono is not None:
            import torch
            w = torch.clamp(mono.weight, min=0.0) * self.mono_dir_
            out = out + self.p['mono_weight'] * (x @ w.T)
        return out

    def _fit_one(self, X, y, seed):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        if not hasattr(self, 'mu_'):
            self.mu_, self.sd_ = (X.mean(0), X.std(0) + 1e-09)
        Xn = (X - self.mu_) / self.sd_
        gains = 2.0 ** np.asarray(y, dtype=float) - 1.0
        xt = torch.tensor(Xn, dtype=torch.float32)
        gt = torch.tensor(gains, dtype=torch.float32)
        layers, in_dim = ([], X.shape[1])
        for _ in range(self.p['depth']):
            layers += [nn.Linear(in_dim, self.p['hidden']), nn.ReLU(), nn.Dropout(self.p['dropout'])]
            in_dim = self.p['hidden']
        layers.append(nn.Linear(in_dim, 1))
        net = nn.Sequential(*layers)
        mono = None
        if self.p['mono_weight'] > 0 and self.directions is not None:
            self.mono_dir_ = torch.tensor(self.directions, dtype=torch.float32)
            mono = nn.Linear(X.shape[1], 1, bias=False)
            nn.init.constant_(mono.weight, 0.01)
        params = list(net.parameters()) + (list(mono.parameters()) if mono is not None else [])
        opt = torch.optim.AdamW(params, lr=self.p['lr'], weight_decay=self.p['weight_decay'])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.p['epochs'])
        g = torch.Generator().manual_seed(seed)
        n = len(y)
        ls = self.p['list_size']
        list_size = n if not ls else min(int(ls), n)
        net.train()
        for _ in range(self.p['epochs']):
            idx = torch.randperm(n, generator=g)[:list_size]
            scores = self._apply(net, mono, xt[idx]).squeeze(-1)
            target = gt[idx]
            if float(target.sum()) <= 0:
                continue
            loss = -(torch.softmax(target, dim=0) * torch.log_softmax(scores, dim=0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        return (net, mono)

class LambdaRanker:

    def __init__(self, num_leaves=15, min_child_samples=40, learning_rate=0.05, n_estimators=300, truncation=4000, monotone=None, seed=SEED):
        self.p = dict(num_leaves=num_leaves, min_child_samples=min_child_samples, learning_rate=learning_rate, n_estimators=n_estimators)
        self.truncation, self.monotone, self.seed = (truncation, monotone, seed)

    def fit(self, X, y):
        import lightgbm as lgb
        kw = {}
        if self.monotone is not None:
            kw['monotone_constraints'] = list(self.monotone)
        self.m_ = lgb.LGBMRanker(objective='lambdarank', metric='ndcg', label_gain=[0, 1, 3], lambdarank_truncation_level=self.truncation, random_state=self.seed, verbose=-1, **self.p, **kw)
        self.m_.fit(X, np.asarray(y, dtype=int), group=[len(y)])
        return self

    def predict(self, X):
        return np.asarray(self.m_.predict(X), dtype=float)

def grouped_oof(make_model, X, y, folds):
    oof = np.zeros(len(y), dtype=float)
    for f in np.unique(folds):
        va, tr = (folds == f, folds != f)
        if tr.sum() < 50 or va.sum() < 5:
            continue
        model = make_model().fit(X[tr], y[tr])
        oof[va] = model.predict(X[va])
    return oof

def extrapolation_score(make_model, X, y, tr_idx, va_idx) -> float:
    model = make_model().fit(X[tr_idx], y[tr_idx])
    return ndcg_at_k(y[va_idx], model.predict(X[va_idx]))

def cv_score(make_model, X, y, fold_sets, extrap=None) -> float:
    grouped = float(np.mean([ndcg_at_k(y, grouped_oof(make_model, X, y, folds)) for folds in fold_sets]))
    if extrap is None:
        return grouped
    return min(grouped, extrapolation_score(make_model, X, y, *extrap))

def time_left() -> float:
    return TIME_BUDGET_S - (time.time() - START_TIME)

def hyperparameter_search(name, build, grid, X, y, fold_sets, log, extrap=None):
    best, best_score = (None, -np.inf)
    for params in grid:
        if time_left() < 300:
            log.append(f'  [{name}] time budget reached, stopping search early')
            break
        score = cv_score(lambda p=params: build(**p), X, y, fold_sets, extrap)
        log.append(f'  [{name}] {params} -> worst-case NDCG@20 {score:.4f}')
        if score > best_score:
            best, best_score = (params, score)
    return (best, best_score)

def blend(a, b, w, stats_a=None, stats_b=None):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mu_a, sd_a = stats_a if stats_a else (a.mean(), a.std() + 1e-09)
    mu_b, sd_b = stats_b if stats_b else (b.mean(), b.std() + 1e-09)
    return w * (a - mu_a) / sd_a + (1 - w) * (b - mu_b) / sd_b

def score_stats(v):
    v = np.asarray(v, dtype=float)
    return (float(v.mean()), float(v.std()) + 1e-09)

def learn_blend_weight(grouped_views, extrap_views, log):
    best_w, best_score, best_parts = (1.0, -np.inf, (0.0, 0.0))
    for w in np.linspace(0.0, 1.0, 21):
        g = float(np.mean([ndcg_at_k(yy, blend(a, b, w)) for a, b, yy in grouped_views]))
        e = float(np.mean([ndcg_at_k(yy, blend(a, b, w)) for a, b, yy in extrap_views]))
        score = min(g, e)
        if score > best_score:
            best_w, best_score, best_parts = (float(w), score, (g, e))
    log.append(f'  learned blend weight w(neural)={best_w:.2f}  over {len(grouped_views)} groupings and {len(extrap_views)} extrapolation splits')
    log.append(f'    grouped mean {best_parts[0]:.4f}  extrapolation mean {best_parts[1]:.4f}  worst-case {best_score:.4f}')
    return (best_w, best_parts[0], best_parts[1], best_score)

def check_submission(sub: pd.DataFrame, test_ids: pd.Series) -> None:
    assert list(sub.columns) == ['id', 'prediction'], f'columns={list(sub.columns)}'
    assert len(sub) == len(test_ids), f'{len(sub)} rows, expected {len(test_ids)}'
    assert sub['id'].duplicated().sum() == 0, 'duplicate ids'
    assert set(sub['id']) == set(test_ids), 'id set does not match test.csv'
    assert (sub['id'].to_numpy() == test_ids.to_numpy()).all(), 'test id order changed'
    assert pd.api.types.is_numeric_dtype(sub['prediction']), 'prediction not numeric'
    assert np.isfinite(sub['prediction'].to_numpy(dtype=float)).all(), 'non-finite prediction'
    assert sub['prediction'].notna().all(), 'missing prediction'
    assert sub['prediction'].nunique() > 1, 'constant prediction carries no ranking'

def main() -> int:
    if len(sys.argv) != 3:
        print('usage: python3 solution.py <public_dir> <submission_out>')
        return 2
    public_dir = Path(sys.argv[1])
    submission_out = Path(sys.argv[2])
    np.random.seed(SEED)
    _self_check_metric()
    train = pd.read_csv(public_dir / 'train.csv')
    targets = pd.read_csv(public_dir / 'train_targets.csv')
    test = pd.read_csv(public_dir / 'test.csv')
    parsed_train = parse_profile(train)
    parsed_test = parse_profile(test)
    merged = parsed_train.merge(targets[['id', 'target']], on='id', how='inner')
    if len(merged) != len(parsed_train):
        raise ValueError(f'{len(parsed_train) - len(merged)} training rows lack a target')
    y = merged['target'].to_numpy(dtype=float)
    if not set(np.unique(y)).issubset({0.0, 1.0, 2.0}):
        raise ValueError('target values outside {0,1,2}')
    merged = merged.drop(columns=['target']).reset_index(drop=True)
    X_train_df = build_feature_frame(merged)
    X_test_df = align_columns(X_train_df, build_feature_frame(parsed_test))
    X = X_train_df.to_numpy(dtype=float)
    X_test = X_test_df.to_numpy(dtype=float)
    log: list[str] = []
    log.append(f'features built: {X.shape[1]}')
    fold_sets = [build_group_folds(X, n_folds=5, n_regions=r, seed=SEED + i) for i, r in enumerate([30, 45, 60])]
    directions = learn_feature_directions(X, y, n_subsets=3, seed=SEED)
    stable = np.abs(directions) >= 0.7
    log.append(f'feature directions learned from data: {int(stable.sum())} of {len(directions)} features have a stable direction (>=0.70 agreement)')
    top = np.argsort(-np.abs(directions))[:6]
    for j in top:
        log.append(f"  {X_train_df.columns[j]:34s} direction {('+' if directions[j] > 0 else '-')} agreement {abs(directions[j]):.2f}")
    mono_all = np.where(stable, np.sign(directions), 0.0)
    excess = np.array([lookup_excess(X[:, j], y, seed=SEED) for j in range(X.shape[1])])
    extrap = extrapolation_split(X, frac=0.35)
    probe = lambda cols: LambdaRanker(monotone=mono_all[cols], seed=SEED, num_leaves=7, min_child_samples=40, n_estimators=250, truncation=4000)
    best_thr, best_thr_score, best_cols = (None, -np.inf, None)
    for thr in [0.02, 0.05, np.inf]:
        cols = np.where(excess <= thr)[0]
        if len(cols) < 5 or time_left() < 600:
            continue
        Xc = X[:, cols]
        mk = lambda c=cols: probe(c)
        grouped = float(np.mean([ndcg_at_k(y, grouped_oof(mk, Xc, y, f)) for f in fold_sets[:2]]))
        ex_scores = [extrapolation_score(mk, Xc, y, *extrapolation_split(Xc, frac=fr)) for fr in (0.25, 0.35, 0.45)]
        score = min(grouped, float(np.mean(ex_scores)))
        label = 'keep all' if not np.isfinite(thr) else f'excess<={thr}'
        log.append(f'  [feature screen] {label}: {len(cols):3d} features -> grouped {grouped:.4f}  extrapolation {np.mean(ex_scores):.4f}  worst-case {score:.4f}')
        if score > best_thr_score:
            best_thr, best_thr_score, best_cols = (thr, score, cols)
    keep = best_cols
    dropped = [X_train_df.columns[j] for j in np.argsort(-excess)[:8] if excess[j] > (best_thr if np.isfinite(best_thr) else np.inf)]
    log.append(f"  selected screen: {('keep all' if not np.isfinite(best_thr) else f'excess<={best_thr}')} -> {len(keep)} of {X.shape[1]} features kept")
    if dropped:
        log.append(f'  highest-excess features dropped: {dropped[:6]}')
    X = X[:, keep]
    X_test = X_test[:, keep]
    mono = mono_all[keep]
    feature_names = [X_train_df.columns[j] for j in keep]
    fold_sets = [build_group_folds(X, n_folds=5, n_regions=r, seed=SEED + i) for i, r in enumerate([30, 45, 60])]
    extrap = extrapolation_split(X, frac=0.35)
    neural_grid = [dict(hidden=64, depth=2, lr=0.003, epochs=600, list_size=None, mono_weight=0.0), dict(hidden=64, depth=2, lr=0.003, epochs=600, list_size=None, mono_weight=0.5), dict(hidden=96, depth=3, lr=0.005, epochs=400, list_size=1024, mono_weight=0.0), dict(hidden=128, depth=2, lr=0.003, epochs=600, list_size=None, mono_weight=0.0), dict(hidden=64, depth=2, lr=0.005, epochs=400, list_size=1024, mono_weight=0.0), dict(hidden=64, depth=2, lr=0.01, epochs=200, list_size=256, mono_weight=0.0)]
    best_neural, s_neural = hyperparameter_search('neural', lambda **kw: NeuralRanker(directions=mono, seed=SEED, **kw), neural_grid, X, y, fold_sets[:1], log, extrap)
    lgbm_grid = [dict(num_leaves=nl, min_child_samples=mcs, n_estimators=ne, truncation=tr) for nl in [7, 15] for mcs in [40, 80] for ne in [300] for tr in [30, 4000]]
    best_lgbm, s_lgbm = hyperparameter_search('lambdarank', lambda **kw: LambdaRanker(monotone=mono, seed=SEED, **kw), lgbm_grid, X, y, fold_sets[:1], log, extrap)
    mk_neural = lambda: NeuralRanker(directions=mono, seed=SEED, **best_neural)
    mk_lgbm = lambda: LambdaRanker(monotone=mono, seed=SEED, **best_lgbm)
    grouped_views, extrap_views = ([], [])
    for folds in fold_sets:
        if time_left() < 420:
            break
        a = grouped_oof(mk_neural, X, y, folds)
        b = grouped_oof(mk_lgbm, X, y, folds)
        grouped_views.append((a, b, y))
        log.append(f'  grouping: neural {ndcg_at_k(y, a):.4f}  lambdarank {ndcg_at_k(y, b):.4f}')
    for frac in (0.25, 0.35, 0.45):
        if time_left() < 300:
            break
        tr_i, va_i = extrapolation_split(X, frac=frac)
        a = mk_neural().fit(X[tr_i], y[tr_i]).predict(X[va_i])
        b = mk_lgbm().fit(X[tr_i], y[tr_i]).predict(X[va_i])
        extrap_views.append((a, b, y[va_i]))
        log.append(f'  extrapolation frac={frac}: neural {ndcg_at_k(y[va_i], a):.4f}  lambdarank {ndcg_at_k(y[va_i], b):.4f}')
    if not grouped_views or not extrap_views:
        raise RuntimeError('no validation views could be built within the time budget')
    w, blend_grouped, blend_extrap, blend_score = learn_blend_weight(grouped_views, extrap_views, log)
    final_neural = mk_neural().fit(X, y)
    final_lgbm = mk_lgbm().fit(X, y)
    stats_n = score_stats(final_neural.predict(X))
    stats_l = score_stats(final_lgbm.predict(X))
    predictions = blend(final_neural.predict(X_test), final_lgbm.predict(X_test), w, stats_n, stats_l)
    submission = pd.DataFrame({'id': parsed_test['id'].to_numpy(), 'prediction': predictions})
    check_submission(submission, test['id'])
    submission_out.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_out, index=False)
    print('\n'.join(log))
    print()
    print('selected method       : neural listwise ranker (ListNet loss) blended')
    print(f'                        with LightGBM LambdaRank, w(neural)={w:.2f}')
    print(f'  neural params       : {best_neural}')
    print(f'  lambdarank params   : {best_lgbm}')
    print(f'validation NDCG@20    : {blend_score:.4f}  (worst case of two views)')
    print(f'  grouped OOF         : {blend_grouped:.4f}  (project-disjoint proxy)')
    print(f'  extrapolation view  : {blend_extrap:.4f}  (trained on small rows, scored on large)')
    print(f'  features kept       : {len(keep)} of {len(excess)} after the lookup screen')
    print(f'train rows            : {len(merged)}')
    print(f'test rows             : {len(parsed_test)}')
    print(f'prediction range      : [{predictions.min():.4f}, {predictions.max():.4f}]')
    print(f'output path           : {submission_out}')
    print(f'runtime               : {time.time() - START_TIME:.1f}s')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
