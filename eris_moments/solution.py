import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, random, itertools
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import GroupKFold
SEED = 2024
DEVICE = 'cuda'
assert torch.cuda.is_available(), 'this solution is planned for the A10G GPU'
N_FOLDS = 5
SEEDS = [0, 1]
EPOCHS = 12
BATCH = 256
LR = 0.002
WD = 0.01
CH = 48
DEPTH = 6
LGB_ROUNDS = 400
NONE_ACT = 15
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(4)
public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])
tr = np.load(public_dir / 'train.npz')
te = np.load(public_dir / 'test.npz')
ycsv = pd.read_csv(public_dir / 'train.csv')
assert (ycsv['id'].values == tr['ids']).all()
STEP_COLS = [f'e_{k:02d}' for k in range(24)]
L = ycsv[STEP_COLS].to_numpy().astype(np.float32)
groups = ycsv['episode_group'].values
Ftr, Atr = (tr['frames'], tr['actions'])
Fte, Ate, test_ids = (te['frames'], te['actions'], te['ids'])
N, NT = (len(Ftr), len(Fte))
print('train', Ftr.shape, 'test', Fte.shape, 'base rate', L.mean())
fold = np.zeros(N, int)
for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(Ftr, groups=groups)):
    fold[va] = k
Lp = np.concatenate([L, np.zeros((N, 1), np.float32)], 1)
Y = np.stack([Lp[:, 6 * v + 1:6 * v + 7] for v in range(4)], 1)
VM = np.ones((4, 6), np.float32)
VM[3, 5] = 0

def interval_inputs(F, A, v):
    n = len(F)
    f = F.astype(np.float32) / 255.0
    z = np.zeros((n, 24, 24), np.float32)
    has1 = 1.0 if v < 3 else 0.0
    f0 = f[:, v]
    f1 = f[:, v + 1] if v < 3 else z
    d = (f1 - f0) * has1
    prev = f0 - f[:, v - 1] if v > 0 else z
    nxt = f[:, v + 2] - f[:, v + 1] if v < 2 else z
    yy, xx = np.meshgrid(np.linspace(-1, 1, 24), np.linspace(-1, 1, 24), indexing='ij')
    ch = [f0, f1, d, np.abs(d), prev, nxt, (f0 < 60 / 255).astype(np.float32), (f1 < 60 / 255).astype(np.float32) * has1, (f0 > 145 / 255).astype(np.float32), (f1 > 145 / 255).astype(np.float32) * has1, np.broadcast_to(yy, (n, 24, 24)), np.broadcast_to(xx, (n, 24, 24)), np.full((n, 24, 24), has1, np.float32)]
    a = A[:, ::6].astype(np.int64)
    act = np.stack([a[:, v], a[:, v - 1] if v > 0 else np.full(n, NONE_ACT), a[:, v + 1] if v < 3 else np.full(n, NONE_ACT), np.full(n, v)], 1)
    return (np.stack(ch, 1).astype(np.float32), act)

def geom_feats(F, v):
    f = F.astype(np.float32)
    n = len(F)
    f0 = f[:, v]
    f1 = f[:, v + 1] if v < 3 else f0
    dark0, dark1, br0, br1 = (f0 < 60, f1 < 60, f0 > 145, f1 > 145)
    nd = dark1 & ~dark0
    gone = br0 & ~br1
    app = br1 & ~br0
    ch = np.abs(f1 - f0) > 20
    yy, xx = np.meshgrid(np.arange(24), np.arange(24), indexing='ij')

    def cstats(m):
        c = m.sum((1, 2)).astype(np.float32)
        cc = np.maximum(c, 1)
        my = (m * yy).sum((1, 2)) / cc
        mx = (m * xx).sum((1, 2)) / cc
        sy = np.sqrt(np.maximum((m * yy ** 2).sum((1, 2)) / cc - my ** 2, 0))
        sx = np.sqrt(np.maximum((m * xx ** 2).sum((1, 2)) / cc - mx ** 2, 0))
        return (c, my, mx, sy, sx)
    c_nd, my_nd, mx_nd, sy_nd, sx_nd = cstats(nd)
    c_g, my_g, mx_g, _, _ = cstats(gone)
    c_ch, my_ch, mx_ch, sy_ch, sx_ch = cstats(ch)
    prevc = (np.abs(f0 - f[:, v - 1]) > 20).sum((1, 2)) if v > 0 else np.zeros(n)
    nextc = (np.abs(f[:, v + 2] - f[:, v + 1]) > 20).sum((1, 2)) if v < 2 else np.zeros(n)
    dist = np.sqrt((my_nd - my_g) ** 2 + (mx_nd - mx_g) ** 2) * (c_nd > 0) * (c_g > 0)
    ident = (np.abs(f1 - f0).sum((1, 2)) == 0).astype(np.float32)
    out = np.stack([c_nd, my_nd, mx_nd, sy_nd, sx_nd, c_g, my_g, mx_g, app.sum((1, 2)), c_ch, my_ch, mx_ch, sy_ch, sx_ch, np.abs(f1 - f0).sum((1, 2)) / 255, prevc, nextc, dist, br0.sum((1, 2)), ident], 1).astype(np.float32)
    if v == 3:
        out[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 17, 19]] = 0
    return out

def build(F, A):
    X, ACT = zip(*[interval_inputs(F, A, v) for v in range(4)])
    return (np.stack(X, 1), np.stack(ACT, 1))
Xtr, ACTtr = build(Ftr, Atr)
Xte, ACTte = build(Fte, Ate)

class IntervalNet(nn.Module):

    def __init__(self, cin=13):
        super().__init__()
        self.emb = nn.Linear(16, 16, bias=False)
        self.semb = nn.Linear(4, 8, bias=False)
        cd = 16 * 3 + 8
        self.inp = nn.Conv2d(cin, CH, 3, 1, 1)
        self.convs = nn.ModuleList([nn.Conv2d(CH, CH, 3, 1, 1) for _ in range(DEPTH)])
        self.norms = nn.ModuleList([nn.GroupNorm(8, CH) for _ in range(DEPTH)])
        self.films = nn.ModuleList([nn.Linear(cd, 2 * CH) for _ in range(DEPTH)])
        self.drop = nn.Dropout(0.1)
        self.pix = nn.Conv2d(CH, 7, 1)
        self.bias = nn.Linear(cd, 7)

    def forward(self, x, a):
        oh = lambda t, k: Fn.one_hot(t, k).float()
        cond = torch.cat([self.emb(oh(a[:, 0], 16)), self.emb(oh(a[:, 1], 16)), self.emb(oh(a[:, 2], 16)), self.semb(oh(a[:, 3], 4))], 1)
        h = Fn.gelu(self.inp(x))
        for conv, gn, film in zip(self.convs, self.norms, self.films):
            gm, bt = film(cond).chunk(2, 1)
            h = h + self.drop(Fn.gelu(gn(conv(h)) * (1 + gm[:, :, None, None]) + bt[:, :, None, None]))
        z = self.pix(h).flatten(2)
        o = torch.logsumexp(z, 2) - float(np.log(576.0)) + self.bias(cond)
        return (o[:, :6], o[:, 6])

def cnn_loss(o, y, vm):
    st, an = o
    l = (Fn.binary_cross_entropy_with_logits(st, y, reduction='none') * vm).sum() / vm.sum()
    return l + 0.3 * Fn.binary_cross_entropy_with_logits(an, (y.sum(1) > 0).float())

def cnn_predict(net, x, a):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 2048):
            xb = torch.from_numpy(x[i:i + 2048]).to(DEVICE)
            ab = torch.from_numpy(a[i:i + 2048]).to(DEVICE)
            out.append(net(xb, ab)[0].float().cpu().numpy())
    return np.concatenate(out)
cnn_oof = np.zeros((N, 4, 6))
cnn_test = np.zeros((NT, 4, 6))
Xte_flat, Ate_flat = (Xte.reshape(-1, 13, 24, 24), ACTte.reshape(-1, 4))
for seed in SEEDS:
    for k in range(N_FOLDS):
        s = SEED + 100 * seed + k
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        trm, vam = (fold != k, fold == k)
        xs = torch.from_numpy(Xtr[trm].reshape(-1, 13, 24, 24)).to(DEVICE)
        as_ = torch.from_numpy(ACTtr[trm].reshape(-1, 4)).to(DEVICE)
        ys = torch.from_numpy(Y[trm].reshape(-1, 6)).to(DEVICE)
        vms = torch.from_numpy(np.tile(VM, (int(trm.sum()), 1))).to(DEVICE)
        net = IntervalNet().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        n = len(xs)
        steps = (n + BATCH - 1) // BATCH
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * steps)
        gen = torch.Generator().manual_seed(s)
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(n, generator=gen).to(DEVICE)
            for b in range(steps):
                i = perm[b * BATCH:(b + 1) * BATCH]
                loss = cnn_loss(net(xs[i], as_[i]), ys[i], vms[i])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sch.step()
        va_logit = cnn_predict(net, Xtr[vam].reshape(-1, 13, 24, 24), ACTtr[vam].reshape(-1, 4))
        cnn_oof[vam] += va_logit.reshape(-1, 4, 6) / len(SEEDS)
        cnn_test += cnn_predict(net, Xte_flat, Ate_flat).reshape(-1, 4, 6) / (len(SEEDS) * N_FOLDS)
        print(f'cnn seed {seed} fold {k} last-batch loss {loss.item():.4f}', flush=True)
        del xs, as_, ys, vms, net, opt

def lgb_design(F, A):
    n = len(F)
    a = A[:, ::6].astype(np.float32)
    geo = [geom_feats(F, v) for v in range(4)]
    z = np.zeros_like(geo[0])
    none = np.full((n, 1), NONE_ACT, np.float32)
    rows = []
    for v in range(4):
        base = np.concatenate([geo[v], geo[v - 1] if v > 0 else z, geo[v + 1] if v < 3 else z, a[:, [v]], a[:, [v - 1]] if v > 0 else none, a[:, [v + 1]] if v < 3 else none, np.full((n, 1), v, np.float32)], 1)
        rows.append(np.stack([np.concatenate([base, np.full((n, 1), k, np.float32)], 1) for k in range(6)], 1))
    return np.stack(rows, 1)
Gtr, Gte = (lgb_design(Ftr, Atr), lgb_design(Fte, Ate))
D = Gtr.shape[-1]
CAT = [60, 61, 62]
vmask = np.broadcast_to(VM.astype(bool), (N, 4, 6))
LGB_PARAMS = dict(objective='binary', learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4, seed=SEED, deterministic=True, force_row_wise=True)
lgb_oof = np.zeros((N, 4, 6))
lgb_test = np.zeros((NT, 4, 6))
for k in range(N_FOLDS):
    trm, vam = (fold != k, fold == k)
    m = vmask[trm]
    ds = lgb.Dataset(Gtr[trm][m], Y[trm][m], categorical_feature=CAT)
    model = lgb.train(LGB_PARAMS, ds, LGB_ROUNDS)
    lgb_oof[vam] = model.predict(Gtr[vam].reshape(-1, D)).reshape(-1, 4, 6)
    lgb_test += model.predict(Gte.reshape(-1, D)).reshape(-1, 4, 6) / N_FOLDS
    print(f'lgb fold {k} done', flush=True)

def logit(p):
    p = np.clip(p, 1e-05, 1 - 1e-05)
    return np.log(p / (1 - p))

def sig(z):
    return 1 / (1 + np.exp(-z))

def to_rows(Pv, step0):
    P = np.zeros((len(Pv), 24))
    P[:, 0] = step0
    for v in range(4):
        hi = min(6 * v + 7, 24)
        P[:, 6 * v + 1:hi] = Pv[:, v, :hi - 6 * v - 1]
    return P

def fit_platt(Z, Yv):
    prm = []
    for v in range(4):
        z = Z[:, v][:, VM[v] > 0].ravel()
        y = Yv[:, v][:, VM[v] > 0].ravel()
        f = lambda w: np.mean((sig(w[0] * z + w[1]) - y) ** 2)
        prm.append(minimize(f, [1.0, 0.0], method='Nelder-Mead', options=dict(xatol=1e-05, fatol=1e-10, maxiter=400)).x)
    return prm

def apply_platt(Z, prm):
    return np.stack([sig(prm[v][0] * Z[:, v] + prm[v][1]) for v in range(4)], 1)

def score(y, p):
    y = y.ravel()
    p = np.clip(p.ravel(), 0, 1)
    base = y.mean()
    sk = 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)
    return 0.01 + 0.99 * sk if sk >= 0 else 0.01 * np.exp(sk)
W_GRID = np.round(np.arange(0.5, 1.0001, 0.1), 2)
A_GRID = np.round(np.arange(0.0, 0.3001, 0.05), 2)

def fit_stage_e(cz, lz, Yv, Lrows):
    pc, pl = (fit_platt(cz, Yv), fit_platt(lz, Yv))
    C, G = (apply_platt(cz, pc), apply_platt(lz, pl))
    base = Lrows.mean(0)
    best = None
    for w, al in itertools.product(W_GRID, A_GRID):
        P = (1 - al) * to_rows(w * C + (1 - w) * G, base[0]) + al * base
        mse = np.mean((P - Lrows) ** 2)
        if best is None or mse < best[0]:
            best = (mse, w, al)
    return (pc, pl, best[1], best[2], base)

def apply_stage_e(cz, lz, prm):
    pc, pl, w, al, base = prm
    B = w * apply_platt(cz, pc) + (1 - w) * apply_platt(lz, pl)
    return np.clip((1 - al) * to_rows(B, base[0]) + al * base, 0, 1)
lgb_oof_z, lgb_test_z = (logit(lgb_oof), logit(lgb_test))
base0 = L.mean(0)
print('OOF score  CNN raw   :', round(score(L, to_rows(sig(cnn_oof), base0[0])), 4))
print('OOF score  LGB raw   :', round(score(L, to_rows(lgb_oof, base0[0])), 4))
nested = np.zeros((N, 24))
for k in range(N_FOLDS):
    t = fold != k
    prm_k = fit_stage_e(cnn_oof[t], lgb_oof_z[t], Y[t], L[t])
    nested[~t] = apply_stage_e(cnn_oof[~t], lgb_oof_z[~t], prm_k)
print('OOF score  blend (nested Stage E):', round(score(L, nested), 4))
prm = fit_stage_e(cnn_oof, lgb_oof_z, Y, L)
print('Stage E: cnn weight', prm[2], 'shrink', prm[3])
pred = apply_stage_e(cnn_test, lgb_test_z, prm)
sub = pd.DataFrame(pred, columns=STEP_COLS)
sub.insert(0, 'id', test_ids)
assert len(sub) == NT == 1478
assert list(sub.columns) == ['id'] + STEP_COLS
assert sub['id'].is_unique and (sub['id'].values == test_ids).all()
vals = sub[STEP_COLS].to_numpy()
assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
submission_out.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(submission_out, index=False, float_format='%.6f')
print('wrote', submission_out, sub.shape)
