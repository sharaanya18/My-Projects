import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.model_selection import GroupKFold

# Fixed, deterministic plan: no wall-clock logic, no host-derived settings, no fallbacks.
SEED = 2024
DEVICE = 'cuda'
assert torch.cuda.is_available(), 'this solution is planned for the A10G GPU'
N_FOLDS = 5
SEEDS = [0, 1]
EPOCHS = 20
BATCH = 128
LR = 3e-3
WD = 0.02
CH = 32
NL = 4
KS = 5
NONE_ACT = 15
CIN = 10
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])
tr = np.load(public_dir / 'train.npz')
te = np.load(public_dir / 'test.npz')
ycsv = pd.read_csv(public_dir / 'train.csv')
assert (ycsv['id'].values == tr['ids']).all()
STEP_COLS = [f'e_{k:02d}' for k in range(24)]
L = ycsv[STEP_COLS].to_numpy().astype(np.float32)
groups = ycsv['episode_group'].values
Ftr, Atr = tr['frames'], tr['actions']
Fte, Ate, test_ids = te['frames'], te['actions'], te['ids']
N, NT = len(Ftr), len(Fte)
print('train', Ftr.shape, 'test', Fte.shape, 'base rate', L.mean(), flush=True)

# Whole attempts are held out together, matching the unseen-episode test split.
fold = np.zeros(N, int)
for k, (_, va) in enumerate(GroupKFold(N_FOLDS).split(Ftr, groups=groups)):
    fold[va] = k

# Span v = steps 6v+1..6v+6: identical frames v, v+1 imply zero events on exactly these steps.
Lp = np.concatenate([L, np.zeros((N, 1), np.float32)], 1)
Y = np.stack([Lp[:, 6 * v + 1:6 * v + 7] for v in range(4)], 1)
VM = np.ones((4, 6), np.float32)
VM[3, 5] = 0  # step 24 does not exist


def inputs(F, A):
    """Per-span image stack: the two bounding pictures, their difference, and neighbouring context."""
    f = (F.astype(np.float32) - 120.0) / 60.0
    n = len(F)
    z = np.zeros((n, 24, 24), np.float32)
    a = A[:, ::6].astype(np.int64)
    X, C = [], []
    for v in range(4):
        h1, hp, hn = v < 3, v > 0, v < 2
        f0 = f[:, v]
        f1 = f[:, v + 1] if h1 else z
        fp = f[:, v - 1] if hp else z
        fn = f[:, v + 2] if hn else z
        ch = [f0, f1, (f1 - f0) if h1 else z, fp, fn, (f0 - fp) if hp else z, (fn - f1) if hn else z,
              np.full_like(z, float(h1)), np.full_like(z, float(hp)), np.full_like(z, float(hn))]
        X.append(np.stack(ch, 1))
        C.append(np.stack([a[:, v], a[:, v - 1] if hp else np.full(n, NONE_ACT),
                           a[:, v + 1] if v < 3 else np.full(n, NONE_ACT), np.full(n, v)], 1))
    return np.stack(X, 1).astype(np.float32), np.stack(C, 1)


class ReachNet(nn.Module):
    """A CNN predicts per-pixel maps (player start, player end, passability, collected-gem potential).
    The player's position distribution is propagated one step at a time with a learned, action-conditioned
    motion kernel, forward from the start and backward from the end. The event probability at step t is the
    expected gem potential under the step-t position posterior, so the model learns *when* inside a span a
    gem was reached from how far it lies along the player's path."""

    def __init__(self):
        super().__init__()
        self.aemb = nn.Embedding(16, 16)
        self.vemb = nn.Embedding(4, 8)
        self.cond = nn.Sequential(nn.Linear(16 * 3 + 8, 64), nn.GELU())
        layers = [nn.Conv2d(CIN + 2, CH, 3, 1, 1), nn.GELU()]
        for i in range(NL - 1):
            dil = 2 if i == NL - 2 else 1
            layers += [nn.Conv2d(CH, CH, 3, 1, dil, dilation=dil), nn.GroupNorm(8, CH), nn.GELU()]
        self.enc = nn.Sequential(*layers)
        self.film = nn.Linear(64, 2 * CH)
        self.maps = nn.Conv2d(CH, 5, 1)
        self.k_gen = nn.Parameter(torch.zeros(KS * KS))
        self.k_act = nn.Embedding(16, KS * KS)
        nn.init.zeros_(self.k_act.weight)
        self.glob = nn.Sequential(nn.Linear(2 * CH + 64, 64), nn.GELU(), nn.Linear(64, 6))
        nn.init.zeros_(self.glob[2].weight)
        nn.init.zeros_(self.glob[2].bias)
        self.tbias = nn.Parameter(torch.zeros(4, 6))
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, 24), torch.linspace(-1, 1, 24), indexing='ij')
        self.register_buffer('coord', torch.stack([yy, xx])[None])

    def step(self, p, k, pas):
        B = p.shape[0]
        if k.dim() == 1:
            w = torch.softmax(k, 0).view(1, 1, KS, KS)
            q = Fn.conv2d(p, w, padding=KS // 2)
        else:
            w = torch.softmax(k, 1).view(B, 1, KS, KS)
            q = Fn.conv2d(p.view(1, B, 24, 24), w, padding=KS // 2, groups=B).view(B, 1, 24, 24)
        q = q * pas
        return q / (q.sum((2, 3), keepdim=True) + 1e-8)

    def forward(self, x, c):
        B = x.shape[0]
        cond = self.cond(torch.cat([self.aemb(c[:, 0]), self.aemb(c[:, 1]), self.aemb(c[:, 2]), self.vemb(c[:, 3])], 1))
        h = self.enc(torch.cat([x, self.coord.expand(B, -1, -1, -1)], 1))
        gm, bt = self.film(cond).chunk(2, 1)
        h = Fn.gelu(h * (1 + gm[:, :, None, None]) + bt[:, :, None, None])
        m = self.maps(h)
        has1 = x[:, 7:8, :1, :1]
        start = torch.softmax(m[:, 0].flatten(1), 1).view(B, 1, 24, 24)
        end = torch.softmax(m[:, 1].flatten(1), 1).view(B, 1, 24, 24)
        end = torch.where(has1 > 0, end, torch.full_like(end, 1.0 / 576))
        pas = torch.sigmoid(m[:, 2:3] + 2.0)
        gem = torch.sigmoid(torch.where(has1 > 0, m[:, 3:4], m[:, 4:5]) - 3.0)
        kA = self.k_act(c[:, 0]) + self.k_gen[None]
        fw, p = [start], start
        for t in range(6):
            p = self.step(p, kA if t == 0 else self.k_gen, pas)
            fw.append(p)
        kgf = self.k_gen.view(KS, KS).flip(0, 1).reshape(-1)
        kAf = kA.view(B, KS, KS).flip(1, 2).reshape(B, -1)
        bw = [None] * 7
        q = end
        bw[6] = end
        for t in range(6, 0, -1):
            q = self.step(q, kAf if t == 1 else kgf, pas)
            bw[t - 1] = q
        outs = []
        for t in range(1, 7):
            post = fw[t] * bw[t]
            post = post / (post.sum((2, 3), keepdim=True) + 1e-12)
            outs.append((post * gem).sum((1, 2, 3)))
        P = torch.stack(outs, 1).clamp(1e-5, 1 - 1e-5)
        z = torch.log(P) - torch.log1p(-P)
        z = z + self.glob(torch.cat([h.mean((2, 3)), h.amax((2, 3)), cond], 1))
        return z + self.tbias[c[:, 3]]


def predict(net, X, C):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 2048):
            out.append(net(torch.from_numpy(X[i:i + 2048]).to(DEVICE), torch.from_numpy(C[i:i + 2048]).to(DEVICE)).float().cpu().numpy())
    return np.concatenate(out)


def to_rows(Pv, step0):
    P = np.zeros((len(Pv), 24))
    P[:, 0] = step0
    for v in range(4):
        hi = min(6 * v + 7, 24)
        P[:, 6 * v + 1:hi] = Pv[:, v, :hi - 6 * v - 1]
    return P


def score(y, p):
    y = y.ravel()
    p = np.clip(p.ravel(), 0, 1)
    sk = 1 - np.mean((p - y) ** 2) / np.mean((y.mean() - y) ** 2)
    return 0.01 + 0.99 * sk if sk >= 0 else 0.01 * np.exp(sk)


Xtr, Ctr = inputs(Ftr, Atr)
Xte, Cte = inputs(Fte, Ate)
Xte_f, Cte_f = Xte.reshape(-1, CIN, 24, 24), Cte.reshape(-1, 4)
VMt = torch.from_numpy(VM).to(DEVICE)
oof = np.zeros((N, 4, 6))
test_z = np.zeros((NT, 4, 6))
for seed in SEEDS:
    for k in range(N_FOLDS):
        s = SEED + 100 * seed + k
        random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
        trm, vam = fold != k, fold == k
        xs = torch.from_numpy(Xtr[trm].reshape(-1, CIN, 24, 24)).to(DEVICE)
        cs = torch.from_numpy(Ctr[trm].reshape(-1, 4)).to(DEVICE)
        ys = torch.from_numpy(Y[trm].reshape(-1, 6)).to(DEVICE)
        vs = VMt[cs[:, 3]]
        net = ReachNet().to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        n = len(xs)
        steps = n // BATCH
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * steps)
        gen = torch.Generator().manual_seed(s)
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(n, generator=gen).to(DEVICE)
            for b in range(steps):
                i = perm[b * BATCH:(b + 1) * BATCH]
                o = net(xs[i], cs[i])
                loss = (Fn.binary_cross_entropy_with_logits(o, ys[i], reduction='none') * vs[i]).sum() / vs[i].sum()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()
                sch.step()
        z = predict(net, Xtr[vam].reshape(-1, CIN, 24, 24), Ctr[vam].reshape(-1, 4)).reshape(-1, 4, 6)
        oof[vam] += z / len(SEEDS)
        test_z += predict(net, Xte_f, Cte_f).reshape(-1, 4, 6) / (len(SEEDS) * N_FOLDS)
        print(f'seed {seed} fold {k} fold-score {score(L[vam], to_rows(1 / (1 + np.exp(-z)), L[trm, 0].mean())):.4f}', flush=True)
        del xs, cs, ys, vs, net, opt

step0 = L[:, 0].mean()  # step 0 is never pictured before it happens; use its training rate
print('OOF score (episode-held-out):', round(score(L, to_rows(1 / (1 + np.exp(-oof)), step0)), 4), flush=True)
pred = np.clip(to_rows(1 / (1 + np.exp(-test_z)), step0), 0, 1)
sub = pd.DataFrame(pred, columns=STEP_COLS)
sub.insert(0, 'id', test_ids)
assert len(sub) == NT and list(sub.columns) == ['id'] + STEP_COLS and sub['id'].is_unique
assert np.isfinite(sub[STEP_COLS].to_numpy()).all()
submission_out.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(submission_out, index=False, float_format='%.6f')
print('wrote', submission_out, sub.shape)
