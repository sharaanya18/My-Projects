"""Piano staff pairing: for every test batch, give each upper-staff part its own lower-staff part.

Approach
--------
1. Every part becomes two piano rolls on a fixed time grid (sounding notes, note onsets).
2. For every (upper, lower) pair of a batch we compute, frame by frame, the histogram of the
   intervals between the notes of the two staves (folded to 12 interval classes x 3 register
   spans). Intervals are unchanged by transposition, so the model sees harmony, not key.
   Extra channels compare the upper staff with the lower staff one beat earlier/later, with the
   local harmony around each frame, with the bass note and the melody note, plus simple per-staff
   activity and register curves.
   A chord/key branch looks at the combined pitch classes of both staves from each of the 12
   possible roots (learned templates, pooled over roots, so again transposition-invariant): it
   judges whether the two staves form coherent chords and a coherent key, not just good intervals.
3. A neural pair scorer (dilated temporal convolutions + attention pooling) turns each pair into
   an embedding; axial attention layers then let every pair look at the competing pairs of its
   row (the other lower parts) and column (the other upper parts) inside the same batch.
4. Training is listwise per batch: each upper must pick its lower among all candidates, and each
   true lower must pick its upper. Extra negatives come from other batches of the same piece.
   Augmentation: transposition of the register channels, identical time shifts of both staves,
   time stretching (doubled note values), octave displacement of the lower staff, note dropout.
5. Models are trained on composer-grouped folds (test composers never appear in training) and
   their held-out accuracy is reported. Test scores of all models are normalised per batch
   (row + column log-softmax), summed, and decoded with the Hungarian algorithm so that every
   lower part is used at most once. Each test batch is decoded from its own parts only.
"""
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold

# ----------------------------------------------------------------------------- configuration
parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default="./dataset/public/")
parser.add_argument("--output", default="./working/submission.csv")
parser.add_argument("--threads", type=int, default=4)
ARGS, _ = parser.parse_known_args()   # ignore any extra platform arguments

SEED = 42
DEVICE = torch.device("cuda")
N_FOLDS = 5
MODEL_SEEDS = [42, 1337]
EPOCHS = 25
LR = 1e-3
WEIGHT_DECAY = 0.05
D_MODEL = 64
DROPOUT = 0.2
HARD_NEGATIVES = 6
STRETCH_PROB = 0.5
OCTAVE_PROB = 0.3
CHORD_DIM = 32

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(ARGS.threads)

DATA = Path(ARGS.data_dir)
OUT = Path(ARGS.output)

# ----------------------------------------------------------------------------- representation
TICKS = 1536                 # a passage is four quarter notes of 384 ticks
RES = 48                     # grid step in ticks
T = TICKS // RES
BEAT = 384 // RES            # frames per quarter note
PLO, P = 21, 88              # piano keyboard A0..C8
KLO, KHI = -12, 60           # upper-minus-lower intervals that are histogrammed
K = KHI - KLO
WIN = 4                      # half-width (frames) of the local-harmony window

_k = np.arange(KLO, KHI)
_span = np.where(_k < 12, 0, np.where(_k < 24, 1, 2))
FOLD_MAT = torch.zeros(K, 36)
FOLD_MAT[np.arange(K), _span * 12 + (_k % 12)] = 1.0
N_PAIR_TYPES = 10
NH = 6
C_IN = N_PAIR_TYPES * 36 + 2 * NH


def part_rolls(starts, lengths, pitches):
    """[2, T, P] rolls: sounding notes and onsets. Out-of-range values are clipped."""
    roll = np.zeros((2, T, P), np.float32)
    for st, ln, p in zip(starts, lengths, pitches):
        st = int(min(max(st, 0), TICKS - 1))
        en = int(min(max(st + max(int(ln), 1), st + 1), TICKS))
        a = st // RES
        b = max(a + 1, -(-en // RES))
        q = int(min(max(p, PLO), PLO + P - 1)) - PLO
        roll[0, a:b, q] = 1.0
        roll[1, a, q] = 1.0
    return roll


def load_batches(notes, upper_table):
    """One entry per batch: upper ids (from the id table), lower ids (from the note file), rolls."""
    by_part = {pid: g for pid, g in notes.groupby("part_id", sort=False)}
    lowers = notes[notes.staff == "lower"].groupby("batch_id").part_id.unique()
    empty = np.zeros((2, T, P), np.float32)
    batches = {}
    for b, g in upper_table.groupby("batch_id", sort=True):
        U = sorted(g.upper_id.unique())
        L = sorted(lowers.get(b, []))
        def roll(pid):
            if pid not in by_part:
                return empty
            n = by_part[pid]
            return part_rolls(n.start.values, n.length.values, n.pitch.values)
        ru = torch.tensor(np.stack([roll(u) for u in U])) if U else torch.zeros(0, 2, T, P)
        rl = torch.tensor(np.stack([roll(l) for l in L])) if L else torch.zeros(0, 2, T, P)
        batches[b] = dict(U=U, L=L, ru=ru.to(DEVICE), rl=rl.to(DEVICE))
    return batches


PITCH = (torch.arange(P, dtype=torch.float32) + PLO).to(DEVICE)
FOLD_MAT = FOLD_MAT.to(DEVICE)


def shifted(x):
    # x [n, T, P] -> [n, T, K, P] with out[..., k, p] = x[..., p + k + KLO] (zero outside the keyboard)
    return F.pad(x, (-KLO, KHI)).unfold(2, P, 1)[:, :, :K, :]


def window_max(x):
    return F.max_pool1d(x.permute(0, 2, 1), 2 * WIN + 1, 1, WIN).permute(0, 2, 1)


def extreme_note(s, lowest):
    """One-hot of the lowest (bass) or highest (melody) sounding key per frame."""
    idx = torch.arange(P, dtype=torch.float32, device=s.device)
    fill = torch.full_like(idx, 999.0 if lowest else -1.0)
    e = torch.where(s > 0, idx, fill)
    e = e.amin(-1, keepdim=True) if lowest else e.amax(-1, keepdim=True)
    return (s > 0).float() * (idx == e).float()


def staff_curves(s, o):
    n = s.sum(-1)
    on = (n > 0).float()
    mean_p = (s * PITCH).sum(-1) / n.clamp(min=1)
    low_p = torch.where(s > 0, PITCH, torch.full_like(PITCH, 200.0)).amin(-1)
    high_p = torch.where(s > 0, PITCH, torch.zeros_like(PITCH)).amax(-1)
    return torch.stack([n / 4, o.sum(-1) / 4, on, (mean_p - 60) / 12 * on,
                        (low_p - 60) / 12 * on, (high_p - 60) / 12 * on], -1)


def pair_features(ru, rl, reg_shift=0.0):
    """ru [nu,2,T,P], rl [nl,2,T,P] -> [nu, nl, T, C_IN]"""
    us, uo, ls, lo = ru[:, 0], ru[:, 1], rl[:, 0], rl[:, 1]
    Us, Uo, Uw = shifted(us), shifted(uo), shifted(window_max(us))
    lw = window_max(ls)
    l_prev = F.pad(ls, (0, 0, BEAT, 0))[:, :T]
    l_next = F.pad(ls, (0, 0, 0, BEAT))[:, BEAT:]
    pairs = [(Us, ls), (Uo, lo), (Uo, ls), (Us, lo), (Us, lw), (Uw, ls), (Us, l_prev), (Us, l_next),
             (Us, extreme_note(ls, True)), (shifted(extreme_note(us, False)), ls)]
    X = torch.cat([torch.einsum("itkp,jtp->ijtk", a, b) @ FOLD_MAT for a, b in pairs], -1)
    X = torch.log1p(X)
    hu, hl = staff_curves(us, uo), staff_curves(ls, lo)
    hu[..., 3:6] += reg_shift * hu[..., 2:3]
    hl[..., 3:6] += reg_shift * hl[..., 2:3]
    nu, nl = ru.shape[0], rl.shape[0]
    return torch.cat([X, hu[:, None].expand(nu, nl, T, NH), hl[None].expand(nu, nl, T, NH)], -1)


# Chord / key coherence inputs: the pitch classes of both staves seen from each of the 12 possible roots.
TO_PC = torch.zeros(P, 12)
TO_PC[torch.arange(P), (torch.arange(P) + PLO) % 12] = 1.0
TO_PC = TO_PC.to(DEVICE)
ROOTS = ((torch.arange(12)[:, None] + torch.arange(12)[None]) % 12).to(DEVICE)   # ROOTS[r, k] = (r + k) % 12
N_CHROMA = 5


def staff_chroma(s, o, lowest):
    """[n, T, 5, 12]: sounding, onsets, local window, whole passage, bass (lower) or melody (upper) note."""
    cs = (s @ TO_PC).clamp(max=1)
    co = (o @ TO_PC).clamp(max=1)
    cw = (window_max(s) @ TO_PC).clamp(max=1)
    cp = s.sum(1, keepdim=True) @ TO_PC
    cp = (cp / cp.amax(-1, keepdim=True).clamp(min=1)).expand_as(cs)
    return torch.stack([cs, co, cw, cp, extreme_note(s, lowest) @ TO_PC], 2)


def chord_inputs(ru, rl):
    """Root-relative views [n, T, 12, 5*12]; templates applied to them are transposition-invariant."""
    cu = staff_chroma(ru[:, 0], ru[:, 1], lowest=False)
    cl = staff_chroma(rl[:, 0], rl[:, 1], lowest=True)
    def rel(c):
        return c[..., ROOTS].permute(0, 1, 3, 2, 4).reshape(c.shape[0], T, 12, -1)
    return rel(cu), rel(cl)


# ----------------------------------------------------------------------------- model
class ConvBlock(nn.Module):
    def __init__(self, d, dilation, p):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, 3, padding=dilation, dilation=dilation)
        self.ff = nn.Sequential(nn.GELU(), nn.Dropout(p), nn.Conv1d(d, d, 1))

    def forward(self, x):
        h = self.norm(x).transpose(1, 2)
        return x + self.ff(self.conv(h)).transpose(1, 2)


class AxialLayer(nn.Module):
    """Self-attention along dim 1 (plain matmuls, deterministic on GPU)."""
    def __init__(self, d, p, heads=4):
        super().__init__()
        self.h = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(p)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(p), nn.Linear(2 * d, d))

    def forward(self, e):
        a, n, d = e.shape
        q, k, v = self.qkv(self.norm1(e)).reshape(a, n, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        w = self.drop(torch.softmax(q @ k.transpose(-1, -2) / (d // self.h) ** 0.5, -1))
        e = e + self.out((w @ v).transpose(1, 2).reshape(a, n, d))
        return e + self.ff(self.norm2(e))


class PairScorer(nn.Module):
    def __init__(self, d=D_MODEL, p=DROPOUT):
        super().__init__()
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(C_IN + 2 * CHORD_DIM, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        # chord/key templates over the combined pitch classes of both staves (summed per staff, then non-linear)
        self.chord_u = nn.Linear(N_CHROMA * 12, CHORD_DIM)
        self.chord_l = nn.Linear(N_CHROMA * 12, CHORD_DIM, bias=False)
        self.chord_mlp = nn.Sequential(nn.GELU(), nn.Linear(CHORD_DIM, CHORD_DIM), nn.GELU())
        self.blocks = nn.ModuleList([ConvBlock(d, dl, p) for dl in (1, 2, 4, 8)])
        self.att = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d))
        self.axial = nn.ModuleList([nn.ModuleList([AxialLayer(d, p), AxialLayer(d, p)]) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, x, chord):
        nu, nl = x.shape[:2]
        vu, vl = chord
        c = self.chord_mlp(self.chord_u(vu)[:, None] + self.chord_l(vl)[None])   # [nu, nl, T, 12 roots, CHORD_DIM]
        x = torch.cat([x, c.amax(3), c.mean(3)], -1)                              # pooled over roots
        h = self.inp(x.reshape(nu * nl, T, -1))
        for blk in self.blocks:
            h = blk(h)
        a = torch.softmax(self.att(h), 1)
        e = self.proj(torch.cat([(a * h).sum(1), h.amax(1)], -1)).reshape(nu, nl, -1)
        for row_layer, col_layer in self.axial:
            e = row_layer(e)                                   # compare candidate lowers of an upper
            e = col_layer(e.transpose(0, 1)).transpose(0, 1)   # compare candidate uppers of a lower
        return self.head(e).squeeze(-1)


def listwise_loss(S, onehot):
    """onehot [nu, nl]: true lower of each upper. Row CE over lowers + column CE over uppers."""
    row = -(torch.log_softmax(S, 1) * onehot).sum(1).mean()
    St = S @ onehot.T                                           # [nu, nu]: column of each true lower
    col = -(torch.log_softmax(St, 0) * torch.eye(S.shape[0], device=S.device)).sum(0).mean()
    return row + col


def normalised(S):
    return torch.log_softmax(S, 1) + torch.log_softmax(S, 0)


def assign(score):
    """One-to-one assignment; any upper left over (only if lowers < uppers) takes its best lower."""
    s = score.cpu().numpy()
    choice = s.argmax(1)
    r, c = linear_sum_assignment(-s)
    choice[r] = c
    return choice


# ----------------------------------------------------------------------------- training
def augment(ru, rl, g):
    if float(torch.rand(1, generator=g)) < STRETCH_PROB:
        # the same music at doubled note values: stretch a random half-passage window of both staves
        a = int(torch.randint(0, T // 2 + 1, (1,), generator=g))
        idx = (a + torch.arange(T) // 2).to(DEVICE)
        ru, rl = ru[:, :, idx].clone(), rl[:, :, idx].clone()
        ru[:, 1, 1::2] = 0
        rl[:, 1, 1::2] = 0
    if float(torch.rand(1, generator=g)) < OCTAVE_PROB:
        # the lower staff an octave higher or lower (pitch classes unchanged), if it stays on the keyboard
        o = 12 if float(torch.rand(1, generator=g)) < 0.5 else -12
        keys = rl[:, 0].amax((0, 1)).nonzero().flatten()
        if len(keys) and int(keys.min()) + o >= 0 and int(keys.max()) + o < P:
            rl = torch.roll(rl, o, -1)
    shift = int(torch.randint(-4, 5, (1,), generator=g)) * 2      # same time shift for both staves
    if shift != 0:
        ru, rl = torch.roll(ru, shift, 2), torch.roll(rl, shift, 2)
        sl = slice(0, shift) if shift > 0 else slice(shift, None)
        ru[:, :, sl] = 0
        rl[:, :, sl] = 0
    keep_u = (torch.rand(ru.shape[0], 1, 1, P, generator=g) > 0.1).float().to(DEVICE)
    keep_l = (torch.rand(rl.shape[0], 1, 1, P, generator=g) > 0.1).float().to(DEVICE)
    return ru * keep_u, rl * keep_l


def train_model(train_ids, B, onehots, siblings, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = PairScorer().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(train_ids), pct_start=0.1)
    for _ in range(EPOCHS):
        model.train()
        for k in torch.randperm(len(train_ids), generator=g).tolist():
            b = train_ids[k]
            ru, rl, oh = B[b]["ru"], B[b]["rl"], onehots[b]
            sib = siblings[b]
            if sib:   # extra negatives: lower parts of another batch of the same piece
                other = B[sib[int(torch.randint(0, len(sib), (1,), generator=g))]]["rl"]
                pick = torch.randperm(other.shape[0], generator=g)[:HARD_NEGATIVES].to(DEVICE)
                rl = torch.cat([rl, other[pick]])
                oh = torch.cat([oh, torch.zeros(oh.shape[0], len(pick), device=DEVICE)], 1)
            ru, rl = augment(ru, rl, g)
            reg_shift = int(torch.randint(-5, 6, (1,), generator=g)) / 12
            loss = listwise_loss(model(pair_features(ru, rl, reg_shift), chord_inputs(ru, rl)), oh)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def score_batches(model, B, ids):
    return {b: normalised(model(pair_features(B[b]["ru"], B[b]["rl"]), chord_inputs(B[b]["ru"], B[b]["rl"]))) for b in ids
            if B[b]["U"] and B[b]["L"]}


def main():
    train_notes = pd.read_csv(DATA / "train_notes.csv")
    test_notes = pd.read_csv(DATA / "test_notes.csv")
    train_up = pd.read_csv(DATA / "train.csv")
    test_up = pd.read_csv(DATA / "test.csv")
    labels = pd.read_csv(DATA / "train_labels.csv")
    groups = pd.read_csv(DATA / "train_groups.csv").set_index("batch_id")
    truth = dict(zip(labels.upper_id, labels.lower_id))

    B = load_batches(train_notes, train_up)
    TB = load_batches(test_notes, test_up)
    print(f"train batches {len(B)}  test batches {len(TB)}", flush=True)

    # training batches usable for supervision: every upper has its labelled lower inside the batch
    onehots = {}
    for b, d in B.items():
        if d["U"] and d["L"] and all(truth.get(u) in d["L"] for u in d["U"]):
            oh = torch.zeros(len(d["U"]), len(d["L"]))
            for i, u in enumerate(d["U"]):
                oh[i, d["L"].index(truth[u])] = 1.0
            onehots[b] = oh.to(DEVICE)
    ids = sorted(onehots)
    composer = groups.composer_id.reindex(ids).fillna("unknown").values
    piece = groups.piece_id.reindex(ids).fillna("unknown").to_dict()
    n_folds = min(N_FOLDS, len(set(composer)))

    test_ids = sorted(TB)
    test_score = {b: torch.zeros(len(TB[b]["U"]), len(TB[b]["L"]), device=DEVICE) for b in test_ids}
    oof_hits = []
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(ids, groups=composer)):
        tr = [ids[i] for i in tr_idx]
        va = [ids[i] for i in va_idx]
        siblings = {b: [o for o in tr if o != b and piece[o] == piece[b] and piece[b] != "unknown"] for b in tr}
        va_score = {b: 0 for b in va}
        for s in MODEL_SEEDS:
            model = train_model(tr, B, onehots, siblings, s + f)
            for b, S in score_batches(model, B, va).items():
                va_score[b] = va_score[b] + S
            for b, S in score_batches(model, TB, test_ids).items():
                test_score[b] += S
        hits = []
        for b in va:
            choice = assign(va_score[b])
            hits += [B[b]["L"][j] == truth[u] for u, j in zip(B[b]["U"], choice)]
        oof_hits += hits
        print(f"fold {f}: held-out composers {sorted(set(composer[va_idx]))}  accuracy {np.mean(hits):.4f}", flush=True)
    print(f"composer-grouped CV accuracy: {np.mean(oof_hits):.4f}", flush=True)

    rows = []
    for b in test_ids:
        U, L = TB[b]["U"], TB[b]["L"]
        if not L:
            continue
        choice = assign(test_score[b])
        rows += [(u, L[j]) for u, j in zip(U, choice)]
    sub = pd.DataFrame(rows, columns=["upper_id", "lower_id"])

    # submission checks: every test upper id exactly once, no blank lower ids
    assert sub.upper_id.is_unique, "repeated upper_id"
    assert set(sub.upper_id) == set(test_up.upper_id), "missing upper_id"
    assert (sub.lower_id.astype(str).str.len() > 0).all(), "blank lower_id"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(OUT, index=False)
    back = pd.read_csv(OUT, keep_default_na=False)
    assert len(back) == len(test_up) and (back.lower_id != "").all()
    print(f"wrote {OUT} with {len(back)} rows", flush=True)


if __name__ == "__main__":
    main()
