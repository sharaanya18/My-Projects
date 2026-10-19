"""Piano staff pairing: for every test batch, give each upper-staff part its own lower-staff part.

Approach
--------
1. Every part becomes two piano rolls on a fixed time grid (sounding notes, note onsets).
2. Pair inputs (all transposition-invariant):
   - frame-by-frame histograms of the intervals between the notes of the two staves (12 interval
     classes x 3 register spans), also against the lower staff one beat earlier/later, against the
     local harmony around each frame, against the bass note and from the melody note;
   - a chord/key branch: learned templates over the combined pitch classes of both staves, seen
     from each of the 12 possible roots and pooled over roots. It judges whether the two staves
     form coherent chords and a coherent key, not just good intervals. One model type follows
     these root-relative chord features through time (harmonic progressions) before pooling;
   - per-staff activity and register curves.
3. A neural pair scorer (dilated temporal convolutions + attention pooling) embeds each pair.
   Batch reasoning inside the network, using only the batch's own parts:
   - analogy: pairs (a, x) and (b, y) exchange messages when upper b is upper a transposed by k
     and lower y is lower x transposed by the same k (sequences and transposed repeats);
   - axial attention over the competing pairs of each row (lowers) and column (uppers).
4. Training is listwise per batch: each upper must pick its lower among all candidates (extra
   negatives from another batch of the same piece), each true lower must pick its upper, plus a
   Sinkhorn (soft one-to-one assignment) term. Augmentation: register transposition, identical
   time shifts, time stretching, octave displacement of the lower staff, note dropout.
5. Models of two types are trained on composer-grouped folds (test composers never appear in
   training); held-out accuracy is reported. Test scores are averaged over time-shift
   augmentation, normalised per batch (row + column log-softmax) and summed over all models.
   A final analogy refinement, whose strength and threshold are tuned on the out-of-fold scores,
   is applied, and each batch is decoded with the Hungarian algorithm so every lower part is used
   at most once. Each test batch is decoded from its own parts only.
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
MODEL_TYPES = [True, False]          # temporal chord branch on / off: both types are always trained
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 0.05
D_MODEL = 64
DROPOUT = 0.2
HARD_NEGATIVES = 6
STRETCH_PROB = 0.5
OCTAVE_PROB = 0.3
CHORD_DIM = 64
SINKHORN_WEIGHT = 1.0
ANALOGY_THRESHOLD = 0.4              # in-network analogy links need this much transposed overlap
TTA_SHIFTS = (0, -2, 2)              # identical time shifts of both staves at prediction time
REFINE_GRID = [(w, thr) for w in (0.0, 4.0, 8.0, 16.0) for thr in (0.3, 0.4, 0.5)]

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


def beat_shift(x, beats):
    """Move [n, T, ...] by whole beats along time (zero fill): positive = later."""
    s = beats * BEAT
    if s > 0:
        return F.pad(x, (0, 0, s, 0))[:, :T]
    return F.pad(x, (0, 0, 0, -s))[:, -s:]


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
    pairs = [(Us, ls), (Uo, lo), (Uo, ls), (Us, lo), (Us, window_max(ls)), (Uw, ls),
             (Us, beat_shift(ls, 1)), (Us, beat_shift(ls, -1)),
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
N_CHROMA_U = 6     # sounding, onsets, local window, whole passage, melody note, batch key
N_CHROMA_L = 8     # sounding, onsets, local window, whole passage, bass note, one beat earlier/later, batch key


def chord_inputs(ru, rl):
    """Root-relative views [n, T, 12, channels*12]; templates applied to them are transposition-invariant."""
    def base(s, o, lowest):
        cs = (s @ TO_PC).clamp(max=1)
        co = (o @ TO_PC).clamp(max=1)
        cw = (window_max(s) @ TO_PC).clamp(max=1)
        cp = s.sum(1, keepdim=True) @ TO_PC
        cp = (cp / cp.amax(-1, keepdim=True).clamp(min=1)).expand_as(cs)
        return [cs, co, cw, cp, extreme_note(s, lowest) @ TO_PC]
    us, ls = ru[:, 0], rl[:, 0]
    key = (us.sum((0, 1)) + ls.sum((0, 1))) @ TO_PC          # pitch-class profile of the whole batch
    key = key / key.max().clamp(min=1)
    lpc = (ls @ TO_PC).clamp(max=1)
    cu = torch.stack(base(us, ru[:, 1], False) + [key.expand(us.shape[0], T, 12)], 2)
    cl = torch.stack(base(ls, rl[:, 1], True) + [beat_shift(lpc, 1), beat_shift(lpc, -1),
                                                 key.expand(ls.shape[0], T, 12)], 2)
    def rel(c):
        return c[..., ROOTS].permute(0, 1, 3, 2, 4).reshape(c.shape[0], T, 12, -1)
    return rel(cu), rel(cl)


def transposition_similarity(onsets):
    """[m, T, P] onsets -> [m, m, 12]: best normalised overlap of part a transposed by k (mod 12) with part b."""
    m = onsets.shape[0]
    nrm = onsets.sum((1, 2)).clamp(min=1)
    den = torch.sqrt(nrm[:, None] * nrm[None])
    out = torch.zeros(m, m, 12, device=onsets.device)
    for k in range(-36, 37):
        moved = F.pad(onsets, (max(k, 0), max(-k, 0)))[..., max(-k, 0):max(-k, 0) + P]
        out[..., k % 12] = torch.maximum(out[..., k % 12], torch.einsum("atp,btp->ab", moved, onsets) / den)
    return out * (1 - torch.eye(m, device=onsets.device))[..., None]


def analogy_inputs(ru, rl, threshold=ANALOGY_THRESHOLD):
    su, sl = transposition_similarity(ru[:, 1]), transposition_similarity(rl[:, 1])
    return torch.where(su >= threshold, su, torch.zeros_like(su)), torch.where(sl >= threshold, sl, torch.zeros_like(sl))


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
    def __init__(self, temporal_chord, d=D_MODEL, p=DROPOUT):
        super().__init__()
        self.temporal_chord = temporal_chord
        self.inp = nn.Sequential(nn.Dropout(0.1), nn.Linear(C_IN + 2 * CHORD_DIM, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))
        # chord/key templates over the combined pitch classes of both staves (summed per staff, then non-linear)
        self.chord_u = nn.Linear(N_CHROMA_U * 12, CHORD_DIM)
        self.chord_l = nn.Linear(N_CHROMA_L * 12, CHORD_DIM, bias=False)
        self.chord_mlp = nn.Sequential(nn.GELU(), nn.Linear(CHORD_DIM, CHORD_DIM), nn.GELU())
        if temporal_chord:   # harmonic progressions: temporal convolutions per root, shared over roots
            self.chord_time = nn.ModuleList([nn.Conv1d(CHORD_DIM, CHORD_DIM, 3, padding=dl, dilation=dl) for dl in (1, 2, 4)])
            self.chord_norm = nn.ModuleList([nn.GroupNorm(1, CHORD_DIM) for _ in range(3)])
        self.blocks = nn.ModuleList([ConvBlock(d, dl, p) for dl in (1, 2, 4, 8)])
        self.att = nn.Linear(d, 1)
        self.proj = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d))
        self.analogy_norm = nn.LayerNorm(d)
        self.analogy_msg = nn.Sequential(nn.Linear(d + 1, d), nn.GELU(), nn.Linear(d, d))
        self.analogy_weight = nn.Parameter(torch.tensor(4.0))
        self.axial = nn.ModuleList([nn.ModuleList([AxialLayer(d, p), AxialLayer(d, p)]) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, 1))

    def forward(self, x, chord, analogy):
        nu, nl = x.shape[:2]
        vu, vl = chord
        c = self.chord_mlp(self.chord_u(vu)[:, None] + self.chord_l(vl)[None])   # [nu, nl, T, 12 roots, CHORD_DIM]
        if self.temporal_chord:
            # eighth-note steps (reshape/expand only, deterministic on GPU)
            z = c.permute(0, 1, 3, 4, 2).reshape(nu * nl * 12, CHORD_DIM, T // 2, 2).mean(-1)
            for conv, norm in zip(self.chord_time, self.chord_norm):
                z = z + conv(F.gelu(norm(z)))
            z = z[..., None].expand(*z.shape, 2).reshape(nu * nl * 12, CHORD_DIM, T)
            c = c + z.reshape(nu, nl, 12, CHORD_DIM, T).permute(0, 1, 4, 2, 3)
        x = torch.cat([x, c.amax(3), c.mean(3)], -1)                              # pooled over roots
        h = self.inp(x.reshape(nu * nl, T, -1))
        for blk in self.blocks:
            h = blk(h)
        a = torch.softmax(self.att(h), 1)
        e = self.proj(torch.cat([(a * h).sum(1), h.amax(1)], -1)).reshape(nu, nl, -1)
        # analogy: pair (a,x) hears pairs (b,y) with b ~ a and y ~ x under the same transposition
        su, sl = analogy
        rel = torch.einsum("abk,xyk->axby", su, sl)
        strength = rel.sum((2, 3))[..., None]
        msg = torch.einsum("axby,byd->axd", rel, self.analogy_norm(e)) / (strength + 1)
        e = e + self.analogy_msg(torch.cat([msg, torch.log1p(strength)], -1))
        for row_layer, col_layer in self.axial:
            e = row_layer(e)                                   # compare candidate lowers of an upper
            e = col_layer(e.transpose(0, 1)).transpose(0, 1)   # compare candidate uppers of a lower
        S = self.head(e).squeeze(-1)
        base = normalised(S)
        N = base
        for _ in range(2):                                     # analogous pairs support each other
            N = base + self.analogy_weight * torch.einsum("axby,by->ax", rel, torch.softmax(N, 1))
        return N


def normalised(S):
    return torch.log_softmax(S, 1) + torch.log_softmax(S, 0)


def sinkhorn_log_assignment(S, iters=10):
    """Soft one-to-one assignment: each upper row sums to one, a dustbin row takes the spare lowers."""
    nu, nl = S.shape
    Z = torch.cat([S, torch.zeros(1, nl, device=S.device)], 0)
    log_r = torch.cat([torch.zeros(nu, device=S.device), torch.full((1,), float(max(nl - nu, 1)), device=S.device).log()])
    u = torch.zeros(nu + 1, device=S.device)
    v = torch.zeros(nl, device=S.device)
    for _ in range(iters):
        u = log_r - torch.logsumexp(Z + v[None], 1)
        v = -torch.logsumexp(Z + u[:, None], 0)
    return (Z + u[:, None] + v[None])[:nu]


def listwise_loss(S, onehot):
    """onehot [nu, nl]: true lower of each upper. Row CE over lowers + column CE over uppers + Sinkhorn term."""
    row = -(torch.log_softmax(S, 1) * onehot).sum(1).mean()
    St = S @ onehot.T                                           # [nu, nu]: column of each true lower
    col = -(torch.log_softmax(St, 0) * torch.eye(S.shape[0], device=S.device)).sum(0).mean()
    sink = -(sinkhorn_log_assignment(S) * onehot).sum(1).mean()
    return row + col + SINKHORN_WEIGHT * sink


def refine(N, su, sl, weight, threshold):
    """Score-level analogy refinement used after ensembling (weight and threshold tuned out of fold)."""
    su = torch.where(su >= threshold, su, torch.zeros_like(su))
    sl = torch.where(sl >= threshold, sl, torch.zeros_like(sl))
    out = N
    for _ in range(2):
        out = N + weight * torch.einsum("abk,xyk,by->ax", su, sl, torch.softmax(out, 1))
    return out


def assign(score):
    """One-to-one assignment; any upper left over (only if lowers < uppers) takes its best lower."""
    s = score.cpu().numpy()
    choice = s.argmax(1)
    r, c = linear_sum_assignment(-s)
    choice[r] = c
    return choice


# ----------------------------------------------------------------------------- training
def time_shift(r, s):
    if s == 0:
        return r
    r = torch.roll(r, s, 2)
    sl = slice(0, s) if s > 0 else slice(s, None)
    r[:, :, sl] = 0
    return r


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
    s = int(torch.randint(-4, 5, (1,), generator=g)) * 2      # same time shift for both staves
    ru, rl = time_shift(ru, s), time_shift(rl, s)
    keep_u = (torch.rand(ru.shape[0], 1, 1, P, generator=g) > 0.1).float().to(DEVICE)
    keep_l = (torch.rand(rl.shape[0], 1, 1, P, generator=g) > 0.1).float().to(DEVICE)
    return ru * keep_u, rl * keep_l


def forward(model, ru, rl, reg_shift=0.0):
    return model(pair_features(ru, rl, reg_shift), chord_inputs(ru, rl), analogy_inputs(ru, rl))


def train_model(train_ids, B, onehots, siblings, temporal_chord, seed):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = PairScorer(temporal_chord).to(DEVICE)
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
            loss = listwise_loss(forward(model, ru, rl, reg_shift), oh)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
    model.eval()
    return model


@torch.no_grad()
def score_batches(model, B, ids):
    """Normalised scores averaged over identical time shifts of both staves (each batch on its own)."""
    out = {}
    for b in ids:
        if not (B[b]["U"] and B[b]["L"]):
            continue
        S = sum(forward(model, time_shift(B[b]["ru"].clone(), s), time_shift(B[b]["rl"].clone(), s)) for s in TTA_SHIFTS)
        out[b] = normalised(S / len(TTA_SHIFTS))
    return out


@torch.no_grad()
def batch_similarities(B, ids):
    return {b: (transposition_similarity(B[b]["ru"][:, 1]), transposition_similarity(B[b]["rl"][:, 1]))
            for b in ids if B[b]["U"] and B[b]["L"]}


def accuracy(scores, sims, B, truth, ids, weight, threshold):
    hits = []
    for b in ids:
        choice = assign(refine(scores[b], *sims[b], weight, threshold))
        hits += [B[b]["L"][j] == truth[u] for u, j in zip(B[b]["U"], choice)]
    return hits


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
    test_score = {b: 0 for b in test_ids}
    oof_score = {}
    fold_of = {}
    for f, (tr_idx, va_idx) in enumerate(GroupKFold(n_splits=n_folds).split(ids, groups=composer)):
        tr = [ids[i] for i in tr_idx]
        va = [ids[i] for i in va_idx]
        siblings = {b: [o for o in tr if o != b and piece[o] == piece[b] and piece[b] != "unknown"] for b in tr}
        for b in va:
            oof_score[b] = 0
            fold_of[b] = f
        for m, temporal in enumerate(MODEL_TYPES):
            model = train_model(tr, B, onehots, siblings, temporal, SEED + 100 * f + m)
            for b, S in score_batches(model, B, va).items():
                oof_score[b] = oof_score[b] + S
            for b, S in score_batches(model, TB, test_ids).items():
                test_score[b] = test_score[b] + S
            del model
        hits = []
        for b in va:
            choice = assign(oof_score[b])
            hits += [B[b]["L"][j] == truth[u] for u, j in zip(B[b]["U"], choice)]
        print(f"fold {f}: held-out composers {sorted(set(composer[va_idx]))}  ensemble accuracy {np.mean(hits):.4f}", flush=True)

    # analogy refinement: weight and threshold chosen on out-of-fold scores (reported nested, chosen on all for test)
    sims = batch_similarities(B, ids)
    base_hits, nested_hits = [], []
    for f in range(n_folds):
        inner = [b for b in ids if fold_of[b] != f]
        outer = [b for b in ids if fold_of[b] == f]
        best = max(REFINE_GRID, key=lambda g: np.mean(accuracy(oof_score, sims, B, truth, inner, *g)))
        base_hits += accuracy(oof_score, sims, B, truth, outer, 0.0, 1.0)
        nested_hits += accuracy(oof_score, sims, B, truth, outer, *best)
    print(f"composer-grouped CV accuracy: ensemble {np.mean(base_hits):.4f}  "
          f"with analogy refinement {np.mean(nested_hits):.4f}", flush=True)
    weight, threshold = max(REFINE_GRID, key=lambda g: np.mean(accuracy(oof_score, sims, B, truth, ids, *g)))
    print(f"analogy refinement for test: weight {weight}  threshold {threshold}", flush=True)

    test_sims = batch_similarities(TB, test_ids)
    rows = []
    for b in test_ids:
        U, L = TB[b]["U"], TB[b]["L"]
        if not L:
            continue
        choice = assign(refine(test_score[b], *test_sims[b], weight, threshold))
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
