import collections, itertools, json, math, os, random, re, sys, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupKFold

T0 = time.time()
TRAIN_DEADLINE = 40 * 60
MAX_ROUNDS = 16
N_FOLDS = 5
ROLE_GATE = 0.98
MAXLEN = 96
CFG = dict(d=96, h=96, dc=96, drop=0.35, wdrop=0.1, board=True, xattn=False, aux_w=0.0, local=False, role_score=False,
           epochs=22, bs=32, lr=3e-3, wd=1e-4, col_w=0.5, clip=2.0, topk=6)

TOK = re.compile(r"\[mask\]|\[hidden\]|<num>|lex_\d+|[a-z0-9]+(?:[-'][a-z0-9]+)*|[^\w\s]")

def log(*a):
    print(f'[{time.time() - T0:7.1f}s]', *a, flush=True)

def tokenize(text):
    return TOK.findall(text.lower())

def crop(toks, mpos, maxlen=MAXLEN):
    if len(toks) <= maxlen:
        return toks, mpos
    start = max(0, min(mpos - maxlen // 2, len(toks) - maxlen))
    return toks[start:start + maxlen], mpos - start

def build_cases(df, labels=None):
    cases = []
    for r in df.itertuples(index=False):
        p = json.loads(r.packet_json)
        sn = {s['snippet_id']: s['text'] for s in p['snippets']}
        toks, mp = [], []
        for i in range(1, 5):
            t = tokenize(sn[f'q{i}'])
            m = t.index('[mask]') if '[mask]' in t else 0
            t, m = crop(t, m)
            toks.append(t); mp.append(m)
        ids = [c['card_id'] for c in p['candidate_cards']]
        c = dict(case_id=r.case_id, toks=toks, mpos=mp, ids=ids, cards=[c['text'] for c in p['candidate_cards']])
        if labels is not None:
            c['tgt'] = [ids.index(a) for a in json.loads(labels[r.case_id])['repair_sequence']]
        cases.append(c)
    return cases

class Vocab:
    def __init__(self, cases, min_freq=2):
        cnt = collections.Counter(w for c in cases for t in c['toks'] for w in t)
        self.w2i = {'<pad>': 0, '<unk>': 1}
        for w, n in cnt.items():
            if n >= min_freq:
                self.w2i[w] = len(self.w2i)
        phrases = sorted({p for c in cases for p in c['cards']})
        self.p2i = {'<unk>': 0}
        for p in phrases:
            self.p2i[p] = len(self.p2i)
        self.cw2i = {'<unk>': 0}
        for w in collections.Counter(w for p in phrases for w in tokenize(p)):
            self.cw2i[w] = len(self.cw2i)

def encode(cases, vocab, role):
    N = len(cases)
    X = np.zeros((N, 4, MAXLEN), np.int64)
    mpos = np.zeros((N, 4), np.int64)
    pid = np.zeros((N, 24), np.int64)
    cw = np.zeros((N, 24, 6), np.int64)
    crole = np.full((N, 24), -1, np.int64)
    for n, c in enumerate(cases):
        for i, t in enumerate(c['toks']):
            X[n, i, :len(t)] = [vocab.w2i.get(w, 1) for w in t]
        mpos[n] = c['mpos']
        for j, p in enumerate(c['cards']):
            pid[n, j] = vocab.p2i.get(p, 0)
            ws = [vocab.cw2i.get(w, 0) + 1 for w in tokenize(p)][:6]
            cw[n, j, :len(ws)] = ws
            crole[n, j] = role.get(p, -1)
    out = dict(x=torch.from_numpy(X), mpos=torch.from_numpy(mpos), pid=torch.from_numpy(pid),
               cw=torch.from_numpy(cw), crole=torch.from_numpy(crole))
    if 'tgt' in cases[0]:
        out['tgt'] = torch.tensor([c['tgt'] for c in cases])
    return out

def batch(E, idx):
    b = {k: v[idx] for k, v in E.items()}
    T = max(int((b['x'] != 0).sum(-1).max()), 1)
    b['x'] = b['x'][:, :, :T]
    return b

def infer_roles(cases):
    co = collections.defaultdict(collections.Counter); freq = collections.Counter()
    for c in cases:
        t = [c['cards'][j] for j in c['tgt']]
        for a in t:
            freq[a] += 1
            for b in t:
                if a != b:
                    co[a][b] += 1
    phr = sorted({p for c in cases for p in c['cards']})
    best = max(cases, key=lambda c: min(freq[c['cards'][j]] for j in c['tgt']))
    seed = [best['cards'][j] for j in best['tgt']]
    role = {p: i for i, p in enumerate(seed)}
    for _ in range(12):
        new = dict(role)
        for p in phr:
            if p in seed:
                continue
            sc = np.zeros(4)
            for q, n in co[p].items():
                if q in role:
                    sc[role[q]] += n
            o = np.sort(sc)
            if o[1] >= 2 and o[0] <= 0.1 * o[1]:
                new[p] = int(np.argmin(sc))
        role = new
    for _ in range(10):
        full = [c for c in cases if all(p in role for p in c['cards'])]
        if not full:
            break
        mode = [collections.Counter(sum(role[p] == r for p in c['cards']) for c in full).most_common(1)[0][0]
                for r in range(4)]
        votes = collections.defaultdict(lambda: np.zeros(4))
        for c in cases:
            unk = [p for p in c['cards'] if p not in role]
            if not unk:
                continue
            cnt = collections.Counter(role[p] for p in c['cards'] if p in role)
            deficit = np.array([max(0, mode[r] - cnt[r]) for r in range(4)], float)
            if deficit.sum() == len(unk) and (deficit > 0).sum() == 1:
                for p in unk:
                    votes[p] += deficit / deficit.sum()
        added = 0
        for p, v in votes.items():
            o = np.sort(v)
            if o[-1] >= 2 and o[-2] <= 0.1 * o[-1]:
                role[p] = int(np.argmax(v)); added += 1
        if added == 0:
            break
    return role

def role_check(cases, role):
    d4 = np.mean([len({role.get(c['cards'][j], -1) for j in c['tgt']} - {-1}) == 4 for c in cases])
    ok = np.mean([(lambda r: len(r) == len(set(r)))([role[c['cards'][j]] for j in c['tgt'] if c['cards'][j] in role])
                  for c in cases])
    return float(d4), float(ok)

class Net(nn.Module):
    def __init__(self, vocab, cfg):
        super().__init__()
        d, h, dc, drop = cfg['d'], cfg['h'], cfg['dc'], cfg['drop']
        self.cfg, self.dc = cfg, dc
        self.emb = nn.Embedding(len(vocab.w2i), d, padding_idx=0)
        self.gru = nn.GRU(d, h, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(drop)
        self.qmlp = nn.Sequential(nn.Linear((8 if cfg['local'] else 6) * h, 2 * dc), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * dc, dc))
        if cfg['xattn']:
            lay = nn.TransformerEncoderLayer(d_model=dc, nhead=4, dim_feedforward=2 * dc, dropout=drop,
                                             batch_first=True, activation='gelu')
            self.snipenc = nn.TransformerEncoder(lay, num_layers=1)
        self.pemb = nn.Embedding(len(vocab.p2i), dc)
        self.pbias = nn.Embedding(len(vocab.p2i), 1)
        nn.init.zeros_(self.pbias.weight)
        self.cwemb = nn.Embedding(len(vocab.cw2i) + 1, dc, padding_idx=0)
        if cfg['board']:
            self.A = nn.Parameter(torch.zeros(len(vocab.p2i), len(vocab.p2i)))
        if cfg['aux_w'] > 0 or cfg['role_score']:
            self.rolehead = nn.Linear(dc, 4)

    def forward(self, b):
        x, mpos, pid, cw = b['x'], b['mpos'], b['pid'], b['cw']
        B, S, T = x.shape
        xf = x.reshape(B * S, T)
        if self.training and self.cfg['wdrop'] > 0:
            drop = (torch.rand(xf.shape) < self.cfg['wdrop']) & (xf > 1)
            xf = torch.where(drop, torch.ones_like(xf), xf)
        pad = (x.reshape(B * S, T) != 0)
        H, _ = self.gru(self.drop(self.emb(xf)))
        hm = H[torch.arange(B * S), mpos.reshape(-1)]
        m = pad.unsqueeze(-1).float()
        mean = (H * m).sum(1) / m.sum(1).clamp(min=1)
        case = mean.view(B, S, -1).mean(1, keepdim=True).expand(B, S, -1).reshape(B * S, -1)
        feats = [hm, mean, case]
        if self.cfg['local']:
            ar = torch.arange(T).unsqueeze(0)
            win = ((ar - mpos.reshape(-1, 1)).abs() <= 3).unsqueeze(-1).float() * m
            feats.append((H * win).sum(1) / win.sum(1).clamp(min=1))
        q = self.qmlp(self.drop(torch.cat(feats, -1))).view(B, S, self.dc)
        if self.cfg['xattn']:
            q = q + self.snipenc(q)
        cand = self.pemb(pid) + self.cwemb(cw).sum(2)
        Sc = torch.einsum('bsd,bcd->bsc', q, cand) + self.pbias(pid).transpose(1, 2)
        if self.cfg['board']:
            Bm = torch.zeros(B, self.A.shape[0]); Bm.scatter_(1, pid, 1.0); Bm[:, 0] = 0
            Sc = Sc + torch.einsum('bcp,bp->bc', self.A[pid], Bm).unsqueeze(1)
        aux = self.rolehead(q) if hasattr(self, 'rolehead') else None
        if self.cfg['role_score']:
            lr = F.log_softmax(aux, -1)
            cr = b['crole']
            g = torch.gather(lr, 2, cr.clamp(min=0).unsqueeze(1).expand(B, S, 24))
            Sc = Sc + g * (cr >= 0).unsqueeze(1).float()
        return Sc, aux

def loss_fn(S, aux, b, cfg):
    B = S.shape[0]; tgt = b['tgt']
    row = F.cross_entropy(S.reshape(B * 4, 24), tgt.reshape(-1))
    cols = torch.gather(S, 2, tgt.unsqueeze(1).expand(B, 4, 4))
    col = F.cross_entropy(cols.permute(0, 2, 1).reshape(B * 4, 4), torch.arange(4).repeat(B))
    loss = row + cfg['col_w'] * col
    if aux is not None and cfg['aux_w'] > 0:
        rt = torch.gather(b['crole'], 1, tgt)
        if (rt >= 0).any():
            loss = loss + cfg['aux_w'] * F.cross_entropy(aux.reshape(B * 4, 4), rt.reshape(-1), ignore_index=-1)
    return loss

def train_model(E, vocab, cfg, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    net = Net(vocab, cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    n = E['x'].shape[0]; bs = cfg['bs']
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=cfg['epochs'] * math.ceil(n / bs),
                                                pct_start=0.15)
    rng = np.random.RandomState(seed)
    for _ in range(cfg['epochs']):
        net.train(); perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = batch(E, torch.from_numpy(perm[i:i + bs]))
            S, aux = net(b)
            loss = loss_fn(S, aux, b, cfg)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), cfg['clip']); opt.step(); sched.step()
    return net

@torch.no_grad()
def predict(net, E, bs=256):
    net.eval(); out = []
    n = E['x'].shape[0]
    for i in range(0, n, bs):
        S, _ = net(batch(E, torch.arange(i, min(n, i + bs))))
        out.append(F.log_softmax(S, -1))
    return torch.cat(out).numpy()

def decode(L, cards_list, role, use_roles, K=6):
    preds = []
    for s, cards in zip(L, cards_list):
        M = (s - np.logaddexp.reduce(s, axis=1, keepdims=True)) + (s - np.logaddexp.reduce(s, axis=0, keepdims=True))
        bp = None
        if use_roles:
            rl = np.array([role.get(p, -1) for p in cards])
            top = np.argsort(-M, axis=1)[:, :K]
            combos = np.array(list(itertools.product(*top)))
            val = M[np.arange(4), combos].sum(1)
            cs = np.sort(combos, 1)
            ok = (cs[:, 1:] != cs[:, :-1]).all(1)
            r = rl[combos]
            for a, b in itertools.combinations(range(4), 2):
                ok &= ~((r[:, a] == r[:, b]) & (r[:, a] >= 0))
            if ok.any():
                bp = combos[np.flatnonzero(ok)[np.argmax(val[ok])]]
        if bp is None:
            ri, ci = linear_sum_assignment(-M); bp = ci[np.argsort(ri)]
        preds.append([int(j) for j in bp])
    return preds

def score_fn(pred, true):
    raw = float(np.mean([p == t for P, T in zip(pred, true) for p, t in zip(P, T)]))
    return max(0.0, min(1.0, (raw - 1 / 24) / (1 - 1 / 24))), raw

def write_submission(path, case_ids, seqs):
    rows = [json.dumps({'repair_sequence': list(s)}, separators=(',', ':')) for s in seqs]
    pd.DataFrame({'case_id': case_ids, 'answer_json': rows}).to_csv(path, index=False)

def validate_submission(path, test_ids):
    sub = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(sub.columns) == ['case_id', 'answer_json'], sub.columns
    assert len(sub) == len(test_ids) and sub.case_id.is_unique and set(sub.case_id) == set(test_ids), 'case ids'
    valid = {f'c{i:02d}' for i in range(24)}
    for a in sub.answer_json:
        pairs = json.loads(a, object_pairs_hook=lambda kv: kv)
        assert [k for k, _ in pairs] == ['repair_sequence'], a
        seq = pairs[0][1]
        assert isinstance(seq, list) and len(seq) == 4 and len(set(seq)) == 4 and all(s in valid for s in seq), a
    return True

def prepare_folds(train_cases, groups, test_cases=None):
    folds = []
    for f, (ti, vi) in enumerate(GroupKFold(N_FOLDS).split(np.zeros(len(train_cases)), groups=groups)):
        tr = [train_cases[i] for i in ti]; va = [train_cases[i] for i in vi]
        vocab = Vocab(tr); role = infer_roles(tr)
        d4, ok = role_check(tr, role); d4v, okv = role_check(va, role)
        use_roles = d4 >= ROLE_GATE
        log(f'fold{f}: train={len(tr)} val={len(va)} vocab={len(vocab.w2i)} roles={len(role)} '
            f'4-distinct-known-roles train={d4:.4f} (held-out {d4v:.4f}) no-conflict train={ok:.4f} '
            f'-> role features {"ON" if use_roles else "OFF"}')
        rr = role if use_roles else {}
        folds.append(dict(vi=vi, vocab=vocab, role=role, use_roles=use_roles,
                          Etr=encode(tr, vocab, rr), Eva=encode(va, vocab, rr),
                          Ete=encode(test_cases, vocab, rr) if test_cases is not None else None,
                          va_cards=[c['cards'] for c in va], va_tgt=[c['tgt'] for c in va]))
    return folds

def oof_score(folds, preds, k, K):
    pred, true = [], []
    for f, fd in enumerate(folds):
        rs = [r for r in preds[f] if r < k]
        if not rs:
            return None
        pred += decode(sum(preds[f][r] for r in rs) / len(rs), fd['va_cards'], fd['role'], fd['use_roles'], K)
        true += fd['va_tgt']
    return score_fn(pred, true)

_FOLDS, _CFG, _WANT_TEST = None, None, False

def _worker_init():
    torch.set_num_threads(1)

def _fit_one(task):
    f, r = task
    t = time.time()
    fd = _FOLDS[f]
    net = train_model(fd['Etr'], fd['vocab'], _CFG, seed=1000 * r + f)
    pv = predict(net, fd['Eva'])
    pt = predict(net, fd['Ete']) if _WANT_TEST else None
    return f, r, pv, pt, time.time() - t

def run_cv(folds, cfg, rounds=None, deadline=None, want_test=False, workers=None):
    global _FOLDS, _CFG, _WANT_TEST
    _FOLDS, _CFG, _WANT_TEST = folds, cfg, want_test
    import multiprocessing as mp
    nf = len(folds); max_r = rounds if rounds is not None else MAX_ROUNDS
    workers = workers or os.cpu_count() or 1
    preds = [dict() for _ in folds]; counts = [0] * nf
    done_r = collections.Counter()
    te_sum, te_n, history = None, 0, []
    tasks = [(f, r) for r in range(max_r) for f in range(nf)]
    t_model, first_logged, next_round = None, False, 0
    ctx = mp.get_context('fork')
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        pending, ti = [], 0
        while ti < len(tasks) or pending:
            while ti < len(tasks) and len(pending) < workers:
                if deadline is not None and t_model is None and ti >= workers:
                    break
                if deadline is not None and t_model is not None and time.time() - T0 + t_model > deadline:
                    log(f'time guard: not starting more models ({ti} started, est. {t_model:.0f}s each)')
                    tasks = tasks[:ti]; break
                pending.append(pool.apply_async(_fit_one, (tasks[ti],))); ti += 1
            if not pending:
                break
            while not any(p.ready() for p in pending):
                time.sleep(0.2)
            for p in [p for p in pending if p.ready()]:
                pending.remove(p)
                f, r, pv, pt, dt = p.get()
                preds[f][r] = pv; counts[f] += 1; done_r[r] += 1
                if pt is not None:
                    te_sum = pt if te_sum is None else te_sum + pt; te_n += 1
                t_model = dt if t_model is None else max(0.7 * t_model + 0.3 * dt, dt)
                if not first_logged and deadline is not None:
                    first_logged = True
                    left = deadline - (time.time() - T0)
                    k = min(max_r, int(workers * max(left, 0) // (nf * dt)) + 1)
                    log(f'first model took {dt:.1f}s ({workers} parallel workers); {left:.0f}s left '
                        f'-> planning about {k} models per fold ({k * nf} total)')
            while done_r[next_round] == nf:
                sc = oof_score(folds, preds, next_round + 1, cfg['topk'])
                history.append((next_round + 1, sc[0], sc[1], time.time() - T0))
                log(f'round {next_round + 1}: models/fold={counts} OOF score={sc[0]:.4f} (raw {sc[1]:.4f})')
                next_round += 1
    if counts and min(counts) > 0 and len(set(counts)) > 1:
        sc = oof_score(folds, preds, max_r, cfg['topk'])
        history.append((min(counts), sc[0], sc[1], time.time() - T0))
        log(f'final (uneven rounds): models/fold={counts} OOF score={sc[0]:.4f} (raw {sc[1]:.4f})')
    return dict(preds=preds, counts=counts, te_sum=te_sum, te_n=te_n, history=history)

def main():
    if len(sys.argv) != 3:
        sys.exit('usage: python3 solution.py PUBLIC_DIR SUBMISSION_OUT')
    pub, out = sys.argv[1], sys.argv[2]
    log(f'cpu_count={os.cpu_count()} torch={torch.__version__}; training uses {os.cpu_count()} worker processes x 1 thread')

    te_df = pd.read_csv(os.path.join(pub, 'test.csv'))
    test_ids = te_df.case_id.tolist()
    write_submission(out, test_ids, [[f'c{i:02d}' for i in range(4)]] * len(test_ids))
    validate_submission(out, test_ids)
    log(f'fallback submission written to {out}')

    tr_df = pd.read_csv(os.path.join(pub, 'train.csv'))
    tg = pd.read_csv(os.path.join(pub, 'train_targets.csv'))
    labels = dict(zip(tg.case_id, tg.answer_json))
    train_cases = build_cases(tr_df, labels)
    test_cases = build_cases(te_df)
    log(f'train={len(train_cases)} test={len(test_cases)}')

    role_full = infer_roles(train_cases)
    d4, ok = role_check(train_cases, role_full)
    use_roles_test = d4 >= ROLE_GATE
    log(f'full-train roles: {len(role_full)} phrases assigned; 4-distinct-known-roles={d4:.4f} '
        f'no-conflict={ok:.4f} -> role decoding for test {"ON" if use_roles_test else "OFF"}')
    if not use_roles_test:
        log('WARNING: role check below gate, role features disabled for test decoding')

    folds = prepare_folds(train_cases, tr_df.validation_group.values, test_cases)
    log(f'config: {CFG}')
    res = run_cv(folds, CFG, deadline=TRAIN_DEADLINE, want_test=True)

    if res['history']:
        rr, sc, raw, _ = res['history'][-1]
        log(f'FINAL out-of-fold score (GroupKFold({N_FOLDS}) by validation_group, {rr} models/fold): '
            f'{sc:.4f} (raw slot accuracy {raw:.4f})')
        if sc < 0.78:
            log('WARNING: out-of-fold score is below the 0.78 go/no-go gate; this will probably not reach 0.75 on test')

    if res['te_n'] > 0:
        L = res['te_sum'] / res['te_n']
        preds = decode(L, [c['cards'] for c in test_cases], role_full, use_roles_test, CFG['topk'])
        seqs = [[c['ids'][j] for j in p] for c, p in zip(test_cases, preds)]
        write_submission(out, test_ids, seqs)
        validate_submission(out, test_ids)
        log(f'submission written and validated: {out} ({res["te_n"]} models averaged)')
    else:
        log('no model finished in time; fallback submission kept')
    log(f'total runtime {time.time() - T0:.1f}s')

if __name__ == '__main__':
    main()
