import sys, os, re, json, math, time, copy, datetime as dt
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel

T_START = time.time()
TRAIN_DEADLINE = 45 * 60
SEED = 42
K_SHORT = 300
BACKBONE = 'intfloat/multilingual-e5-small'
CUT = 8

public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('./dataset/public')
submission_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('./working/submission.csv')

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(max(1, os.cpu_count() or 1))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log(*a):
    print(f'[{time.time() - T_START:7.1f}s]', *a, flush=True)


def _ord(y, m, d):
    try:
        return dt.date(y, m, d).toordinal()
    except ValueError:
        try:
            return dt.date(y, m, min(d, 28)).toordinal()
        except ValueError:
            return None


def parse_meta_date(s):
    if not isinstance(s, str) or not s.strip():
        return (np.nan, np.nan)

    def one(p, end):
        p = p.strip()
        if not p:
            return None
        parts = p.split('-')
        try:
            y = int(parts[0])
            if len(parts) == 1:
                return _ord(y, 12, 31) if end else _ord(y, 1, 1)
            m = int(parts[1])
            if len(parts) == 2:
                return _ord(y, m, 28 if end else 1)
            return _ord(y, m, int(parts[2]))
        except Exception:
            return None

    if '/' in s:
        a, b = s.split('/', 1)
        lo, hi = one(a, False), one(b, True)
        if lo is None and hi is None:
            return (np.nan, np.nan)
        if lo is None:
            lo = hi
        if hi is None:
            hi = lo
        return (float(lo), float(hi))
    lo, hi = one(s, False), one(s, True)
    if lo is None:
        return (np.nan, np.nan)
    return (float(lo), float(hi))


MONTHS = {
    1: 'jan januar jänner jenner janv janvier january januarii',
    2: 'feb febr februar fevr février fevrier february februarii',
    3: 'mär märz mærz maerz marz mart martz merz mars march martii',
    4: 'apr april aprill avril aprilis',
    5: 'mai may maj majus',
    6: 'jun juni juny junius juin june',
    7: 'jul juli july julius juillet juil',
    8: 'aug august augusti août aout augst',
    9: 'sep sept september septbr 7br 7ber 7bre 7b 7bris',
    10: 'oct okt october oktober octob octbr octobre 8br 8ber 8bre 8b 8bris',
    11: 'nov november novbr novembre 9br 9ber 9bre 9b 9bris',
    12: 'dec dez december dezember decbr décembre decembre xbr xber xbre 10br 10ber 10bre 10b xbris',
}
MONTH_OF = {w: m for m, ws in MONTHS.items() for w in ws.split()}
_MW = '|'.join(sorted(map(re.escape, MONTH_OF), key=len, reverse=True))
DAY = r'(\d{1,2})\s*(?:ten|ter|sten|st|en|t|n|e)?\s*[.:]?'
RE_DMY = re.compile(DAY + r'\s*(?:d[.:]\s*)?(' + _MW + r')\b\.?[,]?\s*(\d{4}|\d{2}(?!\d))?', re.I)
RE_REL = re.compile(DAY + r'\s*(v\.?\s*M|vor\.?\s*M|vorigen\s+Monats|d\.?\s*[:.]?\s*M|dieses|huj|h\.?\s*m|c\.?\s*m|curr)\b', re.I)
RE_NUM = re.compile(r'\b(\d{1,2})\s*[./]\s*(\d{1,2})\s*[./]\s*(1[78]\d\d|\d\d)\b')
RE_DAY = re.compile(r'(?:\bd[.:]|\bden|\bam|\bvom|\bunterm|\bbis)\s*(\d{1,2})\s*(?:ten|ter|sten|st|en|t)?\b(?!\s*(?:Uhr|' + _MW + r'))', re.I)
RE_BTW = re.compile(r'\b(?:btw|bntw|beantw\w*|beantwortet|antw)\b\W{0,40}?(?:d\W{0,3})?(\d{1,2})\b(?!\s*(?:\d|' + _MW + r'))', re.I)
RE_NO = re.compile(r'\b(?:No|Nro|Nr|N°|Numero)\s*[:.]?\s*(\d{1,3})\b', re.I)
RE_INT = re.compile(r'(?<![\d/])(\d{1,3})(?![\d/])')
WORD = re.compile(r"[A-Za-zÄÖÜäöüßéèàáíóúçæœ]+")
STOP = set('von van de der den des du la le graf gräfin grafen freiherr freyfrau freifrau frau herr herrn und the of zu zum baron baronin fürst fürstin herzog herzogin prinz prinzessin könig königin königl kgl mad madame mme mlle monsieur sir lord lady dr prof st et al geb geborene gen genannt ritter edler edle'.split())


def _resolve(day, month, year, ref):
    if year is not None:
        return _ord(year, month, day)
    if ref is None or np.isnan(ref):
        return None
    ry = dt.date.fromordinal(int(ref)).year
    best = None
    for y in (ry - 1, ry, ry + 1):
        o = _ord(y, month, day)
        if o is None:
            continue
        diff = o - ref
        cost = abs(diff) if diff <= 20 else abs(diff) * 3
        if best is None or cost < best[0]:
            best = (cost, o)
    return None if best is None else best[1]


def _year(ystr, ref):
    if not ystr:
        return None
    y = int(ystr)
    if y < 100:
        if ref is None or np.isnan(ref):
            y += 1800
        else:
            cy = dt.date.fromordinal(int(ref)).year
            y += (cy // 100) * 100
            if y > cy + 5:
                y -= 100
    return y if 1700 <= y <= 1950 else None


def extract_dates(text, ref):
    out, taken = [], []
    for m in RE_DMY.finditer(text):
        d = int(m.group(1))
        mon = MONTH_OF.get(m.group(2).lower())
        if not (1 <= d <= 31) or mon is None:
            continue
        y = _year(m.group(3), ref)
        o = _resolve(d, mon, y, ref)
        if o is not None:
            out.append((o, m.start(), m.end(), 0 if y else 1))
            taken.append((m.start(), m.end()))
    has_ref = ref is not None and not np.isnan(ref)
    if has_ref:
        rd = dt.date.fromordinal(int(ref))
        for m in RE_REL.finditer(text):
            d = int(m.group(1))
            if not (1 <= d <= 31):
                continue
            y, mo = rd.year, rd.month
            if m.group(2).lower().startswith('v'):
                mo -= 1
                if mo == 0:
                    mo, y = 12, y - 1
            o = _ord(y, mo, d)
            if o is not None:
                out.append((o, m.start(), m.end(), 2))
                taken.append((m.start(), m.end()))
    for m in RE_NUM.finditer(text):
        d, mo = int(m.group(1)), int(m.group(2))
        if not (1 <= d <= 31 and 1 <= mo <= 12):
            continue
        y = _year(m.group(3), ref)
        o = _resolve(d, mo, y, ref)
        if o is not None:
            out.append((o, m.start(), m.end(), 0))
            taken.append((m.start(), m.end()))
    if has_ref:
        rd = dt.date.fromordinal(int(ref))
        for m in RE_BTW.finditer(text):
            d = int(m.group(1))
            if not (1 <= d <= 31) or any(a <= m.start(1) < b for a, b in taken):
                continue
            y, mo = rd.year, rd.month
            if d < rd.day:
                mo += 1
                if mo == 13:
                    mo, y = 1, y + 1
            o = _ord(y, mo, d)
            if o is not None:
                out.append((o, m.start(), m.end(), 4))
                taken.append((m.start(1), m.end(1)))
        for m in RE_DAY.finditer(text):
            if any(a <= m.start() < b for a, b in taken):
                continue
            d = int(m.group(1))
            if not (1 <= d <= 31):
                continue
            y, mo = rd.year, rd.month
            if d > rd.day + 3:
                mo -= 1
                if mo == 0:
                    mo, y = 12, y - 1
            o = _ord(y, mo, d)
            if o is not None:
                out.append((o, m.start(), m.end(), 3))
    return out


def split_people(s):
    if not isinstance(s, str):
        return []
    return [p.strip() for p in s.split(';') if p.strip()]


def name_tokens(p):
    return tuple(t for t in re.findall(r"[a-zäöüßéèàáíóúçæœ]+", p.lower()) if t not in STOP and len(t) > 1)


def surname(p):
    p = p.strip()
    if ',' in p:
        s = p.split(',')[0]
    else:
        toks = p.split()
        s = toks[-1] if toks else p
    return re.sub(r'[^a-zäöüßéèàáíóúçæœ\- ]', '', s.strip().lower()).strip()


def header_number(text):
    m = RE_NO.search(text[:300])
    return int(m.group(1)) if m else -1


def text_surnames(text, vocab, lo=0, hi=None):
    out = set()
    for m in WORD.finditer(text[lo:hi] if hi is not None or lo else text):
        w = m.group(0).lower()
        for c in (w, w[:-1] if w.endswith('s') else None, w[:-2] if w.endswith('en') else None):
            if c and c in vocab:
                out.add(c)
    return out


def ranks_between(mask, cmid, qmid):
    out = np.full(len(cmid), 50.0)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return out
    d = np.sort(cmid[idx])
    for i in idx:
        c = cmid[i]
        if c <= qmid:
            n = np.searchsorted(d, qmid, 'left') - np.searchsorted(d, c, 'right')
        else:
            n = np.searchsorted(d, c, 'left') - np.searchsorted(d, qmid, 'right')
        out[i] = min(max(n, 0), 50)
    return out


class Archive:
    def __init__(self, letters):
        self.df = letters.reset_index(drop=True)
        self.ids = self.df.letter_id.astype(str).values
        self.idx = {l: i for i, l in enumerate(self.ids)}
        N = self.N = len(self.df)
        d = np.array([parse_meta_date(s) for s in self.df.date], dtype=float)
        self.lo, self.hi = d[:, 0], d[:, 1]
        self.has_date = ~np.isnan(self.lo)
        gm = np.nanmedian(self.lo)
        self.lo_f = np.where(self.has_date, self.lo, gm)
        self.hi_f = np.where(self.has_date, self.hi, gm)
        self.mid = (self.lo_f + self.hi_f) / 2.0
        persons = {}

        def pid(p):
            if p not in persons:
                persons[p] = len(persons)
            return persons[p]

        self.snd = [[pid(p) for p in split_people(s)] for s in self.df.sender]
        self.rcp = [[pid(p) for p in split_people(s)] for s in self.df.recipient]
        self.pnames = [None] * len(persons)
        for p, i in persons.items():
            self.pnames[i] = p
        P = len(persons)
        self.ptok = [name_tokens(p) for p in self.pnames]
        self.pgiven = [set(self.ptok[j]) - set(name_tokens(surname(self.pnames[j]))) for j in range(P)]
        tokdf = {}
        for i in range(N):
            for t in set(t for q in self.snd[i] + self.rcp[i] for t in self.ptok[q]):
                tokdf[t] = tokdf.get(t, 0) + 1
        self.tokidf = {t: math.log((N + 1) / (c + 0.5)) for t, c in tokdf.items()}
        pdf = np.zeros(P)
        for i in range(N):
            for q in set(self.snd[i] + self.rcp[i]):
                pdf[q] += 1
        self.pidf = np.log((N + 1) / (pdf + 0.5))

        def mat(lst):
            r = [i for i, l in enumerate(lst) for _ in l]
            c = [q for l in lst for q in l]
            return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(N, P))

        self.Ms, self.Mr = mat(self.snd), mat(self.rcp)
        self._cache = {}

        def sset(lst):
            out = set()
            for q in lst:
                x = surname(self.pnames[q])
                if len(x) >= 4 and x not in STOP and ' ' not in x:
                    out.add(x)
            return out

        self.csur_s = [sset(self.snd[i]) for i in range(N)]
        self.csur_r = [sset(self.rcp[i]) for i in range(N)]
        self.csur = [self.csur_s[i] | self.csur_r[i] for i in range(N)]
        sdf = {}
        for ss in self.csur:
            for s in ss:
                sdf[s] = sdf.get(s, 0) + 1
        self.suridf = {s: math.log((N + 1) / (c + 0.5)) for s, c in sdf.items()}
        self.tsur = [text_surnames(t, self.suridf) for t in self.df.text]
        self.hnum = np.array([header_number(t) for t in self.df.text])
        ci, co = [], []
        for i, t in enumerate(self.df.text):
            ref = self.mid[i] if self.has_date[i] else np.nan
            for o, a, b, k in extract_dates(t, ref):
                ci.append(i)
                co.append(o)
        self.ctd_idx = np.array(ci, dtype=int)
        self.ctd_ord = np.array(co, dtype=float)
        self.repo = self.df.repository.fillna('').astype(str).values
        self.prov = self.df.provenance.fillna('').astype(str).values
        self.shelf_tok = [set(re.findall(r'[a-z]+|\d+', s.lower())) if isinstance(s, str) else set() for s in self.df.shelfmark]
        self.loglen = np.log1p(self.df.text.str.len().values)
        self.tfv = TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, token_pattern=r'(?u)\b\w\w+\b', lowercase=True)
        self.tfX = self.tfv.fit_transform(self.df.text.values)

    def person_sim(self, name):
        if name in self._cache:
            return self._cache[name]
        a = set(name_tokens(name))
        wa = sum(self.tokidf.get(t, 8.0) for t in a)
        v = np.zeros(len(self.pnames))
        if a:
            for j, tb in enumerate(self.ptok):
                inter = a.intersection(tb)
                if inter:
                    wi = sum(self.tokidf.get(t, 8.0) for t in inter)
                    v[j] = wi / (wa + sum(self.tokidf.get(t, 8.0) for t in set(tb) - a))
        self._cache[name] = v
        return v

    def given_sim(self, name):
        key = ('g', name)
        if key in self._cache:
            return self._cache[key]
        a = set(name_tokens(name)) - set(name_tokens(surname(name)))
        v = np.zeros(len(self.pnames))
        if a:
            wa = sum(self.tokidf.get(t, 8.0) for t in a)
            for j, gb in enumerate(self.pgiven):
                inter = a & gb
                if inter:
                    v[j] = sum(self.tokidf.get(t, 8.0) for t in inter) / max(wa, sum(self.tokidf.get(t, 8.0) for t in gb))
        self._cache[key] = v
        return v

    def side_sim(self, people, M):
        best, bestw = np.zeros(self.N), np.zeros(self.N)
        for p in people:
            v = self.person_sim(p)
            best = np.maximum(best, M.multiply(v[None, :]).max(axis=1).toarray().ravel())
            bestw = np.maximum(bestw, M.multiply((v * self.pidf)[None, :]).max(axis=1).toarray().ravel())
        return best, bestw

    def side_given(self, people, M):
        best = np.zeros(self.N)
        for p in people:
            best = np.maximum(best, M.multiply(self.given_sim(p)[None, :]).max(axis=1).toarray().ravel())
        return best


def query_features(A, q, pop_counts):
    N = A.N
    text = q['text'] if isinstance(q['text'], str) else ''
    s0, e0 = int(q['mention_start']), int(q['mention_end'])
    qlo, qhi = parse_meta_date(q['date'])
    qhas = not np.isnan(qlo)
    self_i = A.idx.get(str(q['document_id']), -1)
    if not qhas and self_i >= 0 and A.has_date[self_i]:
        qlo, qhi, qhas = A.lo[self_i], A.hi[self_i], True
    qmid = (qlo + qhi) / 2.0 if qhas else np.nan
    f = {}
    qs, qr = split_people(q['sender']), split_people(q['recipient'])
    ss, ssw = A.side_sim(qs, A.Ms)
    sr, srw = A.side_sim(qs, A.Mr)
    rs, rsw = A.side_sim(qr, A.Ms)
    rr, rrw = A.side_sim(qr, A.Mr)
    f['p_ss'], f['p_sr'], f['p_rs'], f['p_rr'] = ss, sr, rs, rr
    f['pw_ss'], f['pw_sr'], f['pw_rs'], f['pw_rr'] = ssw, srw, rsw, rrw
    rev, same = np.minimum(sr, rs), np.minimum(ss, rr)
    f['p_rev'], f['p_same'], f['p_pair'] = rev, same, np.maximum(rev, same)

    def pidf_of(names):
        best = 0.0
        for p in names:
            v = A.person_sim(p)
            j = int(np.argmax(v))
            best = max(best, A.pidf[j] if v[j] > 0.7 else 8.0)
        return best

    qs_idf, qr_idf = pidf_of(qs), pidf_of(qr)
    cp = np.maximum(sr, ss) if qs_idf >= qr_idf else np.maximum(rs, rr)
    hub = np.maximum(rs, rr) if qs_idf >= qr_idf else np.maximum(sr, ss)
    f['p_cp'], f['p_hub'] = cp, hub
    f['q_idf_max'] = np.full(N, max(qs_idf, qr_idf))
    f['q_idf_min'] = np.full(N, min(qs_idf, qr_idf))
    f['p_any'] = np.maximum.reduce([ss, sr, rs, rr])
    gss, gsr = A.side_given(qs, A.Ms), A.side_given(qs, A.Mr)
    grs, grr = A.side_given(qr, A.Ms), A.side_given(qr, A.Mr)
    f['g_same'] = np.minimum(np.maximum(ss, gss), np.maximum(rr, grr))
    f['g_rev'] = np.minimum(np.maximum(sr, gsr), np.maximum(rs, grs))
    cm = A.mid
    if qhas:
        g = cm - qmid
        f['d_has'] = A.has_date.astype(float)
        f['d_sign'] = np.sign(g) * A.has_date
        f['d_log'] = np.log1p(np.abs(g)) * A.has_date + (~A.has_date) * 8
        f['d_before'] = ((A.hi_f <= qhi) & A.has_date).astype(float)
        f['d_same'] = ((np.abs(g) <= 1) & A.has_date).astype(float)
    else:
        for k in ('d_has', 'd_sign', 'd_log', 'd_before', 'd_same'):
            f[k] = np.zeros(N)
        f['d_log'] = np.full(N, 8.0)
    f['d_cunc'] = np.log1p(A.hi_f - A.lo_f)
    f['d_qunc'] = np.full(N, np.log1p(qhi - qlo) if qhas else 8.0)
    qm = qmid if qhas else float(np.nanmedian(cm))
    f['r_rev'] = np.log1p(ranks_between(rev >= 0.5, cm, qm))
    f['r_same'] = np.log1p(ranks_between(same >= 0.5, cm, qm))
    f['r_pair'] = np.log1p(ranks_between(f['p_pair'] >= 0.5, cm, qm))
    f['r_cp'] = np.log1p(ranks_between(cp >= 0.5, cm, qm))
    ds = extract_dates(text, qmid if qhas else np.nan)

    def best_gap(sel):
        if not sel:
            return np.full(N, 6.0)
        e = np.array(sel, dtype=float)[:, None]
        gap = np.maximum(0, np.maximum(A.lo_f[None, :] - e, e - A.hi_f[None, :]))
        gap = np.where(A.has_date[None, :], gap, 999)
        return np.log1p(gap.min(0))

    def dist(a, b):
        if b >= s0 and a <= e0:
            return 0
        return s0 - b if b < s0 else a - e0

    f['t_in'] = best_gap([o for o, a, b, k in ds if dist(a, b) <= 15])
    f['t_near'] = best_gap([o for o, a, b, k in ds if dist(a, b) <= 150])
    f['t_mid'] = best_gap([o for o, a, b, k in ds if dist(a, b) <= 600])
    f['t_any'] = best_gap([o for o, a, b, k in ds])
    f['t_near_full'] = best_gap([o for o, a, b, k in ds if dist(a, b) <= 150 and k <= 1])
    f['t_n_near'] = np.full(N, float(sum(dist(a, b) <= 150 for o, a, b, k in ds)))
    if qhas and len(A.ctd_ord):
        m = np.full(N, 999.0)
        np.minimum.at(m, A.ctd_idx, np.abs(A.ctd_ord - qmid))
        f['c_mentions_q'] = np.log1p(np.minimum(m, 999))
    else:
        f['c_mentions_q'] = np.full(N, np.log1p(999))
    ment = text[s0:e0]
    near = text[max(0, s0 - 80):e0 + 80]
    mnums = set(int(x) for x in RE_INT.findall(ment))
    nnums = set(int(x) for x in RE_NO.findall(near))
    hn = A.hnum
    f['n_in_mention'] = ((hn >= 0) & np.isin(hn, list(mnums))).astype(float) if mnums else np.zeros(N)
    f['n_near'] = ((hn >= 0) & np.isin(hn, list(nnums))).astype(float) if nnums else np.zeros(N)
    qn = A.hnum[self_i] if self_i >= 0 else -1
    if qn < 0:
        qn = header_number(text)
    f['n_diff'] = np.where((hn >= 0) & (qn >= 0), np.clip(qn - hn, -5, 5), 9).astype(float)
    f['n_mention_has'] = np.full(N, float(bool(mnums)))
    own = set(surname(p) for p in qs + qr)
    pos = {}
    for mw in WORD.finditer(text):
        w, p = mw.group(0).lower(), mw.start()
        for c in (w, w[:-1] if w.endswith('s') else None, w[:-2] if w.endswith('en') else None):
            if c and c in A.suridf and c not in own:
                d = (s0 - p) if p < s0 else max(0, p - e0)
                if c not in pos or d < pos[c]:
                    pos[c] = d
    m_any, m_near, m_prox = np.zeros(N), np.zeros(N), np.zeros(N)
    cl = {'s': np.zeros(N), 'r': np.zeros(N)}
    px = {'s': np.zeros(N), 'r': np.zeros(N)}
    if pos:
        for i in range(N):
            for s in A.csur[i]:
                if s in pos:
                    w, d = A.suridf[s], pos[s]
                    m_any[i] = max(m_any[i], w)
                    if d <= 300:
                        m_near[i] = max(m_near[i], w)
                    m_prox[i] = max(m_prox[i], w * math.exp(-d / 400))
            for side, sets in (('s', A.csur_s), ('r', A.csur_r)):
                for x in sets[i]:
                    if x in pos:
                        w, d = A.suridf[x], pos[x]
                        if d <= 60:
                            cl[side][i] = max(cl[side][i], w)
                        px[side][i] = max(px[side][i], w * math.exp(-d / 150))
    f['m_any'], f['m_near'], f['m_prox'] = m_any, m_near, m_prox
    f['m_close_s'], f['m_prox_s'], f['m_close_r'], f['m_prox_r'] = cl['s'], px['s'], cl['r'], px['r']
    wn = text_surnames(text, A.suridf, max(0, s0 - 300), e0 + 300) - own
    shared, qin = np.zeros(N), np.zeros(N)
    for i in range(N):
        ts = A.tsur[i]
        if wn:
            inter = (wn & ts) - A.csur[i]
            if inter:
                shared[i] = max(A.suridf[x] for x in inter)
        inter2 = own & ts
        if inter2:
            qin[i] = max(A.suridf.get(x, 0) for x in inter2)
    f['x_shared'], f['x_qin'] = shared, qin
    f['tf_win'] = (A.tfX @ A.tfv.transform([text[max(0, s0 - 700):e0 + 700]]).T).toarray().ravel()
    f['tf_full'] = (A.tfX @ A.tfv.transform([text]).T).toarray().ravel()
    if self_i >= 0:
        qrepo, qprov, qsh = A.repo[self_i], A.prov[self_i], A.shelf_tok[self_i]
    else:
        qrepo, qprov, qsh = '', '', set()
    f['h_repo'] = ((A.repo == qrepo) & (qrepo != '')).astype(float)
    f['h_prov'] = ((A.prov == qprov) & (qprov != '')).astype(float)
    f['h_shelf'] = np.array([len(qsh & s) / max(1, len(qsh | s)) for s in A.shelf_tok]) if qsh else np.zeros(N)
    f['c_len'] = A.loglen
    f['c_pop'] = np.log1p(pop_counts)
    f['c_hasnum'] = (hn >= 0).astype(float)
    X = np.stack([f[k] for k in f], 1).astype(np.float32)
    valid = np.ones(N, bool)
    if self_i >= 0:
        valid[self_i] = False
    return X, valid


def first_id(s):
    try:
        v = json.loads(s)
        return str(v[0]) if isinstance(v, list) else str(v)
    except Exception:
        return str(s)


def mrr_at10(S, valid, y):
    S = np.where(valid, S, -1e9)
    t = S[np.arange(len(y)), y][:, None]
    r = 1 + (S > t).sum(1)
    return float(np.mean(np.where(r <= 10, 1.0 / r, 0.0)))


letters = pd.read_csv(public_dir / 'letters.csv', dtype={'letter_id': str})
train = pd.read_csv(public_dir / 'train.csv')
test = pd.read_csv(public_dir / 'test.csv')
val_path = public_dir / 'validation.csv'
val = pd.read_csv(val_path) if val_path.exists() else train.iloc[:0].copy()
log('loaded', len(letters), 'letters', len(train), 'train', len(val), 'val', len(test), 'test')

A = Archive(letters)
log('archive ready')

train = train[train.letter_ids.map(first_id).isin(A.idx)].reset_index(drop=True)
val = val[val.letter_ids.map(first_id).isin(A.idx)].reset_index(drop=True) if len(val) else val
y_tr = np.array([A.idx[first_id(s)] for s in train.letter_ids])
y_va = np.array([A.idx[first_id(s)] for s in val.letter_ids]) if len(val) else np.zeros(0, int)
tr_docs = train.document_id.astype(str).values
pop_all = np.bincount(y_tr, minlength=A.N).astype(float)


def build(df, loo):
    Xs, Vs = [], []
    for i, r in enumerate(df.to_dict('records')):
        pc = pop_all
        if loo:
            pc = pop_all.copy()
            same = y_tr[tr_docs == str(r['document_id'])]
            np.subtract.at(pc, same, 1.0)
        X, v = query_features(A, r, pc)
        Xs.append(X)
        Vs.append(v)
    return np.stack(Xs), np.stack(Vs)


X_tr, V_tr = build(train, True)
X_va, V_va = build(val, False) if len(val) else (np.zeros((0, A.N, X_tr.shape[2]), np.float32), np.zeros((0, A.N), bool))
X_te, V_te = build(test, False)
F = X_tr.shape[2]
log('features', X_tr.shape, X_va.shape, X_te.shape)

mu = X_tr.reshape(-1, F).mean(0)
sd = X_tr.reshape(-1, F).std(0) + 1e-6


def stdz(X):
    return torch.tensor((X - mu) / sd, dtype=torch.float32)


torch.manual_seed(SEED)
lin = nn.Linear(F, 1)
opt = torch.optim.AdamW(lin.parameters(), 3e-3, weight_decay=1e-4)
m_tr = torch.tensor(~V_tr)
yT = torch.tensor(y_tr)
for ep in range(30):
    perm = torch.randperm(len(y_tr))
    for b in range(0, len(y_tr), 32):
        ix = perm[b:b + 32]
        s = lin(stdz(X_tr[ix.numpy()])).squeeze(-1).masked_fill(m_tr[ix], -1e4)
        loss = Fn.cross_entropy(s, yT[ix])
        opt.zero_grad()
        loss.backward()
        opt.step()


@torch.no_grad()
def shortlist(X, V):
    out = []
    for b in range(0, len(X), 64):
        s = lin(stdz(X[b:b + 64])).squeeze(-1).masked_fill(torch.tensor(~V[b:b + 64]), -1e4)
        out.append(torch.topk(s, min(K_SHORT, A.N), 1).indices.numpy())
    return np.concatenate(out) if out else np.zeros((0, K_SHORT), int)


SL_tr, SL_va, SL_te = shortlist(X_tr, V_tr), shortlist(X_va, V_va), shortlist(X_te, V_te)
if len(val):
    log('stage-1 recall@%d val %.4f' % (K_SHORT, np.mean([y_va[i] in SL_va[i] for i in range(len(y_va))])))
for i in range(len(y_tr)):
    if y_tr[i] not in SL_tr[i]:
        SL_tr[i, -1] = y_tr[i]
lab_tr = torch.tensor([int(np.where(SL_tr[i] == y_tr[i])[0][0]) for i in range(len(y_tr))])


def gather(X, SL):
    return stdz(np.take_along_axis(X, SL[:, :, None], 1)) if len(X) else torch.zeros((0, K_SHORT, F))


XS_tr, XS_va, XS_te = gather(X_tr, SL_tr), gather(X_va, SL_va), gather(X_te, SL_te)
del X_tr, X_va, X_te


def short_mrr(S, SL, y):
    rr = []
    for i in range(len(y)):
        j = np.where(SL[i] == y[i])[0]
        if len(j) == 0:
            rr.append(0.0)
            continue
        r = 1 + int((S[i] > S[i][j[0]]).sum())
        rr.append(1.0 / r if r <= 10 else 0.0)
    return float(np.mean(rr)) if rr else 0.0


tok = AutoTokenizer.from_pretrained(BACKBONE)
backbone = AutoModel.from_pretrained(BACKBONE, attn_implementation='eager').to(DEVICE).eval()
HID = backbone.config.hidden_size


def meta(s, r, d):
    return f"{s if isinstance(s, str) else ''} an {r if isinstance(r, str) else ''}, {d if isinstance(d, str) else ''}. "


def qtext(r):
    t = r['text'] if isinstance(r['text'], str) else ''
    s, e = int(r['mention_start']), int(r['mention_end'])
    return 'query: ' + meta(r['sender'], r['recipient'], r['date']) + t[max(0, s - 450):s] + ' «' + t[s:e] + '» ' + t[e:e + 250]


@torch.no_grad()
def encode(texts, lower=False, bs=64):
    embs, hs, ms = [], [], []
    for b in range(0, len(texts), bs):
        en = tok(texts[b:b + bs], max_length=256, truncation=True, padding='max_length' if lower else True, return_tensors='pt').to(DEVICE)
        o = backbone(**en, output_hidden_states=lower)
        m = en['attention_mask'].unsqueeze(-1).float()
        embs.append(Fn.normalize((o.last_hidden_state * m).sum(1) / m.sum(1), dim=-1).cpu())
        if lower:
            hs.append(o.hidden_states[CUT].half().cpu())
            ms.append(en['attention_mask'].cpu())
    if not embs:
        return torch.zeros((0, HID)), torch.zeros((0, 256, HID)).half(), torch.zeros((0, 256), dtype=torch.long)
    return torch.cat(embs), (torch.cat(hs) if lower else None), (torch.cat(ms) if lower else None)


D_emb, _, _ = encode(['passage: ' + meta(r.sender, r.recipient, r.date) + str(r.text)[:800] for r in A.df.itertuples()])
log('archive encoded')
qe_tr, H_tr, M_tr = encode([qtext(r) for r in train.to_dict('records')], lower=True)
qe_va, H_va, M_va = encode([qtext(r) for r in val.to_dict('records')], lower=True)
qe_te, H_te, M_te = encode([qtext(r) for r in test.to_dict('records')], lower=True)
log('queries encoded')


def zero_cos(qe, SL):
    return torch.stack([D_emb[SL[i]] @ qe[i] for i in range(len(SL))]) if len(SL) else torch.zeros((0, K_SHORT))


CZ_tr, CZ_va, CZ_te = zero_cos(qe_tr, SL_tr), zero_cos(qe_va, SL_va), zero_cos(qe_te, SL_te)
E_tr = D_emb[torch.tensor(SL_tr)]
E_va = D_emb[torch.tensor(SL_va)] if len(SL_va) else torch.zeros((0, K_SHORT, HID))
E_te = D_emb[torch.tensor(SL_te)]


class NeuralRanker(nn.Module):
    def __init__(self, top, F, d=32, c=8):
        super().__init__()
        self.top = top
        self.qp = nn.Linear(HID, d)
        self.cp = nn.Linear(HID, d)
        self.qctx = nn.Linear(HID, c)
        self.mlp = nn.Sequential(nn.Linear(F + 2 + c, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.scale = nn.Parameter(torch.tensor(5.0))

    def qenc(self, h, m):
        ext = (1.0 - m[:, None, None, :].float()) * torch.finfo(torch.float32).min
        h = h.float()
        for layer in self.top:
            r = layer(h, attention_mask=ext)
            h = r[0] if isinstance(r, (tuple, list)) else r
        mm = m.unsqueeze(-1).float()
        return (h * mm).sum(1) / mm.sum(1)

    def forward(self, h, m, X, Ec, cz):
        q = self.qenc(h, m)
        qv = Fn.normalize(self.qp(q), dim=-1)
        cv = Fn.normalize(self.cp(Ec), dim=-1)
        dot = (qv[:, None, :] * cv).sum(-1)
        ctx = torch.tanh(self.qctx(q))[:, None, :].expand(-1, X.shape[1], -1)
        z = torch.cat([X, cz[..., None], dot[..., None], ctx], -1)
        return self.mlp(z).squeeze(-1) + self.scale * dot


def predict_neural(model, H, M, XS, E, CZ, bs=64):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(XS), bs):
            out.append(model(H[b:b + bs].to(DEVICE), M[b:b + bs].to(DEVICE), XS[b:b + bs].to(DEVICE), E[b:b + bs].to(DEVICE), CZ[b:b + bs].to(DEVICE)).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, K_SHORT))


torch.manual_seed(SEED)
model = NeuralRanker(copy.deepcopy(backbone.encoder.layer[CUT:]), F).to(DEVICE)
del backbone
head_params = [p for n, p in model.named_parameters() if not n.startswith('top.')]
EPOCHS, BS = 6, 16
opt = torch.optim.AdamW([{'params': model.top.parameters(), 'lr': 2e-5}, {'params': head_params, 'lr': 1e-3}], weight_decay=0.01)
steps = EPOCHS * ((len(y_tr) + BS - 1) // BS)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[2e-5, 1e-3], total_steps=steps, pct_start=0.15)
best_state, best_score, best_ep = copy.deepcopy(model.state_dict()), -1.0, -1
stop = False
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(y_tr))
    tot = 0.0
    for b in range(0, len(y_tr), BS):
        if time.time() - T_START > TRAIN_DEADLINE:
            stop = True
            break
        ix = perm[b:b + BS]
        s = model(H_tr[ix].to(DEVICE), M_tr[ix].to(DEVICE), XS_tr[ix].to(DEVICE), E_tr[ix].to(DEVICE), CZ_tr[ix].to(DEVICE))
        loss = Fn.cross_entropy(s, lab_tr[ix].to(DEVICE), label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item() * len(ix)
    score = short_mrr(predict_neural(model, H_va, M_va, XS_va, E_va, CZ_va), SL_va, y_va) if len(val) else float(ep)
    log(f'neural epoch {ep} loss {tot / len(y_tr):.3f} val MRR@10 {score:.4f}')
    if score > best_score:
        best_score, best_ep, best_state = score, ep, copy.deepcopy(model.state_dict())
    if stop:
        break
model.load_state_dict(best_state)
log(f'neural ranker selected epoch {best_ep} val {best_score:.4f}')
NS_va = predict_neural(model, H_va, M_va, XS_va, E_va, CZ_va)
NS_te = predict_neural(model, H_te, M_te, XS_te, E_te, CZ_te)


class FeatureRanker(nn.Module):
    def __init__(self, F, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F + 1, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, X, cz):
        return self.net(torch.cat([X, cz[..., None]], -1)).squeeze(-1)


def train_feature_ranker(seed, epochs=12):
    torch.manual_seed(seed)
    m = FeatureRanker(F)
    o = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(y_tr))
        for b in range(0, len(y_tr), 32):
            ix = perm[b:b + 32]
            loss = Fn.cross_entropy(m(XS_tr[ix], CZ_tr[ix]), lab_tr[ix])
            o.zero_grad()
            loss.backward()
            o.step()
    m.eval()
    with torch.no_grad():
        return m(XS_va, CZ_va).numpy() if len(val) else np.zeros((0, K_SHORT)), m(XS_te, CZ_te).numpy()


def lsm(S):
    return torch.log_softmax(torch.tensor(S, dtype=torch.float32), 1).numpy() if len(S) else S


FS_va, FS_te = 0, 0
N_FEAT_SEEDS = 5
for sd_ in range(N_FEAT_SEEDS):
    a, b = train_feature_ranker(SEED + sd_)
    FS_va = FS_va + lsm(a) / N_FEAT_SEEDS
    FS_te = FS_te + lsm(b) / N_FEAT_SEEDS
if len(val):
    log(f'feature-ranker ensemble val MRR@10 {short_mrr(FS_va, SL_va, y_va):.4f}')

best_w, best_blend = 1.0, -1.0
for w in [1.0, 0.8, 0.65, 0.5, 0.35, 0.2]:
    sc = short_mrr(w * lsm(NS_va) + (1 - w) * FS_va, SL_va, y_va) if len(val) else 0.0
    if len(val):
        log(f'blend neural weight {w:.2f}: val MRR@10 {sc:.4f}')
    if sc > best_blend + 1e-9:
        best_w, best_blend = w, sc
if len(val):
    best_w = min(max(best_w, 0.35), 1.0)
log(f'selected neural weight {best_w:.2f}')
S_te = best_w * lsm(NS_te) + (1 - best_w) * FS_te

rows = []
for i, r in enumerate(test.itertuples()):
    order = np.argsort(-S_te[i], kind='stable')
    ids, seen = [], set()
    for j in order:
        lid = A.ids[SL_te[i][j]]
        if lid not in seen and lid != str(r.document_id):
            ids.append(lid)
            seen.add(lid)
        if len(ids) == 10:
            break
    rows.append((r.case_id, json.dumps(ids)))
sub = pd.DataFrame(rows, columns=['case_id', 'letter_ids'])

known = set(A.ids)
assert len(sub) == len(test) and sub.case_id.is_unique and set(sub.case_id) == set(test.case_id)
for s in sub.letter_ids:
    v = json.loads(s)
    assert isinstance(v, list) and 1 <= len(v) <= 10 and len(set(v)) == len(v) and all(x in known for x in v)
submission_out.parent.mkdir(parents=True, exist_ok=True)
sub.to_csv(submission_out, index=False)
log('wrote', submission_out, sub.shape)
