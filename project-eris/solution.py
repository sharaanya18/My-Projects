import os
import sys
from pathlib import Path

PUBLIC_DIR = Path(sys.argv[1])
SUBMISSION_OUT = Path(sys.argv[2])
WORK_DIR = SUBMISSION_OUT.parent

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["HF_HOME"] = str(WORK_DIR / "hf_cache")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "1"

import hashlib
import math
import random
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

SEED = 42
BACKBONE = "microsoft/deberta-v3-base"
BACKBONE_REVISION = "8ccc9b6f36199bec6961081d44eb72fb3f7353f3"
DEVICE = "cuda"
AMP_DTYPE = torch.bfloat16

MAX_LEN = 512
EPOCHS = 2
LR_BACKBONE = 2e-5
LR_HEAD = 1e-3
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.10
COUPLETS_PER_STEP = 8
GRAD_ACCUM = 2
GRAD_CLIP = 1.0
DROPOUT = 0.1
HEAD_HIDDEN = 256
BUCKET_CHUNK_BATCHES = 50

N_FOLDS = 5
MODEL_A_VAL_FOLD = 0
MODEL_B_VAL_FOLD = 1
MODELS = (("MODEL_A", MODEL_A_VAL_FOLD),)

MIN_SELECT_DEPTH = 4
DECODE_LAMBDA = 0.5
DECODE_MODE = "max"

WEIGHT_CAP = 5.0
INFER_CHUNK = 64
HAB_CAP, DIST_CAP, SEASON_CAP = 64, 48, 16
VIA_DEPTH = 0
VIA_CAP = 96

TRAIN_CSV = PUBLIC_DIR / "train.csv"
TEST_CSV = PUBLIC_DIR / "test.csv"
LEADS_CSV = PUBLIC_DIR / "key_leads.csv"
SAMPLE_CSV = PUBLIC_DIR / "sample_submission.csv"
SUBMISSION_CSV = SUBMISSION_OUT

_T0 = time.perf_counter()


def log(msg):
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)




def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.set_num_threads(1)


_DASHES = str.maketrans({"-": "–", "—": "–", "‒": "–", "−": "–", "‐": "–", "‑": "–"})


def norm_text(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.translate(_DASHES)
    return re.sub(r"\s+", " ", s).strip()


_ABBREV = {
    "c", "ca", "cf", "e.g", "eg", "i.e", "ie", "var", "subsp", "ssp", "sp", "spp",
    "diam", "approx", "incl", "esp", "fig", "figs", "mt", "mts", "st", "no", "nos",
    "al", "etc", "vs", "min", "max", "sect", "ser", "f", "s.lat", "s.str", "lat",
    "str", "nom", "illeg", "auct", "pers", "comm", "obs", "alt", "nr", "prob",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
    "n", "s", "e", "w", "ne", "nw", "se", "sw", "vic", "nsw", "qld", "tas",
}


def split_clauses(text):
    clauses, start, n = [], 0, len(text)
    i = 0
    while i < n:
        ch = text[i]
        cut = False
        if ch == ";":
            cut = True
        elif ch == "." and i + 2 < n and text[i + 1] == " " and (text[i + 2].isupper() or text[i + 2] in "(\"'"):
            m = re.search(r"([A-Za-z.]+)$", text[start:i])
            word = m.group(1).lower().strip(".") if m else ""
            if word not in _ABBREV and not (len(word) == 1 and word.isalpha()):
                cut = True
        if cut:
            piece = text[start:i + 1].strip()
            if piece:
                clauses.append(piece)
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses or [text]


_STOP = {
    "and", "or", "the", "with", "without", "to", "of", "in", "on", "at", "by", "for",
    "from", "as", "a", "an", "is", "are", "be", "not", "than", "more", "less", "most",
    "mostly", "usually", "often", "sometimes", "rarely", "very", "rather", "somewhat",
    "each", "per", "c", "ca", "mm", "cm", "m", "µm", "μm", "um", "long", "wide",
    "high", "tall", "diam", "across", "about", "up", "only", "also", "when", "which",
    "that", "this", "these", "those", "its", "their", "all", "any", "some", "few",
    "several", "many", "plants", "plant", "taxon", "exceeding", "least", "never",
    "slightly", "much", "generally", "occasionally", "commonly", "frequently",
}


def stem(w):
    w = w.lower()
    if len(w) > 4 and w.endswith("ves"):
        return w[:-3] + "f"
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def content_words(text):
    return [stem(w) for w in re.findall(r"[A-Za-z]+", text) if len(w) >= 3 and w.lower() not in _STOP]


_NUM = r"\d+(?:\.\d+)?"
_UNIT = r"(?:µm|μm|um|mm|cm|m)(?![A-Za-z])"
_RANGE_RE = re.compile(
    rf"(?:\(\s*(?P<xlo>{_NUM})\s*–\s*\)\s*)?"
    rf"(?P<lo>{_NUM})"
    rf"(?:\s*–\s*(?P<hi>{_NUM}))?"
    rf"(?:\s*\(\s*–\s*(?P<xhi>{_NUM})\s*\))?"
    rf"(?:\s*(?P<unit>{_UNIT}))?"
)
_UNIT_MM = {"µm": 1e-3, "μm": 1e-3, "um": 1e-3, "mm": 1.0, "cm": 10.0, "m": 1000.0}
_UPPER_OPS = ("less than", "not exceeding", "not more than", "up to", "to", "<", "under", "≤")
_LOWER_OPS = ("at least", "more than", "exceeding", "over", ">", "≥")
_DIMS = {"long": "long", "length": "long", "wide": "wide", "width": "wide", "broad": "wide",
         "diam": "diam", "diameter": "diam", "high": "high", "tall": "high", "height": "high",
         "thick": "thick", "deep": "deep"}
_BOUND_SPREAD = math.log(4.0)


def extract_numbers(text):
    out = []
    for clause in split_clauses(text):
        head = content_words(clause[:40])[:1]
        for m in _RANGE_RE.finditer(clause):
            s = m.start()
            if s > 0 and (clause[s - 1].isalnum() or clause[s - 1] in ".–"):
                continue
            lo = float(m.group("lo"))
            hi = float(m.group("hi")) if m.group("hi") else lo
            if m.group("xlo"):
                lo = min(lo, float(m.group("xlo")))
            if m.group("xhi"):
                hi = max(hi, float(m.group("xhi")))
            if hi < lo:
                lo, hi = hi, lo
            unit = m.group("unit")
            after = clause[m.end():m.end() + 30]
            if unit is None and re.match(r"\s*(?:%|°|am|pm|years?|yrs?)", after):
                continue
            if unit is not None:
                kind, scale = "len", _UNIT_MM[unit]
            else:
                kind, scale = "cnt", 1.0
            lo, hi = lo * scale, hi * scale
            if lo <= 0 or hi <= 0:
                continue
            before = clause[max(0, s - 40):s].lower()
            op = "range"
            if re.search(r"(?:^|\W)(?:c\.|ca\.|about|approx\.?)\s*$", before):
                op = "approx"
            elif m.group("hi") is None and any(re.search(rf"(?:^|\W){re.escape(o)}\s*$", before) for o in _UPPER_OPS):
                op = "upper"
            elif m.group("hi") is None and any(re.search(rf"(?:^|\W){re.escape(o)}\s*$", before) for o in _LOWER_OPS):
                op = "lower"
            llo, lhi = math.log(lo), math.log(hi)
            if op == "upper":
                llo = lhi - _BOUND_SPREAD
            elif op == "lower":
                lhi = llo + _BOUND_SPREAD
            ctx = set(content_words(clause[:s])[-3:]) | set(head)
            dm = re.match(r"\s*([A-Za-z]+)", after)
            dim = _DIMS.get(dm.group(1).lower().rstrip(".")) if dm else None
            out.append({"kind": kind, "lo": llo, "hi": lhi, "op": op, "ctx": ctx, "dim": dim})
    return out


_PAD = 0.05


def _iou(a, b):
    alo, ahi, blo, bhi = a["lo"] - _PAD, a["hi"] + _PAD, b["lo"] - _PAD, b["hi"] + _PAD
    inter = min(ahi, bhi) - max(alo, blo)
    if inter <= 0:
        return 0.0
    return inter / (max(ahi, bhi) - min(alo, blo))


def _organ_match(a, b):
    return bool(a["ctx"] & b["ctx"]) and (a["dim"] is None or b["dim"] is None or a["dim"] == b["dim"])


def _signed_dist(lead, val):
    if val > lead["hi"]:
        return min(val - lead["hi"], 3.0)
    if val < lead["lo"]:
        return max(val - lead["lo"], -3.0)
    return 0.0


N_BASE_FEATS = 14
N_FEATS = 2 * N_BASE_FEATS


def numeric_features(lead_nums, desc_nums):
    feats = []
    for kind in ("len", "cnt"):
        L = [x for x in lead_nums if x["kind"] == kind]
        D = [x for x in desc_nums if x["kind"] == kind]
        has = 1.0 if L else 0.0
        iou_any = max((_iou(a, b) for a in L for b in D), default=0.0)
        iou_org = max((_iou(a, b) for a in L for b in D if _organ_match(a, b)), default=0.0)
        dists, supported = [], 0
        for a in L:
            vals = [v for b in D if _organ_match(a, b) for v in (b["lo"], b["hi"])]
            if vals:
                supported += 1
                dists.append(min((_signed_dist(a, v) for v in vals), key=lambda d: (abs(d), d)))
        dist = float(np.mean(dists)) if dists else 0.0
        frac = supported / len(L) if L else 0.0
        upper = 1.0 if any(a["op"] == "upper" for a in L) else 0.0
        lower = 1.0 if any(a["op"] == "lower" for a in L) else 0.0
        feats += [has, iou_any, iou_org, dist, frac, upper, lower]
    return feats


def couplet_of(label):
    return re.match(r"^(\d+)", label).group(1)


class Key:
    def __init__(self, df):
        self.lead_text, self.goes_to, self.couplets = {}, {}, {}
        for lead, text, goes in zip(df["lead"], df["text"], df["goes_to"]):
            self.lead_text[lead] = norm_text(text)
            self.goes_to[lead] = goes.strip()
            self.couplets.setdefault(couplet_of(lead), []).append(lead)
        for c in self.couplets:
            self.couplets[c] = sorted(self.couplets[c])
        self.order = sorted(self.couplets, key=int)
        self.parent = {}
        for c in self.order:
            for lead in self.couplets[c]:
                ch = self.child(lead)
                if ch is not None and ch not in self.parent:
                    self.parent[ch] = lead

    def child(self, lead):
        g = self.goes_to.get(lead, "")
        return g if g.isdigit() and g in self.couplets else None

    def ancestors(self, c, depth):
        out, seen = [], {c}
        while len(out) < depth and c in self.parent:
            lead = self.parent[c]
            out.append(lead)
            c = couplet_of(lead)
            if c in seen:
                break
            seen.add(c)
        return out


def build_keys(leads_df):
    keys = {}
    for kid, df in leads_df.groupby("key_id", sort=True):
        keys[kid] = Key(df)
    return keys


def make_key_folds(train, keys):
    parent = {k: k for k in sorted(train["key_id"].unique())}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_key = {}
    for desc, kid in zip(train["desc_norm"], train["key_id"]):
        if desc in first_key:
            ra, rb = find(first_key[desc]), find(kid)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
        else:
            first_key[desc] = kid
    groups = {}
    for k in sorted(parent):
        groups.setdefault(find(k), []).append(k)
    sizes = {g: sum(len(keys[k].couplets) for k in ks) for g, ks in groups.items()}
    order = sorted(groups, key=lambda g: (-sizes[g], g))
    fold_of_key = {}
    for i, g in enumerate(order):
        r, pos = divmod(i, N_FOLDS)
        fold = pos if r % 2 == 0 else N_FOLDS - 1 - pos
        for k in groups[g]:
            fold_of_key[k] = fold
    return fold_of_key


class Encoder:

    def __init__(self, tokenizer, keys):
        self.tok = tokenizer
        self.cls, self.sep = tokenizer.cls_token_id, tokenizer.sep_token_id
        self.pad = tokenizer.pad_token_id
        self.p_choice = self.ids("CHOICE:")
        self.p_other = self.ids("|| OTHER:")
        self.p_bar = self.ids("||")
        self.p_hab = self.ids("HABITAT:")
        self.p_dist = self.ids("DISTRIBUTION:")
        self.p_season = self.ids("SEASON:")
        self.p_via = self.ids("|| VIA:")
        self.lead_ids, self.lead_words, self.lead_nums = {}, {}, {}
        texts = []
        for kid in sorted(keys):
            for lead in sorted(keys[kid].lead_text):
                texts.append(((kid, lead), keys[kid].lead_text[lead]))
        enc = self.tok([t for _, t in texts], add_special_tokens=False)["input_ids"]
        for ((kid, lead), t), ids in zip(texts, enc):
            self.lead_ids[(kid, lead)] = ids
            self.lead_words[(kid, lead)] = set(content_words(t))
            self.lead_nums[(kid, lead)] = extract_numbers(t)
        self.couplet_words = {}
        for kid in sorted(keys):
            for c, leads in keys[kid].couplets.items():
                self.couplet_words[(kid, c)] = set().union(*[self.lead_words[(kid, l)] for l in leads])

    def ids(self, text):
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prepare_rows(self, df):
        rows = []
        clause_lists = [split_clauses(d) for d in df["desc_norm"]]
        flat = [c for cl in clause_lists for c in cl]
        flat_ids = self.tok(flat, add_special_tokens=False)["input_ids"] if flat else []
        meta = {}
        for col, cap, prefix in (("habitat", HAB_CAP, self.p_hab), ("distribution", DIST_CAP, self.p_dist),
                                 ("phenology", SEASON_CAP, self.p_season)):
            vals = [norm_text(v) for v in df[col]]
            enc = self.tok(vals, add_special_tokens=False)["input_ids"]
            meta[col] = [(prefix + e[:cap]) if v else [] for v, e in zip(vals, enc)]
        pos = 0
        for i, cl in enumerate(clause_lists):
            ids = flat_ids[pos:pos + len(cl)]
            pos += len(cl)
            rows.append({
                "clause_ids": ids,
                "clause_words": [set(content_words(c)) for c in cl],
                "meta_ids": meta["habitat"][i] + meta["distribution"][i] + meta["phenology"][i],
                "nums": extract_numbers(df["desc_norm"].iloc[i]),
            })
        return rows

    def seg1(self, kid, cand, others, via):
        s = self.p_choice + self.lead_ids[(kid, cand)] + self.p_other
        for j, o in enumerate(others):
            if j:
                s = s + self.p_bar
            s = s + self.lead_ids[(kid, o)]
        if via:
            s = s + self.p_via + via
        return s

    def via_ids(self, kid, key, c):
        v = []
        for j, lead in enumerate(key.ancestors(c, VIA_DEPTH)):
            if j:
                v = v + self.p_bar
            v = v + self.lead_ids[(kid, lead)]
        return v[:VIA_CAP]

    def seg2(self, row, kid, c, room):
        meta = row["meta_ids"]
        cids = row["clause_ids"]
        total = sum(len(x) for x in cids) + len(meta)
        if total <= room:
            return [t for x in cids for t in x] + meta
        desc_room = room - len(meta)
        keep = [0]
        used = len(cids[0])
        words = self.couplet_words[(kid, c)]
        ranked = sorted(range(1, len(cids)), key=lambda j: (-len(row["clause_words"][j] & words), j))
        for j in ranked:
            if used + len(cids[j]) <= desc_room:
                keep.append(j)
                used += len(cids[j])
        out = [t for j in sorted(keep) for t in cids[j]] + meta
        return out[:room]

    def couplet_sequences(self, row, kid, key, c):
        leads = key.couplets[c]
        via = self.via_ids(kid, key, c)
        s1 = {x: self.seg1(kid, x, [o for o in leads if o != x], via) for x in leads}
        s1_len = max(len(v) for v in s1.values())
        s1_cap = MAX_LEN - 3 - 32
        s1 = {x: v[:s1_cap] for x, v in s1.items()}
        s1_len = min(s1_len, s1_cap)
        s2 = self.seg2(row, kid, c, MAX_LEN - 3 - s1_len)
        return [[self.cls] + s1[x] + [self.sep] + s2 + [self.sep] for x in leads]


def raw_couplet_feats(enc, row, kid, leads):
    base = np.array([numeric_features(enc.lead_nums[(kid, x)], row["nums"]) for x in leads], dtype=np.float32)
    n = len(leads)
    sib_mean = (base.sum(0, keepdims=True) - base) / max(n - 1, 1)
    return np.concatenate([base, base - sib_mean], axis=1)


class Scorer(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h + N_FEATS, HEAD_HIDDEN), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(HEAD_HIDDEN, 1)
        )

    def forward(self, ids, mask, feats):
        with torch.autocast(device_type=DEVICE, dtype=AMP_DTYPE):
            hid = self.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        m = mask.unsqueeze(-1).float()
        pooled = (hid.float() * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(torch.cat([pooled, feats], dim=-1)).squeeze(-1)


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BACKBONE, revision=BACKBONE_REVISION)


def load_backbone():
    return AutoModel.from_pretrained(
        BACKBONE, revision=BACKBONE_REVISION, hidden_dropout_prob=DROPOUT, attention_probs_dropout_prob=DROPOUT
    ).float()


def pad_batch(seqs, pad_id):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


def build_examples(df, rows, keys):
    ex = []
    for i, (kid, path) in enumerate(zip(df["key_id"], df["path"])):
        key = keys[kid]
        for lab in path.split():
            c = couplet_of(lab)
            if c not in key.couplets or lab not in key.couplets[c]:
                continue
            ex.append({"row": i, "kid": kid, "c": c, "target": key.couplets[c].index(lab), "lead": lab})
    return ex


def metric_weights(examples):
    n_c, n_cb = {}, {}
    for e in examples:
        kc = (e["kid"], e["c"])
        n_c[kc] = n_c.get(kc, 0) + 1
        n_cb[kc + (e["lead"],)] = n_cb.get(kc + (e["lead"],), 0) + 1
    m_c = {}
    for (kid, c, _lead) in n_cb:
        m_c[(kid, c)] = m_c.get((kid, c), 0) + 1
    for e in examples:
        kc = (e["kid"], e["c"])
        if m_c[kc] == 1:
            e["w"] = 1.0
        else:
            e["w"] = min(n_c[kc] / (m_c[kc] * n_cb[kc + (e["lead"],)]), WEIGHT_CAP)


def materialise(examples, enc, rows, keys):
    for e in examples:
        key = keys[e["kid"]]
        leads = key.couplets[e["c"]]
        e["seqs"] = [np.asarray(q, dtype=np.int32) for q in enc.couplet_sequences(rows[e["row"]], e["kid"], key, e["c"])]
        e["feats"] = raw_couplet_feats(enc, rows[e["row"]], e["kid"], leads)
        e["len"] = max(len(s) for s in e["seqs"])


def make_batches(examples, gen):
    perm = torch.randperm(len(examples), generator=gen).tolist()
    window = COUPLETS_PER_STEP * BUCKET_CHUNK_BATCHES
    batches = []
    for s in range(0, len(perm), window):
        chunk = sorted(perm[s:s + window], key=lambda i: (examples[i]["len"], i))
        for b in range(0, len(chunk), COUPLETS_PER_STEP):
            batches.append(chunk[b:b + COUPLETS_PER_STEP])
    order = torch.randperm(len(batches), generator=gen).tolist()
    return [batches[i] for i in order]


def train_model(name, examples, feat_mean, feat_std, pad_id):
    log(f"{name}: training on {len(examples)} couplet examples")
    seed_everything(SEED)
    model = Scorer(load_backbone()).to(DEVICE)
    fm = torch.tensor(feat_mean, device=DEVICE)
    fs = torch.tensor(feat_std, device=DEVICE)

    no_decay = ("bias", "LayerNorm.weight", "layernorm.weight", "norm.weight")
    bb_decay = [p for n, p in model.backbone.named_parameters() if not any(k in n for k in no_decay)]
    bb_nodecay = [p for n, p in model.backbone.named_parameters() if any(k in n for k in no_decay)]
    opt = torch.optim.AdamW(
        [
            {"params": bb_decay, "lr": LR_BACKBONE, "weight_decay": WEIGHT_DECAY},
            {"params": bb_nodecay, "lr": LR_BACKBONE, "weight_decay": 0.0},
            {"params": list(model.head.parameters()), "lr": LR_HEAD, "weight_decay": WEIGHT_DECAY},
        ]
    )
    n_batches = math.ceil(len(examples) / COUPLETS_PER_STEP)
    total_steps = math.ceil(n_batches / GRAD_ACCUM) * EPOCHS
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * total_steps), total_steps)
    gen = torch.Generator().manual_seed(SEED)

    step = 0
    for epoch in range(EPOCHS):
        model.train()
        batches = make_batches(examples, gen)
        run_loss, run_n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(batches):
            exs = [examples[i] for i in batch]
            seqs = [s for e in exs for s in e["seqs"]]
            feats = np.concatenate([e["feats"] for e in exs], axis=0)
            ids, mask = pad_batch(seqs, pad_id)
            ids, mask = ids.to(DEVICE), mask.to(DEVICE)
            f = (torch.tensor(feats, device=DEVICE) - fm) / fs
            scores = model(ids, mask, f)
            losses = []
            for e, s in zip(exs, torch.split(scores, [len(e["seqs"]) for e in exs])):
                onehot = torch.zeros_like(s)
                onehot[e["target"]] = 1.0
                losses.append(-(torch.log_softmax(s, dim=0) * onehot).sum() * e["w"])
            loss = torch.stack(losses).mean() / GRAD_ACCUM
            loss.backward()
            run_loss += loss.item() * GRAD_ACCUM
            run_n += 1
            if (bi + 1) % GRAD_ACCUM == 0 or bi + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if (bi + 1) % 200 == 0:
                log(f"{name} epoch {epoch + 1} batch {bi + 1}/{len(batches)} loss {run_loss / run_n:.4f}")
                run_loss, run_n = 0.0, 0
        log(f"{name}: epoch {epoch + 1} done")
    del opt, sched
    model.eval()
    return model


@torch.no_grad()
def score_row(model, enc, row, kid, key, feat_mean, feat_std):
    seqs, feats, labels = [], [], []
    for c in key.order:
        leads = key.couplets[c]
        seqs += enc.couplet_sequences(row, kid, key, c)
        feats.append(raw_couplet_feats(enc, row, kid, leads))
        labels += leads
    feats = (np.concatenate(feats, axis=0) - feat_mean) / feat_std
    order = sorted(range(len(seqs)), key=lambda i: (len(seqs[i]), i))
    out = np.zeros(len(seqs), dtype=np.float64)
    for s in range(0, len(order), INFER_CHUNK):
        idx = order[s:s + INFER_CHUNK]
        ids, mask = pad_batch([seqs[i] for i in idx], enc.pad)
        f = torch.tensor(feats[idx], dtype=torch.float32, device=DEVICE)
        out[idx] = model(ids.to(DEVICE), mask.to(DEVICE), f).float().cpu().numpy()
    assert np.isfinite(out).all(), f"non-finite scores for key {kid}"
    raw = dict(zip(labels, out))
    centred = {}
    for c in key.order:
        leads = key.couplets[c]
        mu = sum(raw[x] for x in leads) / len(leads)
        for x in leads:
            centred[x] = raw[x] - mu
    return centred


def decode(key, s, lam, mode):
    V, on_stack = {}, set()

    def lead_value(x, vals):
        ch = key.child(x)
        return s.get(x, 0.0) + lam * (vals.get(ch, 0.0) if ch is not None else 0.0)

    def value(c):
        if c in V:
            return V[c]
        on_stack.add(c)
        child_vals = {}
        for x in key.couplets[c]:
            ch = key.child(x)
            if ch is not None and ch not in on_stack:
                child_vals[ch] = value(ch)
        terms = [lead_value(x, child_vals) for x in key.couplets[c]]
        if mode == "max":
            v = max(terms)
        else:
            mx = max(terms)
            v = mx + math.log(sum(math.exp(t - mx) for t in terms))
        on_stack.discard(c)
        V[c] = v
        return v

    for c in key.order:
        value(c)
    decisions = {}
    for c in key.order:
        best, best_t = None, None
        for x in key.couplets[c]:
            t = lead_value(x, V)
            if best_t is None or t > best_t:
                best, best_t = x, t
        decisions[c] = best
    return decisions


def challenge_score(items, couplet_filter=None):
    per = {}
    for kid, path, dec in items:
        for lab in path:
            c = couplet_of(lab)
            if couplet_filter is not None and not couplet_filter(kid, c):
                continue
            per.setdefault((kid, c), []).append((lab, dec.get(c)))
    num, den = 0.0, 0
    for kc in sorted(per):
        pairs = per[kc]
        leads = sorted({t for t, _ in pairs})
        m = len(leads)
        if m == 1:
            continue
        recalls = []
        for b in leads:
            tb = [p for t, p in pairs if t == b]
            recalls.append(sum(1 for p in tb if p == b) / len(tb))
        J = (np.mean(recalls) - 1.0 / m) / (1.0 - 1.0 / m)
        num += len(pairs) * J
        den += len(pairs)
    return float(np.clip(num / den, 0.0, 1.0)) if den else 0.0


def raw_accuracy(items):
    hit = tot = 0
    for _kid, path, dec in items:
        for lab in path:
            tot += 1
            hit += int(dec.get(couplet_of(lab)) == lab)
    return hit / tot if tot else 0.0


def feat_stats(examples):
    F = np.concatenate([e["feats"] for e in examples], axis=0).astype(np.float64)
    mean = F.mean(0)
    std = F.std(0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def validate_submission(sub, test, keys, sample):
    assert list(sub.columns) == ["id", "decisions"], sub.columns
    assert list(sample.columns) == ["id", "decisions"], sample.columns
    assert len(sub) == len(test) == len(sample), (len(sub), len(test), len(sample))
    assert sub["id"].is_unique
    assert set(sub["id"]) == set(test["id"]) == set(sample["id"])
    kid_of = dict(zip(test["id"], test["key_id"]))
    for rid, d in zip(sub["id"], sub["decisions"]):
        key = keys[kid_of[rid]]
        labs = d.split(" ")
        assert d == " ".join(labs) and all(labs), rid
        cs = [couplet_of(l) for l in labs]
        assert len(cs) == len(set(cs)), f"couplet answered twice in {rid}"
        assert set(cs) == set(key.couplets), f"couplet coverage mismatch in {rid}"
        assert all(l in key.couplets[couplet_of(l)] for l in labs), f"unknown lead in {rid}"


def main():
    assert torch.cuda.is_available(), "a CUDA GPU is required"
    seed_everything(SEED)
    assert torch.are_deterministic_algorithms_enabled()
    log(f"runtime plan: backbone={BACKBONE}@{BACKBONE_REVISION} seed={SEED} device={DEVICE} "
        f"amp={AMP_DTYPE} epochs={EPOCHS} batch={COUPLETS_PER_STEP}x{GRAD_ACCUM} max_len={MAX_LEN} "
        f"deterministic_algorithms=strict tf32=off workers=0 torch={torch.__version__}")
    os.makedirs(WORK_DIR, exist_ok=True)

    t = time.perf_counter()
    train = pd.read_csv(TRAIN_CSV, keep_default_na=False, dtype=str)
    test = pd.read_csv(TEST_CSV, keep_default_na=False, dtype=str)
    leads_df = pd.read_csv(LEADS_CSV, keep_default_na=False, dtype=str)
    sample = pd.read_csv(SAMPLE_CSV, keep_default_na=False, dtype=str)
    for df in (train, test):
        df["desc_norm"] = [norm_text(d) for d in df["description"]]
    keys = build_keys(leads_df)
    log(f"train {train.shape}, test {test.shape}, keys {len(keys)}, leads {len(leads_df)}")

    fold_of_key = make_key_folds(train, keys)
    train["fold"] = [fold_of_key[k] for k in train["key_id"]]
    for f in range(N_FOLDS):
        ks = sorted({k for k, v in fold_of_key.items() if v == f})
        big = max(ks, key=lambda k: (len(keys[k].couplets), k))
        log(f"fold {f}: {len(ks)} keys, {int((train['fold'] == f).sum())} rows, "
            f"largest key {big} ({len(keys[big].couplets)} couplets)")

    tokenizer = load_tokenizer()
    enc = Encoder(tokenizer, keys)
    train_rows = enc.prepare_rows(train)
    test_rows = enc.prepare_rows(test)
    examples = build_examples(train, train_rows, keys)
    materialise(examples, enc, train_rows, keys)
    log(f"{len(examples)} couplet examples; preprocessing {time.perf_counter() - t:.0f}s")

    fold_arr = train["fold"].to_numpy()
    models, oof = {}, {}
    for name, vfold in MODELS:
        t = time.perf_counter()
        tr_ex = [e for e in examples if fold_arr[e["row"]] != vfold]
        metric_weights(tr_ex)
        fmean, fstd = feat_stats(tr_ex)
        model = train_model(name, tr_ex, fmean, fstd, enc.pad)
        log(f"{name}: trained in {time.perf_counter() - t:.0f}s")
        t = time.perf_counter()
        vidx = [i for i in range(len(train)) if fold_arr[i] == vfold]
        for i in vidx:
            kid = train["key_id"].iloc[i]
            oof[i] = score_row(model, enc, train_rows[i], kid, keys[kid], fmean, fstd)
        log(f"{name}: scored {len(vidx)} fold-{vfold} rows in {time.perf_counter() - t:.0f}s")
        model.to("cpu")
        models[name] = (model, fmean, fstd)
        torch.cuda.empty_cache()

    vrows = sorted(oof)
    paths = {i: train["path"].iloc[i].split() for i in vrows}
    kid_of = {i: train["key_id"].iloc[i] for i in vrows}
    LAM, MODE = DECODE_LAMBDA, DECODE_MODE
    log(f"fixed decoder: mode={MODE}, lambda={LAM}")
    dec = {i: decode(keys[kid_of[i]], oof[i], LAM, MODE) for i in vrows}
    items = [(kid_of[i], paths[i], dec[i]) for i in vrows]
    log(f"fold validation: all {challenge_score(items):.4f} | depth>=4 "
        f"{challenge_score([x for x in items if len(x[1]) >= MIN_SELECT_DEPTH]):.4f} | raw acc {raw_accuracy(items):.4f}")

    t = time.perf_counter()
    for name in models:
        models[name][0].to(DEVICE)
    decisions_out = []
    for i in range(len(test)):
        kid = test["key_id"].iloc[i]
        key = keys[kid]
        per_model = [score_row(m, enc, test_rows[i], kid, key, fm, fs) for m, fm, fs in models.values()]
        avg = {x: float(np.mean([pm[x] for pm in per_model])) for x in per_model[0]}
        dec = decode(key, avg, LAM, MODE)
        decisions_out.append(" ".join(dec[c] for c in key.order))
    log(f"test scored and decoded in {time.perf_counter() - t:.0f}s")

    sub = pd.DataFrame({"id": test["id"].values, "decisions": decisions_out})
    validate_submission(sub, test, keys, sample)
    sub.to_csv(SUBMISSION_CSV, index=False)
    log(f"wrote {SUBMISSION_CSV} ({len(sub)} rows); md5 "
        f"{hashlib.md5(sub.to_csv(index=False).encode()).hexdigest()}")


if __name__ == "__main__":
    main()
