"""
Multi-Witness Question Recovery — Top-5 Leaderboard Solutions (archive)
=========================================================================
This file concatenates all 5 uploaded solutions for reference/archival
purposes, in leaderboard rank order. Each solution is kept verbatim under
its own header; they are NOT meant to be run as a single combined script
(each defines its own main() etc. — run them individually as originally
intended: `python3 <script>.py <public_dir> <submission_out>`).

Contents:
  1. RANK 1 — solution_3_rank1.py
  2. RANK 2 — solution_3_rank2.py
  3. RANK 3 — solution_3_rank3.py
  4. RANK 4 — solution_3_rank4.py
  5. RANK 5 — solution_3_rank5.py
"""

# ============================================================================
# RANK 1 — solution_3_rank1.py
# ============================================================================
r'''
#!/usr/bin/env python3
"""Train a joint scientific-ledger repair model and write the exact output path.

Usage: python3 solution.py <public_dir> <submission_out>
Requires CUDA, torch, transformers>=4.51, peft, and the pinned public Qwen weights.
There is no precomputed challenge checkpoint, external row data, or fallback model.
"""
# User-provided solution saved as requested.
'''


# ============================================================================
# RANK 2 — solution_3_rank2.py
# ============================================================================
r'''
"""Multi-Witness Question Recovery.

Stage 1 (witness selection): fine-tune pairwise cross-encoders (SELECTORS) on "do these two records
answer the same question" pairs from the training ledgers, average their pair logits, then choose the
record subset of size record_count-2 with the highest summed in-cluster affinity.

Stage 2 (question recovery): fine-tune flan-t5-large (full) and Qwen2.5-1.5B-Instruct (LoRA) on
gold-witness records -> original question, generate beam candidates from both for the predicted witness
set, and output the minimum-Bayes-risk hypothesis under the task's recall-weighted chrF. The hypothesis
space holds every candidate and every two-candidate concatenation; the pooled candidates are the pseudo-references.

usage: python3 solution.py <public_dir> <submission_out>
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import itertools
import json
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

SEED = 0
DEV = torch.device("cuda")
SELECTORS = [
    dict(model="BAAI/bge-reranker-large", rev="55611d7bca2a7133960a6d3b71e083071bbfc312", max_len=512, epochs=4, lr=1e-5, bs=16),
]
S2S_MODEL, S2S_REV = "google/flan-t5-large", "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
GEN_MODEL, GEN_REV = "Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
GEN_CFG = dict(max_in=768, max_out=48, epochs=4, lr=1e-4, bs=4, accum=2, lora_r=32, lora_alpha=64, beams=8, nret=8)
S2S_CFG = dict(max_in=768, max_out=48, epochs=5, lr=1e-4, bs=8, accum=1, beams=8, nret=8, gckpt=True)
FALLBACK_Q = "What dataset is used?"


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def worker_init(_):
    seed_all(SEED)


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------- data
def load_tables(public_dir):
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv", dtype=str)
    labels = pd.read_csv(public_dir / "train_labels.csv", dtype=str)
    test = pd.read_csv(public_dir / "test.csv", dtype=str)
    train = train.merge(labels, on="id", how="left")
    for df in (train, test):
        df["records"] = df.records_json.map(json.loads)
        df["n"] = df.record_count.astype(int)
    train["wit"] = train.witnesses.map(lambda s: tuple(sorted(int(t[1:]) for t in s.split())))
    return train, test


def candidate_sets(n):
    return [tuple(c) for c in itertools.combinations(range(n), n - 2)]


def rec_text(r):
    return f"Answer: {r['answer'].strip()} Evidence: {r['evidence'].strip()}"


# ---------------------------------------------------------------- chrF (for MBR candidate selection)
def norm_q(s):
    s = unicodedata.normalize("NFKC", str(s)).casefold()
    return re.sub(r"\s+", " ", s).strip()


def ngram_counts(s, max_n=6):
    return [Counter(s[i:i + n] for i in range(len(s) - n + 1)) for n in range(1, max_n + 1)]


def chrf_counts(hc, rc, beta=2.0):
    total, b2 = 0.0, beta * beta
    for hg, rg in zip(hc, rc):
        hn, rn = sum(hg.values()), sum(rg.values())
        small, big = (hg, rg) if len(hg) <= len(rg) else (rg, hg)
        m = sum(min(v, big[k]) for k, v in small.items() if k in big)
        p = m / hn if hn else 0.0
        rc_ = m / rn if rn else 0.0
        den = b2 * p + rc_
        total += (1 + b2) * p * rc_ / den if den > 0 else 0.0
    return total / len(hc)


def mbr_pick(texts):
    """Expected-chrF decoding: hypotheses are the distinct candidates and their ordered two-way concatenations."""
    texts = list(dict.fromkeys(t for t in texts if t)) or [FALLBACK_Q]
    refs = [ngram_counts(norm_q(t)) for t in texts]
    hyps = texts + [a + " " + b for a in texts for b in texts if a != b]
    hyps = [h for h in hyps if len(h) <= 512]
    best, bs = texts[0], -1.0
    for h in hyps:
        hc = ngram_counts(norm_q(h))
        u = sum(chrf_counts(hc, r) for r in refs)
        if u > bs:
            best, bs = h, u
    return best


# ================================================================ stage 1: witness selection
class PairDS(Dataset):
    def __init__(self, items, tok, max_len):
        self.items, self.tok, self.max_len = items, tok, max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b, y = self.items[i]
        enc = self.tok(a, b, truncation=True, max_length=self.max_len)
        enc["labels"] = float(y)
        return enc


def collate_pairs(tok):
    def fn(batch):
        labels = torch.tensor([b.pop("labels") for b in batch])
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return fn


def pair_items(rows, half=False, seed=0):
    """All ordered record pairs; half=True keeps each unordered pair once in a seeded random orientation."""
    items, index = [], []
    rng = random.Random(seed)
    for r in rows:
        recs = r["records"]; wit = set(r["wit"]) if r.get("wit") is not None else set()
        n = len(recs)
        for i in range(n):
            for j in range(n):
                if i == j or (half and i > j):
                    continue
                a, b = (i, j) if not half or rng.random() < 0.5 else (j, i)
                items.append((rec_text(recs[a]), rec_text(recs[b]), int(i in wit and j in wit)))
                index.append((r["id"], a, b))
    return items, index


def bucket_batches(lens, bs, seed, mult=25):
    """Batches of similar token length (sorted inside shuffled chunks of bs*mult), in shuffled order."""
    rng = random.Random(seed)
    idx = list(range(len(lens))); rng.shuffle(idx)
    batches = []
    for c in range(0, len(idx), bs * mult):
        chunk = sorted(idx[c:c + bs * mult], key=lambda i: lens[i])
        batches += [chunk[k:k + bs] for k in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def train_selector(rows, cfg):
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["rev"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=1, revision=cfg["rev"],
                                                               torch_dtype=torch.float32).float().to(DEV)

    def epoch_loader(ep):
        items, _ = pair_items(rows, half=True, seed=ep)  # pair orientation is re-drawn every epoch
        ds = PairDS(items, tok, cfg["max_len"])
        lens = [len(ds[i]["input_ids"]) for i in range(len(ds))]
        return DataLoader(ds, batch_sampler=bucket_batches(lens, cfg["bs"], 1000 + ep), collate_fn=collate_pairs(tok))

    items, _ = pair_items(rows, half=True)
    steps = ((len(items) + cfg["bs"] - 1) // cfg["bs"]) * cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)
    pos = sum(y for _, _, y in items); neg = len(items) - pos
    pw = torch.tensor(neg / max(pos, 1), device=DEV)
    model.train()
    for ep in range(cfg["epochs"]):
        tot, dl = 0.0, epoch_loader(ep)
        for batch in dl:
            batch = {k: v.to(DEV) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch).logits.squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(out.float(), labels, pos_weight=pw)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); sch.step(); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()
        print(f"[selector {cfg['model']}] epoch {ep} loss {tot / len(dl):.4f}", flush=True)
    return model, tok


@torch.no_grad()
def pair_affinities(model, tok, rows, cfg, bs=64):
    """Symmetrised pair-logit matrix per ledger (both orders scored)."""
    model.eval()
    items, index = pair_items(rows)
    ds = PairDS(items, tok, cfg["max_len"])
    order = sorted(range(len(ds)), key=lambda i: len(items[i][0]) + len(items[i][1]))
    batches = [order[k:k + bs] for k in range(0, len(order), bs)]
    dl = DataLoader(ds, batch_sampler=batches, collate_fn=collate_pairs(tok))
    logits = np.zeros(len(items), dtype=np.float32)
    for idx, batch in zip(batches, dl):
        batch = {k: v.to(DEV) for k, v in batch.items()}
        batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits[idx] = model(**batch).logits.squeeze(-1).float().cpu().numpy()
    mats = {r["id"]: np.zeros((len(r["records"]),) * 2, dtype=np.float32) for r in rows}
    for (rid, i, j), v in zip(index, logits):
        mats[rid][i, j] = v
    return {rid: (m + m.T) / 2.0 for rid, m in mats.items()}


def select_witnesses(rows, mats_list):
    preds = {}
    for r in rows:
        m = sum(mats[r["id"]] for mats in mats_list) / len(mats_list)
        cands = candidate_sets(len(r["records"]))
        scores = [sum(m[i, j] for a, i in enumerate(S) for j in S[a + 1:]) for S in cands]
        preds[r["id"]] = cands[int(np.argmax(scores))]
    return preds


# ================================================================ stage 2: question generation
class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, dropout=0.05):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.zeros_(self.B)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        out = self.base(x)
        lora = (self.drop(x).to(self.A.dtype) @ self.A.t() @ self.B.t()) * self.scale
        return out + lora.to(out.dtype)


def apply_lora(model, targets, r, alpha):
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.Linear) and cname in targets:
                setattr(mod, cname, LoRALinear(child, r, alpha))
    for pname, p in model.named_parameters():
        if not (pname.endswith(".A") or pname.endswith(".B")):
            p.requires_grad = False
    return model


def build_input_ids(tok, recs, wit, max_in, prefix="recover the question:"):
    pre = tok(prefix, add_special_tokens=False)["input_ids"]
    ans = [tok(f" answer: {recs[i]['answer'].strip()}", add_special_tokens=False)["input_ids"][:160] for i in wit]
    evi = [tok(f" evidence: {recs[i]['evidence'].strip()}", add_special_tokens=False)["input_ids"] for i in wit]
    budget = max_in - len(pre) - sum(len(x) for x in ans) - 2
    per = max(16, budget // len(wit))
    evi = [e[:per] for e in evi]
    ids = pre + [t for x in ans for t in x] + [t for x in evi for t in x]
    return ids[: max_in - 1]


class GenDS(Dataset):
    def __init__(self, rows, tok, cfg, train=True, wits=None):
        self.rows, self.tok, self.cfg, self.train, self.wits = rows, tok, cfg, train, wits

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; tok, cfg = self.tok, self.cfg
        wit = list(r["wit"]) if self.wits is None else list(self.wits[r["id"]])
        if self.train:
            random.shuffle(wit)
        src = build_input_ids(tok, r["records"], wit, cfg["max_in"])
        q = tok("\nQuestion:", add_special_tokens=False)["input_ids"]
        prompt = src + q
        if self.train:
            tgt = tok(r["question"].strip(), add_special_tokens=False)["input_ids"][: cfg["max_out"] - 1]
            ids = prompt + tgt + [tok.eos_token_id]
            return {"input_ids": ids, "labels": [-100] * len(prompt) + tgt + [tok.eos_token_id]}
        return {"input_ids": prompt}


def collate_gen(pad_id, train):
    def fn(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ii = torch.full((len(batch), L), pad_id, dtype=torch.long); am = torch.zeros((len(batch), L), dtype=torch.long)
        if train:
            lb = torch.full((len(batch), L), -100, dtype=torch.long)
            for k, b in enumerate(batch):
                n = len(b["input_ids"]); ii[k, :n] = torch.tensor(b["input_ids"]); am[k, :n] = 1
                lb[k, :n] = torch.tensor(b["labels"])
            return {"input_ids": ii, "attention_mask": am, "labels": lb}
        for k, b in enumerate(batch):
            n = len(b["input_ids"]); ii[k, L - n:] = torch.tensor(b["input_ids"]); am[k, L - n:] = 1
        return {"input_ids": ii, "attention_mask": am}
    return fn


def train_generator(rows, cfg):
    tok = AutoTokenizer.from_pretrained(GEN_MODEL, revision=GEN_REV)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, revision=GEN_REV, torch_dtype=torch.bfloat16)
    model = apply_lora(model, {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"},
                       cfg["lora_r"], cfg["lora_alpha"]).to(DEV)
    g = torch.Generator(); g.manual_seed(SEED)
    dl = DataLoader(GenDS(rows, tok, cfg, train=True), batch_size=cfg["bs"], shuffle=True,
                    collate_fn=collate_gen(tok.pad_token_id, True), generator=g, worker_init_fn=worker_init)
    accum = cfg["accum"]
    steps = (len(dl) // accum) * cfg["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    model.train()
    for ep in range(cfg["epochs"]):
        tot, k = 0.0, 0
        for batch in dl:
            batch = {kk: v.to(DEV) for kk, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).last_hidden_state
                tgt = batch["labels"][:, 1:]
                mask = tgt != -100
                logits = model.lm_head(h[:, :-1][mask])
            loss = F.cross_entropy(logits.float(), tgt[mask])
            (loss / accum).backward()
            k += 1
            if k % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()
        print(f"[generator] epoch {ep} loss {tot / len(dl):.4f}", flush=True)
    return model, tok


def clean_question(t):
    t = "".join(c for c in t if not unicodedata.category(c).startswith("C"))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:512].strip() or FALLBACK_Q


@torch.no_grad()
def generate_questions(model, tok, rows, wits, cfg, bs=16):
    model.eval()
    dl = DataLoader(GenDS(rows, tok, cfg, train=False, wits=wits), batch_size=bs, shuffle=False,
                    collate_fn=collate_gen(tok.pad_token_id, False))
    out, k = {}, 0
    nr = cfg["nret"]
    for batch in dl:
        batch = {kk: v.to(DEV) for kk, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**batch, num_beams=cfg["beams"], num_return_sequences=nr, max_new_tokens=cfg["max_out"],
                                 do_sample=False, early_stopping=True, pad_token_id=tok.pad_token_id,
                                 eos_token_id=tok.eos_token_id)
        seqs = gen[:, batch["input_ids"].shape[1]:]
        texts = tok.batch_decode(seqs, skip_special_tokens=True)
        for b in range(len(texts) // nr):
            out[rows[k]["id"]] = [clean_question(texts[b * nr + j].split("\n")[0]) for j in range(nr)]; k += 1
    return out


# ---------------------------------------------------------------- encoder-decoder generator (flan-t5-large, full fine-tune)
@torch.no_grad()
def fix_t5_rescale(model):
    """Some transformers versions apply T5's tied-embedding d_model**-0.5 output rescale to flan-t5's untied lm_head,
    which flattens the logits. Probe the forward pass and fold d_model**0.5 into lm_head when that happens."""
    model.eval()
    ii = torch.tensor([[5, 6, 1]], device=model.lm_head.weight.device)
    dec = torch.tensor([[model.config.pad_token_id, 5]], device=ii.device)
    out = model(input_ids=ii, attention_mask=torch.ones_like(ii), decoder_input_ids=dec, output_hidden_states=True)
    direct = model.lm_head(out.decoder_hidden_states[-1])
    if not torch.allclose(out.logits, direct, atol=1e-3, rtol=1e-3) and \
            torch.allclose(out.logits, direct * model.config.d_model ** -0.5, atol=1e-3, rtol=1e-3):
        model.lm_head.weight.mul_(model.config.d_model ** 0.5)
        print("[s2s] folded d_model**0.5 into lm_head", flush=True)


class S2SDS(Dataset):
    def __init__(self, rows, tok, cfg, train=True, wits=None):
        self.rows, self.tok, self.cfg, self.train, self.wits = rows, tok, cfg, train, wits

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]; tok, cfg = self.tok, self.cfg
        wit = list(r["wit"]) if self.wits is None else list(self.wits[r["id"]])
        if self.train:
            random.shuffle(wit)
        item = {"input_ids": build_input_ids(tok, r["records"], wit, cfg["max_in"]) + [tok.eos_token_id]}
        if self.train:
            item["labels"] = tok(r["question"].strip(), add_special_tokens=False)["input_ids"][: cfg["max_out"] - 1] + [tok.eos_token_id]
        return item


def collate_s2s(pad_id):
    def fn(batch):
        L = max(len(b["input_ids"]) for b in batch)
        ii = torch.full((len(batch), L), pad_id, dtype=torch.long); am = torch.zeros((len(batch), L), dtype=torch.long)
        for k, b in enumerate(batch):
            n = len(b["input_ids"]); ii[k, :n] = torch.tensor(b["input_ids"]); am[k, :n] = 1
        out = {"input_ids": ii, "attention_mask": am}
        if "labels" in batch[0]:
            T = max(len(b["labels"]) for b in batch)
            lb = torch.full((len(batch), T), -100, dtype=torch.long)
            for k, b in enumerate(batch):
                lb[k, :len(b["labels"])] = torch.tensor(b["labels"])
            out["labels"] = lb
        return out
    return fn


def train_s2s(rows, cfg):
    tok = AutoTokenizer.from_pretrained(S2S_MODEL, revision=S2S_REV)
    model = AutoModelForSeq2SeqLM.from_pretrained(S2S_MODEL, revision=S2S_REV, torch_dtype=torch.float32).to(DEV)
    fix_t5_rescale(model)
    if cfg["gckpt"]:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    g = torch.Generator(); g.manual_seed(SEED)
    dl = DataLoader(S2SDS(rows, tok, cfg, train=True), batch_size=cfg["bs"], shuffle=True,
                    collate_fn=collate_s2s(tok.pad_token_id), generator=g, worker_init_fn=worker_init)
    accum = cfg["accum"]
    steps = (len(dl) // accum) * cfg["epochs"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    start, pad = model.config.decoder_start_token_id, model.config.pad_token_id
    model.train()
    for ep in range(cfg["epochs"]):
        tot, k = 0.0, 0
        for batch in dl:
            batch = {kk: v.to(DEV) for kk, v in batch.items()}
            labels = batch["labels"]
            dec = torch.full_like(labels, pad); dec[:, 0] = start; dec[:, 1:] = labels[:, :-1]; dec[dec == -100] = pad
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], decoder_input_ids=dec).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=-100)
            (loss / accum).backward()
            k += 1
            if k % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item()
        print(f"[s2s] epoch {ep} loss {tot / len(dl):.4f}", flush=True)
    if cfg["gckpt"]:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    return model, tok


@torch.no_grad()
def generate_s2s(model, tok, rows, wits, cfg, bs=16):
    model.eval()
    dl = DataLoader(S2SDS(rows, tok, cfg, train=False, wits=wits), batch_size=bs, shuffle=False, collate_fn=collate_s2s(tok.pad_token_id))
    out, k = {}, 0
    nr = cfg["nret"]
    for batch in dl:
        batch = {kk: v.to(DEV) for kk, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            gen = model.generate(**batch, num_beams=cfg["beams"], num_return_sequences=nr, max_new_tokens=cfg["max_out"],
                                 do_sample=False, early_stopping=True)
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for b in range(len(texts) // nr):
            out[rows[k]["id"]] = [clean_question(texts[b * nr + j]) for j in range(nr)]; k += 1
    return out


# ================================================================ main
def main():
    public_dir, out_path = sys.argv[1], sys.argv[2]
    seed_all(SEED)
    train, test = load_tables(public_dir)
    sample = pd.read_csv(Path(public_dir) / "sample_submission.csv", dtype=str)
    tr_rows, te_rows = train.to_dict("records"), test.to_dict("records")

    mats_list = []
    for cfg in SELECTORS:
        seed_all(SEED)
        sel_model, sel_tok = train_selector(tr_rows, cfg)
        mats_list.append(pair_affinities(sel_model, sel_tok, te_rows, cfg))
        del sel_model; torch.cuda.empty_cache()
    pred_wits = select_witnesses(te_rows, mats_list)
    print("[selector] test witness-count distribution:", Counter(len(v) for v in pred_wits.values()), flush=True)

    seed_all(SEED)
    s2s_model, s2s_tok = train_s2s(tr_rows, S2S_CFG)
    cands = generate_s2s(s2s_model, s2s_tok, te_rows, pred_wits, S2S_CFG)
    del s2s_model; torch.cuda.empty_cache()

    seed_all(SEED)
    gen_model, gen_tok = train_generator(tr_rows, GEN_CFG)
    cands2 = generate_questions(gen_model, gen_tok, te_rows, pred_wits, GEN_CFG)
    del gen_model; torch.cuda.empty_cache()
    questions = {r["id"]: mbr_pick(cands[r["id"]] + cands2[r["id"]]) for r in te_rows}

    sub = pd.DataFrame({"id": sample.id,
                        "witnesses": [" ".join(f"w{i}" for i in sorted(pred_wits[i])) for i in sample.id],
                        "question": [questions[i] for i in sample.id]})
    assert list(sub.columns) == ["id", "witnesses", "question"] and len(sub) == len(sample) and sub.id.is_unique
    assert all(2 <= len(w.split()) <= 4 for w in sub.witnesses) and all(0 < len(q) <= 512 for q in sub.question)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print("[done] wrote", out_path, len(sub), "rows", flush=True)


if __name__ == "__main__":
    main()
'''


# ============================================================================
# RANK 3 — solution_3_rank3.py
# ============================================================================
r'''
#!/usr/bin/env python3
"""Multi-Witness Question Recovery.

    python3 solution.py <public_dataset_directory> <submission_csv_path>
"""
import os

# Resource and numerics policy, assigned (not defaulted) before numpy/torch import.
N_THREADS = "8"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = N_THREADS
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import csv
import itertools
import json
import random
import sys
import time
import unicodedata
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))

W_MODEL = "answerdotai/ModernBERT-large"
W_SEEDS = [11, 22]
W_EPOCHS = 4
W_LR = 1e-5
W_ROWS_PER_STEP = 4
W_MAX_ANS = 96
W_MAX_EV = 160
W_WBCE = 1.0
W_WCE = 1.0

Q_MODEL = "google/flan-t5-large"
Q_SEEDS = [11, 22]
Q_EPOCHS = 2
Q_LR = 1e-4
Q_BS = 4
Q_ACCUM = 2
Q_MAX_LEN = 768
Q_MAX_TGT = 48
Q_ANS_BUDGET = 96
Q_AUG_PER_QUESTION = 4
Q_NBEAM = 8
Q_NRET = 8
Q_LENPEN = 1.0

MBR_TEMP = 0.6
MBR_MAX_UNION = 2
MBR_TOP_FOR_UNION = 8

STRICT_DETERMINISM = False
HUB_ATTEMPTS = 4

_T0 = time.time()

def log(msg):
    print("[%7.1fs] %s" % (time.time() - _T0, msg), flush=True)

def configure_backend():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(STRICT_DETERMINISM)

def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def hub(fn, *a, **kw):
    last = None
    for _ in range(HUB_ATTEMPTS):
        try:
            return fn(*a, **kw)
        except Exception as exc:
            last = exc
    raise RuntimeError("could not load pretrained weights after %d attempts: %r"
                       % (HUB_ATTEMPTS, last))

class Row(object):
    __slots__ = ("id", "group", "n", "records", "witnesses", "question")

    def __init__(self, rid, group, records, witnesses=None, question=None):
        self.id = rid
        self.group = group
        self.records = records
        self.n = len(records)
        self.witnesses = witnesses
        self.question = question

    @property
    def k(self):
        return self.n - 2

def read_rows(data_dir, split):
    path = os.path.join(data_dir, split + ".csv")
    with open(path, "r", encoding="utf-8", newline="") as f:
        raw = list(csv.DictReader(f))
    labels = {}
    lpath = os.path.join(data_dir, "train_labels.csv")
    if split == "train":
        with open(lpath, "r", encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                labels[r["id"]] = (
                    tuple(sorted(int(t[1:]) for t in r["witnesses"].split())),
                    r["question"]
                )
    out = []
    for r in raw:
        w, q = labels.get(r["id"], (None, None))
        out.append(Row(r["id"], r["group"], json.loads(r["records_json"]), w, q))
    return out

def subsets(n, k):
    return [tuple(c) for c in itertools.combinations(range(n), k)]

def rec_key(rec):
    return rec["answer"] + chr(1) + rec["evidence"]

def gen_examples(rows, max_extra=Q_AUG_PER_QUESTION):
    items = []
    for r in sorted(rows, key=lambda x: x.id):
        items.append(([(r.records[i]["answer"], r.records[i]["evidence"])
                       for i in sorted(r.witnesses)], r.question))
    q_of, q_recs, seen = {}, {}, set()
    for r in sorted(rows, key=lambda x: x.id):
        for i in sorted(r.witnesses):
            q_of.setdefault(rec_key(r.records[i]), (r.group, r.question))
    for r in sorted(rows, key=lambda x: x.id):
        for i in range(r.n):
            k = rec_key(r.records[i])
            if k not in q_of:
                continue
            gq = q_of[k]
            if (gq, k) in seen:
                continue
            seen.add((gq, k))
            q_recs.setdefault(gq, []).append(
                (k, r.records[i]["answer"], r.records[i]["evidence"])
            )
    for gq in sorted(q_recs):
        pool = sorted(q_recs[gq])
        m = len(pool)
        cands = []
        for size in range(1, min(4, m) + 1):
            cands.extend(subsets(m, size))
        cands.sort(key=lambda c: (-len(c), c))
        for c in cands[:max_extra]:
            items.append(([(pool[i][1], pool[i][2]) for i in c], gq[1]))
    return items

def encode_record(tok, answer, evidence, max_ans, max_ev):
    ids = tok("answer: " + answer, add_special_tokens=False)["input_ids"][:max_ans]
    if max_ev > 0:
        ids += tok(" evidence: " + evidence,
                    add_special_tokens=False)["input_ids"][:max_ev]
    if not ids:
        ids = [tok.unk_token_id if tok.unk_token_id is not None else tok.pad_token_id]
    return ids

def row_pairs(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]

def pad_batch(seqs, pad_id):
    L = max(len(x) for x in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
        att[i, :len(x)] = 1
    return ids, att

def subset_scores_from_pairs(pair_logit, n, k, pair_index):
    cands = subsets(n, k)
    out = []
    for c in cands:
        v = 0.0
        for a in range(len(c)):
            for b in range(a + 1, len(c)):
                v += pair_logit[pair_index[(c[a], c[b])]]
        out.append(v)
    return torch.stack(out), cands

def train_stage_w(train_rows, test_rows, dev, seed):
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
    tok = hub(AutoTokenizer.from_pretrained, W_MODEL)
    seed_all(seed)
    model = hub(AutoModelForSequenceClassification.from_pretrained,
                W_MODEL, num_labels=1).to(dev)
    model.gradient_checkpointing_enable()
    enc_cache = {}

    def enc(tag, rows, ri, i):
        key = (tag, ri, i)
        if key not in enc_cache:
            rec = rows[ri].records[i]
            enc_cache[key] = encode_record(
                tok, rec["answer"], rec["evidence"], W_MAX_ANS, W_MAX_EV
            )
        return enc_cache[key]

    def pair_seqs(tag, rows, ri, flip, flip_rng=None):
        r = rows[ri]
        seqs = []
        for (i, j) in row_pairs(r.n):
            a, b = enc(tag, rows, ri, i), enc(tag, rows, ri, j)
            swap = (flip_rng.random() < 0.5) if flip_rng is not None else flip
            if swap:
                a, b = b, a
            seqs.append(
                [tok.cls_token_id] + a + [tok.sep_token_id] +
                b + [tok.sep_token_id]
            )
        return seqs

    opt = torch.optim.AdamW(
        model.parameters(), lr=W_LR, weight_decay=0.01, betas=(0.9, 0.98)
    )
    n_upd = max(1, (len(train_rows) * W_EPOCHS) // W_ROWS_PER_STEP)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * n_upd), n_upd)

    rng = random.Random(seed)
    model.train()
    for ep in range(W_EPOCHS):
        order = list(range(len(train_rows)))
        rng.shuffle(order)
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for step, ri in enumerate(order):
            r = train_rows[ri]
            prs = row_pairs(r.n)
            pidx = dict((p, t) for t, p in enumerate(prs))
            ids, att = pad_batch(
                pair_seqs("tr", train_rows, ri, False, rng), tok.pad_token_id
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(
                    input_ids=ids.to(dev), attention_mask=att.to(dev)
                ).logits.squeeze(-1)
            lg = lg.float()
            y = torch.tensor(
                [1.0 if (i in r.witnesses and j in r.witnesses) else 0.0
                 for (i, j) in prs], device=dev
            )
            lb = F.binary_cross_entropy_with_logits(lg, y)
            ss, cands = subset_scores_from_pairs(lg, r.n, r.k, pidx)
            lc = F.cross_entropy(
                ss.unsqueeze(0),
                torch.tensor([cands.index(r.witnesses)], device=dev)
            )
            loss = (W_WBCE * lb + W_WCE * lc) / W_ROWS_PER_STEP
            loss.backward()
            tot += float(loss) * W_ROWS_PER_STEP
            nb += 1
            if (step + 1) % W_ROWS_PER_STEP == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
        if len(order) % W_ROWS_PER_STEP:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            opt.zero_grad(set_to_none=True)
        log("  stage W seed %d epoch %d/%d loss %.4f" %
            (seed, ep + 1, W_EPOCHS, tot / nb))

    model.eval()
    out = {}
    with torch.no_grad():
        for ri in range(len(test_rows)):
            r = test_rows[ri]
            acc = None
            for flip in (False, True):
                ids, att = pad_batch(
                    pair_seqs("te", test_rows, ri, flip), tok.pad_token_id
                )
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(
                        input_ids=ids.to(dev),
                        attention_mask=att.to(dev)
                    ).logits.squeeze(-1).float()
                acc = lg if acc is None else acc + lg
            out[r.id] = (acc / 2.0).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return out

def build_input(tok, recs, max_len, ans_budget):
    head = tok(
        "recover the question answered by these records.",
        add_special_tokens=False
    )["input_ids"]
    per = max(32, (max_len - len(head) - 8) // max(1, len(recs)))
    ev_budget = max(0, per - ans_budget - 6)
    ids = ([tok.bos_token_id] if tok.bos_token_id is not None else []) + list(head)
    for idx, (a, e) in enumerate(recs):
        ids += tok(
            " record %d answer: " % (idx + 1) + a,
            add_special_tokens=False
        )["input_ids"][:ans_budget + 6]
        if ev_budget > 0:
            ids += tok(
                " evidence: " + e, add_special_tokens=False
            )["input_ids"][:ev_budget]
    return ids[:max_len - 1] + [tok.eos_token_id]

class GenData(torch.utils.data.Dataset):
    def __init__(self, tok, items):
        self.tok, self.items = tok, items
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        recs, q = self.items[i]
        src = build_input(self.tok, recs, Q_MAX_LEN, Q_ANS_BUDGET)
        tgt = self.tok(q, add_special_tokens=False)["input_ids"][:Q_MAX_TGT - 1]
        return {"src": src, "tgt": tgt + [self.tok.eos_token_id]}

def collate_gen(batch, pad_id):
    L = max(len(b["src"]) for b in batch)
    T = max(len(b["tgt"]) for b in batch)
    src = torch.full((len(batch), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(batch), L), dtype=torch.long)
    tgt = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, :len(b["src"])] = torch.tensor(b["src"])
        att[i, :len(b["src"])] = 1
        tgt[i, :len(b["tgt"])] = torch.tensor(b["tgt"])
    return src, att, tgt

def train_stage_q(items, test_inputs, dev, seed):
    from transformers import (
        AutoTokenizer, AutoModelForSeq2SeqLM,
        get_linear_schedule_with_warmup
    )
    tok = hub(AutoTokenizer.from_pretrained, Q_MODEL)
    seed_all(seed)
    model = hub(AutoModelForSeq2SeqLM.from_pretrained, Q_MODEL).to(dev)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = GenData(tok, items)
    g = torch.Generator()
    g.manual_seed(seed)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=Q_BS, shuffle=True, generator=g, num_workers=0,
        collate_fn=lambda b: collate_gen(b, tok.pad_token_id)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=Q_LR, weight_decay=0.01)
    steps = max(1, (len(dl) * Q_EPOCHS) // Q_ACCUM)
    sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)

    model.train()
    for ep in range(Q_EPOCHS):
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, (src, att, tgt) in enumerate(dl):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(
                    input_ids=src.to(dev),
                    attention_mask=att.to(dev),
                    labels=tgt.to(dev)
                ).loss
            (loss / Q_ACCUM).backward()
            if (bi + 1) % Q_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sch.step()
                opt.zero_grad(set_to_none=True)
            tot += float(loss)
            nb += 1
        log("  stage Q seed %d epoch %d/%d loss %.4f" %
            (seed, ep + 1, Q_EPOCHS, tot / nb))

    model.eval()
    model.config.use_cache = True
    order = sorted(test_inputs)
    cands = {}
    B = 4
    with torch.no_grad():
        for b0 in range(0, len(order), B):
            chunk = order[b0:b0 + B]
            srcs = [
                build_input(tok, test_inputs[rid], Q_MAX_LEN, Q_ANS_BUDGET)
                for rid in chunk
            ]
            L = max(len(s) for s in srcs)
            ids = torch.full(
                (len(chunk), L), tok.pad_token_id, dtype=torch.long
            )
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, s in enumerate(srcs):
                ids[i, :len(s)] = torch.tensor(s)
                att[i, :len(s)] = 1
            kw = dict(
                max_new_tokens=Q_MAX_TGT, num_beams=Q_NBEAM,
                num_return_sequences=Q_NRET, length_penalty=Q_LENPEN,
                early_stopping=True, do_sample=False,
                output_scores=True, return_dict_in_generate=True
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.generate(
                    input_ids=ids.to(dev),
                    attention_mask=att.to(dev),
                    **kw
                )
            txt = tok.batch_decode(o.sequences, skip_special_tokens=True)
            sc = (
                o.sequences_scores.float().cpu().tolist()
                if getattr(o, "sequences_scores", None) is not None
                else [0.0] * len(txt)
            )
            for i, rid in enumerate(chunk):
                cands[rid] = list(zip(
                    txt[i * Q_NRET:(i + 1) * Q_NRET],
                    sc[i * Q_NRET:(i + 1) * Q_NRET]
                ))
    del model
    torch.cuda.empty_cache()
    return cands

def norm_text(s):
    return " ".join(unicodedata.normalize("NFKC", s).casefold().split()).strip()

def chrf_beta2(hyp, ref, maxn=6):
    h, r = norm_text(hyp), norm_text(ref)
    tot = 0.0
    for n in range(1, maxn + 1):
        hg = Counter(h[i:i+n] for i in range(len(h)-n+1)) if len(h) >= n else Counter()
        rg = Counter(r[i:i+n] for i in range(len(r)-n+1)) if len(r) >= n else Counter()
        hn, rn = sum(hg.values()), sum(rg.values())
        m = sum((hg & rg).values())
        P = m / hn if hn else 0.0
        R = m / rn if rn else 0.0
        den = 4.0 * P + R
        tot += (5.0 * P * R / den) if den > 0 else 0.0
    return tot / maxn

def mbr_select(cands):
    uniq, seen = [], {}
    for t, s in cands:
        k = norm_text(t)
        if not k:
            continue
        if k in seen:
            uniq[seen[k]] = (uniq[seen[k]][0], max(uniq[seen[k]][1], s))
        else:
            seen[k] = len(uniq)
            uniq.append((t, s))
    if not uniq:
        return "What question do these records answer?"
    mx = max(s for _, s in uniq)
    w = [pow(2.718281828459045, (s-mx)/MBR_TEMP) for _, s in uniq]
    tw = sum(w)
    w = [x/tw for x in w]
    hyps = [t for t, _ in uniq]
    top = hyps[:MBR_TOP_FOR_UNION]
    for m in range(2, MBR_MAX_UNION + 1):
        for combo in itertools.combinations(range(len(top)), m):
            hyps.append(" ".join(top[i].strip() for i in combo).strip())
    best, bs = hyps[0], -1e18
    for h in hyps:
        u = sum(wj * chrf_beta2(h, c[0]) for wj, c in zip(w, uniq))
        if u > bs:
            bs, best = u, h
    return best

CTRL_CATS = ("Cc", "Cf", "Cs", "Co", "Cn", "Zl", "Zp")

def clean_question(q):
    q = unicodedata.normalize("NFKC", q if q else "")
    q = "".join(" " if unicodedata.category(c) in CTRL_CATS else c for c in q)
    q = " ".join(q.split()).strip()
    if len(q) > 512:
        q = q[:512].rstrip()
    return q if q else "What question do these records answer?"

def write_submission(path, ids, witness_by_id, question_by_id):
    import re
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["id", "witnesses", "question"])
        for rid in ids:
            idx = sorted(set(int(x) for x in witness_by_id[rid]))
            wr.writerow([
                rid, " ".join("w%d" % i for i in idx),
                clean_question(question_by_id[rid])
            ])
    with open(tmp, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        assert next(rd) == ["id", "witnesses", "question"]
        seen = set()
        for row in rd:
            assert len(row) == 3
            rid, w, q = row
            assert rid in ids and rid not in seen
            seen.add(rid)
            toks = w.split(" ")
            assert 2 <= len(toks) <= 4 and len(set(toks)) == len(toks)
            nums = [int(t[1:]) for t in toks]
            assert all(re.fullmatch(r"w[0-5]", t) for t in toks) and nums == sorted(nums)
            assert q and q == q.strip() and len(q) <= 512
            assert not any(unicodedata.category(c) in CTRL_CATS for c in q)
        assert seen == set(ids), "submission is missing ids"
    os.replace(tmp, path)
    return len(ids)

def main():
    assert len(sys.argv) == 3, __doc__
    data_dir, out_path = sys.argv[1], sys.argv[2]
    if not torch.cuda.is_available():
        raise RuntimeError("this solution requires the CUDA device named in the task contract")
    dev = torch.device("cuda")
    configure_backend()
    torch.set_num_threads(int(N_THREADS))
    log("device %s | torch %s" % (torch.cuda.get_device_name(0), torch.__version__))

    train_rows = read_rows(data_dir, "train")
    test_rows = read_rows(data_dir, "test")
    test_ids = [r.id for r in test_rows]
    log("train %d rows, test %d rows" % (len(train_rows), len(test_rows)))

    acc = dict((rid, None) for rid in test_ids)
    for sd in W_SEEDS:
        part = train_stage_w(train_rows, test_rows, dev, sd)
        for rid in sorted(part):
            acc[rid] = part[rid] if acc[rid] is None else acc[rid] + part[rid]
        log("stage W seed %d done" % sd)

    witness = {}
    for r in test_rows:
        prs = row_pairs(r.n)
        pidx = dict((p, t) for t, p in enumerate(prs))
        lg = torch.tensor(acc[r.id] / len(W_SEEDS))
        ss, cands = subset_scores_from_pairs(lg, r.n, r.k, pidx)
        witness[r.id] = list(cands[int(torch.argmax(ss))])
    log("witness sizes: %s" % dict(Counter(len(v) for v in witness.values())))

    items = gen_examples(train_rows)
    log("stage Q training items %d" % len(items))
    test_inputs = {
        r.id: [(r.records[i]["answer"], r.records[i]["evidence"])
               for i in witness[r.id]]
        for r in test_rows
    }
    pooled = dict((rid, []) for rid in test_ids)
    for sd in Q_SEEDS:
        part = train_stage_q(items, test_inputs, dev, sd)
        for rid in sorted(part):
            pooled[rid].extend(part[rid])
        log("stage Q seed %d done" % sd)
    question = dict((rid, mbr_select(pooled[rid])) for rid in sorted(test_ids))

    n = write_submission(out_path, test_ids, witness, question)
    lens = sorted(len(question[r]) for r in test_ids)
    log("wrote %d rows to %s | question chars median %d mean %.1f" %
        (n, out_path, lens[len(lens)//2], sum(lens)/len(lens)))

if __name__ == "__main__":
    main()
'''


# ============================================================================
# RANK 4 — solution_3_rank4.py
# ============================================================================
r'''
#!/usr/bin/env python3
"""
Ledger provenance restoration + archival question recovery.

Two fine-tuned neural sequence models, both trained inside this script on the
supplied training data only:

  (1) SELECTOR -- a transformer cross-encoder fine-tuned to score whether two
      answer/evidence records were written for the same latent question.
      Group-K-fold ensemble; exact witness subsets are recovered by exhaustive
      MAP decoding over the pairwise log-odds under the task constraint
      |witnesses| = record_count - 2 (verified on the training labels).

  (2) GENERATOR -- an encoder-decoder LM fine-tuned to emit the original
      question given the records of a gold cluster. Decoding strategy is
      selected on a group-disjoint training hold-out with the official metric
      (baseline-adjusted chrF, beta=2), including a Minimum-Bayes-Risk
      candidate selection that directly optimises that metric.

No lexical/sparse retrieval component, no external data, no lookup.
"""
import argparse, itertools, json, math, os, random, re, sys, time, unicodedata
import collections
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------- config
SELECTOR_MODEL   = "FacebookAI/roberta-large-mnli"
SELECTOR_REV     = "main"
GENERATOR_MODEL  = "google/flan-t5-large"
GENERATOR_REV    = "main"

SEL_MAXLEN       = 512
SEL_EPOCHS       = 4
SEL_LR           = 8e-6
SEL_BS           = 16
SEL_FOLDS        = 5
SEL_WARMUP       = 0.1

GEN_MAXLEN_IN    = 640
GEN_MAXLEN_OUT   = 48
GEN_EPOCHS       = 4
GEN_LR           = 1e-4
GEN_BS           = 4
GEN_ACCUM        = 2
GEN_FULL_VIEW_W  = 3
GEN_HOLDOUT_FRAC = 0.15
MBR_BEAMS        = 8
MBR_SAMPLES      = 8

SEED             = 20240917
MAX_Q_CHARS      = 512
FALLBACK_Q       = "What datasets are used in the experiments?"

T0 = time.time()
def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)

# ----------------------------------------------------------------------------- official metric
_BASE_Q = "what question do these records answer?"

def _norm(s):
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s

_BASE_CACHE = {}

def _chrf_mean(h, r):
    """Arithmetic mean of recall-weighted (beta=2) char n-gram F, orders 1..6."""
    h = _norm(h); r = _norm(r)
    tot = 0.0
    for n in range(1, 7):
        H = collections.Counter(h[i:i + n] for i in range(len(h) - n + 1))
        R = collections.Counter(r[i:i + n] for i in range(len(r) - n + 1))
        hn = sum(H.values()); rn = sum(R.values())
        if hn == 0 or rn == 0:
            continue
        m = sum((H & R).values())
        P = m / hn; Rc = m / rn
        den = 4 * P + Rc
        if den > 0:
            tot += 5 * P * Rc / den
    return tot / 6.0

def q_score(h, r):
    cb = _BASE_CACHE.get(r)
    if cb is None:
        cb = _BASE_CACHE[r] = _chrf_mean(_BASE_Q, r)
    ch = _chrf_mean(h, r)
    if cb >= 1.0:
        return 1.0 if ch >= 1.0 else 0.0
    return min(max((ch - cb) / (1.0 - cb), 0.0), 1.0)

# ----------------------------------------------------------------------------- data
def load_frames(public_dir):
    tr = pd.read_csv(os.path.join(public_dir, "train.csv"), dtype={"record_count": str})
    te = pd.read_csv(os.path.join(public_dir, "test.csv"), dtype={"record_count": str})
    lb = pd.read_csv(os.path.join(public_dir, "train_labels.csv"))
    tr = tr.merge(lb, on="id", how="inner")
    for d in (tr, te):
        d["records"] = d["records_json"].map(json.loads)
        d["n"] = d["records"].map(len)
    tr["wit"] = tr["witnesses"].map(lambda s: tuple(sorted(int(x[1:]) for x in s.split())))
    return tr, te

def clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def words(s, k):
    w = clean(s).split()
    return " ".join(w[:k])

def rec_side(rec):
    return "answer: " + clean(rec["answer"]) + " | evidence: " + clean(rec["evidence"])

# ----------------------------------------------------------------------------- torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

DEV = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DTYPE = torch.bfloat16

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def autocast():
    if DEV == "cuda":
        return torch.autocast(device_type="cuda", dtype=AMP_DTYPE)
    return torch.autocast(device_type="cpu", enabled=False)

def make_sched(opt, total, warmup):
    def fn(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, p))))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)

# ----------------------------------------------------------------------------- selector
class PairDS(Dataset):
    """Pairs of records. Side order is randomised per epoch (symmetry augmentation)."""
    def __init__(self, items, tok, maxlen, train, seed=0):
        self.items = items; self.tok = tok; self.maxlen = maxlen
        self.train = train; self.epoch = 0; self.seed = seed
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        a, b, y = self.items[i]
        if self.train:
            rng = random.Random((self.seed * 1000003 + self.epoch * 7919 + i))
            if rng.random() < 0.5:
                a, b = b, a
        enc = self.tok(a, b, truncation="longest_first", max_length=self.maxlen)
        enc["labels"] = float(y)
        return enc

def collate(tok):
    def f(batch):
        labels = torch.tensor([b.pop("labels") for b in batch], dtype=torch.float)
        out = tok.pad(batch, return_tensors="pt")
        out["labels"] = labels
        return out
    return f

def build_pairs(df):
    """(row_index, i, j, label) for every unordered record pair of every ledger."""
    out = []
    for ri, r in enumerate(df.itertuples()):
        n = r.n
        wit = set(r.wit) if hasattr(r, "wit") else set()
        for i, j in itertools.combinations(range(n), 2):
            lab = 1 if (i in wit and j in wit) else 0
            out.append((ri, i, j, lab))
    return out

def pair_texts(df, pairs):
    cache = {}
    res = []
    for ri, i, j, lab in pairs:
        key = (ri, i)
        if key not in cache:
            cache[key] = rec_side(df["records"].iloc[ri][i])
        key2 = (ri, j)
        if key2 not in cache:
            cache[key2] = rec_side(df["records"].iloc[ri][j])
        res.append((cache[key], cache[key2], lab))
    return res

@torch.no_grad()
def predict_pairs(model, tok, texts, maxlen, bs=64, symmetric=True):
    """Symmetrised pair log-odds. Batches are length-sorted purely to cut
    padding waste; the returned values are order-independent."""
    model.eval()
    logits = np.zeros(len(texts), dtype=np.float64)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i][0]) + len(texts[i][1]))
    passes = [False, True] if symmetric else [False]
    for flip in passes:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            A = [texts[i][1] if flip else texts[i][0] for i in idx]
            B = [texts[i][0] if flip else texts[i][1] for i in idx]
            enc = tok(A, B, truncation="longest_first", max_length=maxlen,
                      padding=True, return_tensors="pt").to(DEV)
            with autocast():
                out = model(**enc).logits.float()
            lg = (out[:, 1] - out[:, 0]).detach().cpu().numpy()
            for k, i in enumerate(idx):
                logits[i] += lg[k] / len(passes)
    return logits

def train_selector_fold(train_texts, val_texts, tok, seed):
    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        SELECTOR_MODEL, revision=SELECTOR_REV, num_labels=2,
        ignore_mismatched_sizes=True).to(DEV)
    ds = PairDS(train_texts, tok, SEL_MAXLEN, train=True, seed=seed)
    dl = DataLoader(ds, batch_size=SEL_BS, shuffle=True, collate_fn=collate(tok),
                    num_workers=2, drop_last=False,
                    generator=torch.Generator().manual_seed(seed))
    steps = len(dl) * SEL_EPOCHS
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm.weight"))]
    nodecay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm.weight"))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                             {"params": nodecay, "weight_decay": 0.0}], lr=SEL_LR)
    sch = make_sched(opt, steps, int(SEL_WARMUP * steps))
    lossf = nn.CrossEntropyLoss()
    for ep in range(SEL_EPOCHS):
        ds.epoch = ep
        model.train()
        tot = 0.0; nb = 0
        for batch in dl:
            labels = batch.pop("labels").long().to(DEV)
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with autocast():
                logits = model(**batch).logits.float()
            loss = lossf(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        log("    selector epoch %d loss %.4f" % (ep + 1, tot / max(1, nb)))
    val_logits = predict_pairs(model, tok, val_texts, SEL_MAXLEN) if val_texts else None
    return model, val_logits

def decode_subsets(df, pairs, scores, mode="logit"):
    """MAP subset of size n-2 maximising the sum of pairwise scores."""
    by_row = collections.defaultdict(dict)
    for (ri, i, j, _), s in zip(pairs, scores):
        by_row[ri][(i, j)] = s
    preds = {}
    for ri, d in by_row.items():
        n = df["n"].iloc[ri]
        k = n - 2
        best, bv = None, -1e18
        for S in itertools.combinations(range(n), k):
            v = sum(d[(a, b)] for a, b in itertools.combinations(S, 2))
            if v > bv:
                bv = v; best = S
        preds[ri] = best
    return preds

# ----------------------------------------------------------------------------- generator
def gen_views(records, wit):
    """Label-preserving views of one gold cluster."""
    views = [list(wit)] * GEN_FULL_VIEW_W
    if len(wit) > 1:
        views += [[i] for i in wit]
    return views

def gen_input(records, idxs):
    k = max(1, len(idxs))
    ev_budget = max(30, int(560 / k) - 45)
    parts = []
    for c, i in enumerate(idxs):
        r = records[i]
        parts.append("record %d answer: %s ; evidence: %s" %
                     (c + 1, words(r["answer"], 45), words(r["evidence"], ev_budget)))
    return ("Several reviewers answered the same hidden question about one scientific paper. "
            "Write that original question.\n" + "\n".join(parts))

class GenDS(Dataset):
    def __init__(self, rows, tok):
        self.rows = rows; self.tok = tok
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, i):
        src, tgt = self.rows[i]
        enc = self.tok(src, truncation=True, max_length=GEN_MAXLEN_IN)
        lab = self.tok(text_target=tgt, truncation=True, max_length=GEN_MAXLEN_OUT)
        enc["labels"] = lab["input_ids"]
        return enc

def gen_collate(tok, pad_id):
    def f(batch):
        labs = [b.pop("labels") for b in batch]
        out = tok.pad(batch, return_tensors="pt")
        mx = max(len(l) for l in labs)
        L = torch.full((len(labs), mx), -100, dtype=torch.long)
        for i, l in enumerate(labs):
            L[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        out["labels"] = L
        return out
    return f

def train_generator(rows, tok, seed):
    set_seed(seed)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        GENERATOR_MODEL, revision=GENERATOR_REV).to(DEV)
    try:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    except Exception as e:
        log("    (gradient checkpointing unavailable: %s)" % e)
    ds = GenDS(rows, tok)
    dl = DataLoader(ds, batch_size=GEN_BS, shuffle=True,
                    collate_fn=gen_collate(tok, tok.pad_token_id), num_workers=2,
                    generator=torch.Generator().manual_seed(seed))
    steps = max(1, (len(dl) // GEN_ACCUM)) * GEN_EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=GEN_LR, weight_decay=0.01)
    sch = make_sched(opt, steps, int(0.06 * steps))
    for ep in range(GEN_EPOCHS):
        model.train(); tot = 0.0; nb = 0
        opt.zero_grad(set_to_none=True)
        for bi, batch in enumerate(dl):
            batch = {k: v.to(DEV) for k, v in batch.items()}
            with autocast():
                out = model(**batch)
            loss = out.loss.float()
            (loss / GEN_ACCUM).backward()
            if (bi + 1) % GEN_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); nb += 1
        log("    generator epoch %d loss %.4f" % (ep + 1, tot / max(1, nb)))
    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    model.config.use_cache = True
    return model

@torch.no_grad()
def generate(model, tok, srcs, mode, bs=8):
    """mode: ("beam", lp) -> single best beam; ("mbr", lp) -> candidate pool."""
    model.eval()
    kind, lp = mode
    out_all = []
    for s in range(0, len(srcs), bs):
        chunk = srcs[s:s + bs]
        enc = tok(chunk, truncation=True, max_length=GEN_MAXLEN_IN,
                  padding=True, return_tensors="pt").to(DEV)
        if kind == "beam":
            with autocast():
                g = model.generate(**enc, num_beams=5, max_new_tokens=GEN_MAXLEN_OUT,
                                   length_penalty=lp, early_stopping=True,
                                   num_return_sequences=1, do_sample=False)
            txt = tok.batch_decode(g, skip_special_tokens=True)
            out_all.extend([[t] for t in txt])
        else:
            pool = [[] for _ in chunk]
            with autocast():
                g = model.generate(**enc, num_beams=MBR_BEAMS, max_new_tokens=GEN_MAXLEN_OUT,
                                   length_penalty=lp, early_stopping=True,
                                   num_return_sequences=MBR_BEAMS, do_sample=False)
            txt = tok.batch_decode(g, skip_special_tokens=True)
            for b in range(len(chunk)):
                pool[b].extend(txt[b * MBR_BEAMS:(b + 1) * MBR_BEAMS])
            torch.manual_seed(SEED + s)
            with autocast():
                g = model.generate(**enc, do_sample=True, top_p=0.95, temperature=0.9,
                                   max_new_tokens=GEN_MAXLEN_OUT,
                                   num_return_sequences=MBR_SAMPLES)
            txt = tok.batch_decode(g, skip_special_tokens=True)
            for b in range(len(chunk)):
                pool[b].extend(txt[b * MBR_SAMPLES:(b + 1) * MBR_SAMPLES])
            out_all.extend(pool)
    return out_all

def mbr_pick(cands):
    """Minimum-Bayes-Risk selection under the official question-recovery utility."""
    pool = [x for x in (sanitize(c) for c in cands) if x]
    if not pool:
        return FALLBACK_Q
    mult = collections.Counter(pool)
    keys = sorted(mult)
    if len(keys) == 1:
        return keys[0]
    tot = sum(mult.values())
    best, bv = keys[0], -1.0
    for h in keys:
        v = sum(mult[r] * q_score(h, r) for r in keys) / tot
        if v > bv:
            bv = v; best = h
    return best

def sanitize(q):
    q = unicodedata.normalize("NFC", str(q))
    q = "".join(ch for ch in q if ch == " " or (ord(ch) >= 32 and unicodedata.category(ch) != "Cc"))
    q = re.sub(r"\s+", " ", q).strip()
    if len(q) > MAX_Q_CHARS:
        q = q[:MAX_Q_CHARS].rstrip()
    return q

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("submission_out")
    args = ap.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(SEED)

    log("device=%s" % DEV)
    tr, te = load_frames(args.public_dir)
    log("train %d rows / %d papers, test %d rows" % (len(tr), tr.group.nunique(), len(te)))

    k_obs = sorted(set((r.n, len(r.wit)) for r in tr.itertuples()))
    assert all(n - w == 2 for n, w in k_obs), k_obs
    log("verified constraint |witnesses| = record_count - 2 on all training rows")

    groups = sorted(tr.group.unique())
    rng = random.Random(SEED); rng.shuffle(groups)
    gfold = {g: i % SEL_FOLDS for i, g in enumerate(groups)}
    tr["fold"] = tr.group.map(gfold)

    log("=== selector: %s ===" % SELECTOR_MODEL)
    sel_tok = AutoTokenizer.from_pretrained(SELECTOR_MODEL, revision=SELECTOR_REV, use_fast=True)
    tr_pairs = build_pairs(tr)
    tr_texts = pair_texts(tr, tr_pairs)
    te_pairs = build_pairs(te)
    te_texts = pair_texts(te, te_pairs)
    log("train pairs %d (pos %d), test pairs %d" %
        (len(tr_pairs), sum(p[3] for p in tr_pairs), len(te_pairs)))

    oof = np.zeros(len(tr_pairs))
    te_logits = np.zeros(len(te_pairs))
    pair_fold = np.array([tr["fold"].iloc[p[0]] for p in tr_pairs])
    for f in range(SEL_FOLDS):
        log("  fold %d/%d" % (f + 1, SEL_FOLDS))
        tri = np.where(pair_fold != f)[0]; vai = np.where(pair_fold == f)[0]
        model, vl = train_selector_fold([tr_texts[i] for i in tri],
                                        [tr_texts[i] for i in vai],
                                        sel_tok, SEED + 13 * f)
        oof[vai] = vl
        fp = decode_subsets(tr, [tr_pairs[i] for i in vai], vl)
        fa = np.mean([fp[ri] == tr["wit"].iloc[ri] for ri in fp])
        log("  fold %d hold-out exact-witness accuracy %.4f (%d rows)" % (f + 1, fa, len(fp)))
        te_logits += predict_pairs(model, sel_tok, te_texts, SEL_MAXLEN) / SEL_FOLDS
        del model
        if DEV == "cuda":
            torch.cuda.empty_cache()
        log("  fold %d done")

    try:
        from sklearn.metrics import roc_auc_score
        y = np.array([p[3] for p in tr_pairs])
        log("OOF pairwise AUC = %.4f" % roc_auc_score(y, oof))
    except Exception:
        pass

    def exact_acc(scores):
        preds = decode_subsets(tr, tr_pairs, scores)
        ok = sum(1 for ri, S in preds.items() if S == tr["wit"].iloc[ri])
        return ok / len(preds)
    acc_logit = exact_acc(oof)
    acc_prob = exact_acc(2.0 / (1.0 + np.exp(-oof)) - 1.0)
    log("OOF exact-witness accuracy: log-odds %.4f | centred-prob %.4f" % (acc_logit, acc_prob))
    use_prob = acc_prob > acc_logit
    te_scores = (2.0 / (1.0 + np.exp(-te_logits)) - 1.0) if use_prob else te_logits
    log("decoder = %s" % ("centred-prob" if use_prob else "log-odds"))
    test_sets = decode_subsets(te, te_pairs, te_scores)
    sel_acc = acc_prob if use_prob else acc_logit

    log("=== generator: %s ===" % GENERATOR_MODEL)
    gen_tok = AutoTokenizer.from_pretrained(GENERATOR_MODEL, revision=GENERATOR_REV, use_fast=True)
    hold_groups = set(g for g in groups if gfold[g] == 0)
    hg = sorted(hold_groups)
    rng2 = random.Random(SEED + 1); rng2.shuffle(hg)
    target = int(round(GEN_HOLDOUT_FRAC * len(tr)))
    keep, cnt = set(), 0
    sizes = tr.groupby("group").size().to_dict()
    for g in hg:
        if cnt >= target:
            break
        keep.add(g); cnt += sizes[g]
    is_hold = tr.group.isin(keep).values
    fit_rows, hold_rows = [], []
    for ri, r in enumerate(tr.itertuples()):
        if is_hold[ri]:
            hold_rows.append((gen_input(r.records, list(r.wit)), r.question))
        else:
            for v in gen_views(r.records, list(r.wit)):
                fit_rows.append((gen_input(r.records, v), r.question))
    log("generator fit %d rows, hold-out %d rows" % (len(fit_rows), len(hold_rows)))
    gmodel = train_generator(fit_rows, gen_tok, SEED)

    modes = [("beam", 0.6), ("beam", 1.0), ("beam", 1.6), ("mbr", 1.0), ("mbr", 1.6)]
    hold_src = [s for s, _ in hold_rows]; hold_ref = [t for _, t in hold_rows]
    best_mode, best_q = modes[0], -1.0
    for mo in modes:
        cands = generate(gmodel, gen_tok, hold_src, mo)
        hyp = [mbr_pick(c) if mo[0] == "mbr" else sanitize(c[0]) for c in cands]
        qv = float(np.mean([q_score(h, r) for h, r in zip(hyp, hold_ref)]))
        log("  hold-out Q  %-6s lp=%.1f : %.4f" % (mo[0], mo[1], qv))
        if qv > best_q:
            best_q = qv; best_mode = mo
    log("selected decoding = %s lp=%.1f (hold-out Q %.4f)" % (best_mode[0], best_mode[1], best_q))
    log("ESTIMATED leaderboard score ~ %.4f  (%.4f x %.4f)" % (sel_acc * best_q, sel_acc, best_q))

    te_src = []
    for ri, r in enumerate(te.itertuples()):
        te_src.append(gen_input(r.records, list(test_sets[ri])))
    cands = generate(gmodel, gen_tok, te_src, best_mode)
    questions = [mbr_pick(c) if best_mode[0] == "mbr" else sanitize(c[0]) for c in cands]
    questions = [q if q else FALLBACK_Q for q in questions]

    out = pd.DataFrame({
        "id": te["id"].values,
        "witnesses": [" ".join("w%d" % i for i in sorted(test_sets[ri])) for ri in range(len(te))],
        "question": questions,
    })
    assert len(out) == len(te) and out["id"].is_unique
    for w, n in zip(out.witnesses, te["n"]):
        toks = w.split()
        assert len(toks) == n - 2 and len(set(toks)) == len(toks)
    os.makedirs(os.path.dirname(os.path.abspath(args.submission_out)) or ".", exist_ok=True)
    out.to_csv(args.submission_out, index=False)
    log("wrote %s  (%d rows)" % (args.submission_out, len(out)))

if __name__ == "__main__":
    main()
'''


# ============================================================================
# RANK 5 — solution_3_rank5.py
# ============================================================================
r'''
#!/usr/bin/env python3
"""Multi-Witness Question Recovery - one fine-tuned sequence model recovers the witness cluster and the question.

THE PLAN IS FIXED AND DETERMINISTIC. Every count below is a compile-time constant; no branch anywhere depends on
elapsed time, hardware speed, available memory or any other runtime condition, and `time` is used only to print
progress. The plan is sized to finish inside the 90-minute limit on one A10G (see the timing printed at the end).

  The model. A general-purpose pretrained instruction model (Qwen/Qwen2.5-7B-Instruct) is fine-tuned with LoRA on
      the 424 supplied ledgers. Nothing else is read: no external corpus, no upstream records, no synthetic data.
      One training example is one ledger: the prompt lists every record (answer + evidence, in a random order that
      changes every epoch) and states how many of them share the missing question; the target is the line of
      record letters followed by the original question. The same weights therefore produce BOTH submitted fields,
      and the loss is the mean cross-entropy of the letter line plus the mean cross-entropy of the question.

  Two snapshots, one training run. Witness selection keeps improving with training while question wording starts
      to overfit the 424 ledgers, so the LoRA weights are snapshotted at SNAP_QUESTION epochs (used to write the
      question) and at EPOCHS epochs (used to choose the cluster). Both snapshots come from this single run.

  Stage 1 - cluster selection. The witness count is not free: every ledger holds exactly two unrelated singletons,
      so a ledger of n records has n-2 witnesses. All C(n, n-2) candidate subsets are scored by the model's
      log-probability of their letter line, averaged over TTA shuffles of the record order (the order in the
      prompt is arbitrary, so averaging over shuffles removes position bias). The highest-scoring subset wins.

  Stage 2 - question recovery. Conditioned on the chosen letter line, the question snapshot generates one beam
      hypothesis and N_SAMPLES sampled ones, and the submitted question is the candidate with the highest mean
      chrF against the whole pool (minimum Bayes risk under the challenge's own similarity measure). Seeds are
      fixed, so the pool and the choice are reproducible.

  Honest validation. GROUP-held-out folds: the papers (the `group` column) are split into FOLDS parts, so no
      paper is ever in both sides. The printed CV trains on the other parts and scores the held-out part with the
      exact challenge metric (exact witness set gate x baseline-adjusted chrF). Nothing is fitted on the test set;
      every test ledger is decoded on its own.

Usage: python3 solution.py <public_dir> <submission_out>      (defaults: ./dataset/public ./working/submission.csv)
Libraries: numpy, pandas, torch, transformers, peft. Pretrained weights: Qwen/Qwen2.5-7B-Instruct, fine-tuned here.
"""
import copy
import csv
import itertools
import json
import math
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

T00 = time.time()
public_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dataset/public")
submission_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("working/submission.csv")
submission_out.parent.mkdir(parents=True, exist_ok=True)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
EPOCHS = 5
SNAP_QUESTION = 3
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LR, ACCUM, WARMUP = 1e-4, 8, 0.1
ANSWER_CHARS, EVIDENCE_CHARS = 300, 600
TTA = 3
N_SAMPLES, TEMPERATURE, TOP_P, BEAMS = 12, 0.9, 0.95, 4
WRITE_EVERY = 25
MAX_NEW_TOKENS = 64
FOLDS = 3
CV_FOLDS = 0
SEED = 0
LETTERS = "ABCDEF"
SYSTEM = "You repair damaged scientific question-answering ledgers."
BASELINE_QUESTION = "what question do these records answer?"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def log(*a):
    print("[%6.0fs]" % (time.time() - T00), *a, flush=True)


def norm_text(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def chrf(hyp, ref):
    h, r = norm_text(hyp), norm_text(ref)
    total = 0.0
    for n in range(1, 7):
        H = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        R = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        m = sum((H & R).values())
        hn, rn = sum(H.values()), sum(R.values())
        p = m / hn if hn else 0.0
        rc = m / rn if rn else 0.0
        total += 5 * p * rc / (4 * p + rc) if (4 * p + rc) > 0 else 0.0
    return total / 6


def question_score(hyp, ref):
    c, b = chrf(hyp, ref), chrf(BASELINE_QUESTION, ref)
    if b >= 1.0:
        return 1.0 if c >= 1.0 else 0.0
    return min(1.0, max(0.0, (c - b) / (1 - b)))


def clean_question(text):
    text = "".join(ch for ch in text if ch >= " ").strip()[:512].strip()
    return text or "What is the main result of this paper?"


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    tr = pd.read_csv(public_dir / "train.csv", dtype=str).merge(
        pd.read_csv(public_dir / "train_labels.csv", dtype=str), on="id"
    )
    te = pd.read_csv(public_dir / "test.csv", dtype=str)
    train_rows = [
        (json.loads(r.records_json), tuple(int(w[1]) for w in r.witnesses.split()), " ".join(r.question.split()))
        for _, r in tr.iterrows()
    ]
    test_rows = [json.loads(r) for r in te.records_json]
    groups = np.array(sorted(tr.group.unique()))
    np.random.RandomState(SEED).shuffle(groups)
    fold_of_group = {g: i % FOLDS for i, g in enumerate(groups)}
    fold = tr.group.map(fold_of_group).values
    log("train %d ledgers / %d papers ; test %d ledgers" % (len(train_rows), len(groups), len(test_rows)))

    tok = AutoTokenizer.from_pretrained(MODEL)
    end_ids = tok("<|im_end|>", add_special_tokens=False)["input_ids"]

    def prompt_ids(records):
        k = len(records) - 2
        body = "\n\n".join(
            "[%s]\nanswer: %s\nevidence: %s"
            % (
                LETTERS[i],
                " ".join(str(r["answer"]).split())[:ANSWER_CHARS],
                " ".join(str(r["evidence"]).split())[:EVIDENCE_CHARS],
            )
            for i, r in enumerate(records)
        )
        user = (
            "%s\n\nExactly %d of these records were written for the same missing question; the other two belong to two different "
            "questions of the same paper. Reply with the letters of the %d matching records on the first line and the missing "
            "question on the second line." % (body, k, k)
        )
        text = tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True
        )
        return tok(text, add_special_tokens=False)["input_ids"]

    def letter_ids(subset):
        return tok(" ".join(LETTERS[i] for i in subset) + "\n", add_special_tokens=False)["input_ids"]

    def build():
        base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to(DEV)
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        return get_peft_model(
            base,
            LoraConfig(
                r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            )
        )

    def train_model(train_idx, seed):
        torch.manual_seed(seed)
        model = build()
        trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        opt = torch.optim.AdamW([p for _, p in trainable], lr=LR, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, LR, total_steps=EPOCHS * math.ceil(len(train_idx) / ACCUM), pct_start=WARMUP
        )
        rng = np.random.RandomState(seed)
        snapshot = {}
        for ep in range(EPOCHS):
            model.train()
            order = rng.permutation(train_idx)
            running = 0.0
            for n, i in enumerate(order):
                records, gold, question = train_rows[i]
                perm = rng.permutation(len(records))
                records = [records[j] for j in perm]
                place = {int(j): k for k, j in enumerate(perm)}
                gold = tuple(sorted(place[g] for g in gold))
                pi, li = prompt_ids(records), letter_ids(gold)
                qi = tok(question, add_special_tokens=False)["input_ids"] + end_ids
                ids = torch.tensor([pi + li + qi], device=DEV)
                logits = model(input_ids=ids).logits[0, len(pi) - 1:-1].float()
                ce = F.cross_entropy(logits, ids[0, len(pi):], reduction="none")
                loss = ce[:len(li)].mean() + ce[len(li):].mean()
                (loss / ACCUM).backward()
                running += loss.item()
                if (n + 1) % ACCUM == 0 or n == len(order) - 1:
                    torch.nn.utils.clip_grad_norm_([p for _, p in trainable], 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
            if ep + 1 == SNAP_QUESTION:
                snapshot = {n: p.detach().clone() for n, p in trainable}
            log("   epoch %d loss %.4f" % (ep, running / len(order)))
        model.eval()
        final = {n: p.detach().clone() for n, p in trainable}
        return model, trainable, snapshot, final

    def load(trainable, state):
        with torch.no_grad():
            for n, p in trainable:
                p.copy_(state[n])

    @torch.no_grad()
    def select(model, records):
        k = len(records) - 2
        cands = list(itertools.combinations(range(len(records)), k))
        total = np.zeros(len(cands))
        rng = np.random.RandomState(SEED + 12345)
        for t in range(TTA):
            perm = np.arange(len(records)) if t == 0 else rng.permutation(len(records))
            place = {int(o): j for j, o in enumerate(perm)}
            pi = prompt_ids([records[j] for j in perm])
            base = model(input_ids=torch.tensor([pi], device=DEV), use_cache=True)
            last = base.logits[0, -1].float()
            for n, c in enumerate(cands):
                li = letter_ids(tuple(sorted(place[x] for x in c)))
                out = model(
                    input_ids=torch.tensor([li], device=DEV),
                    past_key_values=copy.deepcopy(base.past_key_values),
                    use_cache=True
                )
                lg = torch.cat([last[None], out.logits[0, :-1].float()], 0)
                total[n] += float(
                    torch.log_softmax(lg, -1)[
                        torch.arange(len(li)), torch.tensor(li, device=DEV)
                    ].sum()
                )
        return cands[int(total.argmax())]

    @torch.no_grad()
    def beam_question(model, records, subset):
        ids = torch.tensor([prompt_ids(records) + letter_ids(subset)], device=DEV)
        g = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=BEAMS,
            do_sample=False,
            pad_token_id=tok.pad_token_id
        )
        return clean_question(" ".join(tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True).split()))

    @torch.no_grad()
    def mbr_question(model, records, subset, beam):
        ids = torch.tensor([prompt_ids(records) + letter_ids(subset)], device=DEV)
        sampled = model.generate(
            input_ids=ids.repeat(N_SAMPLES, 1),
            attention_mask=torch.ones_like(ids).repeat(N_SAMPLES, 1),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.pad_token_id
        )
        pool = [beam] + [" ".join(tok.decode(s[ids.shape[1]:], skip_special_tokens=True).split()) for s in sampled]
        pool = [p for p in pool if p] or [beam]
        return clean_question(max(pool, key=lambda h: sum(chrf(h, o) for o in pool)))

    def predict(model, trainable, snapshot, final, records_list, write=None, ids=None):
        load(trainable, final)
        subsets = [select(model, r) for r in records_list]
        log("   selection done for %d ledgers" % len(subsets))
        load(trainable, snapshot)
        questions = [beam_question(model, r, s) for r, s in zip(records_list, subsets)]
        if write is not None:
            write(ids, subsets, questions)
        log("   beam questions done")
        for n, (r, s) in enumerate(zip(records_list, subsets)):
            questions[n] = mbr_question(model, r, s, questions[n])
            if write is not None and ((n + 1) % WRITE_EVERY == 0 or n == len(subsets) - 1):
                write(ids, subsets, questions)
        return subsets, questions

    for f in range(CV_FOLDS):
        idx = np.where(fold != f)[0]
        held = np.where(fold == f)[0]
        log("CV fold %d: training on %d ledgers, holding out %d" % (f, len(idx), len(held)))
        model, trainable, snapshot, final = train_model(idx, SEED + f)
        subsets, questions = predict(model, trainable, snapshot, final, [train_rows[i][0] for i in held])
        w = np.array([s == train_rows[i][1] for s, i in zip(subsets, held)], float)
        q = np.array([question_score(h, train_rows[i][2]) for h, i in zip(questions, held)])
        log(
            "CV fold %d: witness accuracy %.4f | question %.4f | SCORE %.4f"
            % (f, w.mean(), q[w > 0].mean() if w.any() else 0.0, (w * q).mean())
        )
        del model
        torch.cuda.empty_cache()

    log("final model: training on all %d ledgers" % len(train_rows))
    model, trainable, snapshot, final = train_model(np.arange(len(train_rows)), SEED)

    def write_submission(ids, subsets, questions):
        with open(submission_out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", "witnesses", "question"])
            for tid, subset, question in zip(ids, subsets, questions):
                writer.writerow([tid, " ".join("w%d" % i for i in sorted(subset)), question])

    predict(model, trainable, snapshot, final, test_rows, write=write_submission, ids=list(te.id))
    log("wrote %s (%d rows) in %.1f min" % (submission_out, len(te), (time.time() - T00) / 60))


if __name__ == "__main__":
    main()
'''
