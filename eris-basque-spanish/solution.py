"""
Eris challenge: Basque/Spanish code-switched speech continuity (cross-lingual retrieval, MRR).

Run:  python3 solution.py <public_dir> <submission_out>

COMPLIANCE HEADER (maps to the challenge's own Rules section)
- Hardware: one CUDA GPU (A10G class) is required; the script raises if none is present.
  There is no CPU fallback (a missing GPU makes model loading fail) and no branching on hardware
  or on wall-clock time.
- Pretrained weights: one public multilingual encoder loaded from the Hugging Face hub at a pinned
  commit SHA. No self-hosted or previously fine-tuned weights, no GitHub installs, no external APIs.
- Training data: ONLY the rows of train.csv joined to train_labels.csv (Spanish query paragraph ->
  the Basque paragraph of the same speech) plus paragraphs.csv text. No external data, no synthetic
  or generated examples, no pseudo-labels.
- Training: the encoder is fine-tuned end to end inside this script with a contrastive loss whose
  negatives are the other Basque candidates of the query's own gallery (the test-time situation).
- Test data: each test query is ranked from its own Spanish paragraph and its own gallery's
  candidates only. No statistic, vocabulary or model state is fitted on test text.
- Prohibited signals, none used: query_id / paragraph_id / gallery_id strings; row order of any
  file; order of ids inside candidate_paragraph_ids; how often a candidate appears across
  galleries; the fact that each candidate answers at most one query; reallocating candidates
  between queries; outside copies of the proceedings; external translation; private models.
- Every hyperparameter below is a fixed constant chosen offline. Nothing in this file reads a
  clock back into a decision; timing is written to timing_log.txt as evidence only.
- Determinism: fixed seeds, TF32 off, deterministic algorithms on, eager (non-fused) attention.
"""
import os
import sys
import time
import random
from pathlib import Path

# Must be set before CUDA initialises (deterministic cuBLAS).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ----------------------------------------------------------------------------------------------
# Fixed constants (chosen offline; see CHALLENGE_NOTES.md for the experiments behind each value)
# ----------------------------------------------------------------------------------------------
CFG = dict(
    model="intfloat/multilingual-e5-base",
    revision="d128750597153bb5987e10b1c3493a34e5a4502a",
    prefix="query: ",        # e5 convention; applied to both languages (symmetric task)
    max_len=192,
    epochs=4,
    lr=2e-5,
    weight_decay=0.01,
    warmup_frac=0.1,
    queries_per_step=16,     # queries drawn from ONE gallery per step
    temperature=0.05,
    grad_clip=1.0,
    seed=42,
    eval_batch=64,
    threads=4,
    amp=True,                # fp16 autocast + GradScaler (same code path on T4 validation and A10G)
)


def set_determinism(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    # pinned, non-fused attention (also selected via attn_implementation="eager" at load time)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def load_tables(public_dir):
    """Returns paragraph_id -> text, train frame (with answer), test frame."""
    pdir = Path(public_dir)
    para = pd.read_csv(pdir / "paragraphs.csv")
    texts = dict(zip(para["paragraph_id"], para["text"]))
    train = pd.read_csv(pdir / "train.csv").merge(pd.read_csv(pdir / "train_labels.csv"), on="query_id")
    test = pd.read_csv(pdir / "test.csv")
    return texts, train, test


def build_galleries(df, with_answers):
    """gallery_id -> dict(cands=[ids], queries=[(query_id, query_paragraph_id, answer_or_None)]).
    Candidates are sorted so the order of the shuffled input list is never used."""
    out = {}
    for row in df.itertuples(index=False):
        g = out.setdefault(row.gallery_id, dict(cands=sorted(row.candidate_paragraph_ids.split()), queries=[]))
        ans = row.same_speech_paragraph_id if with_answers else None
        g["queries"].append((row.query_id, row.query_paragraph_id, ans))
    for g in out.values():
        g["queries"].sort(key=lambda q: q[0])  # deterministic base order; shuffled with a seeded RNG below
    return out


def load_model(device, cfg=CFG):
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    model = AutoModel.from_pretrained(cfg["model"], revision=cfg["revision"], attn_implementation="eager")
    assert model.config.model_type == "xlm-roberta", model.config.model_type
    torch.manual_seed(cfg["seed"])  # re-seed after load (no new heads here, kept for safety)
    return tok, model.to(device)


def embed(model, tok, texts, device, cfg=CFG):
    """Mean-pooled, L2-normalised embeddings for a list of raw texts (grad flows if enabled)."""
    batch = tok([cfg["prefix"] + t for t in texts], padding=True, truncation=True,
                max_length=cfg["max_len"], return_tensors="pt").to(device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=cfg["amp"]):
        h = model(**batch).last_hidden_state
    m = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize(((h * m).sum(1) / m.sum(1)).float(), dim=-1)


@torch.no_grad()
def embed_many(model, tok, texts, device, cfg=CFG):
    model.eval()
    order = sorted(range(len(texts)), key=lambda i: (len(texts[i]), i))
    out = torch.zeros(len(texts), model.config.hidden_size)
    for s in range(0, len(order), cfg["eval_batch"]):
        idx = order[s:s + cfg["eval_batch"]]
        out[idx] = embed(model, tok, [texts[i] for i in idx], device, cfg).cpu()
    return out


def make_steps(galleries, cfg, epoch_rng):
    """One epoch of (gallery_id, [query tuples]) steps: each step uses queries of a single gallery."""
    steps = []
    for gid in sorted(galleries):
        qs = list(galleries[gid]["queries"])
        epoch_rng.shuffle(qs)
        for s in range(0, len(qs), cfg["queries_per_step"]):
            steps.append((gid, qs[s:s + cfg["queries_per_step"]]))
    epoch_rng.shuffle(steps)
    return steps


def fine_tune(model, tok, galleries, texts, device, cfg=CFG, log=print):
    """Contrastive fine-tuning: softmax over ALL Basque candidates of the query's gallery."""
    rng = random.Random(cfg["seed"])
    total = sum(len(make_steps(galleries, cfg, random.Random(cfg["seed"] + e))) for e in range(cfg["epochs"]))
    no_decay = ("bias", "LayerNorm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg["lr"])
    warm = max(1, int(cfg["warmup_frac"] * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])
    p0 = next(model.parameters()).detach().clone()
    step = 0
    for ep in range(cfg["epochs"]):
        model.train()
        run, n = 0.0, 0
        for gid, qs in make_steps(galleries, cfg, random.Random(cfg["seed"] + ep)):
            cands = galleries[gid]["cands"]
            pos = {c: i for i, c in enumerate(cands)}
            emb = embed(model, tok, [texts[q[1]] for q in qs] + [texts[c] for c in cands], device, cfg)
            qe, ce = emb[:len(qs)], emb[len(qs):]
            target = torch.tensor([pos[q[2]] for q in qs], device=device)
            loss = F.cross_entropy(qe @ ce.T / cfg["temperature"], target)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step == 0:  # live invariant: the backbone really moved after the first update
                assert not torch.equal(p0, next(model.parameters()).detach().cpu().to(p0.device)), "no update"
            run, n, step = run + loss.item(), n + 1, step + 1
        log(f"epoch {ep + 1}/{cfg['epochs']} mean train loss {run / n:.4f}")
    return model


def rank_gallery_queries(model, tok, gal, texts, device, cfg=CFG):
    """For each query of a gallery: its own candidates, ranked by cosine similarity.
    Candidate embeddings are per-paragraph (independent of which query is asked)."""
    cands = gal["cands"]
    ce = embed_many(model, tok, [texts[c] for c in cands], device, cfg)
    qe = embed_many(model, tok, [texts[q[1]] for q in gal["queries"]], device, cfg)
    scores = qe @ ce.T
    ranked = {}
    for i, q in enumerate(gal["queries"]):
        order = sorted(range(len(cands)), key=lambda j: (-scores[i, j].item(), cands[j]))
        ranked[q[0]] = [cands[j] for j in order]
    return ranked


def mrr(ranked, answers):
    return float(np.mean([1.0 / (ranked[q].index(a) + 1) for q, a in answers.items()]))


def main():
    public_dir, sub_out = Path(sys.argv[1]), Path(sys.argv[2])
    sub_out.parent.mkdir(parents=True, exist_ok=True)
    timing = open(sub_out.parent / "timing_log.txt", "w")  # evidence only, never read back

    def log(msg):
        print(msg, flush=True)
        timing.write(f"{time.time():.1f} {msg}\n")
        timing.flush()

    device = torch.device("cuda")  # no fallback: without a GPU, model.to(device) fails loudly
    set_determinism(CFG["seed"], CFG["threads"])

    texts, train, test = load_tables(public_dir)
    tr_gal = build_galleries(train, with_answers=True)
    te_gal = build_galleries(test, with_answers=False)
    log(f"train galleries {len(tr_gal)} queries {len(train)} | test galleries {len(te_gal)} queries {len(test)}")

    tok, model = load_model(device)
    model = fine_tune(model, tok, tr_gal, texts, device, log=log)

    ranked = {}
    for gid in sorted(te_gal):
        ranked.update(rank_gallery_queries(model, tok, te_gal[gid], texts, device))
    log("inference done")

    sub = pd.DataFrame({"query_id": test["query_id"],
                        "ranked_paragraph_ids": [" ".join(ranked[q]) for q in test["query_id"]]})
    # format validation: one row per query, every candidate exactly once, no duplicates
    assert len(sub) == len(test) and sub["query_id"].is_unique
    for r, c in zip(sub["ranked_paragraph_ids"], test["candidate_paragraph_ids"]):
        assert sorted(r.split()) == sorted(c.split())
    sub.to_csv(sub_out, index=False)
    log(f"wrote {sub_out} ({len(sub)} rows)")


if __name__ == "__main__":
    main()
