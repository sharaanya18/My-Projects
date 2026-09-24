"""Single-fold smoke test + timing measurement for the fine-tuned encoder
branch. This CPU dev box has no GPU, so this deliberately trains on a
subsample for a few steps: the goal here is (a) prove the fine-tuning loop
is correct end-to-end and (b) measure real per-step wall time to size the
epoch/runtime budget for the actual A10G run in solution.ipynb -- not to
reproduce the final score.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import build_tag_vocab, load_train
from grouping import cluster_groups, group_kfold_splits
from metric import mean_average_precision
from model import TagRankerHead, multi_positive_softmax_loss
from model_encoder import EncoderTitleEmbedder, load_backbone, tokenize_titles

DATA_DIR = "/tmp/claude-0/-home-user-My-Projects/36034f05-2813-539a-97cd-f1d3748dec95/scratchpad/eris_tag_ranking/public"


def make_pool_tensors(pools, answers, vocab):
    oov = len(vocab)
    n = len(pools)
    idx = np.full((n, 80), oov, dtype=np.int64)
    for i, pool in enumerate(pools):
        for j, t in enumerate(pool):
            idx[i, j] = vocab.get(t, oov)
    pos = None
    if answers is not None:
        pos = np.zeros((n, 80), dtype=bool)
        for i, pool in enumerate(pools):
            for j, t in enumerate(pool):
                if t in answers[i]:
                    pos[i, j] = True
    return idx, pos


def main(subsample=2000, epochs=2, batch_size=16, max_length=48):
    t0 = time.time()
    cases = load_train(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/train_labels.csv")
    groups = cluster_groups(cases.titles, n_clusters=24, seed=0)
    splits = group_kfold_splits(groups, n_splits=5, seed=0)
    tr_idx, va_idx = splits[0]
    rng = np.random.default_rng(0)
    if subsample and len(tr_idx) > subsample:
        tr_idx = rng.choice(tr_idx, size=subsample, replace=False)
    va_idx = va_idx[: min(500, len(va_idx))]
    print(f"setup {time.time()-t0:.1f}s; train={len(tr_idx)} (subsampled) val={len(va_idx)}")

    titles_tr = [cases.titles[i] for i in tr_idx]
    titles_va = [cases.titles[i] for i in va_idx]
    pools_tr = [cases.pools[i] for i in tr_idx]
    pools_va = [cases.pools[i] for i in va_idx]
    ans_tr = [cases.answers[i] for i in tr_idx]
    ans_va = [cases.answers[i] for i in va_idx]
    va_case_ids = [cases.case_ids[i] for i in va_idx]

    vocab = build_tag_vocab(pools_tr)
    idx_tr, pos_tr = make_pool_tensors(pools_tr, ans_tr, vocab)
    idx_va, _ = make_pool_tensors(pools_va, None, vocab)

    t0 = time.time()
    name, tok, enc = load_backbone()
    print(f"backbone load: {time.time()-t0:.1f}s")

    device = "cpu"
    embedder = EncoderTitleEmbedder(enc, out_dim=256).to(device)
    head = TagRankerHead(in_dim=256, num_tags=len(vocab) + 1, emb_dim=256, dropout=0.1).to(device)
    # EncoderTitleEmbedder already projects to 256; feed that straight through
    # the head's projection (in_dim=256) so the head still learns a fusion op.
    opt = torch.optim.AdamW(list(embedder.parameters()) + list(head.parameters()), lr=2e-5)

    idx_t = torch.tensor(idx_tr)
    pos_t = torch.tensor(pos_tr)
    n = len(titles_tr)

    step_times = []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for bstart in range(0, n, batch_size):
            bi = perm[bstart : bstart + batch_size]
            batch_titles = [titles_tr[i] for i in bi]
            enc_in = tokenize_titles(tok, batch_titles, max_length=max_length)
            t1 = time.time()
            title_feat = embedder(enc_in["input_ids"], enc_in["attention_mask"])
            logits = head(title_feat, idx_t[bi])
            loss = multi_positive_softmax_loss(logits, pos_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step_times.append(time.time() - t1)
            if len(step_times) % 10 == 0:
                print(f"  step {len(step_times)} loss={loss.item():.4f} avg_step={np.mean(step_times[-10:]):.3f}s")

    print(f"median step time: {np.median(step_times):.3f}s over batch_size={batch_size}")
    per_example = np.median(step_times) / batch_size
    print(f"~{per_example*1000:.1f} ms/example -> full 15003-row epoch would take ~{per_example*15003/60:.1f} min on this CPU")

    embedder.eval()
    head.eval()
    with torch.no_grad():
        enc_in = tokenize_titles(tok, titles_va, max_length=max_length)
        title_feat = embedder(enc_in["input_ids"], enc_in["attention_mask"])
        logits = head(title_feat, torch.tensor(idx_va))
    order = np.argsort(-logits.numpy(), axis=1)
    rankings = {va_case_ids[k]: [pools_va[k][j] for j in order[k]] for k in range(len(va_case_ids))}
    answers_d = {va_case_ids[k]: ans_va[k] for k in range(len(va_case_ids))}
    score = mean_average_precision(rankings, answers_d)
    print(f"smoke-test val MAP (subsampled, {epochs} epoch(s), NOT the final number): {score:.4f}")


if __name__ == "__main__":
    main()
