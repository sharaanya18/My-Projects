"""Fast single-fold diagnostic loop to tune the sparse branch: per-epoch val
MAP, and an optional co-occurrence-SVD warm start for the tag embeddings
(trainable parameters initialised from a real, train-only association
statistic -- not a scoring rule, see the audit doc)."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import build_tag_vocab, load_train
from features_sparse import SparseTitleFeaturizer
from grouping import cluster_groups, group_kfold_splits
from metric import mean_average_precision
from model import TagRankerHead, multi_positive_softmax_loss

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


def cooccurrence_init(feat_tr, ans_tr, vocab, emb_dim):
    feat_dim = feat_tr.shape[1]
    num_tags = len(vocab)
    assoc = np.zeros((num_tags, feat_dim), dtype=np.float64)
    counts = np.zeros(num_tags)
    for i, ans in enumerate(ans_tr):
        for t in ans:
            j = vocab.get(t)
            if j is not None:
                assoc[j] += feat_tr[i]
                counts[j] += 1
    assoc /= counts[:, None].clip(min=1)
    svd = TruncatedSVD(n_components=min(emb_dim, feat_dim - 1), random_state=0)
    reduced = svd.fit_transform(assoc)
    if reduced.shape[1] < emb_dim:
        pad = np.zeros((num_tags, emb_dim - reduced.shape[1]))
        reduced = np.concatenate([reduced, pad], axis=1)
    norm = np.linalg.norm(reduced, axis=1, keepdims=True)
    reduced = reduced / (norm + 1e-8) * 0.1
    return reduced.astype(np.float32)


def run(feat_tr, idx_tr, pos_tr, feat_va, idx_va, ans_va, va_case_ids, pools_va, num_tags, in_dim,
        epochs, emb_dim, dropout, lr, wd, tag_init=None, seed=0, device="cpu", eval_every=2):
    torch.manual_seed(seed)
    model = TagRankerHead(in_dim=in_dim, num_tags=num_tags + 1, emb_dim=emb_dim, dropout=dropout).to(device)
    if tag_init is not None:
        with torch.no_grad():
            model.tag_emb.weight[:num_tags] = torch.tensor(tag_init)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = feat_tr.shape[0]
    batch_size = 128
    feat_t = torch.tensor(feat_tr, device=device)
    idx_t = torch.tensor(idx_tr, device=device)
    pos_t = torch.tensor(pos_tr, device=device)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    feat_va_t = torch.tensor(feat_va, device=device)
    idx_va_t = torch.tensor(idx_va, device=device)

    best = 0.0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        total_loss = 0.0
        for b in range(0, n, batch_size):
            bi = perm[b : b + batch_size]
            logits = model(feat_t[bi], idx_t[bi])
            loss = multi_positive_softmax_loss(logits, pos_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(bi)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(feat_va_t, idx_va_t).cpu().numpy()
            order = np.argsort(-logits, axis=1)
            rankings = {va_case_ids[k]: [pools_va[k][j] for j in order[k]] for k in range(len(va_case_ids))}
            answers_d = {va_case_ids[k]: ans_va[k] for k in range(len(va_case_ids))}
            score = mean_average_precision(rankings, answers_d)
            best = max(best, score)
            print(f"    epoch {ep+1}/{epochs} loss={total_loss/n:.4f} val_MAP={score:.4f}")
    return best


def main():
    t0 = time.time()
    cases = load_train(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/train_labels.csv")
    groups = cluster_groups(cases.titles, n_clusters=24, seed=0)
    splits = group_kfold_splits(groups, n_splits=5, seed=0)
    tr_idx, va_idx = splits[0]
    print(f"setup in {time.time()-t0:.1f}s; train={len(tr_idx)} val={len(va_idx)}")

    titles_tr = [cases.titles[i] for i in tr_idx]
    titles_va = [cases.titles[i] for i in va_idx]
    pools_tr = [cases.pools[i] for i in tr_idx]
    pools_va = [cases.pools[i] for i in va_idx]
    ans_tr = [cases.answers[i] for i in tr_idx]
    ans_va = [cases.answers[i] for i in va_idx]
    va_case_ids = [cases.case_ids[i] for i in va_idx]

    featurizer = SparseTitleFeaturizer(word_dim=200, char_dim=200, seed=0).fit(titles_tr)
    feat_tr = featurizer.transform(titles_tr)
    feat_va = featurizer.transform(titles_va)
    vocab = build_tag_vocab(pools_tr)
    idx_tr, pos_tr = make_pool_tensors(pools_tr, ans_tr, vocab)
    idx_va, _ = make_pool_tensors(pools_va, None, vocab)

    configs = [
        dict(name="baseline-more-epochs", epochs=24, emb_dim=256, dropout=0.2, lr=1e-3, wd=1e-5, init=False),
        dict(name="cooc-init", epochs=24, emb_dim=256, dropout=0.2, lr=1e-3, wd=1e-5, init=True),
        dict(name="cooc-init-lowdrop-longer", epochs=36, emb_dim=256, dropout=0.1, lr=1e-3, wd=1e-6, init=True),
    ]

    tag_init = cooccurrence_init(feat_tr, ans_tr, vocab, emb_dim=256)

    for cfg in configs:
        print(f"== {cfg['name']} ==")
        t1 = time.time()
        best = run(
            feat_tr, idx_tr, pos_tr, feat_va, idx_va, ans_va, va_case_ids, pools_va,
            num_tags=len(vocab), in_dim=featurizer.dim,
            epochs=cfg["epochs"], emb_dim=cfg["emb_dim"], dropout=cfg["dropout"],
            lr=cfg["lr"], wd=cfg["wd"], tag_init=tag_init if cfg["init"] else None, seed=0,
        )
        print(f"  best val MAP for {cfg['name']}: {best:.4f}  ({time.time()-t1:.1f}s)")


if __name__ == "__main__":
    main()
