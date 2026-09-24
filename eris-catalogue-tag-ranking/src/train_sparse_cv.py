"""Group-CV for the sparse (TF-IDF -> SVD -> learned projection + tag
embeddings) branch. Everything here is fit fold-locally on the fold's own
training split; validation pools/titles are only ever used for scoring.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import build_tag_vocab, load_train
from features_sparse import SparseTitleFeaturizer
from grouping import cluster_groups, group_kfold_splits
from metric import mean_average_precision
from model import TagRankerHead, multi_positive_softmax_loss

DATA_DIR = "/tmp/claude-0/-home-user-My-Projects/36034f05-2813-539a-97cd-f1d3748dec95/scratchpad/eris_tag_ranking/public"


def make_pool_tensors(pools: list[list[str]], answers: list[set[str]] | None, vocab: dict[str, int]):
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
            ans = answers[i]
            for j, t in enumerate(pool):
                if t in ans:
                    pos[i, j] = True
    return idx, pos


def train_one(feat_tr, idx_tr, pos_tr, num_tags, in_dim, epochs=8, batch_size=128, lr=1e-3, seed=0, device="cpu"):
    torch.manual_seed(seed)
    model = TagRankerHead(in_dim=in_dim, num_tags=num_tags + 1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = feat_tr.shape[0]
    feat_t = torch.tensor(feat_tr, device=device)
    idx_t = torch.tensor(idx_tr, device=device)
    pos_t = torch.tensor(pos_tr, device=device)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)
    model.train()
    for ep in range(epochs):
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
        print(f"    epoch {ep+1}/{epochs} loss={total_loss/n:.4f}")
    return model


@torch.no_grad()
def rank_pools(model, feat_val, idx_val, device="cpu"):
    model.eval()
    feat_t = torch.tensor(feat_val, device=device)
    idx_t = torch.tensor(idx_val, device=device)
    logits = model(feat_t, idx_t).cpu().numpy()
    order = np.argsort(-logits, axis=1)
    return order


def main():
    t0 = time.time()
    cases = load_train(f"{DATA_DIR}/train.csv", f"{DATA_DIR}/train_labels.csv")
    print(f"loaded {len(cases.case_ids)} training cases in {time.time()-t0:.1f}s")

    print("clustering titles for proxy group-CV ...")
    t0 = time.time()
    groups = cluster_groups(cases.titles, n_clusters=24, seed=0)
    print(f"  clustered into {len(set(groups))} groups in {time.time()-t0:.1f}s")
    splits = group_kfold_splits(groups, n_splits=5, seed=0)

    fold_scores = []
    for fold, (tr_idx, va_idx) in enumerate(splits[:2]):  # 2 of 5 folds for a quick honest read
        print(f"-- fold {fold} -- train={len(tr_idx)} val={len(va_idx)}")
        titles_tr = [cases.titles[i] for i in tr_idx]
        titles_va = [cases.titles[i] for i in va_idx]
        pools_tr = [cases.pools[i] for i in tr_idx]
        pools_va = [cases.pools[i] for i in va_idx]
        ans_tr = [cases.answers[i] for i in tr_idx]
        ans_va = [cases.answers[i] for i in va_idx]

        featurizer = SparseTitleFeaturizer(word_dim=200, char_dim=200, seed=0).fit(titles_tr)
        feat_tr = featurizer.transform(titles_tr)
        feat_va = featurizer.transform(titles_va)

        vocab = build_tag_vocab(pools_tr)
        idx_tr, pos_tr = make_pool_tensors(pools_tr, ans_tr, vocab)
        idx_va, _ = make_pool_tensors(pools_va, None, vocab)

        cov = np.mean([sum(1 for t in p if t in vocab) / 80 for p in pools_va])
        print(f"    val pool tag coverage by fold-train vocab: {cov:.3f}")

        model = train_one(feat_tr, idx_tr, pos_tr, num_tags=len(vocab), in_dim=featurizer.dim, epochs=8, seed=fold)
        order = rank_pools(model, feat_va, idx_va)

        rankings = {}
        answers_d = {}
        for k, i in enumerate(va_idx):
            cid = cases.case_ids[i]
            ranked = [pools_va[k][j] for j in order[k]]
            rankings[cid] = ranked
            answers_d[cid] = ans_va[k]
        score = mean_average_precision(rankings, answers_d)
        print(f"    fold {fold} MAP = {score:.4f}")
        fold_scores.append(score)

    print(f"mean CV MAP over {len(fold_scores)} folds: {np.mean(fold_scores):.4f}  (scores={fold_scores})")


if __name__ == "__main__":
    main()
