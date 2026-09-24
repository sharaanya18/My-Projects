"""Same single-fold diagnostic as train_sparse_diag.py, but the sparse-text
features stay full-dimensional (no SVD) and the model's own input linear
layer does the dimensionality reduction under the task's loss."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from data import build_tag_vocab, load_train
from features_sparse_raw import RawTitleFeaturizer
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


def run(Xtr, idx_tr, pos_tr, Xva, idx_va, ans_va, va_case_ids, pools_va, num_tags, in_dim,
        epochs, emb_dim, dropout, input_dropout, lr, wd, seed=0, batch_size=128, eval_every=3):
    torch.manual_seed(seed)
    model = TagRankerHead(in_dim=in_dim, num_tags=num_tags + 1, emb_dim=emb_dim, dropout=dropout, input_dropout=input_dropout)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = Xtr.shape[0]
    idx_t = torch.tensor(idx_tr)
    pos_t = torch.tensor(pos_tr)
    steps_per_epoch = (n + batch_size - 1) // batch_size
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, steps_per_epoch=steps_per_epoch, epochs=epochs)

    idx_va_t = torch.tensor(idx_va)
    Xva_dense = torch.tensor(Xva.toarray(), dtype=torch.float32)

    best = 0.0
    for ep in range(epochs):
        model.train()
        perm = np.random.permutation(n)
        total_loss = 0.0
        for b in range(0, n, batch_size):
            bi = perm[b : b + batch_size]
            feat = torch.tensor(Xtr[bi].toarray(), dtype=torch.float32)
            logits = model(feat, idx_t[bi])
            loss = multi_positive_softmax_loss(logits, pos_t[bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total_loss += loss.item() * len(bi)
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                logits = model(Xva_dense, idx_va_t).numpy()
            order = np.argsort(-logits, axis=1)
            rankings = {va_case_ids[k]: [pools_va[k][j] for j in order[k]] for k in range(len(va_case_ids))}
            answers_d = {va_case_ids[k]: ans_va[k] for k in range(len(va_case_ids))}
            score = mean_average_precision(rankings, answers_d)
            best = max(best, score)
            print(f"    epoch {ep+1}/{epochs} loss={total_loss/n:.4f} val_MAP={score:.4f} temp={model.log_temp.exp().item():.2f}")
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

    t1 = time.time()
    featurizer = RawTitleFeaturizer(max_word_features=40000, max_char_features=40000).fit(titles_tr)
    Xtr = featurizer.transform(titles_tr)
    Xva = featurizer.transform(titles_va)
    print(f"featurized: dim={featurizer.dim} in {time.time()-t1:.1f}s")

    vocab = build_tag_vocab(pools_tr)
    idx_tr, pos_tr = make_pool_tensors(pools_tr, ans_tr, vocab)
    idx_va, _ = make_pool_tensors(pools_va, None, vocab)

    configs = [
        dict(name="raw-tfidf-linear", epochs=15, emb_dim=256, dropout=0.1, input_dropout=0.3, lr=2e-3, wd=1e-6),
        dict(name="raw-tfidf-linear-bigger-emb", epochs=15, emb_dim=384, dropout=0.1, input_dropout=0.3, lr=2e-3, wd=1e-6),
    ]
    for cfg in configs:
        print(f"== {cfg['name']} ==")
        t1 = time.time()
        best = run(
            Xtr, idx_tr, pos_tr, Xva, idx_va, ans_va, va_case_ids, pools_va,
            num_tags=len(vocab), in_dim=featurizer.dim,
            epochs=cfg["epochs"], emb_dim=cfg["emb_dim"], dropout=cfg["dropout"],
            input_dropout=cfg["input_dropout"], lr=cfg["lr"], wd=cfg["wd"], seed=0,
        )
        print(f"  best val MAP for {cfg['name']}: {best:.4f}  ({time.time()-t1:.1f}s)")


if __name__ == "__main__":
    main()
