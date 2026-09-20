import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from collections import Counter
import ledger as L

DATA = "/tmp/claude-0/-home-user-My-Projects/53ada2a2-dae3-5edd-96c4-b4c2eeaa8655/scratchpad/work"

CFG = dict(
    ans_cap=48, evi_cap=72, max_tgt=26, min_tgt=5,
    d_model=128, heads=4, d_ff=384, enc_layers=2, rec_layers=2, dec_layers=2, xrec_layers=2,
    dropout=0.15, attn_dropout=0.1, label_smooth=0.1,
    lr=2.5e-3, weight_decay=0.01, clip=1.0, warmup=0.08, lr_floor=0.05,
    batch=8, epochs=24, lambda_pair=1.0, lambda_un=0.5, sub_mix=0.7, rvec_use_evi=1, witness_cond=True,
    beam=16, n_cand=28, gen_temp=1.0, len_norm=0.8, mbr_tau=3.0, no_repeat=3, dbs_groups=4, dbs_lambda=0.5,
    sel_w_un=0.3, sel_w_out=0.3,
    mlm_epochs=8, mlm_lr=1.5e-3, mlm_batch=24, mlm_rate=0.15, pretrain=1,
    src_min_freq=2, src_cap=16000, tgt_min_freq=1, tgt_cap=6000,
)

def load():
    tr = pd.read_csv(f"{DATA}/train.csv"); lb = pd.read_csv(f"{DATA}/train_labels.csv")
    te = pd.read_csv(f"{DATA}/test.csv")
    labels = {str(r.id): {"witnesses": r.witnesses, "question": r.question}
              for r in lb.itertuples(index=False)}
    return L.parse_rows(tr), labels, L.parse_rows(te), te

def build_vocabs(rows, labels, cfg):
    sc, tc = Counter(), Counter()
    for row in rows:
        for a in row["ans"]: sc.update(a)
        for e in row["evi"]: sc.update(e)
        tc.update(L.words(labels[row["id"]]["question"]))
    return L.Vocab(sc, cfg["src_min_freq"], cfg["src_cap"]), L.Vocab(tc, cfg["tgt_min_freq"], cfg["tgt_cap"])

def folds_by_group(rows, k, seed=0):
    groups = sorted({r["group"] for r in rows})
    perm = np.random.default_rng(seed).permutation(len(groups))
    return {groups[g]: i % k for i, g in enumerate(perm)}
