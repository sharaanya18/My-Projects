# Chosen on the frequent-pair compositional folds during development
# (see README).  Training length is *not* fixed here: it comes from early
# stopping on the in-script holdout.
FOLD_SEED = 0
FINAL_CFG = dict(
    src_vocab=6000, tgt_vocab=1500,
    d_model=192, nhead=4, enc_layers=2, plan_layers=2, dec_layers=2, ff=384,
    dropout=0.25, lr=3e-4, warmup=300, batch_size=48, label_smooth=0.1,
    prev_head_drop=0.3, len_balance_alpha=0.0, bag_weight=0.0,
    epochs=30,           # upper bound for the validation model; early stopping picks the best
    eval_every=5,
    max_members=3,
)
if os.environ.get("ERIS_SMOKE"):      # quick plumbing check only; never used for the real run
    FINAL_CFG.update(epochs=2, eval_every=1, max_members=1)
