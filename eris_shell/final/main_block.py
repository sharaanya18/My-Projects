# =============================================================================
# End-to-end run: raw CSVs -> vocab/spacing fit -> in-script validation &
# early stopping -> ensemble training -> per-row inference -> checked CSV.
# =============================================================================
import os
import random as _random
import sys as _sys
import time as _time

import numpy as _np
import pandas as _pd

T_START = _time.time()
TRAIN_STOP_SEC = 52 * 60        # watchdog: stop all training by ~52 min (Guidebook 3.5)
HARD_STOP_SEC = 80 * 60         # inference must be done well inside 1.5 h

torch.set_num_threads(max(1, min(8, os.cpu_count() or 4)))
DEVICE = torch.device("cpu")    # the challenge specifies CPU computation


def elapsed():
    return _time.time() - T_START


def log(*a):
    print(f"[{elapsed():7.1f}s]", *a, flush=True)


def find_public_dir():
    for cand in ("./dataset/public", "../dataset/public", "./data", "../data"):
        if os.path.exists(os.path.join(cand, "train.csv")):
            return cand
    raise FileNotFoundError("dataset/public/train.csv not found")


def seed_everything(seed):
    _random.seed(seed)
    _np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(train_rows, cfg, seed):
    """Fit vocabularies on these rows only and build a fresh model."""
    seed_everything(seed)
    ex = build_examples(train_rows)
    src_v, tgt_v, head_v = build_vocabs(train_rows, [e["heads"] for e in ex],
                                        src_max=cfg["src_vocab"], tgt_max=cfg["tgt_vocab"])
    for e in ex:
        prepare(e, src_v, tgt_v, head_v)
    apply_length_weights(ex, cfg["len_balance_alpha"], len(head_v))
    model = ShellSynth(len(src_v), len(tgt_v), len(head_v), cfg).to(DEVICE)
    return model, ex, (src_v, tgt_v, head_v)


def validate(model, vocabs, rows, spacing, dec):
    src_v, tgt_v, head_v = vocabs
    model.eval()
    preds, truths = [], []
    for e in build_examples(rows):
        prepare(e, src_v, tgt_v, head_v, with_labels=False)
        toks, _ = predict_one([model], e, src_v, tgt_v, head_v, DEVICE, **dec)
        preds.append(finalize(toks, spacing))
    truths = [r["output"] for r in rows]
    return score_breakdown(preds, truths)


def main():
    pub = find_public_dir()
    out_dir = "./working"
    os.makedirs(out_dir, exist_ok=True)
    train_df = _pd.read_csv(os.path.join(pub, "train.csv"), dtype=str, keep_default_na=False)
    test_df = _pd.read_csv(os.path.join(pub, "test.csv"), dtype=str, keep_default_na=False)
    rows = train_df[["id", "input", "output"]].to_dict("records")
    test_rows = test_df[["id", "input"]].to_dict("records")
    log(f"train {len(rows)} rows | test {len(test_rows)} rows | data dir {pub}")

    cfg = dict(DEFAULT_CFG)
    cfg.update(FINAL_CFG)
    spacing = SpacingModel().fit([r["output"] for r in rows])

    # ---------------------------------------------------------------------
    # 1) In-script validation model.  A compositional holdout is carved out
    #    of train.csv exactly the way the evaluation split is described:
    #    whole ordered head->head pairs (frequent ones, whose heads keep >=20
    #    training rows) are removed from training.  Model A trains on the rest;
    #    its holdout score drives early stopping (best checkpoint kept) and the
    #    choice of decoding settings.  Nothing here reads test.csv.
    # ---------------------------------------------------------------------
    fold = make_frequent_pair_folds(rows, n_folds=1, pairs_per_fold=6, seed=FOLD_SEED)[0]
    tr_rows = [rows[i] for i in fold["train"]]
    val_rows = [rows[i] for i in fold["val"]]
    log(f"holdout pairs {fold['held_pairs']} -> {len(val_rows)} validation rows")

    budget_a = TRAIN_STOP_SEC * 0.45
    model_a, ex_a, voc_a = build_model(tr_rows, cfg, seed=0)
    best = {"score": -1.0, "state": None, "epoch": -1}
    base_dec = dict(plan_beam_size=4, rz_beam_size=4, min_stages=1)

    def ev_a(m, ep):
        sc, sim, ex_ = validate(m, voc_a, val_rows, spacing, base_dec)
        if sc > best["score"]:
            best.update(score=sc, epoch=ep,
                        state={k: v.detach().clone() for k, v in m.state_dict().items()})
        return {"holdout": round(sc, 4), "sim": round(sim, 4), "exact": round(ex_, 4)}

    train(model_a, ex_a, dict(cfg), deadline=T_START + budget_a,
          log=lambda r: log("A", r), eval_fn=ev_a, eval_every=cfg["eval_every"])
    model_a.load_state_dict(best["state"])
    log(f"model A best holdout {best['score']:.4f} at epoch {best['epoch'] + 1}")

    # decoding choice (min pipeline length, realize length normalisation) is
    # made on the holdout, not fixed by hand
    dec_grid = [dict(plan_beam_size=4, rz_beam_size=4, min_stages=ms) for ms in (1, 2)]
    dec_scores = []
    for d in dec_grid:
        sc = validate(model_a, voc_a, val_rows, spacing, d)[0]
        dec_scores.append(sc)
        log("decode", d, "holdout", round(sc, 4))
    dec = dec_grid[int(_np.argmax(dec_scores))]
    log("chosen decoding", dec)

    # ---------------------------------------------------------------------
    # 2) Full-data models.  Same architecture, all of train.csv, trained for
    #    the epoch count model A's early stopping found, as many seeds as the
    #    watchdog allows.
    # ---------------------------------------------------------------------
    full_epochs = best["epoch"] + 1
    members = []
    seed = 1
    while True:
        remaining = TRAIN_STOP_SEC - elapsed()
        per_model = getattr(main, "_per_model", None)
        if members and (per_model is None or remaining < per_model * 1.05):
            break
        if not members and remaining < 120:
            break
        t0 = elapsed()
        c = dict(cfg)
        c["epochs"] = full_epochs
        c["seed"] = seed
        m, ex_f, voc_f = build_model(rows, c, seed=seed)
        train(m, ex_f, c, deadline=T_START + TRAIN_STOP_SEC, log=lambda r: log(f"F{seed}", r))
        m.eval()
        members.append(m)
        main._per_model = elapsed() - t0
        log(f"full-data model seed {seed} done in {main._per_model:.0f}s")
        seed += 1
        if len(members) >= cfg["max_members"]:
            break
    vocabs = voc_f   # identical for every full-data member (same rows, deterministic)
    src_v, tgt_v, head_v = vocabs
    if not members:
        # the watchdog left no time for a full-data model: fall back to model A
        members, vocabs = [model_a], voc_a
        src_v, tgt_v, head_v = voc_a
    log(f"ensemble of {len(members)} full-data model(s); training stopped at {elapsed():.0f}s")

    # ---------------------------------------------------------------------
    # 3) Inference -- one test row at a time, nothing pooled across rows.
    # ---------------------------------------------------------------------
    preds = []
    for i, r in enumerate(test_rows):
        e = build_examples([r], with_labels=False)[0]
        prepare(e, src_v, tgt_v, head_v, with_labels=False)
        if elapsed() < HARD_STOP_SEC:
            toks, _ = predict_one(members, e, src_v, tgt_v, head_v, DEVICE, **dec)
        else:
            toks, _ = predict_one(members[:1], e, src_v, tgt_v, head_v, DEVICE,
                                  plan_beam_size=1, rz_beam_size=1,
                                  min_stages=dec["min_stages"])
        preds.append(finalize(toks, spacing))
        if (i + 1) % 40 == 0:
            log(f"predicted {i + 1}/{len(test_rows)}")

    sub = _pd.DataFrame({"id": [r["id"] for r in test_rows], "output": preds})

    # ---------------------------------------------------------------------
    # 4) Pre-submission checklist, enforced as assertions (build spec sec. 8)
    # ---------------------------------------------------------------------
    test_ids = [r["id"] for r in test_rows]
    assert list(sub.columns) == ["id", "output"]
    assert len(sub) == len(test_ids) and sub["id"].is_unique
    assert set(sub["id"]) == set(test_ids)
    for o in sub["output"]:
        assert isinstance(o, str) and o.strip() and len(o) <= 4096 and "\x00" not in o
        shlex.split(o)                      # balanced POSIX quoting
    path = os.path.join(out_dir, "submission.csv")
    sub.to_csv(path, index=False)
    back = _pd.read_csv(path, dtype=str, keep_default_na=False)
    assert back.equals(sub), "CSV round-trip changed the submission"
    log(f"wrote {path} ({len(sub)} rows); total runtime {elapsed() / 60:.1f} min")
    for o in sub["output"].head(12):
        log("  sample:", o)


if __name__ == "__main__":
    main()
