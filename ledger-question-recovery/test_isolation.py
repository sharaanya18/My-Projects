"""
Row-independence ("no sibling behaviour"), unseen-data robustness and determinism
checks for solution.py.

    python3 test_isolation.py <public_dir>

Uses an untrained (randomly initialised) network: these are properties of the
data path and must hold regardless of the learned weights.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import solution as S


def main(public_dir):
    public_dir = Path(public_dir)
    train = pd.read_csv(public_dir / "train.csv")
    labels_df = pd.read_csv(public_dir / "train_labels.csv")
    test = pd.read_csv(public_dir / "test.csv")
    labels = {str(r.id): {"witnesses": r.witnesses, "question": r.question}
              for r in labels_df.itertuples(index=False)}
    cfg = S.sub_cfg(S.CFG, "qst")
    cfg.update(beam=8, dbs_groups=2, n_cand=6, max_tgt=12)
    svoc, tvoc = S.build_vocabs(S.parse_rows(train), labels, cfg)
    model = S.Model(cfg, len(svoc), len(tvoc), 123)
    model.training = False
    exs = S.build_examples(S.parse_rows(test), cfg, svoc, tvoc)   # no labels at all
    fails = 0

    print("=== 1. batch-composition invariance (no sibling / cross-row leakage) ===")
    for R in (4, 5, 6):
        pool = [e for e in exs if e["R"] == R]
        if len(pool) < 4:
            continue
        t = pool[0]
        base_w = S.select_for(model, [t], cfg)
        base_q = S.mbr_select(S.generate_for(model, [t], cfg, tvoc, base_w)[0], cfg)
        variants = {
            "with 3 partners":     [t] + pool[1:4],
            "with 3 others":       [t] + (pool[4:7] if len(pool) >= 7 else pool[1:4]),
            "last in batch of 4":  pool[1:4] + [t],
        }
        for name, batch in variants.items():
            w = S.select_for(model, batch, cfg)
            q = [S.mbr_select(c, cfg) for c in S.generate_for(model, batch, cfg, tvoc, w)]
            p = batch.index(t)
            ok_w, ok_q = w[p] == base_w[0], q[p] == base_q
            print(f"  R={R} {name:20s} witnesses={'OK' if ok_w else 'DIFFER'} "
                  f"question={'OK' if ok_q else 'DIFFER'}")
            fails += (not ok_w) + (not ok_q)

    print("\n=== 2. logit invariance when batch padding changes ===")
    for R in (4, 5):
        pool = [e for e in exs if e["R"] == R]
        if len(pool) < 3:
            continue
        t = pool[0]
        longest = max(pool[1:], key=lambda e: max(len(r) for r in e["recs"]))
        ref, reflr = None, None
        for label, batch in (("alone", [t]), ("padded", [t, longest])):
            bt = S.make_batch(batch, with_target=False)
            model.start()
            tok, ctx = model.encode(bt)
            pair, _ = model.witness_scores(ctx, bt)
            v = pair.d.reshape(len(batch), len(bt["pairs"]))[0]
            if label == "alone":
                ref, reflr = v.copy(), bt["Lr"]
            else:
                d = float(np.max(np.abs(v - ref)))
                print(f"  R={R}: Lr {reflr} -> {bt['Lr']}, max |delta| = {d:.2e} "
                      f"{'OK' if d < 1e-4 else 'LEAK'}")
                fails += (d >= 1e-4)

    print("\n=== 3. unseen-data robustness ===")
    cases = [
        ("empty answer", "", "some evidence text"),
        ("empty evidence", "an answer", ""),
        ("both empty", "", ""),
        ("punctuation only", "!!! ??? ...", "***"),
        ("unseen tokens", "zzqq xylophonic", "wubbaflorp zzzznot"),
        ("cjk / unicode", "漢字 テスト", "éàü — “q”"),
        ("4000-word evidence", "ans", " ".join("word%d" % i for i in range(4000))),
        ("numeric only", "12.5 %", "3.4 5.6"),
    ]
    for nm, ans, evi in cases:
        row = {"id": "x", "group": "g", "R": 4,
               "ans": [S.words(ans)] + [S.words("normal answer")] * 3,
               "evi": [S.words(evi)] + [S.words("normal evidence passage")] * 3}
        try:
            ex = S.build_examples([row], cfg, svoc, tvoc)
            w = S.select_for(model, ex, cfg)
            q = S.format_question(S.mbr_select(S.generate_for(model, ex, cfg, tvoc, w)[0], cfg))
            ok = (len(w[0]) == 2 and q and q == q.strip() and len(q) <= 512
                  and not any(ord(c) < 32 for c in q))
            print(f"  {nm:20s} witnesses={' '.join('w%d' % i for i in w[0])} "
                  f"{'OK' if ok else 'BAD'}")
            fails += (not ok)
        except Exception as exc:
            print(f"  {nm:20s} EXCEPTION {type(exc).__name__}: {exc}")
            fails += 1

    print("\n=== 4. determinism ===")
    b = [e for e in exs if e["R"] == 4][:3]
    w1 = S.select_for(model, b, cfg)
    q1 = [S.mbr_select(c, cfg) for c in S.generate_for(model, b, cfg, tvoc, w1)]
    w2 = S.select_for(model, b, cfg)
    q2 = [S.mbr_select(c, cfg) for c in S.generate_for(model, b, cfg, tvoc, w2)]
    same = w1 == w2 and q1 == q2
    print(f"  repeated run identical: {'OK' if same else 'DIFFER'}")
    fails += (not same)

    print("\n" + ("ALL CHECKS PASSED" if fails == 0 else f"{fails} CHECK(S) FAILED"))
    return fails


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "public"))
