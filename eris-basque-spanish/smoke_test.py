"""CPU smoke test of the solution's train/rank/format path on tiny slices (not a performance measurement)."""
import sys, random, tempfile
import torch, pandas as pd
import solution as S

pub = sys.argv[1]
cfg = {**S.CFG, "epochs": 1, "max_len": 48, "queries_per_step": 4, "eval_batch": 16}
texts, train, test = S.load_tables(pub)
g_tr = S.build_galleries(train, True)
g_te = S.build_galleries(test, False)
sub_tr = {g: dict(cands=g_tr[g]["cands"][:12], queries=[q for q in g_tr[g]["queries"] if q[2] in g_tr[g]["cands"][:12]][:6]) for g in sorted(g_tr)[:2]}
dev = torch.device("cpu")
torch.set_num_threads(4)
tok, model = S.load_model(dev, cfg)
model = S.fine_tune(model, tok, sub_tr, texts, dev, cfg)
gid = sorted(g_te)[0]
tiny = dict(cands=g_te[gid]["cands"][:10], queries=g_te[gid]["queries"][:3])
r = S.rank_gallery_queries(model, tok, tiny, texts, dev, cfg)
assert all(sorted(v) == sorted(tiny["cands"]) for v in r.values()) and len(r) == 3
# metric sanity: perfect ranking -> 1, reversed 2-item ranking -> 0.5
assert S.mrr({"a": ["x", "y"]}, {"a": "x"}) == 1.0 and S.mrr({"a": ["y", "x"]}, {"a": "x"}) == 0.5
print("SMOKE OK", {k: v[:2] for k, v in r.items()})
