"""Dev only: re-evaluate a saved checkpoint under different decoding settings."""
import sys, json, itertools, torch
import decode as D
from train_dev import build, evaluate
from dataset import read_csv
from detok import SpacingModel
from splits import make_compositional_folds, random_fold

torch.set_num_threads(4)
tag = sys.argv[1]
ck = torch.load(f"ckpt/{tag}.pt", weights_only=False)
cfg = ck["cfg"]
rows = read_csv("data/train.csv")
comp = make_compositional_folds(rows, 6, 180, 0)[0]
rnd = random_fold(rows, 0.08, seed=7)
tr = [rows[i] for i in comp["train"]]
cval = [rows[i] for i in comp["val"]]
tset = set(comp["train"])
rv = [rows[i] for i in rnd["val"] if i in tset][:150]
tr2 = [r for r in tr if r not in rv]
spacing = SpacingModel().fit([r["output"] for r in tr2])
model, ex, sv, tv, hv = build(tr2, cfg, 0)
model.load_state_dict(ck["model"]); model.eval()

orig = D.realize_beam
for ln, beam, ms in itertools.product([0.8, 1.2, 1.6], [4], [1, 2]):
    def rb(*a, **k):
        k["len_norm"] = ln
        return orig(*a, **k)
    D.realize_beam = rb
    import train_dev; train_dev.predict_one = D.predict_one
    r, _, _ = evaluate(model, cval, sv, tv, hv, spacing, torch.device("cpu"), 150,
                       rz_beam_size=beam, min_stages=ms)
    print(json.dumps({"len_norm": ln, "beam": beam, "min_stages": ms,
                      **{k: round(v, 4) for k, v in r.items()}}), flush=True)
