# Experiment (not the submission): frozen-feature linear probes for several
# pretrained backbones under the same site-grouped CV and exact metric, to pick
# which pretraining carries the flood-damage signal before fine-tuning.
import json, os, time, warnings
from collections import Counter
import numpy as np, pandas as pd, torch, timm
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
torch.manual_seed(42); np.random.seed(42)
DATA = _sys.argv[1]; DEV = os.environ.get("SWEEP_DEVICE", "cuda:0")
L = ["support_scour", "debris_obstruction_or_impact", "approach_or_embankment_washout", "structural_displacement_or_collapse"]
tr = pd.read_csv(f"{DATA}/train.csv"); tg = pd.read_csv(f"{DATA}/train_targets.csv")
t = tg.set_index("id").loc[tr.id, "target"].apply(json.loads)
Y = np.stack([t.apply(lambda d: d[l]).values for l in L], 1)
site = tr.id.str.split("-", n=1).str[0].values; n = len(tr)
strat = (Y * [1, 2, 4, 8]).sum(1)

def metric(idx, P):
    c = Counter(site[idx]); w = np.array([1 / c[s] for s in site[idx]])
    return [average_precision_score(Y[idx, k], P[:, k], sample_weight=w) for k in range(4)]

def sw(idx):
    c = Counter(site[idx]); w = np.array([1 / c[s] for s in site[idx]]); return w / w.mean()

CANDIDATES = [  # (timm name, H, W)
    ("convnext_tiny.fb_in22k_ft_in1k", 384, 576),
    ("convnext_base.fb_in22k_ft_in1k", 384, 576),
    ("vit_base_patch16_clip_224.openai", 384, 576),
    ("vit_large_patch14_clip_336.openai", 336, 504),
    ("eva02_base_patch14_448.mim_in22k_ft_in22k_in1k", 448, 672),
    ("vit_base_patch14_dinov2.lvd142m", 378, 560),
    ("tf_efficientnetv2_m.in21k_ft_in1k", 384, 576),
]
cache = {}
for name, H, W in CANDIDATES:
    t0 = time.time()
    try:
        kw = {"img_size": (H, W)} if ("vit" in name or "eva" in name) else {}
        m = timm.create_model(name, pretrained=True, num_classes=0, **kw).to(DEV).eval().half()
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"], device=DEV).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=DEV).view(1, 3, 1, 1)
        feats = []
        with torch.no_grad():
            for i in range(0, n, 16):
                b = [np.asarray(ImageOps.pad(ImageOps.exif_transpose(Image.open(f"{DATA}/{p}")).convert("RGB"), (W, H), method=Image.BICUBIC)) for p in tr.image_path[i:i + 16]]
                x = ((torch.from_numpy(np.stack(b)).to(DEV).permute(0, 3, 1, 2).float() / 255 - mean) / std).half()
                feats.append(((m(x) + m(torch.flip(x, dims=[3]))) / 2).float().cpu().numpy())
        X = np.concatenate(feats); del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name}: FAILED {e}", flush=True); continue
    cache[name] = X
    best = None
    for C in [0.0003, 0.001, 0.003, 0.01]:
        pooled, folds = [], []
        for rep in range(3):
            oof = np.zeros(Y.shape)
            for tri, vai in StratifiedGroupKFold(5, shuffle=True, random_state=42 + rep).split(X, strat, site):
                sc = StandardScaler().fit(X[tri]); a, b2 = sc.transform(X[tri]), sc.transform(X[vai])
                for k in range(4):
                    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(a, Y[tri, k], sample_weight=sw(tri))
                    oof[vai, k] = clf.predict_proba(b2)[:, 1]
                folds.append(np.mean(metric(vai, oof[vai])))
            pooled.append(metric(np.arange(n), oof))
        pooled = np.array(pooled)
        line = f"{name:48s} C={C:<6} pooled {pooled.mean():.4f} | per-label {np.round(pooled.mean(0), 3)} | fold mean {np.mean(folds):.3f} [{np.min(folds):.3f},{np.max(folds):.3f}]"
        print(line, flush=True)
    print(f"  ({time.time() - t0:.0f}s)", flush=True)
