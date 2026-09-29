"""
validate.py -- local CV harness for solution.py, cotton-module view classifier.

Runs the exact same model/augmentation/imbalance-handling as solution.py but only
reports out-of-fold metrics -- it never writes a submission. Use it to gauge whether
a change (architecture, augmentation, epochs) is actually an improvement before
spending a submission credit on the real script.

Usage:
    python3 validate.py <public_dir> [--folds 5] [--epochs 22] [--quick]

--quick shrinks folds/epochs for a fast sanity pass while iterating; drop it for a
number you'd actually trust before submitting.
"""

import argparse
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import timm
from sklearn.model_selection import StratifiedKFold

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_H, IMG_W = 144, 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RandomNightSim:
    def __init__(self, p=0.15, gamma_range=(1.8, 4.5)):
        self.p = p
        self.gamma_range = gamma_range

    def __call__(self, img):
        if random.random() < self.p:
            img = TF.adjust_gamma(img, random.uniform(*self.gamma_range))
        return img


train_tf = T.Compose([
    T.RandomResizedCrop((IMG_H, IMG_W), scale=(0.75, 1.0), ratio=(1.6, 1.9)),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(8),
    T.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.25, hue=0.15),
    T.RandomGrayscale(p=0.1),
    RandomNightSim(p=0.15),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.5))], p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
eval_tf = T.Compose([T.Resize((IMG_H, IMG_W)), T.ToTensor(), T.Normalize(MEAN, STD)])


class ModuleImageDataset(Dataset):
    def __init__(self, df, root, transform):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(f"{self.root}/{row['image_path']}").convert("RGB")
        return self.transform(img), int(row["target"])


def macro_f1_and_costs(y_true, y_pred):
    f1s = []
    for c in range(3):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1 = 1.0 if (tp + fp + fn) == 0 else (2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))
    tp1 = np.sum((y_true == 1) & (y_pred == 1)); fn1 = np.sum((y_true == 1) & (y_pred != 1))
    end_recall = float(tp1 / (tp1 + fn1)) if (tp1 + fn1) else 0.0
    tp2 = np.sum((y_true == 2) & (y_pred == 2)); fp2 = np.sum((y_true != 2) & (y_pred == 2))
    side_prec = float(tp2 / (tp2 + fp2)) if (tp2 + fp2) else 0.0
    score = 0.70 * macro_f1 + 0.15 * end_recall + 0.15 * side_prec
    return score, macro_f1, end_recall, side_prec


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    out = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        probs = torch.softmax(model(xb), dim=1)
        probs_flip = torch.softmax(model(torch.flip(xb, dims=[3])), dim=1)
        out.append(((probs + probs_flip) / 2).cpu().numpy())
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("public_dir")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.folds, args.epochs = 2, 4

    train = pd.read_csv(f"{args.public_dir}/train.csv").merge(
        pd.read_csv(f"{args.public_dir}/train_targets.csv"), on="id"
    )
    print(f"train={train.shape} folds={args.folds} epochs={args.epochs} device={DEVICE}")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    oof_probs = np.zeros((len(train), 3))
    t0 = time.time()

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train["target"])):
        tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
        tr_ds = ModuleImageDataset(tr_df, args.public_dir, train_tf)
        va_ds = ModuleImageDataset(va_df, args.public_dir, eval_tf)

        counts = tr_df["target"].value_counts().reindex([0, 1, 2]).values
        weights = tr_df["target"].map(lambda c: 1.0 / counts[c]).values
        sampler = WeightedRandomSampler(weights, num_samples=len(tr_df), replacement=True)
        tr_loader = DataLoader(tr_ds, batch_size=32, sampler=sampler, num_workers=2, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=2)

        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=3, drop_rate=0.3).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

        best_score, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
            sched.step()
            va_probs = predict_proba(model, va_loader)
            score, mf1, erec, sprec = macro_f1_and_costs(va_df["target"].values, va_probs.argmax(1))
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"fold {fold} epoch {epoch+1}/{args.epochs} score={score:.4f} "
                  f"macro_f1={mf1:.4f} end_recall={erec:.4f} side_prec={sprec:.4f} "
                  f"elapsed={time.time()-t0:.0f}s")

        model.load_state_dict(best_state)
        oof_probs[va_idx] = predict_proba(model, va_loader)
        print(f"fold {fold} best_score={best_score:.4f}")

    score, mf1, erec, sprec = macro_f1_and_costs(train["target"].values, oof_probs.argmax(1))
    print(f"\n=== Overall OOF (no bias tuning) ===")
    print(f"score={score:.4f} macro_f1={mf1:.4f} end_recall={erec:.4f} side_prec={sprec:.4f}")
    print(f"total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
