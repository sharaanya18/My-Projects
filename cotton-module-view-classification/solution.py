import os
import sys
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
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

GLOBAL_START = time.time()

PUBLIC_DIR = sys.argv[1] if len(sys.argv) > 1 else "./dataset/public"
SUBMISSION_OUT = sys.argv[2] if len(sys.argv) > 2 else "./working/submission.csv"
os.makedirs(os.path.dirname(SUBMISSION_OUT) or ".", exist_ok=True)

print("Loading data...")
train = pd.read_csv(f"{PUBLIC_DIR}/train.csv").merge(
    pd.read_csv(f"{PUBLIC_DIR}/train_targets.csv"), on="id"
)
test = pd.read_csv(f"{PUBLIC_DIR}/test.csv")
print(f"Train: {train.shape}, Test: {test.shape}")
print("Class balance:", train["target"].value_counts().sort_index().to_dict())

N_CLASSES = 3
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
eval_tf = T.Compose([
    T.Resize((IMG_H, IMG_W)),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])


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
        x = self.transform(img)
        y = int(row["target"]) if "target" in row else -1
        return x, y


def build_model():
    m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=N_CLASSES, drop_rate=0.3)
    return m.to(DEVICE)


@torch.no_grad()
def predict_proba(model, loader, tta=True):
    model.eval()
    all_probs = []
    for xb, _ in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        if tta:
            logits_flip = model(torch.flip(xb, dims=[3]))
            probs = (probs + torch.softmax(logits_flip, dim=1)) / 2
        all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def macro_f1_and_costs(y_true, y_pred):
    f1s = []
    for c in range(3):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        f1 = 1.0 if (tp + fp + fn) == 0 else (2 * tp / (2 * tp + fp + fn) if tp else 0.0)
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    tp1 = np.sum((y_true == 1) & (y_pred == 1))
    fn1 = np.sum((y_true == 1) & (y_pred != 1))
    end_recall = float(tp1 / (tp1 + fn1)) if (tp1 + fn1) else 0.0

    tp2 = np.sum((y_true == 2) & (y_pred == 2))
    fp2 = np.sum((y_true != 2) & (y_pred == 2))
    side_prec = float(tp2 / (tp2 + fp2)) if (tp2 + fp2) else 0.0

    score = 0.70 * macro_f1 + 0.15 * end_recall + 0.15 * side_prec
    return score, macro_f1, end_recall, side_prec


def apply_bias_and_argmax(probs, bias):
    return np.argmax(probs * np.array(bias)[None, :], axis=1)


def search_best_bias(oof_probs, y_true):
    best_score, best_bias = -1.0, (1.0, 1.0, 1.0)
    for b1 in np.arange(0.9, 1.81, 0.05):
        for b2 in np.arange(0.85, 1.21, 0.05):
            bias = (1.0, b1, b2)
            preds = apply_bias_and_argmax(oof_probs, bias)
            score, *_ = macro_f1_and_costs(y_true, preds)
            if score > best_score:
                best_score, best_bias = score, bias
    return best_bias, best_score


N_FOLDS = 5
EPOCHS_PER_FOLD = 22
PATIENCE = 5
BATCH_SIZE = 32

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(train), N_CLASSES))
test_probs_sum = np.zeros((len(test), N_CLASSES))
folds_run = 0

test_ds = ModuleImageDataset(test.assign(target=-1), PUBLIC_DIR, eval_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

for fold, (tr_idx, va_idx) in enumerate(skf.split(train, train["target"])):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    tr_ds = ModuleImageDataset(tr_df, PUBLIC_DIR, train_tf)
    va_ds = ModuleImageDataset(va_df, PUBLIC_DIR, eval_tf)

    class_counts = tr_df["target"].value_counts().reindex([0, 1, 2]).values
    sample_weights = tr_df["target"].map(lambda c: 1.0 / class_counts[c]).values
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(tr_df), replacement=True)

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2, drop_last=True)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS_PER_FOLD)
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_score, best_state, no_improve = -1.0, None, 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()

        va_probs = predict_proba(model, va_loader, tta=False)
        va_pred = va_probs.argmax(1)
        score, mf1, erec, sprec = macro_f1_and_costs(va_df["target"].values, va_pred)

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        elapsed = time.time() - GLOBAL_START
        print(f"fold {fold} epoch {epoch+1}/{EPOCHS_PER_FOLD} "
              f"score={score:.4f} macro_f1={mf1:.4f} end_recall={erec:.4f} side_prec={sprec:.4f} "
              f"elapsed={elapsed:.0f}s")

        if no_improve >= PATIENCE:
            print(f"fold {fold}: no improvement for {PATIENCE} epochs, stopping this fold early.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    oof_probs[va_idx] = predict_proba(model, va_loader, tta=True)
    test_probs_sum += predict_proba(model, test_loader, tta=True)
    folds_run += 1
    print(f"fold {fold} done | best_val_score={best_score:.4f} | total elapsed={time.time()-GLOBAL_START:.0f}s")

assert folds_run > 0, "No fold finished training -- time budget too tight for even one fold."
test_probs = test_probs_sum / folds_run

oof_mask = oof_probs.sum(axis=1) > 0
best_bias, best_oof_score = search_best_bias(oof_probs[oof_mask], train["target"].values[oof_mask])
print(f"Best bias (class0, class1, class2) = {best_bias} | OOF score = {best_oof_score:.4f}")

final_preds = apply_bias_and_argmax(test_probs, best_bias)

submission = pd.DataFrame({"id": test["id"], "prediction": final_preds.astype(int)})

assert len(submission) == len(test), f"Row count mismatch: {len(submission)} vs {len(test)}"
assert submission["prediction"].isin([0, 1, 2]).all(), "Predictions out of range"
assert submission["id"].is_unique, "Duplicate ids in submission"

submission.to_csv(SUBMISSION_OUT, index=False)
print(f"\nSubmission written to {SUBMISSION_OUT} | shape={submission.shape}")
print(submission["prediction"].value_counts().sort_index())
print(f"Total runtime: {time.time() - GLOBAL_START:.0f}s")
