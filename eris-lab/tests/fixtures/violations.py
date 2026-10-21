# Deliberately non-compliant script used to test the auditor. Do not submit.
import os
import time
import subprocess

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE

subprocess.run("pip install fancy-lib", shell=True)
DEBUG = False
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_WORKERS = os.cpu_count()
MODE = os.environ.get("MODE", "fast")
T0 = time.time()

train = pd.read_csv("./dataset/public/train.csv")
test = pd.read_csv("./dataset/public/test.csv")
answers = pd.read_csv("../private/answers.csv")

try:
    model = torch.hub.load("someone/repo", "model")
except Exception:
    model = RandomForestClassifier()

vec = TfidfVectorizer()
vec.fit(test["text"])
full = pd.concat([train, test])
X_res, y_res = SMOTE().fit_resample(train[["a"]], train["y"])
emb = np.load("cached_embeddings.npy")

for epoch in range(100):
    if time.time() - T0 > 3000:
        break

study.optimize(objective, timeout=600)
pseudo_labels = model.predict(test)
thr = np.quantile(test_pred, 0.9)
for name in set(train.columns):
    pass
m = RandomForestClassifier(n_estimators=100)
m.fit(train[["a"]], train["y"])
