import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

SEED = 42
np.random.seed(SEED)

PUBLIC_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./dataset/public")
SUBMISSION_OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("./working/submission.csv")

MODE_LABELS = ("S", "A", "B", "C", "D", "E", "F")
HORIZON_STEPS = 10
N_CLASSES = len(MODE_LABELS)
LABEL_TO_IDX = {m: i for i, m in enumerate(MODE_LABELS)}
LABELS = np.array(MODE_LABELS)

HGB_GRID = [
    dict(max_iter=150, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=20),
    dict(max_iter=250, learning_rate=0.03, max_leaf_nodes=15, l2_regularization=2.0, min_samples_leaf=30),
]
ALPHA_GRID = [round(a, 2) for a in np.arange(0.0, 1.01, 0.1)]
N_FOLDS = 5


def f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 1.0


def positions(row, mode):
    return [step for step, value in enumerate(row) if value == mode]


def macro_f1(reference, prediction, tolerant):
    scores = []
    for mode in MODE_LABELS:
        tp = total_true = total_pred = 0
        for target, guess in zip(reference, prediction):
            true_steps = positions(target, mode)
            predicted_steps = positions(guess, mode)
            total_true += len(true_steps)
            total_pred += len(predicted_steps)
            if not tolerant:
                tp += len(set(true_steps) & set(predicted_steps))
            else:
                unused = set(true_steps)
                for step in predicted_steps:
                    candidates = [other for other in unused if abs(other - step) <= 1]
                    if candidates:
                        unused.remove(min(candidates, key=lambda other: (abs(other - step), other)))
                        tp += 1
        scores.append(f1(tp, total_pred - tp, total_true - tp))
    return sum(scores) / len(scores)


def event_step(row):
    for step, value in enumerate(row):
        if value != "S":
            return step
    return HORIZON_STEPS


def onset_utility(reference, prediction):
    utilities = []
    for target, guess in zip(reference, prediction):
        target_step = event_step(target)
        guess_step = event_step(guess)
        if target_step == guess_step:
            utilities.append(1.0)
        elif target_step == HORIZON_STEPS or guess_step == HORIZON_STEPS:
            utilities.append(0.0)
        else:
            utilities.append(max(0.0, 1.0 - abs(target_step - guess_step) / HORIZON_STEPS))
    return sum(utilities) / len(utilities) if utilities else 0.0


def total_score(reference, prediction):
    exact = macro_f1(reference, prediction, False)
    tolerant = macro_f1(reference, prediction, True)
    onset = onset_utility(reference, prediction)
    return 0.30 * exact + 0.50 * tolerant + 0.20 * onset, exact, tolerant, onset


def history_features(history):
    cards = np.array([[ch == "b" for ch in card] for card in history.split("|")], dtype=np.float32)
    raw = cards.ravel()
    mean_all = cards.mean(0)
    mean_last2 = cards[-2:].mean(0)
    delta_last = cards[-1] - cards[-2]
    delta_span = cards[-1] - cards[0]
    toggles = np.abs(np.diff(cards, axis=0)).sum(0)
    recency = np.array([next((k for k in range(4) if cards[3 - k, j]), 4) for j in range(7)],
                       dtype=np.float32)
    per_card_total = cards.sum(1)
    return np.concatenate([raw, mean_all, mean_last2, delta_last, delta_span,
                           toggles, recency, per_card_total])


def build_matrix(histories):
    return np.stack([history_features(h) for h in histories])


def decode(proba, class_weights):
    return ["".join(LABELS[row]) for row in (proba * class_weights).argmax(2)]


def make_model(cfg):
    return HistGradientBoostingClassifier(random_state=SEED, early_stopping=False, **cfg)


def fit_predict(cfg, X_train, Y_train, X_eval):
    proba = np.zeros((len(X_eval), HORIZON_STEPS, N_CLASSES))
    for step in range(HORIZON_STEPS):
        model = make_model(cfg).fit(X_train, Y_train[:, step])
        proba[:, step, model.classes_] = model.predict_proba(X_eval)
    return proba


def main():
    train = pd.read_csv(PUBLIC_DIR / "train.csv")
    test = pd.read_csv(PUBLIC_DIR / "test.csv")
    print(f"train {train.shape}, test {test.shape}", flush=True)

    X_train = build_matrix(train["history"])
    X_test = build_matrix(test["history"])
    Y_train = np.array([[LABEL_TO_IDX[c] for c in s] for s in train["prediction"]])
    reference = list(train["prediction"])
    prior = np.bincount(Y_train.ravel(), minlength=N_CLASSES) / Y_train.size

    groups = train["history"].factorize()[0]
    folds = list(GroupKFold(n_splits=N_FOLDS).split(X_train, groups=groups))

    oof_by_config = []
    for ci, cfg in enumerate(HGB_GRID):
        oof = np.zeros((len(X_train), HORIZON_STEPS, N_CLASSES))
        for tr_idx, va_idx in folds:
            oof[va_idx] = fit_predict(cfg, X_train[tr_idx], Y_train[tr_idx], X_train[va_idx])
        oof_by_config.append(oof)
        plain = total_score(reference, decode(oof, np.ones(N_CLASSES)))
        print(f"config {ci}: grouped-CV plain argmax total={plain[0]:.4f}", flush=True)

    candidates = [(ci, alpha) for ci in range(len(HGB_GRID)) for alpha in ALPHA_GRID]
    reference_arr = np.array(reference)
    nested_pred = np.empty(len(reference), dtype=object)
    for tr_idx, va_idx in folds:
        inner_ref = list(reference_arr[tr_idx])
        ci, alpha = max(candidates, key=lambda c: total_score(
            inner_ref, decode(oof_by_config[c[0]][tr_idx], prior ** -c[1]))[0])
        nested_pred[va_idx] = decode(oof_by_config[ci][va_idx], prior ** -alpha)
    nested = total_score(reference, list(nested_pred))
    print(f"nested grouped-CV estimate: total={nested[0]:.4f} exact={nested[1]:.4f} "
          f"tolerant={nested[2]:.4f} onset={nested[3]:.4f}", flush=True)

    best_ci, best_alpha = max(candidates, key=lambda c: total_score(
        reference, decode(oof_by_config[c[0]], prior ** -c[1]))[0])
    best = total_score(reference, decode(oof_by_config[best_ci], prior ** -best_alpha))
    print(f"selected config {best_ci}, alpha={best_alpha}: grouped-CV total={best[0]:.4f} "
          f"exact={best[1]:.4f} tolerant={best[2]:.4f} onset={best[3]:.4f}", flush=True)

    test_proba = fit_predict(HGB_GRID[best_ci], X_train, Y_train, X_test)
    predictions = decode(test_proba, prior ** -best_alpha)
    submission = pd.DataFrame({"id": test["id"], "prediction": predictions})

    allowed = set(MODE_LABELS)
    assert len(submission) == len(test) == submission["id"].nunique()
    assert set(submission["id"]) == set(test["id"])
    assert all(len(p) == HORIZON_STEPS and set(p) <= allowed for p in submission["prediction"])

    SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_OUT, index=False)
    print(f"wrote {SUBMISSION_OUT} ({len(submission)} rows)", flush=True)


if __name__ == "__main__":
    main()
