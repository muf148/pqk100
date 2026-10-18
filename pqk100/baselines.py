"""Classical baselines and the shared scoring routine."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC

N_JOBS = -1      # set to 1 when running several sweeps side by side
C_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
G_MULT = [0.03, 0.1, 0.3, 1, 3, 10]


def _cv(seed=0):
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def fit_rbf_svm(Xtr, ytr, seed=0):
    """RBF-SVM with C and gamma tuned by 5-fold CV. Gamma grid is relative to sklearn 'scale'."""
    g0 = 1.0 / (Xtr.shape[1] * Xtr.var())
    gs = GridSearchCV(SVC(kernel="rbf", class_weight="balanced"),
                      {"C": C_GRID, "gamma": [g0 * m for m in G_MULT]},
                      cv=_cv(seed), scoring="roc_auc", n_jobs=N_JOBS)
    return gs.fit(Xtr, ytr)


def fit_logreg(Xtr, ytr, seed=0):
    gs = GridSearchCV(LogisticRegression(max_iter=5000, class_weight="balanced"),
                      {"C": [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3]},
                      cv=_cv(seed), scoring="roc_auc", n_jobs=N_JOBS)
    return gs.fit(Xtr, ytr)


def fit_gbm(Xtr, ytr, seed=0):
    gs = GridSearchCV(HistGradientBoostingClassifier(random_state=seed, class_weight="balanced"),
                      {"learning_rate": [0.03, 0.1], "max_depth": [2, 3], "max_iter": [100, 300]},
                      cv=_cv(seed), scoring="roc_auc", n_jobs=N_JOBS)
    return gs.fit(Xtr, ytr)


def score(model, X, y):
    s = model.decision_function(X)
    return {"auc": roc_auc_score(y, s), "bacc": balanced_accuracy_score(y, model.predict(X))}


def evaluate(fit, Xtr, ytr, sets: dict, seed=0):
    """Fit on train, return CV AUC plus AUC / balanced accuracy on each named set."""
    m = fit(Xtr, ytr, seed)
    out = {"cv_auc": m.best_score_, "params": m.best_params_}
    for name, (X, y) in sets.items():
        for k, v in score(m, X, y).items():
            out[f"{name}_{k}"] = v
    return out
