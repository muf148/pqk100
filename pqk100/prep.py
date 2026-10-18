"""Splits, cross-platform scaling, gene selection, and angle encoding.

Leakage rules used throughout:
  * genes are selected on the training samples only (labels of nothing else are seen);
  * scaling never uses labels;
  * 'cohort' scaling centers each gene within each cohort. That uses unlabeled validation
    data, which is standard for cross-platform work but is transductive. Use 'rank' for a
    single-sample alternative and report both.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata, norm
from sklearn.feature_selection import f_classif


@dataclass
class Split:
    Xtr: np.ndarray; ytr: np.ndarray       # training cohort, training subset
    Xte: np.ndarray; yte: np.ndarray       # training cohort, held-out (in-distribution)
    Xex: np.ndarray; yex: np.ndarray       # independent validation cohort
    genes: np.ndarray


def _strat_pick(y, n, rng):
    """Pick n indices with the class balance of y."""
    idx = []
    for c in (0, 1):
        pool = np.flatnonzero(y == c)
        k = int(round(n * len(pool) / len(y)))
        idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def make_split(train, valid, task, n_train=200, n_test=200, n_ext=400, seed=0,
               scaling="cohort", n_prefilter=5000) -> Split:
    """Draw the training subset, the held-out subset, and the validation subset, and scale them.

    Only the `n_prefilter` most variable genes of the training subset are kept (an
    unsupervised filter), which also keeps memory low.
    """
    rng = np.random.default_rng(seed)
    ya, yb = train.labels[task].values, valid.labels[task].values
    A, Bi = np.flatnonzero(~np.isnan(ya)), np.flatnonzero(~np.isnan(yb))
    ya, yb = ya[A].astype(int), yb[Bi].astype(int)
    Ra, Rb = train.X.values, valid.X.values                  # float32, all genes
    genes = train.X.columns.values

    tr = _strat_pick(ya, n_train, rng)
    rest = np.setdiff1d(np.arange(len(ya)), tr)
    te = rest[_strat_pick(ya[rest], min(n_test, len(rest)), rng)]
    ex = _strat_pick(yb, min(n_ext, len(yb)), rng)

    raw_var = Ra[A[tr]].astype(np.float64).var(0)            # training subset only, no labels
    keep = np.argsort(-raw_var)[:n_prefilter]

    if scaling == "cohort":            # gene-wise z-score inside each cohort, no labels
        Xa = Ra[A][:, keep].astype(np.float64)
        Xb = Rb[Bi][:, keep].astype(np.float64)
        Xa = (Xa - Xa.mean(0)) / (Xa.std(0) + 1e-9)
        Xb = (Xb - Xb.mean(0)) / (Xb.std(0) + 1e-9)
    elif scaling == "rank":            # per-sample rank over all genes -> normal scores, then train-fit z-score
        def normal_scores(R, rows, step=256):
            out = np.empty((len(rows), len(keep)))
            for s in range(0, len(rows), step):
                block = R[rows[s:s + step]].astype(np.float64)
                out[s:s + step] = norm.ppf((rankdata(block, axis=1) - 0.5) / block.shape[1])[:, keep]
            return out
        Xa, Xb = normal_scores(Ra, A), normal_scores(Rb, Bi)
        mu, sd = Xa[tr].mean(0), Xa[tr].std(0) + 1e-9
        Xa, Xb = (Xa - mu) / sd, (Xb - mu) / sd
    else:
        raise ValueError(scaling)
    return Split(Xa[tr], ya[tr], Xa[te], ya[te], Xb[ex], yb[ex], genes[keep])


def select_genes(X, y, k, max_corr=0.8):
    """Top-k genes by ANOVA F on the training set, skipping near-duplicates.

    Returns column indices, best first. `max_corr=None` disables the redundancy filter.
    """
    F, _ = f_classif(X, y)
    order = np.argsort(-np.nan_to_num(F))
    if max_corr is None:
        return order[:k]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    chosen = []
    for j in order:
        if chosen:
            c = np.abs(Z[:, chosen].T @ Z[:, j]) / len(Z)
            if c.max() > max_corr:
                continue
        chosen.append(j)
        if len(chosen) == k:
            break
    return np.array(chosen)


def to_angles(Z, bandwidth=1.0, clip=3.0):
    """z-scores -> rotation angles in radians: bandwidth * clip(z)."""
    return bandwidth * np.clip(Z, -clip, clip)
