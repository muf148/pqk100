"""Projected quantum kernel, its diagnostics, and a first-order noise proxy."""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel

from .baselines import fit_rbf_svm


def features(P):
    """(N, 3, n) projections -> (N, 3n) feature matrix."""
    return np.asarray(P).reshape(len(P), -1)


def fit_pqk(Ftr, ytr, seed=0):
    """PQK-SVM. The PQK is an RBF kernel on the measured 1-RDM features (Huang et al. 2021),
    so it is tuned with exactly the same grid as the classical RBF-SVM."""
    return fit_rbf_svm(Ftr, ytr, seed)


def default_gamma(F):
    return 1.0 / (F.shape[1] * F.var())


def pqk_matrix(Fa, Fb=None, gamma=None):
    return rbf_kernel(Fa, Fb, gamma=default_gamma(Fa) if gamma is None else gamma)


def distance_cv(F):
    """std / mean of squared pairwise distances. Scale-free; small means all points look
    equally far apart, which is what a concentrated kernel is."""
    G = F @ F.T
    d2 = np.diag(G)[:, None] + np.diag(G)[None, :] - 2 * G
    off = d2[~np.eye(len(F), dtype=bool)]
    return float(off.std() / off.mean())


def concentration(P):
    """How much the kernel and its inputs flatten out.

    bloch_norm      mean length of the single-qubit Bloch vectors (1 = pure, 0 = fully mixed)
    feature_spread  mean std across samples of each <P_k>; shot noise must sit well below it
    d2_cv           see distance_cv
    k_offdiag_var   variance of off-diagonal kernel entries at gamma = 1 / (3n * var)
    """
    P = np.asarray(P)
    F = features(P)
    K = pqk_matrix(F)
    off = K[~np.eye(len(K), dtype=bool)]
    return {
        "bloch_norm": float(np.sqrt((P ** 2).sum(1)).mean()),
        "feature_spread": float(F.std(0).mean()),
        "d2_cv": distance_cv(F),
        "k_offdiag_var": float(off.var()),
    }


def geometric_difference(Kc, Kq, lam=1e-3):
    """g(K_c || K_q) of Huang et al. (Eq. F19), kernels normalized to trace N.

    g near 1: the classical kernel already covers what the quantum one does.
    g near sqrt(N): room for a prediction gap (necessary, not sufficient).
    """
    N = len(Kc)
    Kc, Kq = Kc * N / np.trace(Kc), Kq * N / np.trace(Kq)
    sq, sc = np.real(sqrtm(Kq)), np.real(sqrtm(Kc))
    inv = np.linalg.inv(Kc + lam * np.eye(N))
    M = sq @ sc @ inv @ inv @ sc @ sq
    return float(np.sqrt(np.linalg.norm(M, 2)))


def alignment(K, y):
    """Kernel-target alignment with centered kernel, labels in {0, 1}."""
    s = 2.0 * np.asarray(y) - 1.0
    H = np.eye(len(K)) - 1.0 / len(K)
    Kc = H @ K @ H; Y = np.outer(s - s.mean(), s - s.mean())
    return float((Kc * Y).sum() / (np.linalg.norm(Kc) * np.linalg.norm(Y)))


def noisy(P, shots, atten=1.0, rng=None):
    """First-order hardware proxy, NOT a device simulation.

    Each expectation value is shrunk by `atten` (scalar or per-qubit array, e.g.
    (1 - eps_2q) ** gates_in_cone) and gets binomial shot noise from `shots` shots per basis.
    Readout error is assumed mitigated without bias.
    """
    rng = np.random.default_rng() if rng is None else rng
    m = np.clip(np.asarray(P) * atten, -1, 1)
    p = (1 + m) / 2
    return 2 * rng.binomial(shots, p) / shots - 1
