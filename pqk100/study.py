"""The numerical experiments, as functions the notebooks call.

Every function returns a DataFrame and, if `out` is given, writes it to CSV. Nothing here
touches quantum hardware; the lattice patch only fixes which qubits and couplers the
feature map uses.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel

from . import baselines as B
from .backends import local_backend, median_cz_error
from .data import load_cohorts
from .featuremap import FEATURE_MAPS, cone_stats, strip_entanglers
from .kernel import alignment, concentration, distance_cv, features, fit_pqk, geometric_difference, noisy
from .lattice import patch
from .lightcone import projections
from .prep import make_split, select_genes, to_angles

SIZES = [20, 40, 60, 80, 100, 127, 156]


def _quiet(n_threads=1):
    warnings.filterwarnings("ignore")
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(n_threads)
    except Exception:
        return None


def _parallel(fn, items, n_jobs):
    """Run fn over items, one process per item when n_jobs != 1."""
    if n_jobs == 1 or len(items) == 1:
        return [fn(x) for x in items]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, verbose=5)(delayed(fn)(x) for x in items)


def _drop(r):
    return {k: v for k, v in r.items() if k != "params"}


# ------------------------------------------------------------------ task screen
def task_screen(tasks=("ER", "lumAB", "TP53"), scalings=("cohort", "rank"), seeds=(0, 1, 2), ks=(1, 20, 156),
                n_train=200, n_test=200, n_ext=400, data_dir="data", cache_dir="cache", out=None):
    """Classical models only: is the task hard, multi-gene, and shifted between cohorts?"""
    _quiet(); B.N_JOBS = -1
    tcga, meta = load_cohorts(data_dir, cache_dir)
    rows = []
    for task, scaling, seed in itertools.product(tasks, scalings, seeds):
        sp = make_split(tcga, meta, task, n_train, n_test, n_ext, seed, scaling)
        for k in ks:
            idx = select_genes(sp.Xtr, sp.ytr, k)
            sets = {"in": (sp.Xte[:, idx], sp.yte), "ext": (sp.Xex[:, idx], sp.yex)}
            for name, fit in (("rbf_svm", B.fit_rbf_svm), ("logreg", B.fit_logreg), ("gbm", B.fit_gbm)):
                if k == 1 and name != "logreg":
                    continue
                r = B.evaluate(fit, sp.Xtr[:, idx], sp.ytr, sets, seed)
                rows.append(dict(task=task, scaling=scaling, seed=seed, k=k, model=name,
                                 top_gene=sp.genes[idx[0]], **_drop(r)))
    df = pd.DataFrame(rows)
    if out:
        df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------------ scaling sweep
def _sweep_seed(a):
    (seed, task, scaling, sizes, reps_list, bandwidths, fmap, n_train, n_test, n_ext, patches, data_dir, cache_dir, inner) = a
    lim = _quiet(); B.N_JOBS = inner
    tcga, meta = load_cohorts(data_dir, cache_dir)
    sp = make_split(tcga, meta, task, n_train, n_test, n_ext, seed, scaling); del tcga, meta
    ntr, nte = len(sp.ytr), len(sp.yte)
    order = select_genes(sp.Xtr, sp.ytr, max(sizes))             # nested panels: top-n prefix
    base, rows = dict(task=task, seed=seed, scaling=scaling), []
    r = B.evaluate(B.fit_logreg, sp.Xtr, sp.ytr, {"in": (sp.Xte, sp.yte), "ext": (sp.Xex, sp.yex)}, seed)
    rows.append(dict(**base, n=sp.Xtr.shape[1], model="logreg_all_genes", **_drop(r)))
    for n in sizes:
        idx = order[:n]
        Ztr, Zte, Zex = sp.Xtr[:, idx], sp.Xte[:, idx], sp.Xex[:, idx]
        sets = {"in": (Zte, sp.yte), "ext": (Zex, sp.yex)}
        cl = {}
        for name, fit in (("rbf_svm", B.fit_rbf_svm), ("logreg", B.fit_logreg), ("gbm", B.fit_gbm)):
            cl[name] = B.evaluate(fit, Ztr, sp.ytr, sets, seed)
            rows.append(dict(**base, n=n, model=name, d2_cv=distance_cv(Ztr), **_drop(cl[name])))
        Kc = rbf_kernel(Ztr, gamma=cl["rbf_svm"]["params"]["gamma"]); lam = 1.0 / cl["rbf_svm"]["params"]["C"]
        phys, edges = patches[n]
        for reps, bw in itertools.product(reps_list, bandwidths):
            ops = FEATURE_MAPS[fmap](n, edges, reps=reps)
            size, n2q, ncz = cone_stats(n, ops)
            P, _ = projections(n, ops, to_angles(np.vstack([Ztr, Zte, Zex]), bw), max_cone=26)
            F = features(P)
            Ftr, Fte, Fex = F[:ntr], F[ntr:ntr + nte], F[ntr + nte:]
            r = B.evaluate(fit_pqk, Ftr, sp.ytr, {"in": (Fte, sp.yte), "ext": (Fex, sp.yex)}, seed)
            Kq = rbf_kernel(Ftr, gamma=r["params"]["gamma"])
            rows.append(dict(**base, n=n, model="pqk", fmap=fmap, reps=reps, bandwidth=bw,
                             cone_max=int(size.max()), cone_mean=float(size.mean()), cz_in_cone=float(ncz.mean()),
                             **concentration(P[:ntr]), geo_diff=geometric_difference(Kc, Kq, lam),
                             align_q=alignment(Kq, sp.ytr), align_c=alignment(Kc, sp.ytr), **_drop(r)))
    return rows


def scaling_sweep(seeds=(0, 1, 2, 3, 4), sizes=SIZES, reps=(1, 2, 3), bandwidths=(0.25, 0.5, 1.0), fmap="zz",
                  task="TP53", scaling="cohort", n_train=200, n_test=200, n_ext=400, backend=None,
                  data_dir="data", cache_dir="cache", out=None, n_jobs=1):
    """Noiseless PQK against classical models from 20 to 156 qubits, by exact simulation.

    Train on a TCGA-BRCA subset; score on held-out TCGA and on METABRIC. One gene per qubit;
    every model sees the same n genes. With n_jobs > 1 each seed runs in its own process.
    """
    backend = backend or local_backend()
    patches = {n: patch(backend, n) for n in sizes}
    inner = 1 if (n_jobs != 1 and len(seeds) > 1) else -1
    args = [(s, task, scaling, list(sizes), list(reps), list(bandwidths), fmap, n_train, n_test, n_ext,
             patches, data_dir, cache_dir, inner) for s in seeds]
    df = pd.DataFrame([r for rows in _parallel(_sweep_seed, args, n_jobs) for r in rows])
    df["lattice"] = backend.name
    if out:
        df.to_csv(out, index=False)
    return df


def selected(df):
    """Per seed and size, the PQK whose (reps, bandwidth) has the best CV AUC on the training set."""
    q = df[df.model == "pqk"]
    return q.loc[q.groupby(["seed", "n"]).cv_auc.idxmax()]


# ------------------------------------------------------------------ entanglement ablation
def _ablation_seed(a):
    seed, task, scaling, sizes, reps_list, bandwidths, fmap, patches, data_dir, cache_dir, inner = a
    lim = _quiet(); B.N_JOBS = inner
    tcga, meta = load_cohorts(data_dir, cache_dir)
    sp = make_split(tcga, meta, task, 200, 200, 400, seed, scaling); del tcga, meta
    order = select_genes(sp.Xtr, sp.ytr, max(sizes)); ntr, nte = len(sp.ytr), len(sp.yte)
    rows = []
    for n in sizes:
        Z = np.vstack([sp.Xtr[:, order[:n]], sp.Xte[:, order[:n]], sp.Xex[:, order[:n]]])
        sets = lambda M: {"in": (M[ntr:ntr + nte], sp.yte), "ext": (M[ntr + nte:], sp.yex)}
        for name, fit, M in (("RBF-SVM", B.fit_rbf_svm, Z), ("RBF-SVM, clipped inputs", B.fit_rbf_svm, np.clip(Z, -3, 3)),
                             ("Logistic regression", B.fit_logreg, Z)):
            rows.append(dict(seed=seed, n=n, scaling=scaling, model=name, **_drop(B.evaluate(fit, M[:ntr], sp.ytr, sets(M), seed))))
        phys, edges = patches[n]
        for label, strip in (("PQK", False), ("PQK, entanglers removed", True)):
            best = None
            for r_, bw in itertools.product([1] if strip else reps_list, bandwidths):
                ops = FEATURE_MAPS[fmap](n, edges, reps=r_)
                F = features(projections(n, strip_entanglers(ops) if strip else ops, to_angles(Z, bw))[0])
                F = F[:, F.std(0) > 1e-12]
                r = B.evaluate(fit_pqk, F[:ntr], sp.ytr, sets(F), seed)
                if best is None or r["cv_auc"] > best["cv_auc"]:
                    best = r
            rows.append(dict(seed=seed, n=n, scaling=scaling, model=label, **_drop(best)))
    return rows


def ablation(scaling="cohort", seeds=(0, 1, 2, 3, 4), sizes=(20, 100, 156), reps=(1, 2), bandwidths=(0.25, 0.5),
             fmap="zz", task="TP53", backend=None, data_dir="data", cache_dir="cache", out=None, n_jobs=1):
    """PQK with and without its two-qubit gates, against classical models with and without
    input clipping. A gain that survives stripping the entanglers is not a quantum effect."""
    backend = backend or local_backend()
    patches = {n: patch(backend, n) for n in sizes}
    inner = 1 if (n_jobs != 1 and len(seeds) > 1) else -1
    args = [(s, task, scaling, list(sizes), list(reps), list(bandwidths), fmap, patches, data_dir, cache_dir, inner) for s in seeds]
    df = pd.DataFrame([r for rows in _parallel(_ablation_seed, args, n_jobs) for r in rows])
    if out:
        df.to_csv(out, index=False)
    return df


# ------------------------------------------------------------------ noise proxy
def _proxy_seed(a):
    seed, grp, task, scaling, shots_list, eps, fmap, patches, data_dir, cache_dir, inner = a
    lim = _quiet(); B.N_JOBS = inner
    tcga, meta = load_cohorts(data_dir, cache_dir)
    sp = make_split(tcga, meta, task, 200, 200, 400, seed, scaling); del tcga, meta
    order = select_genes(sp.Xtr, sp.ytr, max(n for n, _, _ in grp)); ntr, nte = len(sp.ytr), len(sp.yte)
    rows = []
    for n, reps, bw in grp:
        Z = np.vstack([sp.Xtr[:, order[:n]], sp.Xte[:, order[:n]], sp.Xex[:, order[:n]]])
        phys, edges = patches[n]; ops = FEATURE_MAPS[fmap](n, edges, reps=reps)
        size, n2q, ncz = cone_stats(n, ops)
        P, _ = projections(n, ops, to_angles(Z, bw), max_cone=26)
        att = (1 - eps) ** ncz
        for shots in [None] + list(shots_list):
            Pv = P if shots is None else noisy(P, shots, att, np.random.default_rng(seed))
            F = features(Pv)
            r = B.evaluate(fit_pqk, F[:ntr], sp.ytr, {"in": (F[ntr:ntr + nte], sp.yte), "ext": (F[ntr + nte:], sp.yex)}, seed)
            rows.append(dict(seed=seed, n=n, reps=reps, bandwidth=bw, shots="exact" if shots is None else shots,
                             retained=float(att.mean()), spread=float(features(P[:ntr]).std(0).mean()),
                             in_auc=r["in_auc"], ext_auc=r["ext_auc"]))
    return rows


def noise_proxy(sweep, shots=(4096, 1024, 256), eps=None, fmap="zz", task="TP53", scaling="cohort", backend=None,
                data_dir="data", cache_dir="cache", out=None, n_jobs=1):
    """Shot noise and gate error at the feature maps CV selected in the sweep.

    Each expectation value is shrunk by (1 - eps) ** (CZ gates in its light cone), then
    sampled with a finite number of shots. A planning tool for the shot budget, not a device
    simulation: no crosstalk, no leakage, no readout bias.
    """
    backend = backend or local_backend()
    eps = median_cz_error(backend) if eps is None else eps
    best = selected(sweep)
    patches = {int(n): patch(backend, int(n)) for n in sorted(best.n.unique())}
    groups = {int(s): [(int(b.n), int(b.reps), float(b.bandwidth)) for _, b in g.iterrows()] for s, g in best.groupby("seed")}
    inner = 1 if (n_jobs != 1 and len(groups) > 1) else -1
    args = [(s, g, task, scaling, list(shots), eps, fmap, patches, data_dir, cache_dir, inner) for s, g in groups.items()]
    df = pd.DataFrame([r for rows in _parallel(_proxy_seed, args, n_jobs) for r in rows])
    df["eps_cz"] = eps
    if out:
        df.to_csv(out, index=False)
    return df
