"""Tables and figures from the result files. Used by the notebooks and by the paper."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .study import selected

MODELS = [("pqk_selected", "PQK"), ("rbf_svm", "RBF-SVM"), ("logreg", "Logistic regression"), ("gbm", "Gradient boosting")]
COLORS = {"pqk_selected": "#2a78d6", "rbf_svm": "#eb6834", "logreg": "#1baf7a", "gbm": "#8a8983"}
MARKS = {"pqk_selected": "o", "rbf_svm": "s", "logreg": "^", "gbm": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def sem(s):
    return s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan


def models_frame(sweep):
    """Classical rows plus the CV-selected PQK, one row per seed, size, and model."""
    cl = sweep[sweep.model.isin(["rbf_svm", "logreg", "gbm"])]
    return pd.concat([cl, selected(sweep).assign(model="pqk_selected")], ignore_index=True)


def main_table(sweep, metric="auc"):
    """Mean and standard error over resamples, held-out TCGA and METABRIC, by number of qubits."""
    m = models_frame(sweep)
    g = m.groupby(["n", "model"])[[f"in_{metric}", f"ext_{metric}"]].agg(["mean", sem])
    return g.unstack("model")


def paired_diff(sweep, refs=("rbf_svm", "logreg")):
    """Selected PQK minus a classical model, paired by resample."""
    sel = selected(sweep); out = []
    for ref in refs:
        c = sweep[sweep.model == ref]
        m = sel.merge(c, on=["seed", "n"], suffixes=("", "_c"))
        for col in ("in_auc", "ext_auc"):
            g = (m[col] - m[col + "_c"]).groupby(m.n)
            out.append(pd.DataFrame({"mean": g.mean(), "se": g.apply(sem)}).assign(vs=ref, metric=col))
    return pd.concat(out).reset_index()


def selection_counts(sweep):
    return selected(sweep).groupby(["reps", "bandwidth"]).size().unstack(fill_value=0)


def depth_bandwidth(sweep):
    q = sweep[sweep.model == "pqk"]
    cols = ["cv_auc", "in_auc", "ext_auc", "bloch_norm", "feature_spread", "geo_diff", "align_q"]
    return q.groupby(["reps", "bandwidth"])[cols].mean()


def concentration_by_n(sweep, bandwidth=0.5):
    q = sweep[(sweep.model == "pqk") & (sweep.bandwidth == bandwidth)]
    return q.groupby(["reps", "n"])[["bloch_norm", "feature_spread", "d2_cv", "cone_max"]].mean()


def all_genes_reference(sweep):
    return sweep[sweep.model == "logreg_all_genes"][["in_auc", "ext_auc"]].mean()


def cone_table(backend, n=156, reps=(1, 2, 3, 4, 5, 6), fmap="zz"):
    """Light-cone size and gate counts for the full-width circuit."""
    from .featuremap import FEATURE_MAPS, cone_stats, CZ_COST
    from .lattice import patch
    phys, edges = patch(backend, n); rows = []
    for r in reps:
        ops = FEATURE_MAPS[fmap](n, edges, reps=r)
        size, n2q, ncz = cone_stats(n, ops)
        rows.append({"reps": r, "largest cone (qubits)": int(size.max()), "mean cone (qubits)": round(float(size.mean()), 1),
                     "CZ gates in largest cone": int(ncz.max()), "CZ gates in circuit": int(sum(CZ_COST.get(o.gate, 0) for o in ops)),
                     "CZ depth": 3 * 2 * r if fmap == "zz" else 3 * r})
    return pd.DataFrame(rows).set_index("reps")


# ------------------------------------------------------------------ figures
def _style(paper):
    import matplotlib.pyplot as plt
    bg = "white" if paper else "#fcfcfb"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9 if paper else 10, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK if paper else MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.facecolor": bg, "axes.facecolor": bg, "savefig.facecolor": bg, "pdf.fonttype": 42})
    return bg


def fig_scaling(sweep, paper=False, title=True):
    """AUC against number of qubits, held-out TCGA and METABRIC side by side."""
    import matplotlib.pyplot as plt
    bg = _style(paper)
    m, ref = models_frame(sweep), all_genes_reference(sweep)
    ns = sorted(m.n.unique()); seeds = m.seed.nunique()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.3) if paper else (11, 4.6), sharey=True)
    for ax, col, ttl in zip(axes, ("in_auc", "ext_auc"), ("Held-out TCGA-BRCA (same cohort)", "METABRIC (independent cohort)")):
        for (key, lab), dx in zip(MODELS, (-2.4, -0.8, 0.8, 2.4)):
            g = m[m.model == key].groupby("n")[col]
            mu, se = g.mean().reindex(ns), g.apply(sem).reindex(ns)
            x = np.array(ns) + dx
            ax.errorbar(x, mu, yerr=se, fmt="none", ecolor=COLORS[key], elinewidth=1.0, alpha=0.75, zorder=2)
            ax.plot(x, mu, "--" if key == "gbm" else "-", color=COLORS[key], lw=1.6 if paper else 2, marker=MARKS[key],
                    ms=4.5 if paper else 6, mec=bg, mew=1.0, label=lab, zorder=3)
        ax.axhline(ref[col], color=MUTED, lw=1.1, ls=":", label="Logistic regression, 5,000 genes", zorder=1)
        ax.set_title(ttl, color=INK, fontsize=9.5 if paper else 11, loc="left")
        ax.set_xlabel("Qubits (genes in the panel)"); ax.set_xticks(ns)
        ax.tick_params(axis="x", labelsize=7.5 if paper else 10)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("ROC AUC")
    lo = min(m.groupby(["n", "model"])[c].mean().min() for c in ("in_auc", "ext_auc"))
    axes[0].set_ylim(np.floor((lo - 0.02) * 50) / 50, 0.96)
    h, l = axes[0].get_legend_handles_labels()
    top = 0.80 if paper else (0.88 if title else 0.93)
    fig.legend(h, l, frameon=False, loc="upper left", bbox_to_anchor=(0.005, 1.0 if (paper or not title) else 0.925),
               ncols=3 if paper else 5, labelcolor=INK, fontsize=8 if paper else 9.5, handlelength=2.6, columnspacing=1.6)
    if title and not paper:
        fig.suptitle(f"TP53 mutation status from expression: train on 200 TCGA tumors, mean of {seeds} resamples (bars = 1 s.e.)",
                     x=0.01, ha="left", color=INK, fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, top))
    return fig


def fig_bandwidth(sweep, paper=False):
    """METABRIC AUC and Bloch-vector length against bandwidth, one line per depth."""
    import matplotlib.pyplot as plt
    bg = _style(paper)
    q = sweep[sweep.model == "pqk"]
    cols = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}; mk = {1: "o", 2: "s", 3: "^"}
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 3.9), sharex=True) if paper else plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, lab in zip(axes, ("ext_auc", "bloch_norm"), ("METABRIC ROC AUC", "Mean Bloch-vector length")):
        for r in sorted(q.reps.unique()):
            g = q[q.reps == r].groupby(["bandwidth", "seed"])[col].mean().groupby("bandwidth")
            mu, se = g.mean(), g.apply(sem)
            ax.errorbar(mu.index, mu, yerr=se, color=cols.get(int(r), MUTED), marker=mk.get(int(r), "o"), ms=5, mec=bg, mew=1.0,
                        lw=1.6, elinewidth=1.0, label=f"{int(r)} repetition" + ("s" if r > 1 else ""))
        ax.set_xscale("log", base=2); ax.set_xticks(sorted(q.bandwidth.unique()))
        ax.set_xticklabels([f"{b:g}" for b in sorted(q.bandwidth.unique())])
        ax.set_ylabel(lab)
        if not paper or ax is axes[-1]:
            ax.set_xlabel("Bandwidth (rad per standard deviation)")
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(frameon=False, labelcolor=INK, fontsize=8 if paper else 9.5, loc="lower left")
    fig.tight_layout()
    return fig


def lattice_positions(backend):
    """(x, y) for each qubit. Heron r2 (ibm_fez): 8 rows of 16 qubits, joined by 4 bridge qubits per gap.
    Falls back to a generic graph layout for other devices."""
    nbr = {q: set() for q in range(backend.num_qubits)}
    for a, b in backend.coupling_map.get_edges():
        nbr[a].add(b); nbr[b].add(a)
    pos = {q: (q % 20, -2 * (q // 20)) for q in range(backend.num_qubits) if q % 20 < 16}
    for q in range(backend.num_qubits):
        if q not in pos:
            row = [p for p in nbr[q] if p in pos]
            if not row:
                import networkx as nx
                return nx.kamada_kawai_layout(nx.Graph(list(backend.coupling_map.get_edges())))
            up = max(row, key=lambda p: pos[p][1])          # a bridge sits just below its upper row
            pos[q] = (pos[up][0], pos[up][1] - 1)
    return pos


def fig_lattice(backend, highlight=(), title=None):
    """The device lattice with the qubits in `highlight` and the couplers between them in blue.
    Matplotlib only: Qiskit's plot_gate_map needs the separate Graphviz program for 156-qubit devices."""
    import matplotlib.pyplot as plt
    _style(False)
    pos, inside = lattice_positions(backend), set(highlight)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for a, b in {tuple(sorted(e)) for e in backend.coupling_map.get_edges()}:
        on = a in inside and b in inside
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=COLORS["pqk_selected"] if on else GRID,
                lw=2.2 if on else 1.2, zorder=1)
    for q in range(backend.num_qubits):
        on = q in inside
        ax.scatter(*pos[q], s=150, color=COLORS["pqk_selected"] if on else "#d9d8d2", edgecolor="white", linewidth=0.8, zorder=2)
        ax.text(*pos[q], str(q), ha="center", va="center", fontsize=5.5, color="white" if on else MUTED, zorder=3)
    ax.set_title(title or backend.name, loc="left", color=INK)
    ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
    return fig
