"""Write the LaTeX tables and number macros that paper/pqk100_preliminary.tex includes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import report
from .study import selected

KEYS = ["pqk_selected", "rbf_svm", "logreg", "gbm"]


def _pm(mu, se, d=3):
    """0.909(8): value with the standard error in units of the last digit."""
    return f"{mu:.{d}f}({int(round(se * 10 ** d))})" if np.isfinite(se) else f"{mu:.{d}f}"


def _tabular(spec, header, rows):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n"


def table_cohorts(data_dir="data", cache_dir="cache"):
    from .data import load_cohorts
    rows = []
    for c, platform in zip(load_cohorts(data_dir, cache_dir), ("RNA-seq (RSEM)", "Illumina HT-12 microarray")):
        y = c.labels["TP53"].dropna()
        rows.append([c.name, platform, f"{len(c.X):,}", f"{len(y):,}", f"{100 * y.mean():.1f}"])
    return _tabular("llrrr", r"Cohort & Platform & Tumors & \textit{TP53} call & Mutated (\%) \\", rows)


def table_task_screen(screen):
    s = screen[screen.scaling == "cohort"]
    names = {"ER": "ER status (IHC)", "lumAB": "Luminal A vs B (PAM50)", "TP53": "TP53 mutation"}
    rows = []
    for task in ("ER", "lumAB", "TP53"):
        g = s[s.task == task]
        one = g[(g.k == 1) & (g.model == "logreg")][["in_auc", "ext_auc"]].mean()
        many = g[(g.k == 156) & (g.model == "logreg")][["in_auc", "ext_auc"]].mean()
        gene = g[g.k == 1].top_gene.value_counts().index[0]
        rows.append([names[task], rf"\textit{{{gene}}}", f"{one.in_auc:.3f}", f"{one.ext_auc:.3f}", f"{many.in_auc:.3f}", f"{many.ext_auc:.3f}"])
    head = (r" & & \multicolumn{2}{c}{Best single gene} & \multicolumn{2}{c}{156 genes} \\" "\n"
            r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}" "\n"
            r"Task & Top gene & TCGA & METABRIC & TCGA & METABRIC \\")
    return _tabular("llcccc", head, rows)


def table_main(sweep):
    t = report.main_table(sweep)
    rows = [[str(int(n))] + [_pm(t.loc[n, (c, "mean", k)], t.loc[n, (c, "sem", k)]) for c in ("in_auc", "ext_auc") for k in KEYS]
            for n in sorted(t.index)]
    head = (r" & \multicolumn{4}{c}{Held-out TCGA-BRCA} & \multicolumn{4}{c}{METABRIC} \\" "\n"
            r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}" "\n"
            r"Qubits & PQK & RBF-SVM & LogReg & GBM & PQK & RBF-SVM & LogReg & GBM \\")
    return _tabular("r" + "c" * 8, head, rows)


def table_paired(sweep):
    d = report.paired_diff(sweep)
    f = lambda vs, m, n: (lambda r: f"${r['mean'].iloc[0]:+.3f}\\pm{r['se'].iloc[0]:.3f}$")(d[(d.vs == vs) & (d.metric == m) & (d.n == n)])
    rows = [[str(int(n)), f("rbf_svm", "in_auc", n), f("rbf_svm", "ext_auc", n), f("logreg", "in_auc", n), f("logreg", "ext_auc", n)]
            for n in sorted(d.n.unique())]
    head = (r" & \multicolumn{2}{c}{PQK $-$ RBF-SVM} & \multicolumn{2}{c}{PQK $-$ LogReg} \\" "\n"
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" "\n"
            r"Qubits & TCGA & METABRIC & TCGA & METABRIC \\")
    return _tabular("rcccc", head, rows)


def table_depth_bandwidth(sweep):
    t, cnt = report.depth_bandwidth(sweep), selected(sweep).groupby(["reps", "bandwidth"]).size()
    rows = [[str(int(r)), f"{b:g}", f"{v.in_auc:.3f}", f"{v.ext_auc:.3f}", f"{v.bloch_norm:.3f}", f"{v.feature_spread:.3f}",
             f"{v.geo_diff:.2f}", str(int(cnt.get((r, b), 0)))] for (r, b), v in t.iterrows()]
    head = r"$R$ & $\beta$ & TCGA & METABRIC & $\overline{|\mathbf{r}|}$ & Spread & $g$ & Selected \\"
    return _tabular("cccccccc", head, rows)


def table_cones(backend):
    t = report.cone_table(backend)
    rows = [[str(r), str(int(v["largest cone (qubits)"])), f'{v["mean cone (qubits)"]:.1f}', str(int(v["CZ gates in largest cone"])),
             f'{int(v["CZ gates in circuit"]):,}', str(int(v["CZ depth"]))] for r, v in t.iterrows()]
    head = r"$R$ & Largest cone & Mean cone & CZ in largest cone & CZ in circuit & CZ depth \\"
    return _tabular("cccccc", head, rows)


def table_ablation(results_dir):
    order = ["PQK", "PQK, entanglers removed", "RBF-SVM", "RBF-SVM, clipped inputs", "Logistic regression"]
    blocks = []
    for sc, label in (("cohort", "Cohort-wise $z$-score"), ("rank", "Per-sample rank")):
        a = pd.read_csv(Path(results_dir) / f"ablation_{sc}.csv")
        g = a.groupby(["model", "n"]).ext_auc.agg(["mean", report.sem])
        ns = sorted(a.n.unique())
        blocks.append((label, ns, [[m] + [_pm(*g.loc[(m, n)]) for n in ns] for m in order]))
    ns = blocks[0][1]
    lines = []
    for label, _, rows in blocks:
        lines.append(rf"\multicolumn{{{len(ns) + 1}}}{{l}}{{\textit{{{label}}}}} \\")
        lines += [" & ".join(r) + r" \\" for r in rows]
        lines.append(r"\addlinespace")
    head = "Model & " + " & ".join(f"$n={n}$" for n in ns) + r" \\"
    return ("\\begin{tabular}{l" + "c" * len(ns) + "}\n\\toprule\n" + head + "\n\\midrule\n" + "\n".join(lines[:-1]) +
            "\n\\bottomrule\n\\end{tabular}\n")


def table_proxy(results_dir):
    p = pd.read_csv(Path(results_dir) / "noise_proxy.csv"); p["shots"] = p.shots.astype(str)
    t = p.pivot_table(index="n", columns="shots", values="ext_auc")
    cols = ["exact", "4096", "1024", "256"]
    rows = [[str(int(n))] + [f"{t.loc[n, c]:.3f}" for c in cols] for n in t.index]
    return _tabular("rcccc", r"Qubits & Exact & 4096 shots & 1024 shots & 256 shots \\", rows)


def numbers(sweep, results_dir, backend, data_dir="data", cache_dir="cache"):
    """Macros for every number quoted in the prose."""
    from .data import load_cohorts
    from .featuremap import FEATURE_MAPS
    from .hardware import build, circuit_report, estimate_minutes
    from .lattice import patch
    from .backends import median_cz_error
    tcga, meta = load_cohorts(data_dir, cache_dir)
    d, best, ref = report.paired_diff(sweep), selected(sweep), report.all_genes_reference(sweep)
    cnt = best.groupby("bandwidth").size(); q = sweep[sweep.model == "pqk"]
    db = report.depth_bandwidth(sweep)
    rng_n = q.groupby(["reps", "bandwidth", "n"])[["bloch_norm", "feature_spread"]].mean().groupby(level=[0, 1]).agg(lambda x: x.max() - x.min()).max()
    phys, edges = patch(backend, 156)
    rep = circuit_report(build(156, FEATURE_MAPS["zz"](156, edges, reps=2), phys, backend)[0], backend)
    p = pd.read_csv(Path(results_dir) / "noise_proxy.csv")
    m = {
        "NTcga": f"{len(tcga.X):,}", "NMeta": f"{len(meta.X):,}", "NGenes": f"{tcga.X.shape[1]:,}",
        "NSeeds": str(sweep.seed.nunique()), "Lattice": backend.name.replace("fake_", "ibm\\_"),
        "MaxAbsDiff": f"{d['mean'].abs().max():.3f}", "SeMin": f"{d.se.min():.3f}", "SeMax": f"{d.se.max():.3f}",
        "SelSmall": str(int(cnt.get(0.25, 0))), "SelMid": str(int(cnt.get(0.5, 0))), "SelLarge": str(int(cnt.get(1.0, 0))),
        "SelTotal": str(len(best)),
        "RefIn": f"{ref.in_auc:.3f}", "RefExt": f"{ref.ext_auc:.3f}",
        "BlochSmallMin": f"{db.xs(0.25, level='bandwidth').bloch_norm.min():.2f}", "BlochSmallMax": f"{db.xs(0.25, level='bandwidth').bloch_norm.max():.2f}",
        "BlochLargeMin": f"{db.xs(1.0, level='bandwidth').bloch_norm.min():.2f}", "BlochLargeMax": f"{db.xs(1.0, level='bandwidth').bloch_norm.max():.2f}",
        "LossLargeMin": f"{(db.xs(0.25, level='bandwidth').ext_auc - db.xs(1.0, level='bandwidth').ext_auc).min():.2f}",
        "LossLargeMax": f"{(db.xs(0.25, level='bandwidth').ext_auc - db.xs(1.0, level='bandwidth').ext_auc).max():.2f}",
        "ConcBloch": f"{rng_n.bloch_norm:.2f}", "ConcSpread": f"{rng_n.feature_spread:.2f}",
        "GeoMeanMin": f"{db.geo_diff.min():.1f}", "GeoMeanMax": f"{db.geo_diff.max():.1f}", "GeoMax": f"{q.geo_diff.max():.1f}",
        "EpsCZ": f"{median_cz_error(backend):.4f}", "ProxyEps": f"{p.eps_cz.iloc[0]:.4f}",
        "ShotUs": f"{rep['shot_s'] * 1e6:.0f}", "RepDelayUs": f"{rep['rep_delay_s'] * 1e6:.0f}", "CircuitUs": f"{rep['circuit_s'] * 1e6:.1f}",
        "MinFull": f"{estimate_minutes(rep, 600, 1024):.0f}", "MinFullHigh": f"{estimate_minutes(rep, 600, 4096):.0f}",
        "MinPilot": f"{estimate_minutes(rep, 80, 1024):.1f}",
    }
    mps = Path(results_dir) / "mps_check.csv"
    if mps.exists():
        for _, r in pd.read_csv(mps).iterrows():
            e = f"{r['max_abs_error']:.1e}".split("e"); words = {8: "Eight", 32: "ThirtyTwo", 64: "SixtyFour"}[int(r["chi"])]
            m[f"Mps{words}"] = rf"{e[0]}\times10^{{{int(e[1])}}}"
    return m


def write_tables(sweep, results_dir="results", out_dir="paper/tables", backend=None, data_dir="data", cache_dir="cache"):
    from .backends import local_backend
    backend = backend or local_backend()
    R, out = Path(results_dir), Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    files = {
        "cohorts.tex": table_cohorts(data_dir, cache_dir),
        "task_screen.tex": table_task_screen(pd.read_csv(R / "task_screen.csv")),
        "main.tex": table_main(sweep),
        "paired.tex": table_paired(sweep),
        "depth_bandwidth.tex": table_depth_bandwidth(sweep),
        "cones.tex": table_cones(backend),
        "ablation.tex": table_ablation(R),
        "proxy.tex": table_proxy(R),
    }
    nums = numbers(sweep, R, backend, data_dir, cache_dir)
    files["numbers.tex"] = "% Generated by notebook 5. Do not edit by hand.\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in nums.items())
    for name, text in files.items():
        (out / name).write_text(text)
    return files
