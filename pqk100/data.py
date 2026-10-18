"""Load the two breast cancer cohorts and build task labels.

Training cohort : TCGA-BRCA (RNA-seq, RSEM), cBioPortal PanCancer Atlas expression,
                  labels from the PanCancer Atlas and the 2015 Cell paper study.
Validation cohort: METABRIC (Illumina HT-12 microarray), cBioPortal.

Both are read from tab-separated cBioPortal files. Nothing here touches labels of the
validation cohort except to hand them back for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NONSILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "In_Frame_Del", "In_Frame_Ins", "Nonstop_Mutation",
    "Translation_Start_Site",
}


@dataclass
class Cohort:
    name: str
    X: pd.DataFrame          # samples x genes, log2 scale
    labels: pd.DataFrame     # samples x tasks, values in {0, 1, NaN}


def _read(path, **kw):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False, **kw)


def _read_case_list(path) -> list[str]:
    for line in Path(path).read_text().splitlines():
        if line.startswith("case_list_ids:"):
            return line.split(":", 1)[1].split()
    raise ValueError(f"no case_list_ids in {path}")


def _load_expression(path, log2p1: bool) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    df = df.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"])
    df = df.set_index("Hugo_Symbol").astype("float32")
    if log2p1:
        df = np.log2(df.clip(lower=0) + 1.0)
    # drop genes with missing values, then keep one row per symbol (largest variance)
    df = df.dropna(axis=0, how="any")
    df = df.iloc[np.argsort(-df.var(axis=1).values, kind="stable")]
    df = df[~df.index.duplicated(keep="first")]
    return df.sort_index().T  # samples x genes


def load_tcga(d) -> Cohort:
    d = Path(d)
    X = _load_expression(d / "expr.txt", log2p1=True)
    lab = pd.DataFrame(index=X.index)

    # PAM50 (PanCancer Atlas clinical file), luminal A = 0, luminal B = 1
    pat = _read(d / "clin_patient.txt").set_index("PATIENT_ID")
    sub = pat["SUBTYPE"].reindex(lab.index.str[:12]).values
    lab["lumAB"] = np.select([sub == "BRCA_LumA", sub == "BRCA_LumB"], [0.0, 1.0], np.nan)

    # ER by immunohistochemistry (2015 study)
    smp = _read(d / "clin_sample_pub2015.txt").set_index("SAMPLE_ID")
    er = smp["ER_STATUS_BY_IHC"].reindex(lab.index).values
    lab["ER"] = np.select([er == "Negative", er == "Positive"], [0.0, 1.0], np.nan)

    # TP53 non-silent somatic mutation (2015 study), only for sequenced samples
    mut = _read(d / "mutations_pub2015.txt", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"])
    seq = set(_read_case_list(d / "cases_sequenced_pub2015.txt"))
    hit = set(mut[(mut.Hugo_Symbol == "TP53") & mut.Variant_Classification.isin(NONSILENT)].Tumor_Sample_Barcode)
    lab["TP53"] = [np.nan if s not in seq else float(s in hit) for s in lab.index]
    return Cohort("TCGA-BRCA", X, lab)


def load_metabric(d) -> Cohort:
    d = Path(d)
    X = _load_expression(d / "expr.txt", log2p1=False)
    lab = pd.DataFrame(index=X.index)
    pat = _read(d / "clin_patient.txt").set_index("PATIENT_ID")

    sub = pat["CLAUDIN_SUBTYPE"].reindex(lab.index).values
    lab["lumAB"] = np.select([sub == "LumA", sub == "LumB"], [0.0, 1.0], np.nan)

    er = pat["ER_IHC"].reindex(lab.index).values      # 'Positve' is the spelling in the file
    lab["ER"] = np.select([er == "Negative", er == "Positve"], [0.0, 1.0], np.nan)

    mut = _read(d / "mutations.txt", usecols=["Hugo_Symbol", "Variant_Classification", "Tumor_Sample_Barcode"])
    seq = set(_read_case_list(d / "cases_sequenced.txt"))
    hit = set(mut[(mut.Hugo_Symbol == "TP53") & mut.Variant_Classification.isin(NONSILENT)].Tumor_Sample_Barcode)
    lab["TP53"] = [np.nan if s not in seq else float(s in hit) for s in lab.index]
    return Cohort("METABRIC", X, lab)


def load_cohorts(data_dir="data", cache_dir="cache") -> tuple[Cohort, Cohort]:
    """Return (train cohort, validation cohort) restricted to their common genes."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    f = cache_dir / "cohorts.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        out = []
        for k in ("tcga", "metabric"):
            X = pd.DataFrame(z[f"{k}_X"], index=z[f"{k}_s"], columns=z["genes"])
            lab = pd.DataFrame(z[f"{k}_y"], index=z[f"{k}_s"], columns=z["tasks"])
            out.append(Cohort(str(z[f"{k}_name"]), X, lab))
        return tuple(out)
    a, b = load_tcga(data_dir / "tcga_brca"), load_metabric(data_dir / "metabric")
    genes = sorted(set(a.X.columns) & set(b.X.columns))
    a.X, b.X = a.X[genes], b.X[genes]
    cache_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        f, genes=np.array(genes), tasks=np.array(a.labels.columns),
        tcga_X=a.X.values, tcga_s=a.X.index.values.astype(str), tcga_y=a.labels.values, tcga_name=a.name,
        metabric_X=b.X.values, metabric_s=b.X.index.values.astype(str), metabric_y=b.labels.values, metabric_name=b.name,
    )
    return a, b
