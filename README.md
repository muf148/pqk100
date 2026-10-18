# PQK100

Projected quantum kernels on 20 to 156 qubits for genomic classification, with an
independent validation cohort.

- Task: TP53 mutation status of breast tumors, predicted from gene expression.
- Training cohort: TCGA-BRCA (RNA-seq). Validation cohort: METABRIC (microarray).
- One gene per qubit. Feature map native to the heavy-hex lattice of ibm_fez. X, Y, Z on every qubit.

Everything you run is a notebook. Open any of them from VS Code or Jupyter; each one finds this folder by itself.

## Run order

| Notebook | What it does | Time |
|---|---|---|
| `00_setup_and_data.ipynb` | Checks the Python packages and your saved IBM account, downloads the data. Run once per computer. | a few minutes the first time |
| `01_data_and_task.ipynb` | Loads both cohorts and shows why TP53 is the task. | under 2 min |
| `02_feature_map_and_simulation.ipynb` | Patch on ibm_fez, the circuit, light cones, checks of the exact simulator, one 156-qubit experiment. | a few minutes |
| `03_scaling_study.ipynb` | Scaling from 20 to 156 qubits, depth and bandwidth, ablation, noise proxy. Reads saved results unless you ask it to recompute. | seconds |
| `04_hardware_ibm_fez.ipynb` | General hardware runner. `USE_HARDWARE = False` uses FakeFez; `True` uses ibm_fez through `Faryad-free-ibm`. Submits nothing until `SUBMIT = True`. | 4 min locally |
| `05_tables_and_figures.ipynb` | Writes the tables, figures and number macros that the paper includes. | seconds |
| `06_self_checks.ipynb` | Checks the simulator against Qiskit and the hardware code path on FakeFez. Run after changing `pqk100/`. | about 2 min |
| `push_to_github.ipynb` | Commits and pushes this folder to github.com/muf148/pqk100. | under a minute |

## IBM Quantum Credits application

`credits/` holds the five notebooks for the Credits application. See `credits/README.md`.
The one to run now is `credits/c3_pilot_ibm_fez.ipynb`: Run All submits the 2-minute pilot to ibm_fez.
Then run `credits/c4_budget_for_form.ipynb`.

The application itself is in `application/` (Word). It is kept off GitHub.

## Folders

| Folder | What is in it |
|---|---|
| `pqk100/` | The shared code the notebooks import. Not something you run. |
| `results/` | Saved results that notebooks 01, 03 and 05 read. |
| `credits/` | Credits notebooks and their outputs. |
| `paper/` | `pqk100_preliminary.tex`, with tables and figures written by notebook 05. |
| `application/` | The Credits application (Word). Not pushed to GitHub. |
| `data/`, `cache/` | Downloaded cohorts and a loading cache. Not pushed to GitHub. |
| `junk/` | Old scripts and files nothing uses any more. Not pushed to GitHub. Delete it whenever you like. |
| `.venv/` | The Python environment VS Code uses for these notebooks. |

## Paper

Compile `paper/pqk100_preliminary.tex` (REVTeX 4.2) with

    cd paper
    pdflatex pqk100_preliminary
    pdflatex pqk100_preliminary

## The shared code in `pqk100/`

    data.py        cohorts and labels (TP53, ER, luminal A vs B)
    prep.py        splits, cross-platform scaling, gene selection, angle encoding
    backends.py    device name and saved account, in one place
    lattice.py     nested connected patches of the device, best qubits kept longest
    featuremap.py  feature maps as gate lists; light cones; entangler ablation
    lightcone.py   exact <X>, <Y>, <Z> by light-cone statevector simulation
    kernel.py      PQK, concentration measures, geometric difference, noise proxy
    baselines.py   RBF-SVM, logistic regression, gradient boosting
    study.py       the experiments of notebook 03 as functions
    report.py      summary tables and figures
    paper.py       LaTeX tables and number macros
    hardware.py    cost estimate and helpers for the hardware path

## Adding a feature map from the NAS

A feature map is a function `(n, edges, reps=..., **kw) -> list[Op]`. An `Op` is
`(gate, qubits, coeff, feats)` with angle `coeff * prod(x[f] for f in feats)`.
Gates: h, rx, ry, rz, rzz, rxx, ryy, cz. Register it in `FEATURE_MAPS` in `pqk100/featuremap.py`.
The Qiskit circuit and the classical simulation are both built from that list.
Then run `06_self_checks.ipynb`.

## Data sources

TCGA-BRCA and METABRIC, as distributed by cBioPortal (github.com/cBioPortal/datahub).
Cite TCGA (Nature 2012; Ciriello et al., Cell 2015), METABRIC (Curtis et al., Nature
2012; Pereira et al., Nat. Commun. 2016), and cBioPortal.
