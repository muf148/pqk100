# IBM Quantum Credits: notebooks for the application

These five notebooks produce the numbers the Credits form still needs. Each one finds the project folder by
itself. They use the same environment and data as notebooks 01 to 05; if something does not import, run
`00_setup_and_data.ipynb` in the project folder first.

| Run | Notebook | What it does | Writes to `credits/outputs/` | Time |
|---|---|---|---|---|
| 1 | `c1_fixed_panel.ipynb` | Picks the 156 genes once, on 410 TCGA tumors that are then set aside. The other 401 TCGA and 600 METABRIC tumors are the measured set. | `fixed_panel.npz`, `fixed_panel.json` | 2 min |
| 2 | `c2_fixed_panel_simulation.ipynb` | Reruns the scaling study with that fixed panel, 10 resamples. Gives the AUC values to quote. | `numbers_for_application.md`, `fixed_panel_sweep.csv`, `fixed_panel_scaling.png`, `exact/` | 30 to 60 min |
| 3 | `c3_pilot_ibm_fez.ipynb` | The 2-minute pilot. Measures billed time per shot, per-job overhead, and signal retained. | `pilot_summary.json` (job ids go to `results/jobs/`) | 4 min locally, about 2 min of QPU |
| 4 | `c4_budget_for_form.ipynb` | Turns the pilot's rate into the hour estimate and counts two-qubit gates after transpilation. | `budget_for_form.md` | seconds |
| 5 | `c5_prior_work_arxiv.ipynb` | Searches arXiv for anything that would undercut the "first PQK genomic classification above 61 qubits" claim. Independent of the others. | `arxiv_prior_work.csv` | 1 to 3 min |

C2 caches its results. Rerunning it is instant unless you set `RECOMPUTE = True`.

## Running the pilot (C3)

As delivered, the notebook runs on ibm_fez through the account saved on your computer as `Faryad-free-ibm`.

1. Open it and choose Run All. It checks the saved account, prints the cost, and submits 3 jobs if the
   estimate is under `MAX_MINUTES` (5). Expect about 2 minutes of Open Plan time.
2. If the jobs are still queued, close the notebook and come back later. Run All again: it finds the saved
   job ids, reuses the same qubits, and does not resubmit.
3. When it finishes, run C4.

To try it on the laptop first, set `USE_HARDWARE = False` (FakeFez, 10 qubits, nothing goes to IBM).
To see the cost only, set `SUBMIT = False`.

The 80 tumors go in as jobs of 10, 20 and 50. The unequal sizes let the notebook separate the fixed cost per job
from the cost per shot.

## Where each number goes

| Output | Field in the form, or section of the application doc |
|---|---|
| `numbers_for_application.md` | Project abstract; Section 3, first answer |
| `pilot_summary.json` (C3.9 prints the sentence) | Section 4: "Were you able to determine feasibility via the Open Plan?" |
| `budget_for_form.md` | Section 5: qubits to date, 2-qubit gates, estimated compute time, and the block table |
| `arxiv_prior_work.csv` (C5.3 lists what to read) | Section 2 abstract and Section 3: the "first" claim |

## Your IBM Quantum account is on your Gmail address

The saved account `Faryad-free-ibm` is the IBMid on your Gmail address. That is fine for the pilot, which runs
on the Open Plan.

The Credits form is different. It asks for your institutional email, and IBM says applications from private email
domains will not be considered. Credits are then attached to an IBMid. You have two options:

- Create a second IBM Quantum account with your LUMS email before you apply, and use that address in the form.
  This is the safe choice.
- Or email quantum.programs@ibm.com first and ask whether credits for a LUMS application can go on the existing
  Gmail account.

If the credits arrive on a LUMS account, save it once under its own name and change `ACCOUNT` in C3 (and in
`pqk100/backends.py` for notebooks 01 to 05):

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    token="<API key of the LUMS account>",
    instance="<CRN of the Credits instance>",
    name="Faryad-lums-credits",
    channel="ibm_quantum_platform",
)
```

`save_account` stores the key in your user folder. Do not type the key into a notebook you will share or upload.
