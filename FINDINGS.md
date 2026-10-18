# PQK100: first findings (Oct 11, 2026)

Status: pipeline built and tested in simulation. Nothing has run on IBM hardware yet.

Updated the same day: the target device is now ibm_fez, and all simulations were rerun on the Fez lattice. The numbers below are from that rerun. The code is now driven from five notebooks, and the preliminary results are also in `paper/pqk100_preliminary.tex`.

## Bottom line

- The slot still looks open, with one caveat on how to word the claim.
- Task chosen: TP53 mutation status in breast cancer. Train on TCGA-BRCA, validate on METABRIC.
- In exact noiseless simulation from 20 to 156 qubits, the PQK ties tuned classical models on both cohorts. No advantage, no penalty.
- The shallow heavy-hex circuits in the plan are exactly simulable. The 156-qubit kernel is reproduced on a laptop.
- The 180-minute pool covers the scaling study at 1,024 shots. It does not at 4,096.

## 1. Checks the plan asked for

**Prior work.** arXiv's own search blocks automated access, so this is a web-search check, not an exhaustive listing. Do one manual pass on arXiv before you write the introduction.

| Paper | Qubits | What it is |
|---|---|---|
| arXiv:2507.22710 (IBM) | 61 | PQK, CAR T-cell cytotoxicity, ibm_marrakesh. Says "largest PQK application to date". 100 min of QPU time. |
| arXiv:2605.24252 (WISER, E.ON) | 100 | PQK inside a Gaussian process, electricity time series, ibm_marrakesh. Regression, not classification. |
| arXiv:2412.07915 (Agliardi et al.) | 156 | Fidelity kernels with bit-flip tolerance, synthetic and EV-scheduling data. Says "largest experimental demonstration of quantum machine learning on IBM devices to date". |

I found no PQK study on genomic data above 61 qubits, and none with an independent validation cohort. The third paper matters for wording. "Largest quantum kernel classification" is taken. "First PQK genomic classification at 100+ qubits with an independent cohort" is not.

**Runtime.** All three numbers in the plan hold.

- Open Plan: 10 minutes per 28 days. One-time 180 minutes for 12 months after you log 20 minutes in any 12-month period.
- ibm_kingston (156 qubits, Heron r2) has been on the Open Plan since March 2026.
- IBM's PQK tutorial: 80 minutes on a Heron r3, for 246 samples on 60 qubits with ZNE at three noise factors.

You may already qualify from the course notebooks on ibm_fez. Check the usage page.

## 2. Decisions I made (overrule any of them)

**Disease and cohorts.** Breast cancer. TCGA-BRCA (RNA-seq, 1,082 tumors) for training. METABRIC (Illumina microarray, 1,980 tumors) for validation. Different countries, different platform, 17,139 genes in common. I used METABRIC from cBioPortal instead of a GEO series because both cohorts then come with clean labels from one source.

**Task.** I screened three with classical models only (200 training tumors, 3 resamples). Each cell gives AUC on held-out TCGA, then on METABRIC.

| Task | Best single gene | 156 genes, logistic regression | Verdict |
|---|---|---|---|
| ER status | 0.957, 0.957 (ESR1) | 0.951, 0.941 | Solved by one gene. Rejected. |
| Luminal A vs B | 0.882, 0.840 | 0.973, 0.929 | Label is computed from expression. Little headroom. |
| TP53 mutation | 0.864, 0.791 | 0.925, 0.874 | Chosen. |

TP53 has the right shape. The label comes from DNA sequencing, not from the expression data. No single gene solves it. There is a visible drop on the independent cohort. Class balance matches across cohorts (34.4% and 34.6% mutated; 811 and 1,904 labeled tumors).

**Normalization.** Each gene is z-scored within its own cohort. No labels are used, but the validation cohort's unlabeled data is. That is standard for cross-platform work and has to be stated in the paper. A single-sample alternative (rank normalization) is in the code.

**Feature map.** A placeholder until your NAS map goes in: H, RZ(x), then RZZ(x_i x_j) on the lattice edges, repeated. The angle is the bandwidth times the clipped z-score.

## 3. Simulation results

200 TCGA training tumors, 200 held-out TCGA, 400 METABRIC. Five resamples. Genes are chosen on the training set only. Every model sees the same n genes. PQK depth (1 to 3 repetitions) and bandwidth are picked by cross-validation on the training set. Figure: `results/scaling_auc.png`.

| Qubits | Held-out TCGA: PQK / RBF-SVM / LogReg | METABRIC: PQK / RBF-SVM / LogReg |
|---|---|---|
| 20 | 0.909 / 0.904 / 0.911 | 0.884 / 0.873 / 0.880 |
| 40 | 0.907 / 0.915 / 0.914 | 0.881 / 0.888 / 0.887 |
| 60 | 0.910 / 0.917 / 0.911 | 0.869 / 0.886 / 0.878 |
| 80 | 0.915 / 0.912 / 0.916 | 0.875 / 0.869 / 0.870 |
| 100 | 0.913 / 0.919 / 0.917 | 0.882 / 0.883 / 0.875 |
| 127 | 0.914 / 0.917 / 0.912 | 0.866 / 0.876 / 0.863 |
| 156 | 0.917 / 0.917 / 0.918 | 0.888 / 0.881 / 0.879 |

Paired differences between the PQK and either classical model stay within 0.017 AUC and change sign from size to size. Their standard errors are 0.001 to 0.021. Gradient boosting trails by 0.01 to 0.03 on held-out TCGA and is at or below the others on METABRIC. Logistic regression on all 5,000 prefiltered genes gives 0.918 and 0.890, so 20 genes already carry most of the signal.

Four things worth knowing before the hardware run:

1. **Cross-validation picks the least quantum circuit.** The smallest bandwidth (0.25) wins 28 of 35 times. The largest (1.0) never wins and costs 0.03 to 0.06 AUC on METABRIC. At bandwidth 0.25 the mean Bloch vector length is 0.98 to 0.99, so the qubits are barely entangled.
2. **No concentration with qubit count at fixed depth.** Bloch length and feature spread are flat from 20 to 156 qubits. Each feature depends only on its light cone, so nothing changes with n. The geometric difference to the classical kernel averages 1.1 to 1.9, against sqrt(N) = 14.
3. **Entanglement does not change the score.** The PQK and the same circuit with its two-qubit gates removed agree within one standard error at 20, 100, and 156 qubits, under both normalizations. Under rank normalization the kernel models are less stable: in one of five resamples the plain RBF-SVM loses more than 0.1 AUC on METABRIC at 100 and 156 qubits, and the PQK does so at 100. Clipped inputs, the entangler-free circuit, and logistic regression are affected less. An earlier run on the Kingston lattice showed the PQK ahead of RBF-SVM by about 0.02 under rank normalization. That gap did not survive the rerun on the Fez lattice, so do not rely on it.
4. **The validation sample size is the weak point.** With 200 METABRIC tumors the AUC has a standard deviation near 0.03 from sampling alone. With 400 it is 0.02. That is larger than any model difference above.

The two controls in point 3 (clip the classical inputs; strip the entanglers) are cheap. I would run both on the cytology data before the new paper leans on that result.

## 4. The simulation reproduces the kernel

The plan says to say so if it does. It does, exactly.

A single-qubit expectation value depends only on the gates in its backward light cone. On the 156-qubit patch that cone holds:

| Repetitions | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Qubits in the largest cone | 4 | 7 | 13 | 19 | 31 | 41 |
| CZ gates at 156 qubits | 352 | 704 | 1,056 | 1,408 | 1,760 | 2,112 |

Up to 4 repetitions every 1-RDM is a small statevector calculation. The simulator agrees with Qiskit to 1e-13 on full circuits. A matrix product state on the whole 156-qubit circuit at 2 repetitions converges to the same values (error 3e-2, 4.5e-4, 1.9e-5 at bond dimension 8, 32, 64).

This is the central tension of the paper. The circuits shallow enough to survive noise and to classify well are the ones a laptop reproduces. Cones that are not trivially simulable need 5 or more repetitions, and accuracy falls as the circuit does more.

## 5. Hardware budget

From the ibm_fez calibration snapshot in Qiskit (FakeFez), not a live reading.

- Circuits are 1 to 2.4 microseconds long. The shot is dominated by the 250 microsecond repetition delay.
- So cost does not depend on qubit count or depth. It is samples x 3 bases x shots.
- 600 samples at 4,096 shots: about 47 minutes per configuration. At 1,024 shots: about 12 minutes. Both include a 1.5x allowance for overhead, which is a guess.
- IBM's tutorial implies about 72 microseconds per shot on a Heron r3, so the real cost may be lower. Measure it in the pilot.
- A crude noise proxy (gate error in the light cone plus shot noise) shows no measurable loss at 1,024 shots, or even at 256.

At 1,024 shots, seven sizes at one depth cost about 82 minutes. Four sizes at two depths cost about 94. Either fits in 180 with room for one deeper stress test.

Readout mitigation only (TREX), as planned. No ZNE. The circuit is transpiled once and all samples go in as a parameter array.

## 6. What is not done

- No job has been submitted. The live path is tested up to submission, on a local noisy model of Fez (correlation 0.997 with exact values on 10 qubits).
- Your NAS feature map is not in. The interface for it is a short gate list.
- Five resamples of the same two cohorts are not independent. The standard errors are optimistic.
- The noise proxy ignores crosstalk, leakage, and readout bias.

## 7. Next steps

1. Check whether you already have 20 logged minutes in the last 12 months. Opt in to the 180 if so.
2. Pilot, about 2 minutes: open `04_hardware_ibm_fez.ipynb`, set `USE_HARDWARE = True`, and run it. It prints the cost and stops. Set `SUBMIT = True` to submit. It reports the billed time per shot and the signal retained on each qubit against the exact values.
3. Port the NAS map into `FEATURE_MAPS`.
4. Choose the run list (sizes, depths, samples) once the pilot gives the real rate.
5. Decide the framing: a scaling and dequantization study at 1 to 3 repetitions, with one deep point at 5 or 6 repetitions where the cones are no longer trivially simulable.

## Sources

- Utro et al., Enhanced Prediction of CAR T-Cell Cytotoxicity with Quantum-Kernel Methods: https://arxiv.org/abs/2507.22710
- Hybrid Quantum-Classical Machine Learning Algorithms for Multi-Output Time-Series Forecasting at Utility Scale: https://arxiv.org/abs/2605.24252
- Agliardi et al., Mitigating exponential concentration in covariant quantum kernels: https://arxiv.org/abs/2412.07915
- IBM tutorial, Enhance feature classification using projected quantum kernels: https://quantum.cloud.ibm.com/docs/en/tutorials/projected-quantum-kernels
- IBM Quantum blog, Open Plan updates: https://www.ibm.com/quantum/blog/open-plan-updates
- Quantum Computing Report, IBM special promotion to Open Plan users: https://quantumcomputingreport.com/ibm-makes-special-promotion-to-open-plan-users/
- cBioPortal datahub (TCGA-BRCA, METABRIC): https://github.com/cBioPortal/datahub

Code, notebooks, results, and the LaTeX draft are in the project folder.
