# Credits form numbers (2026-10-18)

Rate used: 284 us per shot + 11.4 s per job (measured on ibm_fez, 2026-10-18).

| Block | What it tests | Configurations | Tumors | Shots | QPU hours |
| --- | --- | --- | --- | --- | --- |
| A. Scaling | 20, 40, 60, 80, 100, 127, 156 qubits at 1 and 2 layers | 14 | 1,000 | 1,024 | 3.8 |
| B. Deep and strong regime | 156 qubits at 4, 5, 6 layers; 2 layers at bandwidth 0.5 and 1.0 | 5 | 1,000 | 1,024 | 1.4 |
| C. Shot noise | 156 qubits at 1 and 2 layers | 2 | 300 | 4,096 | 0.6 |
| D. NAS feature map | 20, 100, 156 qubits | 3 | 1,000 | 1,024 | 0.8 |
| E. Reproducibility | 156 qubits, 1 layer, second Heron device | 1 | 1,000 | 1,024 | 0.3 |
| Contingency | 12% of the blocks |  |  |  | 0.8 |
| Total |  | 25 |  |  | 7.7 |

## Form fields

- Were you able to determine feasibility via the Open Plan: Yes. Pilot on ibm_fez: 156 qubits, 284 µs billed per shot, 89% of the exact signal retained on average across qubits.
- Number of physical qubits to date: 156 in simulation on the ibm_fez lattice; 156 on ibm_fez hardware (pilot)
- Target number of physical qubits: 156
- Number of 2-qubit gates (after transpilation): 352 CZ per layer at 156 qubits. Main study: 352 to 704. Deep stress test: up to 2,112.
- Estimate the quantum compute time: 8 hours
