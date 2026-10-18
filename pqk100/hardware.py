"""Hardware helpers: one parameterized ISA circuit, all samples bound as a parameter array.

Notebook 4 shows these steps cell by cell. The functions here do the same thing and are
what tests/test_hardware_path.py exercises.

The feature map is transpiled once onto the chosen patch. Each sample is a row of angles,
so a run of N samples costs N x 3 measurement bases x shots, with no per-sample transpile.
Mitigation is readout only (TREX, resilience_level=1), as in the plan. No ZNE.

Nothing here submits a job unless you call `submit`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager

from .featuremap import to_circuit


def build(n, ops, phys, backend, optimization_level=1):
    """Return (isa circuit, parameter column order, observables with shape (3n, 1))."""
    qc, x = to_circuit(n, ops)
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                      initial_layout=list(phys), seed_transpiler=0)
    isa = pm.run(qc)
    final = isa.layout.final_index_layout()
    if list(final) != list(phys):
        raise RuntimeError("routing moved qubits; the patch edges are not native couplers")
    cols = [p.index for p in isa.parameters]
    obs = [[SparsePauliOp.from_sparse_list([(P, [k], 1.0)], num_qubits=n).apply_layout(isa.layout)]
           for P in "XYZ" for k in range(n)]
    return isa, cols, obs


def circuit_report(isa, backend):
    """Gate counts, two-qubit depth, and duration of one shot."""
    two = lambda inst: inst.operation.num_qubits == 2
    t = backend.target
    dur = isa.estimate_duration(t)
    meas = float(np.median([p.duration for p in t["measure"].values() if p is not None and p.duration]))
    cfg = backend.configuration()
    rep = getattr(cfg, "default_rep_delay", None) or 250e-6
    return {"qubits": int(len(isa.layout.final_index_layout())), "twoq_gates": int(isa.size(two)),
            "twoq_depth": int(isa.depth(two)), "circuit_s": float(dur), "measure_s": meas,
            "rep_delay_s": float(rep), "shot_s": float(dur + meas + rep)}


def estimate_minutes(report, n_samples, shots, bases=3, overhead=1.5):
    """Rough QPU minutes. `overhead` covers readout calibration and per-circuit latency.
    Treat this as a planning number and check it against a small pilot job."""
    return n_samples * bases * shots * report["shot_s"] * overhead / 60.0


def make_estimator(backend, shots=4096, dynamical_decoupling=False, rep_delay=None):
    """EstimatorV2 with readout mitigation only: TREX on, ZNE off, no gate twirling."""
    from qiskit_ibm_runtime import EstimatorV2
    est = EstimatorV2(mode=backend)
    o = est.options
    o.default_shots = shots
    o.resilience_level = 1
    o.resilience.measure_mitigation = True
    o.resilience.zne_mitigation = False
    o.twirling.enable_measure = True
    o.twirling.enable_gates = False
    o.dynamical_decoupling.enable = dynamical_decoupling
    if rep_delay is not None:               # seconds; shorter means cheaper shots, check reset quality first
        o.execution.rep_delay = rep_delay
    return est


def submit(isa, cols, obs, angles, backend, shots=4096, chunk=100, outdir="results/jobs", tag="run",
           dynamical_decoupling=False, rep_delay=None):
    """Submit all samples in chunks. Job ids are written to disk before anything is awaited."""
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    est = make_estimator(backend, shots, dynamical_decoupling, rep_delay)
    angles = np.asarray(angles)[:, cols]
    jobs, meta = [], {"tag": tag, "backend": backend.name, "shots": shots, "n_samples": len(angles),
                      "created": time.strftime("%Y-%m-%d %H:%M:%S"), "jobs": []}
    for s in range(0, len(angles), chunk):
        job = est.run([(isa, obs, angles[s:s + chunk])])
        jobs.append(job)
        meta["jobs"].append({"id": job.job_id(), "start": s, "stop": min(s + chunk, len(angles))})
        (out / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    return jobs


def collect(jobs, n):
    """Stack job results into projections with shape (N, 3, n), plus standard errors."""
    P, S = [], []
    for job in jobs:
        d = job.result()[0].data
        P.append(np.asarray(d.evs).reshape(3, n, -1).transpose(2, 0, 1))
        S.append(np.asarray(d.stds).reshape(3, n, -1).transpose(2, 0, 1))
    return np.concatenate(P), np.concatenate(S)


def collect_saved(path, service, n):
    """Same as `collect`, from the job-id file written by `submit` (for a later session)."""
    meta = json.loads(Path(path).read_text())
    return collect([service.job(j["id"]) for j in meta["jobs"]], n)
