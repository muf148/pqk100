"""Where circuits run. One place to change the device or the saved account."""
from __future__ import annotations

ACCOUNT = "Faryad-free-ibm"      # name of the saved IBM Quantum account
DEVICE = "ibm_fez"               # 156-qubit Heron r2


def local_backend():
    """FakeFez: the ibm_fez lattice and a calibration snapshot, simulated on the laptop."""
    from qiskit_ibm_runtime.fake_provider import FakeFez
    return FakeFez()


def ibm_backend(account: str = ACCOUNT, device: str = DEVICE):
    """The real device, through the saved account. Returns (service, backend)."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(name=account)
    return service, service.backend(device)


def median_cz_error(backend) -> float:
    import numpy as np
    return float(np.median([p.error for p in backend.target["cz"].values() if p is not None and p.error is not None]))
