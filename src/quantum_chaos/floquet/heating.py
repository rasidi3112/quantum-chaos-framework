from dataclasses import dataclass

import numpy as np

from ..hamiltonians.hubbard_ed import HubbardRingED
from ..utils.helpers import hermitian_eigh
from .propagator import floquet_operator


@dataclass(frozen=True)
class DrivenRingBlock:
    nn_hopping: np.ndarray
    nnn_hopping: np.ndarray
    double_occupancy: np.ndarray

    def static(self, J0: float, tp: float, U: float) -> np.ndarray:
        return J0 * self.nn_hopping + tp * self.nnn_hopping + U * self.double_occupancy


def driven_ring_block(
    n_sites: int, n_up: int, n_down: int, momentum: int = 0, with_nnn: bool = True
) -> DrivenRingBlock:
    ed = HubbardRingED(n_sites, n_up, n_down, t=1.0, tp=0.0, U=0.0)
    V = ed.momentum_basis(momentum)
    Vh = V.conj().T

    def project(op):
        return (Vh @ op @ V).toarray()

    nnn = ed.hopping_operator(2) if with_nnn else 0 * ed.hopping_operator(1)
    return DrivenRingBlock(
        project(ed.hopping_operator(1)), project(nnn), project(ed.double_occupancy())
    )


def stroboscopic_energies(
    floquet: np.ndarray, initial: np.ndarray, observable: np.ndarray, n_periods: int
) -> np.ndarray:
    E, V = hermitian_eigh(np.asarray(observable))
    ev, evec = np.linalg.eig(floquet)
    coeffs = np.linalg.solve(evec, initial)
    out = np.empty(n_periods + 1)
    for n in range(n_periods + 1):
        psi = evec @ (coeffs * ev**n)
        out[n] = np.real(np.vdot(psi, observable @ psi))
    return out


def heating_time(
    energies: np.ndarray, e_initial: float, e_infinite: float, threshold: float = 0.5
) -> float:
    q = (np.asarray(energies) - e_initial) / (e_infinite - e_initial)
    above = np.flatnonzero(q >= threshold)
    return float(above[0]) if above.size else float("inf")


def heating_scan(
    block: DrivenRingBlock,
    omegas,
    J0: float = 1.0,
    J1: float = 1.0,
    tp: float = 0.0,
    U: float = 4.0,
    n_periods: int = 1000,
    n_steps: int = 64,
) -> dict:
    H0 = block.static(J0, tp, U)
    E, V = hermitian_eigh(H0)
    psi0 = V[:, 0]
    e_inf = float(np.real(np.trace(H0)) / H0.shape[0])
    rows = {}
    for omega in omegas:
        T = 2 * np.pi / omega

        def drive(t: float, w: float = float(omega)) -> np.ndarray:
            return H0 + J1 * np.cos(w * t) * block.nn_hopping

        F = floquet_operator(drive, T, n_steps=n_steps)
        energies = stroboscopic_energies(F, psi0, H0, n_periods)
        rows[float(omega)] = {
            "energies": energies,
            "heating_time": heating_time(energies, E[0], e_inf),
            "late_energy_fraction": float(
                (np.mean(energies[-n_periods // 10 :]) - E[0]) / (e_inf - E[0])
            ),
            "time_averaged_fraction": float(
                (np.mean(energies[n_periods // 10 :]) - E[0]) / (e_inf - E[0])
            ),
        }
    return {"e_ground": float(E[0]), "e_infinite": e_inf, "dim": H0.shape[0], "omegas": rows}
