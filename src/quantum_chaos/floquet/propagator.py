from collections.abc import Callable

import numpy as np
from scipy.linalg import expm

GAUSS_C = (0.5 - np.sqrt(3) / 6, 0.5 + np.sqrt(3) / 6)


def floquet_operator(
    hamiltonian: Callable[[float], np.ndarray],
    period: float,
    n_steps: int = 100,
    method: str = "magnus4",
) -> np.ndarray:
    dt = period / n_steps
    dim = np.asarray(hamiltonian(0.0)).shape[0]
    F = np.eye(dim, dtype=complex)
    for k in range(n_steps):
        t = k * dt
        if method == "midpoint":
            step = expm(-1j * dt * np.asarray(hamiltonian(t + dt / 2)))
        elif method == "magnus4":
            A1 = np.asarray(hamiltonian(t + GAUSS_C[0] * dt))
            A2 = np.asarray(hamiltonian(t + GAUSS_C[1] * dt))
            omega = -1j * dt / 2 * (A1 + A2) - (np.sqrt(3) / 12) * dt**2 * (A2 @ A1 - A1 @ A2)
            step = expm(omega)
        else:
            raise ValueError(f"unknown method {method!r}")
        F = step @ F
    return F


def quasi_energies(floquet: np.ndarray, period: float) -> np.ndarray:
    return np.sort(-np.angle(np.linalg.eigvals(floquet)) / period)


def fold(energies: np.ndarray, omega: float) -> np.ndarray:
    return np.sort(((np.asarray(energies) + omega / 2) % omega) - omega / 2)


def quasi_energy_distance(a: np.ndarray, b: np.ndarray, omega: float) -> float:
    a = np.sort(fold(a, omega))
    b = np.sort(fold(b, omega))
    d = np.abs(a - b)
    return float(np.max(np.minimum(d, omega - d)))
