from math import comb

import numpy as np
from scipy.sparse import spmatrix


def syk_eta(n_majorana: int, q: int = 4) -> float:
    total = sum((-1) ** r * comb(q, r) * comb(n_majorana - q, q - r) for r in range(q + 1))
    return total / comb(n_majorana, q)


def syk_kurtosis_exact(n_majorana: int, q: int = 4) -> float:
    return syk_eta(n_majorana, q) - 1.0


def syk_kurtosis_garcia_garcia_verbaarschot(n_majorana: int) -> float:
    n = n_majorana
    return -32 * (n - 4) * (n * n - 11 * n + 36) / (n * (n - 1) * (n - 2) * (n - 3))


def syk_second_moment(
    n_majorana: int, coupling_strength: float = 1.0, chi_squared: float = 1.0, q: int = 4
) -> float:
    variance = coupling_strength**2 * 6.0 / n_majorana**3
    return comb(n_majorana, q) * variance * chi_squared**q


def normalized_moments(hamiltonian: np.ndarray | spmatrix) -> tuple[float, float]:
    H = hamiltonian
    dim = H.shape[0]
    H2 = H @ H
    m2 = float(np.real(H2.diagonal().sum())) / dim
    if isinstance(H2, np.ndarray):
        m4 = float(np.real(np.sum(H2 * H2.T))) / dim
    else:
        m4 = float(np.real(H2.multiply(H2.T).sum())) / dim
    return m2, m4
