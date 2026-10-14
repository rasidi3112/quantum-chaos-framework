from collections.abc import Sequence

import numpy as np

from ..utils.helpers import hermitian_eigh


def eigenbasis_matrix_elements(hamiltonian, operator) -> tuple[np.ndarray, np.ndarray]:
    H = hamiltonian.toarray() if hasattr(hamiltonian, "toarray") else np.asarray(hamiltonian)
    E, V = hermitian_eigh(H)
    return E, V.conj().T @ np.asarray(operator) @ V


def _central(E: np.ndarray, fraction: float) -> np.ndarray:
    n = E.size
    cut = int(np.floor((1 - fraction) / 2 * n))
    return np.arange(cut, n - cut)


def diagonal_fluctuation(
    E: np.ndarray, O: np.ndarray, fraction: float = 0.5, degree: int = 3
) -> float:
    idx = _central(E, fraction)
    if idx.size <= degree + 2:
        return float("nan")
    diag = np.real(np.diag(O))[idx]
    x = (E[idx] - E[idx].mean()) / max(E[idx].std(), 1e-300)
    smooth = np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyfit(x, diag, degree))
    return float(np.std(diag - smooth, ddof=degree + 1))


def offdiagonal_moment_ratio(
    E: np.ndarray, O: np.ndarray, fraction: float = 0.5, omega_max: float | None = None
) -> float:
    idx = _central(E, fraction)
    block = O[np.ix_(idx, idx)]
    omega = np.abs(np.subtract.outer(E[idx], E[idx]))
    mask = ~np.eye(idx.size, dtype=bool)
    if omega_max is not None:
        mask &= omega <= omega_max
    a2 = np.abs(block[mask]) ** 2
    return float(np.mean(a2**2) / np.mean(a2) ** 2)


def page_entropy(dim_a: int, dim_b: int) -> float:
    m, n = sorted((dim_a, dim_b))
    k = np.arange(n + 1, m * n + 1)
    return float(np.sum(1.0 / k) - (m - 1) / (2 * n))


def entanglement_entropy_bipartite(state: np.ndarray, dim_a: int) -> float:
    psi = np.asarray(state).reshape(dim_a, -1)
    s = np.linalg.svd(psi, compute_uv=False) ** 2
    s = s[s > 1e-15]
    return float(-np.sum(s * np.log(s)))


def eigenstate_entanglement(
    eigenvectors: np.ndarray,
    n_qubits: int,
    subsystem_sizes: Sequence[int],
    indices: Sequence[int],
) -> dict[int, np.ndarray]:
    return {
        l: np.array([entanglement_entropy_bipartite(eigenvectors[:, k], 2**l) for k in indices])
        for l in subsystem_sizes
    }
