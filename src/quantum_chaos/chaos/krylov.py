import numpy as np
from scipy.linalg import expm

from ..utils.helpers import hermitian_eigh


def _liouvillian_frame(hamiltonian, operator):
    H = hamiltonian.toarray() if hasattr(hamiltonian, "toarray") else np.asarray(hamiltonian)
    E, V = hermitian_eigh(H)
    O = V.conj().T @ np.asarray(operator) @ V
    return np.subtract.outer(E, E), O


def lanczos_coefficients(
    hamiltonian, operator, n_max: int = 50, tol: float = 1e-4, reorthogonalize: bool = True
) -> np.ndarray:
    omega, O = _liouvillian_frame(hamiltonian, operator)
    dim = O.shape[0]

    def inner(a, b):
        return np.vdot(a, b) / dim

    basis = [O / np.sqrt(inner(O, O).real)]
    b: list[float] = []
    previous = np.zeros_like(O)
    b_prev = 0.0
    for _ in range(n_max):
        A = omega * basis[-1] - b_prev * previous
        if reorthogonalize:
            for _ in range(2):
                for vec in basis:
                    A = A - inner(vec, A) * vec
        norm = np.sqrt(max(inner(A, A).real, 0.0))
        scale = b[0] if b else np.sqrt(max(inner(omega * basis[0], omega * basis[0]).real, 1e-300))
        if norm < tol * scale:
            break
        b.append(norm)
        previous, b_prev = basis[-1], norm
        basis.append(A / norm)
    return np.array(b)


def krylov_wavefunction(b: np.ndarray, times: np.ndarray) -> np.ndarray:
    n = len(b) + 1
    M = np.zeros((n, n))
    for k, bk in enumerate(b, start=1):
        M[k, k - 1] = bk
        M[k - 1, k] = -bk
    e0 = np.zeros(n)
    e0[0] = 1.0
    return np.array([expm(M * t) @ e0 for t in np.asarray(times, dtype=float)])


def krylov_complexity(b: np.ndarray, times: np.ndarray) -> np.ndarray:
    phi = krylov_wavefunction(b, times)
    return (phi**2) @ np.arange(phi.shape[1])


def autocorrelation(hamiltonian, operator, times: np.ndarray) -> np.ndarray:
    omega, O = _liouvillian_frame(hamiltonian, operator)
    weights = np.abs(O) ** 2
    weights = weights / weights.sum()
    return np.array([np.sum(weights * np.exp(1j * omega * t)) for t in np.asarray(times)])
