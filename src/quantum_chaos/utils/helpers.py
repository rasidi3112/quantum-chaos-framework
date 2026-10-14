from itertools import combinations

import numpy as np
from scipy.linalg import expm

RngLike = None | int | np.random.Generator


def as_rng(seed: RngLike = None) -> np.random.Generator:
    return np.random.default_rng(seed)


def hermitian_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from scipy.linalg import eigh

    m = np.asarray(matrix)
    evals, evecs = eigh((m + m.conj().T) / 2, driver="evr")
    return evals, evecs


def hermitian_eigvalsh(matrix: np.ndarray) -> np.ndarray:
    from scipy.linalg import eigh

    m = np.asarray(matrix)
    evals: np.ndarray = eigh((m + m.conj().T) / 2, eigvals_only=True, driver="evr")
    return evals


def generate_random_coupling(
    n_majorana: int, variance: float = 1.0, seed: RngLike = None, antisymmetric: bool = True
) -> np.ndarray:
    rng = as_rng(seed)
    sigma = np.sqrt(variance * 6.0 / (n_majorana**3))
    J = np.zeros((n_majorana, n_majorana, n_majorana, n_majorana))

    if antisymmetric:
        for indices in combinations(range(n_majorana), 4):
            value = rng.normal(0, sigma)
            for perm in _signed_permutations(indices):
                idx, sign = perm
                J[idx] = sign * value
    else:
        J = rng.normal(0, sigma, (n_majorana, n_majorana, n_majorana, n_majorana))

    return J


def _signed_permutations(indices: tuple[int, ...]) -> list[tuple[tuple[int, ...], int]]:
    from itertools import permutations

    result = []
    index_list = list(indices)

    for perm in permutations(range(4)):
        permuted = tuple(index_list[p] for p in perm)
        sign = 1
        for i in range(4):
            for j in range(i + 1, 4):
                if perm[i] > perm[j]:
                    sign *= -1
        result.append((permuted, sign))

    return result


def compute_commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def compute_anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B + B @ A


def matrix_exponential(H: np.ndarray, t: float = 1.0) -> np.ndarray:
    return expm(-1j * H * t)


def partial_trace(rho: np.ndarray, keep: list[int], dims: list[int]) -> np.ndarray:
    dims = [int(d) for d in dims]
    n = len(dims)
    total = int(np.prod(dims))
    if rho.shape != (total, total):
        raise ValueError(f"rho has shape {rho.shape}, expected {(total, total)} for dims={dims}")
    keep = sorted(set(keep))
    if any(k < 0 or k >= n for k in keep):
        raise ValueError(f"keep={keep} out of range for {n} subsystems")
    if n > 26:
        raise ValueError("partial_trace supports at most 26 subsystems")

    row = [chr(ord("a") + i) for i in range(n)]
    col = [row[i] if i not in keep else chr(ord("A") + i) for i in range(n)]
    out = [row[i] for i in keep] + [col[i] for i in keep]
    spec = "".join(row) + "".join(col) + "->" + "".join(out)
    reduced = np.einsum(spec, rho.reshape(dims + dims))

    kept_dim = int(np.prod([dims[i] for i in keep]))
    return reduced.reshape(kept_dim, kept_dim)


def _clip_psd_spectrum(evals: np.ndarray) -> np.ndarray:
    tol = evals.size * np.finfo(float).eps * max(np.max(np.abs(evals)), 1e-300)
    return np.where(evals > tol, evals, 0.0)


def _psd_sqrt(rho: np.ndarray) -> np.ndarray:
    rho = (rho + rho.conj().T) / 2
    evals, evecs = np.linalg.eigh(rho)
    return (evecs * np.sqrt(_clip_psd_spectrum(evals))) @ evecs.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    sqrt_rho = _psd_sqrt(np.asarray(rho, dtype=complex))
    inner = sqrt_rho @ np.asarray(sigma, dtype=complex) @ sqrt_rho
    inner = (inner + inner.conj().T) / 2
    evals = _clip_psd_spectrum(np.linalg.eigvalsh(inner))
    return float(np.sum(np.sqrt(evals)) ** 2)


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    overlap = np.abs(np.vdot(psi, phi)) ** 2
    return float(overlap)


def entanglement_entropy(
    state: np.ndarray, subsystem_qubits: list[int], total_qubits: int
) -> float:
    rho = np.outer(state, state.conj())
    dims = [2] * total_qubits
    rho_A = partial_trace(rho, subsystem_qubits, dims)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)


def spectral_density(
    eigenvalues: np.ndarray, bins: int = 50, normalize: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(eigenvalues, dtype=float)
    if normalize:
        density, bin_edges = np.histogram(values, bins=int(bins), density=True)
    else:
        density, bin_edges = np.histogram(values, bins=int(bins), density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, density


def level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    from ..spectral.statistics import mean_spacing_ratio

    return mean_spacing_ratio(eigenvalues)
