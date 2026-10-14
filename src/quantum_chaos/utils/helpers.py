from itertools import combinations

import numpy as np
from scipy.linalg import expm

RngLike = None | int | np.random.Generator


def as_rng(seed: RngLike = None) -> np.random.Generator:
    """Return a ``numpy.random.Generator`` from None, an int seed or a Generator."""
    return np.random.default_rng(seed)


def hermitian_eigh(matrix: np.ndarray, eigvals_only: bool = False):
    """Eigen-decomposition of a dense Hermitian matrix with LAPACK ``*heevr``.

    The divide-and-conquer driver (``*heevd``, used by ``numpy.linalg.eigh``)
    of Apple Accelerate fails for complex matrices of size n >~ 3000
    ("illegal value in argument 10"; observed with NumPy 2.5.4 / SciPy 1.18.1
    on macOS); the MRRR driver does not.
    """
    from scipy.linalg import eigh

    matrix = np.asarray(matrix)
    return eigh((matrix + matrix.conj().T) / 2, eigvals_only=eigvals_only, driver="evr")


def generate_random_coupling(
    n_majorana: int, variance: float = 1.0, seed: RngLike = None, antisymmetric: bool = True
) -> np.ndarray:
    """Gaussian SYK couplings with <J_ijkl^2> = 3! * variance / N^3.

    ``seed`` may be None, an int or a ``numpy.random.Generator``; the global
    NumPy RNG is never touched.
    """
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
    indices = list(indices)

    for perm in permutations(range(4)):
        permuted = tuple(indices[p] for p in perm)
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
    """Reduced density matrix on the subsystems ``keep``.

    ``rho`` acts on the tensor product of subsystems with dimensions ``dims``
    (subsystem 0 is the most significant factor of the Kronecker product).
    The kept subsystems are returned in increasing index order.
    """
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

    # Row indices get letters a.., column indices A..; traced subsystems share a letter.
    row = [chr(ord("a") + i) for i in range(n)]
    col = [row[i] if i not in keep else chr(ord("A") + i) for i in range(n)]
    out = [row[i] for i in keep] + [col[i] for i in keep]
    spec = "".join(row) + "".join(col) + "->" + "".join(out)
    reduced = np.einsum(spec, rho.reshape(dims + dims))

    kept_dim = int(np.prod([dims[i] for i in keep]))
    return reduced.reshape(kept_dim, kept_dim)


def _clip_psd_spectrum(evals: np.ndarray) -> np.ndarray:
    """Zero out eigenvalues that are round-off noise of a PSD matrix.

    The square root amplifies noise (sqrt(1e-17) ~ 3e-9), so eigenvalues
    below ``dim * eps * max|lambda|`` are treated as exact zeros.
    """
    tol = evals.size * np.finfo(float).eps * max(np.max(np.abs(evals)), 1e-300)
    return np.where(evals > tol, evals, 0.0)


def _psd_sqrt(rho: np.ndarray) -> np.ndarray:
    """Square root of a Hermitian positive semi-definite matrix.

    Uses an eigendecomposition (stable for rank-deficient states, unlike
    ``scipy.linalg.sqrtm``).
    """
    rho = (rho + rho.conj().T) / 2
    evals, evecs = np.linalg.eigh(rho)
    return (evecs * np.sqrt(_clip_psd_spectrum(evals))) @ evecs.conj().T


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Uhlmann fidelity F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.

    Squared convention: F(|psi><psi|, |phi><phi|) = |<psi|phi>|^2.
    """
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
    density, bin_edges = np.histogram(eigenvalues, bins=bins, density=normalize)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, density


def level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    """Mean spacing ratio of one symmetry sector; see :mod:`quantum_chaos.spectral`.

    Exact degeneracies are kept (r = 0) and reported with a
    ``DegenerateSpectrumWarning``; ``nan`` for fewer than three levels.
    """
    from ..spectral.statistics import mean_spacing_ratio

    return mean_spacing_ratio(eigenvalues)
