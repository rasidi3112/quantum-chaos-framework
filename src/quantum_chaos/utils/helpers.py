import numpy as np
from typing import Tuple, Optional, List
from scipy.linalg import expm
from itertools import combinations


def generate_random_coupling(
    n_majorana: int,
    variance: float = 1.0,
    seed: Optional[int] = None,
    antisymmetric: bool = True
) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    
    sigma = np.sqrt(variance * 6.0 / (n_majorana ** 3))
    J = np.zeros((n_majorana, n_majorana, n_majorana, n_majorana))
    
    if antisymmetric:
        for indices in combinations(range(n_majorana), 4):
            value = np.random.normal(0, sigma)
            for perm in _signed_permutations(indices):
                idx, sign = perm
                J[idx] = sign * value
    else:
        J = np.random.normal(0, sigma, (n_majorana, n_majorana, n_majorana, n_majorana))
    
    return J


def _signed_permutations(indices: Tuple[int, ...]) -> List[Tuple[Tuple[int, ...], int]]:
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


def partial_trace(rho: np.ndarray, keep: List[int], dims: List[int]) -> np.ndarray:
    n_subsystems = len(dims)
    keep = sorted(keep)
    trace_out = [i for i in range(n_subsystems) if i not in keep]
    
    shape = dims + dims
    rho_reshaped = rho.reshape(shape)
    
    for i in sorted(trace_out, reverse=True):
        n_remaining = len(dims) - len(trace_out) + len([t for t in trace_out if t > i])
        rho_reshaped = np.trace(rho_reshaped, axis1=i, axis2=i + n_remaining)
        shape = [d for j, d in enumerate(shape) if j != i and j != i + n_remaining]
    
    kept_dim = int(np.prod([dims[i] for i in keep]))
    return rho_reshaped.reshape(kept_dim, kept_dim)


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    from scipy.linalg import sqrtm
    
    sqrt_rho = sqrtm(rho)
    prod = sqrt_rho @ sigma @ sqrt_rho
    sqrt_prod = sqrtm(prod)
    
    return np.real(np.trace(sqrt_prod)) ** 2


def state_fidelity(psi: np.ndarray, phi: np.ndarray) -> float:
    overlap = np.abs(np.vdot(psi, phi)) ** 2
    return float(overlap)


def entanglement_entropy(state: np.ndarray, subsystem_qubits: List[int], total_qubits: int) -> float:
    rho = np.outer(state, state.conj())
    dims = [2] * total_qubits
    rho_A = partial_trace(rho, subsystem_qubits, dims)
    eigenvalues = np.linalg.eigvalsh(rho_A)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    entropy = -np.sum(eigenvalues * np.log2(eigenvalues))
    return float(entropy)


def spectral_density(eigenvalues: np.ndarray, bins: int = 50, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    density, bin_edges = np.histogram(eigenvalues, bins=bins, density=normalize)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, density


def level_spacing_ratio(eigenvalues: np.ndarray) -> float:
    sorted_eigs = np.sort(eigenvalues)
    spacings = np.diff(sorted_eigs)
    spacings = spacings[spacings > 1e-12]
    
    if len(spacings) < 2:
        return 0.0
    
    ratios = []
    for i in range(len(spacings) - 1):
        r = min(spacings[i], spacings[i+1]) / max(spacings[i], spacings[i+1])
        ratios.append(r)
    
    return float(np.mean(ratios))
