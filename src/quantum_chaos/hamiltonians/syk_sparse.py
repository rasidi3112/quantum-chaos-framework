from itertools import combinations
from math import comb

import numpy as np
from scipy.sparse import csc_matrix

from ..utils.helpers import RngLike
from .syk_hamiltonian import _pauli_product, _popcount, majorana_pauli


def syk_couplings(
    n_majorana: int, seed: RngLike = None, coupling_strength: float = 1.0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(coupling_strength**2 * 6.0 / n_majorana**3)
    return np.array([rng.normal(0, sigma) for _ in range(comb(n_majorana, 4))])


def syk_terms_by_mask(
    n_majorana: int, couplings: np.ndarray, chi_squared: float = 1.0
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    n = n_majorana // 2
    chis = [majorana_pauli(a, n) for a in range(n_majorana)]
    grouped: dict[int, tuple[list[int], list[complex]]] = {}
    for J, (i, j, k, l) in zip(couplings, combinations(range(n_majorana), 4), strict=True):
        if J == 0:
            continue
        x, z, phase = _pauli_product(
            _pauli_product(chis[i], chis[j]), _pauli_product(chis[k], chis[l])
        )
        zs, cs = grouped.setdefault(x, ([], []))
        zs.append(z)
        cs.append(chi_squared**2 * J * phase)
    return {
        x: (np.array(zs, dtype=np.uint64), np.array(cs, dtype=complex))
        for x, (zs, cs) in grouped.items()
    }


def parity_sector_states(n_qubits: int, sector: str = "even") -> np.ndarray:
    basis = np.arange(2**n_qubits, dtype=np.uint64)
    parity = _popcount(basis) % 2
    return basis[parity == (0 if sector == "even" else 1)]


def sparse_syk_couplings(
    n_majorana: int, k: float, seed: RngLike = None, coupling_strength: float = 1.0
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_terms = comb(n_majorana, 4)
    p = min(1.0, k * n_majorana / n_terms)
    sigma = np.sqrt(coupling_strength**2 * 6.0 / n_majorana**3 / p)
    keep = rng.random(n_terms) < p
    return np.where(keep, rng.normal(0, sigma, n_terms), 0.0)


def syk_sector_hamiltonian(
    n_majorana: int,
    seed: RngLike = None,
    sector: str = "even",
    coupling_strength: float = 1.0,
    chi_squared: float = 1.0,
    dtype: type = np.complex128,
    couplings: np.ndarray | None = None,
) -> csc_matrix:
    n = n_majorana // 2
    if couplings is None:
        couplings = syk_couplings(n_majorana, seed, coupling_strength)
    groups = syk_terms_by_mask(n_majorana, couplings, chi_squared)
    states = parity_sector_states(n, sector)
    dim = states.size
    lookup = np.full(2**n, -1, dtype=np.int64)
    lookup[states.astype(np.int64)] = np.arange(dim)
    masks = sorted(groups)
    indices = np.empty((dim, len(masks)), dtype=np.int32)
    data: np.ndarray = np.empty((dim, len(masks)), dtype=dtype)
    for col, x in enumerate(masks):
        zs, cs = groups[x]
        signs = 1.0 - 2.0 * (_popcount(states[None, :] & zs[:, None]) % 2)
        data[:, col] = cs @ signs
        indices[:, col] = lookup[(states ^ np.uint64(x)).astype(np.int64)]
    indptr = np.arange(0, dim * len(masks) + 1, len(masks), dtype=np.int64)
    return csc_matrix((data.ravel(), indices.ravel(), indptr), shape=(dim, dim))


def n_flip_masks(n_majorana: int) -> int:
    n = n_majorana // 2
    return comb(n, 4) + comb(n, 2) + 1


def sector_memory_bytes(n_majorana: int, itemsize: int = 16) -> int:
    return 2 ** (n_majorana // 2 - 1) * n_flip_masks(n_majorana) * (itemsize + 4)
