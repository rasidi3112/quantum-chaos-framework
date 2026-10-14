from itertools import combinations
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from ..spectral.rmt import R_MEAN, ensemble_mean_r
from ..spectral.statistics import bootstrap_mean, spacing_ratios
from ..utils.helpers import (
    RngLike,
    generate_random_coupling,
    hermitian_eigh,
    hermitian_eigvalsh,
    spectral_density,
)
from ..utils.jordan_wigner import JordanWignerMapper

_CLASS_BY_N_MOD_8 = {0: "GOE", 2: "GUE", 4: "GSE", 6: "GUE"}


def syk_symmetry_class(n_majorana: int) -> str:
    if n_majorana % 2:
        raise ValueError("N must be even")
    return _CLASS_BY_N_MOD_8[n_majorana % 8]


def _popcount(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(a).astype(np.int64)
    count = np.zeros(a.shape, dtype=np.int64)
    while np.any(a):
        count += (a & np.uint64(1)).astype(np.int64)
        a = a >> np.uint64(1)
    return count


def majorana_pauli(index: int, n_qubits: int) -> tuple[int, int, complex]:
    site = index // 2

    def bit(q: int) -> int:
        return 1 << (n_qubits - 1 - q)

    z = 0
    for q in range(site):
        z |= bit(q)
    if index % 2 == 0:
        return bit(site), z, 1.0
    return bit(site), z | bit(site), 1j


def _pauli_product(a, b):
    xa, za, pa = a
    xb, zb, pb = b
    sign = -1.0 if bin(za & xb).count("1") % 2 else 1.0
    return xa ^ xb, za ^ zb, pa * pb * sign


def pauli_terms_to_sparse(terms, n_qubits: int) -> csr_matrix:
    dim = 2**n_qubits
    basis = np.arange(dim, dtype=np.uint64)
    by_x: dict[int, np.ndarray] = {}
    for x, z, c in terms:
        signs = 1.0 - 2.0 * (_popcount(basis & np.uint64(z)) % 2)
        by_x[x] = by_x.get(x, 0) + c * signs
    rows, cols, vals = [], [], []
    for x, column_values in by_x.items():
        rows.append((basis ^ np.uint64(x)).astype(np.int64))
        cols.append(basis.astype(np.int64))
        vals.append(np.asarray(column_values, dtype=complex))
    H = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)
    ).tocsr()
    H.eliminate_zeros()
    return H


class SYKHamiltonian:
    def __init__(
        self,
        n_majorana: int,
        coupling_strength: float = 1.0,
        seed: RngLike = None,
        use_sparse: bool = False,
        chi_squared: float = 1.0,
    ):
        if n_majorana < 4:
            raise ValueError("SYK model requires at least 4 Majorana fermions")
        if n_majorana % 2 != 0:
            raise ValueError("Number of Majorana fermions must be even")
        if chi_squared not in (1.0, 0.5):
            raise ValueError("chi_squared must be 1 (Pauli strings) or 1/2 (Maldacena-Stanford)")

        self.n_majorana = n_majorana
        self.n_qubits = n_majorana // 2
        self.coupling_strength = coupling_strength
        self.seed = seed
        self.use_sparse = use_sparse
        self.chi_squared = chi_squared
        self.symmetry_class = syk_symmetry_class(n_majorana)

        self._jw_mapper = JordanWignerMapper(self.n_qubits)
        self._couplings: np.ndarray | None = None
        self._sparse_hamiltonian: csr_matrix | None = None
        self._hamiltonian_matrix: np.ndarray | None = None
        self._eigenvalues: np.ndarray | None = None
        self._eigenvectors: np.ndarray | None = None
        self._sector_levels: dict[str, np.ndarray] = {}

    @property
    def couplings(self) -> np.ndarray:
        if self._couplings is None:
            self._couplings = generate_random_coupling(
                self.n_majorana,
                variance=self.coupling_strength**2,
                seed=self.seed,
                antisymmetric=True,
            )
        return self._couplings

    @property
    def sparse_hamiltonian(self) -> csr_matrix:
        if self._sparse_hamiltonian is None:
            self._sparse_hamiltonian = self._build_hamiltonian()
        return self._sparse_hamiltonian

    @property
    def hamiltonian_matrix(self):
        if self.use_sparse:
            return self.sparse_hamiltonian
        if self._hamiltonian_matrix is None:
            self._hamiltonian_matrix = self.sparse_hamiltonian.toarray()
        return self._hamiltonian_matrix

    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._diagonalize()
        assert self._eigenvalues is not None
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        if self._eigenvectors is None:
            self._diagonalize()
        assert self._eigenvectors is not None
        return self._eigenvectors

    def _build_hamiltonian(self) -> csr_matrix:
        n = self.n_qubits
        chis = [majorana_pauli(a, n) for a in range(self.n_majorana)]
        J = self.couplings
        scale = self.chi_squared**2
        terms = []
        for i, j, k, l in combinations(range(self.n_majorana), 4):
            x, z, phase = _pauli_product(
                _pauli_product(chis[i], chis[j]), _pauli_product(chis[k], chis[l])
            )
            terms.append((x, z, scale * J[i, j, k, l] * phase))
        return pauli_terms_to_sparse(terms, n)

    def _diagonalize(self, n_eigenvalues: int | None = None):
        if self.use_sparse and n_eigenvalues is not None:
            self._eigenvalues, self._eigenvectors = eigsh(
                self.sparse_hamiltonian, k=n_eigenvalues, which="SA"
            )
        else:
            self._eigenvalues, self._eigenvectors = hermitian_eigh(
                self.sparse_hamiltonian.toarray()
            )

    def get_ground_state(self) -> tuple[float, np.ndarray]:
        return self.eigenvalues[0], self.eigenvectors[:, 0]

    def get_spectral_density(self, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        return spectral_density(self.eigenvalues, bins=bins)

    def get_level_spacing_ratio(self, trim: float = 0.0) -> float:
        r = self.spacing_ratios(trim=trim)
        return float(np.mean(r)) if r.size else float("nan")

    def get_thermal_state(self, temperature: float) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        beta = 1.0 / temperature
        boltzmann_factors = np.exp(-beta * (self.eigenvalues - self.eigenvalues[0]))
        Z = np.sum(boltzmann_factors)
        thermal_diag = np.diag(boltzmann_factors / Z)
        rho = self.eigenvectors @ thermal_diag @ self.eigenvectors.conj().T
        return rho

    def time_evolution_operator(self, t: float) -> np.ndarray:
        phases = np.exp(-1j * self.eigenvalues * t)
        U = self.eigenvectors @ np.diag(phases) @ self.eigenvectors.conj().T
        return U

    def evolve_state(self, initial_state: np.ndarray, t: float) -> np.ndarray:
        U = self.time_evolution_operator(t)
        return U @ initial_state

    def get_majorana_operator(self, index: int) -> np.ndarray:
        return self._jw_mapper.majorana_operator(index)

    def parity_sector_indices(self, sector: str = "even") -> np.ndarray:
        if sector not in ("even", "odd"):
            raise ValueError("sector must be 'even' or 'odd'")
        basis = np.arange(2**self.n_qubits, dtype=np.uint64)
        parity = _popcount(basis) % 2
        return np.flatnonzero(parity == (0 if sector == "even" else 1))

    def parity_projection(self, sector: str = "even") -> np.ndarray:
        idx = self.parity_sector_indices(sector)
        return self.sparse_hamiltonian[idx][:, idx].toarray()

    def sector_eigenvalues(self, sector: str = "even") -> np.ndarray:
        if sector not in self._sector_levels:
            self._sector_levels[sector] = hermitian_eigvalsh(self.parity_projection(sector))
        return self._sector_levels[sector]

    def particle_hole_unitary(self) -> np.ndarray:
        n = self.n_qubits
        x, z, phase = 0, 0, 1.0 + 0j
        for j in range(n):
            x, z, phase = _pauli_product((x, z, phase), majorana_pauli(2 * j, n))
        return pauli_terms_to_sparse([(x, z, phase)], n).toarray()

    def _sectors_for_statistics(self) -> list[tuple[str, bool]]:
        if self.symmetry_class == "GUE":
            return [("even", False)]
        kramers = self.symmetry_class == "GSE"
        return [("even", kramers), ("odd", kramers)]

    def spacing_ratios(self, trim: float = 0.0) -> np.ndarray:
        return np.concatenate(
            [
                spacing_ratios(self.sector_eigenvalues(sector), kramers=kramers, trim=trim)
                for sector, kramers in self._sectors_for_statistics()
            ]
        )

    def get_level_spacing_ratio_parity(self, sector: str = "even", trim: float = 0.0) -> float:
        r = spacing_ratios(
            self.sector_eigenvalues(sector), kramers=self.symmetry_class == "GSE", trim=trim
        )
        return float(np.mean(r)) if r.size else float("nan")

    def summary(self) -> dict:
        return {
            "n_majorana": self.n_majorana,
            "n_qubits": self.n_qubits,
            "hilbert_dim": 2**self.n_qubits,
            "coupling_strength": self.coupling_strength,
            "chi_squared": self.chi_squared,
            "ground_state_energy": float(self.eigenvalues[0]),
            "spectral_width": float(self.eigenvalues[-1] - self.eigenvalues[0]),
            "symmetry_class": self.symmetry_class,
            "rmt_reference": R_MEAN[self.symmetry_class.lower()],
            "level_spacing_ratio": self.get_level_spacing_ratio(),
            "level_spacing_ratio_even": self.get_level_spacing_ratio_parity("even"),
        }

    def __repr__(self) -> str:
        return (
            f"SYKHamiltonian(N={self.n_majorana}, n_qubits={self.n_qubits}, "
            f"J={self.coupling_strength})"
        )


def sector_level_count(n_majorana: int) -> int:
    dim = 2 ** (n_majorana // 2 - 1)
    return dim // 2 if syk_symmetry_class(n_majorana) == "GSE" else dim


def disorder_average_lsr(
    n_majorana: int,
    n_realizations: int = 20,
    seed_start: int = 0,
    trim: float = 0.0,
    coupling_strength: float = 1.0,
    rng: RngLike = None,
) -> dict:
    per_realization = np.array(
        [
            SYKHamiltonian(
                n_majorana, coupling_strength=coupling_strength, seed=seed
            ).get_level_spacing_ratio(trim=trim)
            for seed in range(seed_start, seed_start + n_realizations)
        ]
    )
    stats: dict[str, Any] = bootstrap_mean(per_realization, rng=rng)
    cls = syk_symmetry_class(n_majorana)
    stats.update(
        {
            "symmetry_class": cls,
            "reference": R_MEAN[cls.lower()],
            "levels_per_sector": sector_level_count(n_majorana),
            "trim": trim,
            "per_realization": per_realization,
        }
    )
    return stats


def syk_rmt_reference(
    n_majorana: int, n_samples: int = 500, trim: float = 0.0, rng: RngLike = None
) -> dict:
    cls = syk_symmetry_class(n_majorana)
    res: dict[str, Any] = ensemble_mean_r(
        cls, sector_level_count(n_majorana), n_samples, trim=trim, rng=rng
    )
    res["symmetry_class"] = cls
    return res


def syk2_hamiltonian(
    n_majorana: int, seed: RngLike = None, coupling_strength: float = 1.0, chi_squared: float = 1.0
) -> csr_matrix:
    rng = np.random.default_rng(seed)
    n = n_majorana // 2
    chis = [majorana_pauli(a, n) for a in range(n_majorana)]
    sigma = coupling_strength / np.sqrt(n_majorana)
    terms = []
    for i, j in combinations(range(n_majorana), 2):
        x, z, phase = _pauli_product(chis[i], chis[j])
        terms.append((x, z, 1j * rng.normal(0, sigma) * chi_squared * phase))
    return pauli_terms_to_sparse(terms, n)
