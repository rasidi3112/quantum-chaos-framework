"""Sachdev-Ye-Kitaev model of N Majorana fermions with q = 4.

Conventions
-----------
H = sum_{i<j<k<l} J_ijkl chi_i chi_j chi_k chi_l, with independent Gaussian
J_ijkl of variance 3! J^2 / N^3 (Maldacena & Stanford, PRD 94, 106002 (2016);
Cotler et al., JHEP 05 (2017) 118). The Majoranas are realised by the
Jordan-Wigner strings chi_{2j} = Z_0...Z_{j-1} X_j and
chi_{2j+1} = Z_0...Z_{j-1} Y_j, i.e. chi^2 = 1 ({chi_a, chi_b} = 2 delta_ab).
The references above use {chi_a, chi_b} = delta_ab (chi^2 = 1/2); with the same
couplings their Hamiltonian is H/4. Pass ``chi_squared=0.5`` to work in that
normalisation.

Symmetry classes (Cotler et al. 2017, sec. 3.1): the fermion parity
Gamma = prod_j Z_j splits the spectrum into two sectors, and the antiunitary
particle-hole operator P gives, per sector, GOE for N mod 8 = 0 (P^2 = +1),
GSE for N mod 8 = 4 (P^2 = -1, Kramers pairs inside each sector) and GUE for
N mod 8 = 2, 6 (P exchanges the sectors, whose spectra are then identical).
"""

from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from ..spectral.rmt import R_MEAN, ensemble_mean_r
from ..spectral.statistics import bootstrap_mean, spacing_ratios
from ..utils.helpers import RngLike, generate_random_coupling, hermitian_eigh, spectral_density
from ..utils.jordan_wigner import JordanWignerMapper

_CLASS_BY_N_MOD_8 = {0: "GOE", 2: "GUE", 4: "GSE", 6: "GUE"}


def syk_symmetry_class(n_majorana: int) -> str:
    """Random-matrix class of one parity sector of the q=4 SYK model (N even)."""
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
    """(x_mask, z_mask, phase) with chi_index = phase * X^x Z^z.

    Bit (n_qubits - 1 - q) of a mask refers to qubit q, so masks act directly
    on computational-basis indices in Kronecker order (qubit 0 most significant).
    """
    site = index // 2

    def bit(q: int) -> int:
        return 1 << (n_qubits - 1 - q)

    z = 0
    for q in range(site):
        z |= bit(q)
    if index % 2 == 0:
        return bit(site), z, 1.0
    return bit(site), z | bit(site), 1j  # Y = i X Z


def _pauli_product(a, b):
    xa, za, pa = a
    xb, zb, pb = b
    sign = -1.0 if bin(za & xb).count("1") % 2 else 1.0
    return xa ^ xb, za ^ zb, pa * pb * sign


def pauli_terms_to_sparse(terms, n_qubits: int) -> csr_matrix:
    """Sparse matrix of sum_k c_k * X^{x_k} Z^{z_k} from (x, z, c) triples."""
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
    """
    SYK (Sachdev-Ye-Kitaev) model, q = 4, built as a sparse matrix from
    Pauli strings (see module docstring for conventions and symmetry classes).
    """

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
        self._couplings = None
        self._sparse_hamiltonian = None
        self._hamiltonian_matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
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
        """Dense matrix (or CSR if ``use_sparse``)."""
        if self.use_sparse:
            return self.sparse_hamiltonian
        if self._hamiltonian_matrix is None:
            self._hamiltonian_matrix = self.sparse_hamiltonian.toarray()
        return self._hamiltonian_matrix

    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._diagonalize()
        return self._eigenvalues

    @property
    def eigenvectors(self) -> np.ndarray:
        if self._eigenvectors is None:
            self._diagonalize()
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
        """Sector-resolved mean spacing ratio (see :meth:`spacing_ratios`)."""
        r = self.spacing_ratios(trim=trim)
        return float(np.mean(r)) if r.size else float("nan")

    def get_thermal_state(self, temperature: float) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")

        beta = 1.0 / temperature
        # shift by E_0 so that the largest weight is exp(0) = 1 (no overflow)
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
        """Basis indices with fermion parity Gamma = prod_j Z_j = +1 ('even') or -1 ('odd')."""
        if sector not in ("even", "odd"):
            raise ValueError("sector must be 'even' or 'odd'")
        basis = np.arange(2**self.n_qubits, dtype=np.uint64)
        parity = _popcount(basis) % 2
        return np.flatnonzero(parity == (0 if sector == "even" else 1))

    def parity_projection(self, sector: str = "even") -> np.ndarray:
        """Dense Hamiltonian block of one fermion-parity sector."""
        idx = self.parity_sector_indices(sector)
        return self.sparse_hamiltonian[idx][:, idx].toarray()

    def sector_eigenvalues(self, sector: str = "even") -> np.ndarray:
        if sector not in self._sector_levels:
            self._sector_levels[sector] = hermitian_eigh(
                self.parity_projection(sector), eigvals_only=True
            )
        return self._sector_levels[sector]

    def particle_hole_unitary(self) -> np.ndarray:
        """Unitary part U of the antiunitary particle-hole operator P = U K.

        U = chi_0 chi_2 ... chi_{N-2} and K is complex conjugation in the
        computational basis. Then P chi_a P^-1 = (-1)^(N/2-1) chi_a, so
        U H^* U^dag = H, P^2 = U U^* = (-1)^(n(n-1)/2) with n = N/2, and
        U Gamma U^dag = (-1)^n Gamma (P exchanges the parity sectors iff n is odd).
        """
        n = self.n_qubits
        x, z, phase = 0, 0, 1.0 + 0j
        for j in range(n):
            x, z, phase = _pauli_product((x, z, phase), majorana_pauli(2 * j, n))
        return pauli_terms_to_sparse([(x, z, phase)], n).toarray()

    def _sectors_for_statistics(self) -> list[tuple[str, bool]]:
        """(sector, kramers) pairs entering the level statistics of this N."""
        if self.symmetry_class == "GUE":
            return [("even", False)]  # the odd sector has the identical spectrum
        kramers = self.symmetry_class == "GSE"
        return [("even", kramers), ("odd", kramers)]

    def spacing_ratios(self, trim: float = 0.0) -> np.ndarray:
        """Spacing ratios of the independent parity sectors, Kramers-reduced for GSE.

        Sectors are never merged: ratios are computed inside each sector and then
        concatenated.
        """
        return np.concatenate(
            [
                spacing_ratios(self.sector_eigenvalues(sector), kramers=kramers, trim=trim)
                for sector, kramers in self._sectors_for_statistics()
            ]
        )

    def get_level_spacing_ratio_parity(self, sector: str = "even", trim: float = 0.0) -> float:
        """Mean spacing ratio inside one parity sector (Kramers-reduced for GSE)."""
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
    """Number of distinct levels per parity sector used in the statistics."""
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
    """Disorder-averaged, sector-resolved mean spacing ratio.

    Realisation ``k`` uses seed ``seed_start + k``. Each realisation contributes
    the mean ratio of its independent sectors; mean, standard error and a
    percentile-bootstrap 95% interval are taken over realisations
    (``rng`` seeds the bootstrap). ``reference`` is the large-N value of the
    symmetry class (Atas et al. 2013); see :func:`syk_rmt_reference` for a
    finite-size reference with the same number of levels.
    """
    per_realization = np.array(
        [
            SYKHamiltonian(
                n_majorana, coupling_strength=coupling_strength, seed=seed
            ).get_level_spacing_ratio(trim=trim)
            for seed in range(seed_start, seed_start + n_realizations)
        ]
    )
    stats = bootstrap_mean(per_realization, rng=rng)
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
    """Finite-size RMT <r> of the SYK class with the same number of levels and trimming."""
    cls = syk_symmetry_class(n_majorana)
    res = ensemble_mean_r(cls, sector_level_count(n_majorana), n_samples, trim=trim, rng=rng)
    res["symmetry_class"] = cls
    return res
