"""Exact diagonalisation of the 1D Hubbard ring, resolved by symmetry sectors.

Model (L sites, periodic boundary conditions for the fermion operators)::

    H = -t  sum_{j,s} (c+_{j+1,s} c_{j,s} + h.c.)
        -t' sum_{j,s} (c+_{j+2,s} c_{j,s} + h.c.)
        + U sum_j n_{j,up} n_{j,down}

Resolved symmetries
-------------------
* N_up, N_down (the basis is built inside one (N_up, N_down) sector);
* lattice momentum K = 2 pi m / L (translation T, T c+_j T^-1 = c+_{j+1},
  T|K> = exp(-iK)|K>); only m = 0..L//2 are needed because K and -K are
  related by reflection;
* site reflection j -> -j (mod L) at K = 0 and K = pi;
* total spin S (S^2) - the Hamiltonian is SU(2) invariant for any t';
* eta-spin (eta^2, eta^+ = sum_j (-1)^j c+_{j,up} c+_{j,down}) when t' = 0
  and L is even. The next-nearest-neighbour hopping t' connects sites of the
  same sublattice, breaks the eta-SU(2) symmetry and the Lieb-Wu
  integrability of the t' = 0 chain.

For t' != 0 the Hamiltonian is real in the occupation basis, so time reversal
times reflection is an antiunitary symmetry squaring to +1 inside every
momentum sector: the expected class is GOE, also at K != 0, pi.

Mode ordering: up modes 0..L-1, down modes L..2L-1; a configuration is an
integer whose bit p is the occupation of mode p, and states are
c+_{p1} c+_{p2} ... |0> with p1 < p2 < ... .
"""

import warnings
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from ..spectral.statistics import spacing_ratios
from ..utils.helpers import hermitian_eigh

_POP_LUT = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.int64)


def _popcount(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.int64)
    out = np.zeros(a.shape, dtype=np.int64)
    while np.any(a):
        out += _POP_LUT[a & 0xFFFF]
        a = a >> 16
    return out


def _ladder(configs, mode: int, create: bool):
    """Apply c+_mode (create=True) or c_mode to every configuration.

    Returns (new_configs, sign, valid) with the Jordan-Wigner sign
    (-1)^(number of occupied modes below ``mode``).
    """
    bit = np.int64(1) << mode
    occupied = (configs & bit) != 0
    valid = ~occupied if create else occupied
    sign = 1 - 2 * (_popcount(configs & (bit - 1)) % 2)
    return configs ^ bit, sign, valid


def _apply_word(configs, word: Sequence[Tuple[bool, int]]):
    """Apply a product of ladder operators, written left to right as ``word``."""
    out = configs.copy()
    amp = np.ones(configs.shape, dtype=np.int64)
    valid = np.ones(configs.shape, dtype=bool)
    for create, mode in reversed(word):
        out, sign, ok = _ladder(out, mode, create)
        amp *= sign
        valid &= ok
    return out, amp, valid


class HubbardRingED:
    """Symmetry-resolved exact diagonalisation of the 1D Hubbard ring.

    Parameters
    ----------
    n_sites : L (>= 3; >= 5 if ``tp != 0`` so that next-nearest-neighbour
        bonds are not double counted).
    n_up, n_down : particle numbers per spin species.
    t, tp, U : nearest-neighbour hopping, next-nearest-neighbour hopping and
        on-site interaction.
    """

    def __init__(
        self, n_sites: int, n_up: int, n_down: int, t: float = 1.0, tp: float = 0.0, U: float = 4.0
    ):
        if n_sites < 3:
            raise ValueError("need L >= 3 so that nearest-neighbour bonds are distinct")
        if tp != 0 and n_sites < 5:
            raise ValueError(
                "t' != 0 needs L >= 5 so that next-nearest-neighbour bonds are distinct"
            )
        if not (0 <= n_up <= n_sites and 0 <= n_down <= n_sites):
            raise ValueError("particle numbers out of range")
        self.L, self.n_up, self.n_down = n_sites, n_up, n_down
        self.t, self.tp, self.U = t, tp, U

        L = n_sites
        ups = [sum(1 << j for j in occ) for occ in combinations(range(L), n_up)]
        downs = [sum(1 << j for j in occ) for occ in combinations(range(L), n_down)]
        self.configs = np.sort(np.array([a | (b << L) for a in ups for b in downs], dtype=np.int64))
        self.dim = self.configs.size
        self._H = None

    # -- modes and generic operators ---------------------------------------------

    def mode(self, site: int, spin: str) -> int:
        return (site % self.L) + (0 if spin == "up" else self.L)

    def operator(self, terms: List[Tuple[complex, Sequence[Tuple[bool, int]]]]) -> csr_matrix:
        """Sparse matrix in this (N_up, N_down) sector of sum_k coef_k * word_k.

        Every word must conserve N_up and N_down.
        """
        rows, cols, vals = [], [], []
        for coef, word in terms:
            new, amp, valid = _apply_word(self.configs, word)
            if not np.any(valid):
                continue
            idx = np.searchsorted(self.configs, new[valid])
            if np.any(idx >= self.dim) or np.any(
                self.configs[np.minimum(idx, self.dim - 1)] != new[valid]
            ):
                raise ValueError("operator leaves the (N_up, N_down) sector")
            rows.append(idx)
            cols.append(np.flatnonzero(valid))
            vals.append(coef * amp[valid])
        if not rows:
            return csr_matrix((self.dim, self.dim), dtype=complex)
        M = coo_matrix(
            (np.concatenate(vals).astype(complex), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, self.dim),
        ).tocsr()
        M.sum_duplicates()
        return M

    def site_permutation(self, site_map: Sequence[int]) -> csr_matrix:
        """Unitary U with U c+_{j,s} U^-1 = c+_{site_map[j], s} (fermionic signs included)."""
        L = self.L
        perm = np.array([site_map[p] if p < L else L + site_map[p - L] for p in range(2 * L)])
        new = np.zeros_like(self.configs)
        for p in range(2 * L):
            occ = (self.configs >> p) & 1
            new |= occ << perm[p]
        inversions = np.zeros(self.dim, dtype=np.int64)
        for p in range(2 * L):
            for q in range(p + 1, 2 * L):
                if perm[p] > perm[q]:
                    inversions += ((self.configs >> p) & 1) & ((self.configs >> q) & 1)
        idx = np.searchsorted(self.configs, new)
        sign = 1 - 2 * (inversions % 2)
        return coo_matrix(
            (sign.astype(complex), (idx, np.arange(self.dim))), shape=(self.dim, self.dim)
        ).tocsr()

    # -- physical operators ----------------------------------------------------------

    @property
    def hamiltonian(self) -> csr_matrix:
        if self._H is None:
            self._H = self._build_hamiltonian()
        return self._H

    def _build_hamiltonian(self) -> csr_matrix:
        L = self.L
        terms = []
        for distance, amplitude in ((1, self.t), (2, self.tp)):
            if amplitude == 0:
                continue
            for j in range(L):
                for s in ("up", "down"):
                    a, b = self.mode(j, s), self.mode(j + distance, s)
                    terms.append((-amplitude, [(True, b), (False, a)]))
                    terms.append((-amplitude, [(True, a), (False, b)]))
        H = self.operator(terms)
        doublons = _popcount(self.configs & (self.configs >> L) & ((1 << L) - 1))
        return (
            H
            + coo_matrix(
                (self.U * doublons.astype(complex), (np.arange(self.dim), np.arange(self.dim))),
                shape=(self.dim, self.dim),
            )
        ).tocsr()

    def translation(self) -> csr_matrix:
        return self.site_permutation([(j + 1) % self.L for j in range(self.L)])

    def reflection(self) -> csr_matrix:
        return self.site_permutation([(-j) % self.L for j in range(self.L)])

    def spin_squared(self) -> csr_matrix:
        """S^2 = S^- S^+ + S_z^2 + S_z."""
        L = self.L
        sz = (self.n_up - self.n_down) / 2
        terms = [
            (
                1.0,
                [
                    (True, self.mode(i, "down")),
                    (False, self.mode(i, "up")),
                    (True, self.mode(j, "up")),
                    (False, self.mode(j, "down")),
                ],
            )
            for i in range(L)
            for j in range(L)
        ]
        return (self.operator(terms) + (sz * sz + sz) * _identity(self.dim)).tocsr()

    def eta_squared(self) -> csr_matrix:
        """eta^2 = eta^- eta^+ + eta_z^2 + eta_z, eta_z = (N - L)/2 (bipartite ring, L even)."""
        if self.L % 2:
            raise ValueError("eta-pairing needs a bipartite ring (even L)")
        L = self.L
        ez = (self.n_up + self.n_down - L) / 2
        terms = [
            (
                (-1.0) ** (i + j),
                [
                    (False, self.mode(i, "down")),
                    (False, self.mode(i, "up")),
                    (True, self.mode(j, "up")),
                    (True, self.mode(j, "down")),
                ],
            )
            for i in range(L)
            for j in range(L)
        ]
        return (self.operator(terms) + (ez * ez + ez) * _identity(self.dim)).tocsr()

    @property
    def has_eta_symmetry(self) -> bool:
        return self.tp == 0 and self.L % 2 == 0

    # -- momentum sectors ---------------------------------------------------------------

    def momentum_basis(self, m: int) -> csr_matrix:
        """Isometry V (dim x n_m) onto the sector T|psi> = exp(-2 pi i m / L)|psi>."""
        T = self.translation().tocoo()
        target = np.empty(self.dim, dtype=np.int64)
        sign = np.empty(self.dim)
        target[T.col] = T.row
        sign[T.col] = T.data.real
        K = 2 * np.pi * m / self.L
        visited = np.zeros(self.dim, dtype=bool)
        rows, cols, vals = [], [], []
        n_col = 0
        for start in range(self.dim):
            if visited[start]:
                continue
            states, signs = [start], [1.0]
            current, acc = start, 1.0
            while True:
                acc *= sign[current]
                current = target[current]
                if current == start:
                    break
                states.append(current)
                signs.append(acc)
            visited[states] = True
            R = len(states)
            if not np.isclose(np.exp(1j * K * R), acc):  # T^R|r> = acc |r>
                continue
            d = np.arange(R)
            rows.append(np.array(states))
            cols.append(np.full(R, n_col))
            vals.append(np.array(signs) * np.exp(1j * K * d) / np.sqrt(R))
            n_col += 1
        if n_col == 0:
            return csr_matrix((self.dim, 0), dtype=complex)
        return coo_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.dim, n_col),
        ).tocsr()

    def sector_blocks(
        self,
        momenta: Optional[Sequence[int]] = None,
        shifts: Tuple[float, float, float] = (0.1234567, 0.0765432, 0.0321987),
    ) -> List[Dict]:
        """Eigenvalues of every symmetry block (m, S, eta, reflection parity).

        Within each momentum sector, H + a S^2 + b eta^2 + c R is diagonalised
        (generic a, b, c lift coincidences between blocks); each eigenvector is
        labelled by its S^2, eta^2 and R expectation values, which are checked
        to be exact quantum numbers. Default momenta: m = 0..L//2.
        """
        L = self.L
        momenta = range(L // 2 + 1) if momenta is None else momenta
        H, S2 = self.hamiltonian, self.spin_squared()
        eta2 = self.eta_squared() if self.has_eta_symmetry else None
        Rop = self.reflection()
        blocks: Dict[Tuple, List[float]] = {}
        for m in momenta:
            V = self.momentum_basis(m)
            if V.shape[1] == 0:
                continue
            Vh = V.conj().T
            mats = {"H": (Vh @ H @ V).toarray(), "S2": (Vh @ S2 @ V).toarray()}
            if eta2 is not None:
                mats["eta2"] = (Vh @ eta2 @ V).toarray()
            with_reflection = (2 * m) % L == 0
            if with_reflection:
                mats["R"] = (Vh @ Rop @ V).toarray()
            M = mats["H"] + shifts[0] * mats["S2"]
            if "eta2" in mats:
                M = M + shifts[1] * mats["eta2"]
            if "R" in mats:
                M = M + shifts[2] * mats["R"]
            _, vecs = hermitian_eigh(M)
            labels = {}
            for key in ("S2", "eta2", "R"):
                if key in mats:
                    labels[key] = _expectations(vecs, mats[key])
            energies = _expectations(vecs, mats["H"])
            S = _casimir_to_spin(labels["S2"], "S")
            eta = _casimir_to_spin(labels["eta2"], "eta") if "eta2" in labels else None
            R = _round_checked(labels["R"], (-1.0, 1.0), "R") if "R" in labels else None
            for k in range(len(energies)):
                key = (m, S[k], None if eta is None else eta[k], None if R is None else R[k])
                blocks.setdefault(key, []).append(energies[k])
        return [
            {
                "m": key[0],
                "K": 2 * np.pi * key[0] / L,
                "S": key[1],
                "eta": key[2],
                "reflection": key[3],
                "energies": np.sort(np.array(levels)),
            }
            for key, levels in sorted(blocks.items(), key=lambda kv: str(kv[0]))
        ]


def _expectations(vecs: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Real parts of <v_k|op|v_k> for every column v_k (one BLAS matmul)."""
    return np.real(np.sum(vecs.conj() * (op @ vecs), axis=0))


def _identity(dim: int) -> csr_matrix:
    return coo_matrix(
        (np.ones(dim, dtype=complex), (np.arange(dim), np.arange(dim))), shape=(dim, dim)
    ).tocsr()


def _round_checked(values, allowed, name, tol=1e-6):
    allowed = np.asarray(allowed)
    nearest = allowed[np.argmin(np.abs(values[:, None] - allowed[None, :]), axis=1)]
    if np.max(np.abs(values - nearest)) > tol:
        raise RuntimeError(
            f"{name} is not a good quantum number of the computed eigenvectors "
            f"(deviation {np.max(np.abs(values - nearest)):.2e}); change `shifts`"
        )
    return nearest


def _casimir_to_spin(casimir, name, tol=1e-6):
    """Map j(j+1) to j (in steps of 1/2), checking that the value is exact."""
    casimir = np.asarray(casimir)
    j = np.round(2 * (-1 + np.sqrt(1 + 4 * np.clip(casimir, 0, None))) / 2) / 2
    deviation = np.max(np.abs(casimir - j * (j + 1))) if casimir.size else 0.0
    if deviation > tol:
        raise RuntimeError(
            f"{name} is not a good quantum number of the computed eigenvectors "
            f"(deviation {deviation:.2e}); change `shifts`"
        )
    return j


def hubbard_level_statistics(
    n_sites: int,
    n_up: int,
    n_down: int,
    t: float = 1.0,
    tp: float = 0.0,
    U: float = 4.0,
    trim: float = 0.0,
    min_levels: int = 10,
) -> Dict:
    """Spacing ratios pooled over fully resolved symmetry blocks (m = 0..L//2).

    Blocks with fewer than ``min_levels`` levels are skipped. The standard error
    treats ratios as independent (neighbouring ratios are weakly correlated, so
    it is a slight underestimate). ``degenerate_fraction`` counts exactly
    degenerate spacings left after symmetry resolution (should be ~0 for a
    generic non-integrable model).
    """
    ed = HubbardRingED(n_sites, n_up, n_down, t=t, tp=tp, U=U)
    blocks = ed.sector_blocks()
    ratios, sizes, n_degenerate, n_spacings = [], [], 0, 0
    for b in blocks:
        e = b["energies"]
        if e.size < min_levels:
            continue
        s = np.diff(e)
        n_degenerate += int(np.sum(s <= 1e-10 * np.mean(s)))
        n_spacings += s.size
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratios.append(spacing_ratios(e, trim=trim))
        sizes.append(e.size)
    r = np.concatenate(ratios) if ratios else np.array([])
    return {
        "mean": float(np.mean(r)) if r.size else float("nan"),
        "stderr": float(np.std(r, ddof=1) / np.sqrt(r.size)) if r.size > 1 else float("nan"),
        "n_ratios": int(r.size),
        "n_blocks": len(sizes),
        "block_sizes": sizes,
        "degenerate_fraction": n_degenerate / n_spacings if n_spacings else float("nan"),
        "eta_resolved": ed.has_eta_symmetry,
        "sector_dim": ed.dim,
        "params": {
            "L": n_sites,
            "n_up": n_up,
            "n_down": n_down,
            "t": t,
            "tp": tp,
            "U": U,
            "trim": trim,
        },
    }
