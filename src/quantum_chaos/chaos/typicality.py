from collections.abc import Sequence

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import expm_multiply

from ..hamiltonians.syk_hamiltonian import _popcount, majorana_pauli
from ..hamiltonians.syk_sparse import parity_sector_states, syk_sector_hamiltonian
from ..utils.helpers import RngLike

SECTORS = ("even", "odd")
OTHER = {"even": "odd", "odd": "even"}


class SectorSYK:
    def __init__(
        self,
        n_majorana: int,
        seed: RngLike = None,
        chi_squared: float = 1.0,
        dtype: type = np.complex128,
    ):
        self.n_majorana = n_majorana
        self.n_qubits = n_majorana // 2
        self.chi_squared = chi_squared
        self.hamiltonian = {
            s: syk_sector_hamiltonian(n_majorana, seed, s, chi_squared=chi_squared, dtype=dtype)
            for s in SECTORS
        }
        self.states = {s: parity_sector_states(self.n_qubits, s) for s in SECTORS}
        self.dim = 2**self.n_qubits
        self._lookup = np.full(self.dim, -1, dtype=np.int64)
        for s in SECTORS:
            self._lookup[self.states[s].astype(np.int64)] = np.arange(self.states[s].size)
        self._majorana_cache: dict[tuple[int, str], csc_matrix] = {}

    def majorana(self, index: int, sector: str) -> csc_matrix:
        key = (index, sector)
        if key not in self._majorana_cache:
            x, z, phase = majorana_pauli(index, self.n_qubits)
            src = self.states[sector]
            values = phase * (1.0 - 2.0 * (_popcount(src & np.uint64(z)) % 2))
            values = values * np.sqrt(self.chi_squared)
            rows = self._lookup[(src ^ np.uint64(x)).astype(np.int64)]
            self._majorana_cache[key] = csc_matrix(
                (values, (rows, np.arange(src.size))), shape=(src.size, src.size)
            )
        return self._majorana_cache[key]

    def apply_majorana(self, index: int, psi: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {OTHER[s]: self.majorana(index, s) @ v for s, v in psi.items()}

    def evolve(self, psi: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:
        return {s: expm_multiply(-1j * t * self.hamiltonian[s], v) for s, v in psi.items()}

    def imaginary_evolve(self, psi: dict[str, np.ndarray], tau: float) -> dict[str, np.ndarray]:
        return {s: expm_multiply(-tau * self.hamiltonian[s], v) for s, v in psi.items()}

    def random_state(self, rng: np.random.Generator) -> dict[str, np.ndarray]:
        psi = {
            s: rng.normal(size=self.states[s].size) + 1j * rng.normal(size=self.states[s].size)
            for s in SECTORS
        }
        norm = np.sqrt(sum(np.vdot(v, v).real for v in psi.values()))
        return {s: v / norm for s, v in psi.items()}


def _inner(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> complex:
    return complex(sum(np.vdot(a[s], b[s]) for s in a if s in b))


def typical_autocorrelation(
    model: SectorSYK, index: int, times: Sequence[float], n_vectors: int = 4, rng: RngLike = None
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rng)
    samples = []
    for _ in range(n_vectors):
        r = model.random_state(rng)
        values = []
        for t in times:
            a = model.evolve(model.apply_majorana(index, r), t)
            b = model.evolve(r, t)
            values.append(_inner(b, model.apply_majorana(index, a)))
        samples.append(values)
    s = np.array(samples) / model.chi_squared
    return {"mean": s.mean(axis=0), "stderr": s.std(axis=0, ddof=1) / np.sqrt(n_vectors)}


def typical_otoc(
    model: SectorSYK,
    w: int,
    v: int,
    times: Sequence[float],
    n_vectors: int = 4,
    beta: float = 0.0,
    rng: RngLike = None,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(rng)
    samples = []
    norms = []
    for _ in range(n_vectors):
        r0 = model.random_state(rng)
        r = model.imaginary_evolve(r0, beta / 8) if beta > 0 else r0
        half = model.imaginary_evolve(r0, beta / 2) if beta > 0 else r0
        norms.append(_inner(half, half).real)
        values = []
        for t in times:

            def y(psi):
                return model.imaginary_evolve(psi, beta / 4) if beta > 0 else psi

            def wt(psi, t=t):
                return model.evolve(model.apply_majorana(w, model.evolve(psi, t)), -t)

            phi = y(model.apply_majorana(v, r))
            phi = y(wt(phi))
            phi = y(model.apply_majorana(v, phi))
            phi = wt(phi)
            values.append(_inner(r, phi))
        samples.append(values)
    s = np.array(samples)
    z = np.array(norms)
    ratio = s / z.mean()
    return {"mean": ratio.mean(axis=0), "stderr": ratio.std(axis=0, ddof=1) / np.sqrt(n_vectors)}
