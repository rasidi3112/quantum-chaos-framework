from itertools import combinations

import numpy as np
import pytest
from scipy.sparse.linalg import eigsh

from quantum_chaos.chaos.kpm import (
    chebyshev_moments,
    jackson_kernel,
    kpm_density,
    moments_to_power_moments,
)
from quantum_chaos.chaos.krylov import autocorrelation
from quantum_chaos.chaos.otoc import thermal_otoc
from quantum_chaos.chaos.typicality import SectorSYK, typical_autocorrelation, typical_otoc
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.hamiltonians.syk_sparse import (
    n_flip_masks,
    sector_memory_bytes,
    syk_couplings,
    syk_sector_hamiltonian,
)
from quantum_chaos.utils.helpers import generate_random_coupling


def test_couplings_reproduce_the_dense_model():
    J = generate_random_coupling(10, seed=3)
    flat = np.array([J[c] for c in combinations(range(10), 4)])
    np.testing.assert_array_equal(flat, syk_couplings(10, 3))


@pytest.mark.parametrize("N", [8, 10, 12])
@pytest.mark.parametrize("sector", ["even", "odd"])
def test_sector_matrix_equals_dense_block(N, sector):
    dense = SYKHamiltonian(N, seed=5, chi_squared=0.5).parity_projection(sector)
    sparse = syk_sector_hamiltonian(N, 5, sector, chi_squared=0.5).toarray()
    np.testing.assert_allclose(sparse, dense, atol=1e-15)


def test_memory_prediction_and_mask_count():
    H = syk_sector_hamiltonian(16, seed=0)
    assert H.nnz == H.shape[0] * n_flip_masks(16)
    assert H.data.nbytes + H.indices.nbytes == sector_memory_bytes(16)


def test_lanczos_ground_state_matches_dense():
    syk = SYKHamiltonian(16, seed=1)
    e_sparse = min(
        eigsh(syk_sector_hamiltonian(16, 1, s), k=1, which="SA", return_eigenvectors=False)[0]
        for s in ("even", "odd")
    )
    assert e_sparse == pytest.approx(syk.eigenvalues[0], abs=1e-10)


def test_kpm_moments_density_and_exact_second_moment():
    H = syk_sector_hamiltonian(14, seed=2, chi_squared=0.5)
    exact = np.sum(syk_couplings(14, 2) ** 2) * 0.5**4
    mom = chebyshev_moments(H, 120, n_vectors=30, rng=0)
    mean, second = moments_to_power_moments(mom)
    assert second == pytest.approx(exact, rel=0.03)
    assert abs(mean) < 0.05
    grid = np.linspace(mom["shift"] - mom["scale"], mom["shift"] + mom["scale"], 4001)
    rho = kpm_density(mom, grid)
    assert np.trapezoid(rho, grid) == pytest.approx(1.0, abs=0.02)
    assert jackson_kernel(10)[0] == pytest.approx(1.0)


def test_typicality_reproduces_exact_correlators():
    N = 12
    model = SectorSYK(N, seed=3, chi_squared=0.5)
    syk = SYKHamiltonian(N, seed=3, chi_squared=0.5)
    times = np.array([0.0, 1.0, 2.5])
    exact_auto = autocorrelation(syk.hamiltonian_matrix, syk.get_majorana_operator(0), times)
    typ_auto = typical_autocorrelation(model, 0, times, n_vectors=8, rng=1)
    assert np.all(np.abs(typ_auto["mean"] - exact_auto) < 5 * typ_auto["stderr"] + 1e-3)
    half = [syk.get_majorana_operator(a) * np.sqrt(0.5) for a in (0, 1)]
    for beta in (0.0, 2.0):
        exact = thermal_otoc(syk.hamiltonian_matrix, half[0], half[1], times, beta)
        typ = typical_otoc(model, 0, 1, times, n_vectors=12, beta=beta, rng=0)
        assert np.all(np.abs(typ["mean"] - exact) < 5 * typ["stderr"] + 2e-3)
