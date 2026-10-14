"""SYK symmetry classes and sector-resolved level statistics (PLAN P1-2, P1-3)."""

import numpy as np
import pytest

from quantum_chaos.hamiltonians import syk_hamiltonian as syk_module
from quantum_chaos.hamiltonians.syk_hamiltonian import SYKHamiltonian
from quantum_chaos.spectral import R_MEAN


def test_level_spacing_ratio_keeps_degeneracies():
    from quantum_chaos.utils.helpers import level_spacing_ratio

    with pytest.warns(UserWarning):
        assert level_spacing_ratio(np.array([0.0, 1.0, 1.0, 3.0])) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "N, expected", [(8, "GOE"), (10, "GUE"), (12, "GSE"), (14, "GUE"), (16, "GOE")]
)
def test_disorder_average_reports_class_and_reference(N, expected):
    stats = syk_module.disorder_average_lsr(N, n_realizations=2)
    assert stats["symmetry_class"] == expected
    assert stats["reference"] == pytest.approx(R_MEAN[expected.lower()])
    assert SYKHamiltonian(N, seed=0).summary()["symmetry_class"] == expected


def test_cross_class_finite_size_extrapolation_removed():
    assert not hasattr(syk_module, "finite_size_scaling")


# --- construction ----------------------------------------------------------


def jordan_wigner_reference(syk):
    """Original dense construction: products of Jordan-Wigner Majorana matrices."""
    from itertools import combinations

    chi = [syk.get_majorana_operator(a) for a in range(syk.n_majorana)]
    H = np.zeros((2**syk.n_qubits,) * 2, dtype=complex)
    for i, j, k, l in combinations(range(syk.n_majorana), 4):
        H += syk.couplings[i, j, k, l] * chi[i] @ chi[j] @ chi[k] @ chi[l]
    return H


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12])
def test_pauli_string_construction_matches_jordan_wigner(N):
    syk = SYKHamiltonian(N, seed=N)
    np.testing.assert_allclose(syk.hamiltonian_matrix, jordan_wigner_reference(syk), atol=1e-12)
    np.testing.assert_allclose(syk.hamiltonian_matrix, syk.hamiltonian_matrix.conj().T, atol=0)


def test_maldacena_stanford_normalisation_is_a_quarter():
    H1 = SYKHamiltonian(10, seed=3).hamiltonian_matrix
    H_half = SYKHamiltonian(10, seed=3, chi_squared=0.5).hamiltonian_matrix
    np.testing.assert_allclose(H_half, H1 / 4, atol=1e-14)


def test_hamiltonian_variance_matches_coupling_convention():
    # chi^2 = 1: Tr H^2 / D = sum_{i<j<k<l} J_ijkl^2 exactly (Pauli strings are orthonormal)
    from itertools import combinations

    syk = SYKHamiltonian(12, seed=5)
    H = syk.hamiltonian_matrix
    J2 = sum(syk.couplings[c] ** 2 for c in combinations(range(12), 4))
    assert np.trace(H @ H).real / H.shape[0] == pytest.approx(J2, rel=1e-12)


# --- particle-hole symmetry ------------------------------------------------------


@pytest.mark.parametrize("N", [8, 10, 12, 14, 16])
def test_particle_hole_operator(N):
    syk = SYKHamiltonian(N, seed=1)
    H = syk.hamiltonian_matrix
    U = syk.particle_hole_unitary()
    n = N // 2
    np.testing.assert_allclose(U @ H.conj() @ U.conj().T, H, atol=1e-12)  # [P, H] = 0
    np.testing.assert_allclose(
        U @ U.conj(), (-1) ** (n * (n - 1) // 2) * np.eye(2**n), atol=1e-12
    )  # P^2
    gamma = np.diag((-1.0) ** np.array([bin(b).count("1") for b in range(2**n)]))
    np.testing.assert_allclose(U @ gamma @ U.conj().T, (-1) ** n * gamma, atol=1e-12)
    p2 = (-1) ** (n * (n - 1) // 2)
    expected = {
        "GOE": (p2 == 1 and n % 2 == 0),
        "GSE": (p2 == -1 and n % 2 == 0),
        "GUE": n % 2 == 1,
    }
    assert expected[syk.symmetry_class]


@pytest.mark.parametrize("N", [10, 14])
def test_gue_sectors_are_degenerate_with_each_other(N):
    syk = SYKHamiltonian(N, seed=2)
    np.testing.assert_allclose(
        syk.sector_eigenvalues("even"), syk.sector_eigenvalues("odd"), atol=1e-10
    )


@pytest.mark.parametrize("N", [12, 20])
def test_gse_sectors_consist_of_kramers_pairs(N):
    syk = SYKHamiltonian(N, seed=2)
    for sector in ("even", "odd"):
        e = syk.sector_eigenvalues(sector)
        np.testing.assert_allclose(e[0::2], e[1::2], atol=1e-10)
        assert np.min(np.diff(e[0::2])) > 1e-8


@pytest.mark.parametrize("N", [8, 16])
def test_goe_sectors_have_no_systematic_degeneracy(N):
    syk = SYKHamiltonian(N, seed=2)
    for sector in ("even", "odd"):
        assert np.min(np.diff(syk.sector_eigenvalues(sector))) > 1e-8


# --- statistics vs random-matrix theory ---------------------------------------


def _compare_with_rmt(N, n_realizations, n_rmt):
    from quantum_chaos.hamiltonians.syk_hamiltonian import syk_rmt_reference

    data = syk_module.disorder_average_lsr(N, n_realizations=n_realizations, rng=0)
    ref = syk_rmt_reference(N, n_samples=n_rmt, rng=1)
    z = (data["mean"] - ref["mean"]) / np.hypot(data["stderr"], ref["stderr"])
    assert abs(z) < 3, (data["mean"], ref["mean"], z)
    # large-N Atas value within statistical error + measured finite-size offset (< 5e-3)
    assert abs(data["mean"] - data["reference"]) < 3 * data["stderr"] + 5e-3


@pytest.mark.parametrize("N, n_realizations", [(12, 200), (14, 150), (16, 100)])
def test_sector_statistics_match_dimension_matched_rmt(N, n_realizations):
    _compare_with_rmt(N, n_realizations, n_rmt=600)


@pytest.mark.slow
@pytest.mark.parametrize("N, n_realizations", [(18, 100), (20, 40)])
def test_sector_statistics_match_rmt_larger_n(N, n_realizations):
    _compare_with_rmt(N, n_realizations, n_rmt=300)
