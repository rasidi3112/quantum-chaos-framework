import numpy as np
import pytest

from quantum_chaos.chaos.eth import (
    diagonal_fluctuation,
    eigenbasis_matrix_elements,
    eigenstate_entanglement,
    entanglement_entropy_bipartite,
    offdiagonal_moment_ratio,
    page_entropy,
)
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.utils.helpers import hermitian_eigh


def sector_operator(N, majoranas):
    syk = SYKHamiltonian(N, seed=0)
    op = np.eye(2 ** (N // 2), dtype=complex)
    for a in majoranas:
        op = op @ syk.get_majorana_operator(a)
    idx = syk.parity_sector_indices("even")
    return syk, op[np.ix_(idx, idx)]


def test_page_formula_exact_value_and_haar_average():
    assert page_entropy(2, 2) == pytest.approx(1 / 3)
    assert page_entropy(4, 16) == page_entropy(16, 4)
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(1500):
        v = rng.normal(size=64) + 1j * rng.normal(size=64)
        samples.append(entanglement_entropy_bipartite(v / np.linalg.norm(v), 4))
    assert np.mean(samples) == pytest.approx(page_entropy(4, 16), abs=4 * np.std(samples) / 40)


def test_entanglement_of_simple_states():
    bell = np.array([1, 0, 0, 1]) / np.sqrt(2)
    assert entanglement_entropy_bipartite(bell, 2) == pytest.approx(np.log(2))
    assert entanglement_entropy_bipartite(np.kron([1, 0], [0, 1]), 2) == pytest.approx(0.0)


def test_particle_hole_odd_operator_has_vanishing_diagonal_for_goe():
    syk, O = sector_operator(16, [0, 1])
    E, Om = eigenbasis_matrix_elements(syk.parity_projection("even"), -1j * O)
    np.testing.assert_allclose(np.diag(Om), 0, atol=1e-10)


def test_diagonal_fluctuations_decay_like_inverse_sqrt_dimension():
    dims, sigmas = [], []
    for N in (10, 12, 14, 16):
        syk, O = sector_operator(N, [0, 1, 2, 3])
        E, Om = eigenbasis_matrix_elements(syk.parity_projection("even"), O)
        dims.append(E.size)
        sigmas.append(diagonal_fluctuation(E, Om))
    slope = np.polyfit(np.log(dims), np.log(sigmas), 1)[0]
    assert slope == pytest.approx(-0.5, abs=0.15)


@pytest.mark.parametrize("N, expected", [(16, 3.0), (18, 2.0)])
def test_offdiagonal_elements_are_gaussian_with_class_dependent_ratio(N, expected):
    syk, O = sector_operator(N, [0, 1, 2, 3])
    E, Om = eigenbasis_matrix_elements(syk.parity_projection("even"), O)
    assert offdiagonal_moment_ratio(E, Om) == pytest.approx(expected, rel=0.1)


def test_mid_spectrum_eigenstates_follow_the_page_curve():
    N = 20
    n = N // 2
    syk = SYKHamiltonian(N, seed=0)
    idx = syk.parity_sector_indices("even")
    E, V = hermitian_eigh(syk.parity_projection("even"))
    mid = np.arange(E.size // 2 - 10, E.size // 2 + 10)
    full = np.zeros((2**n, mid.size), dtype=complex)
    full[idx] = V[:, mid]
    entropies = eigenstate_entanglement(full, n, range(1, 6), range(mid.size))
    for l, values in entropies.items():
        assert values.mean() == pytest.approx(page_entropy(2**l, 2 ** (n - l)), abs=0.01)
