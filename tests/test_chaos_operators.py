from itertools import combinations

import numpy as np
import pytest

from quantum_chaos.chaos.krylov import (
    autocorrelation,
    krylov_complexity,
    krylov_wavefunction,
    lanczos_coefficients,
)
from quantum_chaos.chaos.operator_size import (
    mean_size_from_anticommutators,
    pauli_coefficients,
    size_distribution,
)
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.hamiltonians.syk_hamiltonian import syk2_hamiltonian
from quantum_chaos.utils.jordan_wigner import get_pauli_string_representation


def heisenberg(H, O, t):
    E, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * E * t)) @ V.conj().T
    return U.conj().T @ O @ U


def test_first_lanczos_coefficient_is_exact():
    syk = SYKHamiltonian(10, seed=3)
    b = lanczos_coefficients(syk.hamiltonian_matrix, syk.get_majorana_operator(0), n_max=5)
    J = syk.couplings
    exact = 4 * sum(J[c] ** 2 for c in combinations(range(10), 4) if 0 in c)
    assert b[0] ** 2 == pytest.approx(exact, rel=1e-12)


def test_krylov_wavefunction_reproduces_autocorrelation():
    syk = SYKHamiltonian(10, seed=1)
    chi = syk.get_majorana_operator(2)
    b = lanczos_coefficients(syk.hamiltonian_matrix, chi, n_max=80)
    times = np.linspace(0, 2, 9)
    phi = krylov_wavefunction(b, times)
    np.testing.assert_allclose(
        phi[:, 0], autocorrelation(syk.hamiltonian_matrix, chi, times).real, atol=1e-8
    )
    np.testing.assert_allclose((phi**2).sum(axis=1), 1, atol=1e-10)
    assert krylov_complexity(b, np.array([0.0]))[0] == pytest.approx(0.0)


def test_free_syk_krylov_space_terminates():
    N = 12
    chi = SYKHamiltonian(N, seed=0).get_majorana_operator(0)
    b_free = lanczos_coefficients(syk2_hamiltonian(N, seed=1), chi, n_max=60)
    b_int = lanczos_coefficients(SYKHamiltonian(N, seed=1).hamiltonian_matrix, chi, n_max=60)
    assert len(b_free) <= N - 1
    assert len(b_int) == 60


def test_fast_pauli_coefficients_match_slow_decomposition():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    fast = np.sort(np.abs(pauli_coefficients(A)).ravel())
    slow = np.sort(np.abs([c for c, _ in get_pauli_string_representation(A, 3, threshold=0)]))
    np.testing.assert_allclose(fast, slow, atol=1e-12)


def test_majorana_size_distribution():
    chi = [SYKHamiltonian(8, seed=0).get_majorana_operator(a) for a in range(8)]
    for a in range(8):
        assert size_distribution(chi[a])[1] == pytest.approx(1.0)
    assert size_distribution(chi[0] @ chi[1] @ chi[5])[3] == pytest.approx(1.0)
    rng = np.random.default_rng(2)
    odd = sum(rng.normal() * chi[a] for a in range(8)) + 0.7 * chi[1] @ chi[3] @ chi[6]
    p = size_distribution(odd)
    assert p @ np.arange(p.size) == pytest.approx(mean_size_from_anticommutators(odd, chi))


def test_operator_growth_free_versus_interacting():
    N = 10
    chi = [SYKHamiltonian(N, seed=0).get_majorana_operator(a) for a in range(N)]
    H_free = syk2_hamiltonian(N, seed=4).toarray()
    H_int = np.mean([SYKHamiltonian(N, seed=s).hamiltonian_matrix for s in range(1)], axis=0)
    for t in (0.5, 3.0):
        assert size_distribution(heisenberg(H_free, chi[0], t))[1] == pytest.approx(1.0)
    late = np.mean(
        [
            mean_size_from_anticommutators(
                heisenberg(SYKHamiltonian(N, seed=s).hamiltonian_matrix, chi[0], 20.0), chi
            )
            for s in range(4)
        ]
    )
    assert late == pytest.approx(N / 2, rel=0.15)
    assert mean_size_from_anticommutators(heisenberg(H_int, chi[0], 0.0), chi) == pytest.approx(1)
