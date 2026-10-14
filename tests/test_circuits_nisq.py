import numpy as np
import pytest

from quantum_chaos.circuits.noisy_otoc import (
    NoisyTrotterOTOC,
    richardson_extrapolate,
    zero_noise_extrapolated,
)
from quantum_chaos.circuits.otoc_calculator import OTOCCalculator
from quantum_chaos.circuits.resources import pauli_rotation_cost, syk_pauli_terms, trotter_step_cost
from quantum_chaos.circuits.trotter_bounds import first_order_commutator_bound, trotter_error
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.hamiltonians.syk_sparse import sparse_syk_couplings, syk_couplings
from quantum_chaos.noise.noise_channels import NoiseModel
from quantum_chaos.utils.jordan_wigner import pauli_operators

P = pauli_operators()


def pauli_matrix(label):
    m = np.array([[1.0 + 0j]])
    for c in label:
        m = np.kron(m, P[c])
    return m


def test_richardson_is_exact_for_polynomials():
    folds = [1, 3, 5]
    assert richardson_extrapolate(folds, [2 + 3 * f - 0.5 * f**2 for f in folds]) == pytest.approx(
        2
    )


@pytest.mark.parametrize("N", [6, 8, 10])
def test_pauli_terms_reproduce_the_hamiltonian(N):
    terms = syk_pauli_terms(N, syk_couplings(N, 2))
    H = sum(c * pauli_matrix(s) for c, s in terms)
    np.testing.assert_allclose(H, SYKHamiltonian(N, seed=2).hamiltonian_matrix, atol=1e-14)


def test_gate_costs():
    assert pauli_rotation_cost("ZZZZ") == {"cnot": 6, "single_qubit": 1}
    assert pauli_rotation_cost("XIYI") == {"cnot": 2, "single_qubit": 5}
    cost = trotter_step_cost([(1.0, "XX"), (0.5, "ZI"), (0.2, "II")], order=2)
    assert cost["rotations"] == 4 and cost["cnot"] == 4


def test_sparse_syk_keeps_kN_terms_and_the_second_moment():
    N, k = 16, 4
    kept, second = [], []
    for s in range(200):
        J = sparse_syk_couplings(N, k, s)
        kept.append(np.count_nonzero(J))
        second.append(np.sum(J**2))
    dense = np.mean([np.sum(syk_couplings(N, s) ** 2) for s in range(200)])
    assert np.mean(kept) == pytest.approx(k * N, rel=0.05)
    assert np.mean(second) == pytest.approx(dense, rel=0.05)


def test_first_order_commutator_bound_holds():
    terms = [c * pauli_matrix(s) for c, s in syk_pauli_terms(8, syk_couplings(8, 1))]
    for steps in (1, 4, 16):
        assert trotter_error(terms, 2.0, steps) <= first_order_commutator_bound(terms, 2.0, steps)
    ratio = first_order_commutator_bound(terms, 2.0, 4) / first_order_commutator_bound(
        terms, 2.0, 16
    )
    assert ratio == pytest.approx(4.0)
    assert trotter_error(terms, 2.0, 16, order=2) < trotter_error(terms, 2.0, 16, order=1)


@pytest.fixture(scope="module")
def syk6():
    syk = SYKHamiltonian(6, seed=1)
    return syk, syk_pauli_terms(6, syk_couplings(6, 1))


def test_noiseless_trotter_otoc_matches_exact_and_echo_is_one(syk6):
    syk, terms = syk6
    calc = OTOCCalculator(3, syk.hamiltonian_matrix)
    exact = calc.compute_otoc_exact(calc.local_pauli("Z", 0), calc.local_pauli("X", 2), [1.0])
    proto = NoisyTrotterOTOC(terms, 3, n_steps=12, order=2)
    assert proto.otoc(("Z", 0), ("X", 2))(1.0) == pytest.approx(exact[0].real, abs=2e-3)
    assert proto.echo(("X", 2))(1.0) == pytest.approx(1.0, abs=1e-10)
    assert proto.otoc(("Z", 0), ("X", 2), fold=3)(1.0) == pytest.approx(
        proto.otoc(("Z", 0), ("X", 2))(1.0), abs=1e-10
    )


@pytest.mark.slow
def test_noise_reduces_the_signal_and_zne_recovers_part_of_it(syk6):
    syk, terms = syk6
    ideal = NoisyTrotterOTOC(terms, 3, n_steps=6).otoc(("Z", 0), ("X", 2))(0.8)
    noisy = NoisyTrotterOTOC(
        terms, 3, n_steps=6, noise=NoiseModel(depolarizing_rate=0.01, t1=np.inf, t2=np.inf)
    )
    raw = noisy.otoc(("Z", 0), ("X", 2))(0.8)
    echo = noisy.echo(("X", 2))(0.8)
    zne = zero_noise_extrapolated(lambda f: noisy.otoc(("Z", 0), ("X", 2), fold=f)(0.8))
    assert echo < 1
    assert abs(raw) < abs(ideal)
    assert abs(zne["extrapolated"] - ideal) < abs(raw - ideal) / 3
    assert abs(raw / echo - ideal) < abs(raw - ideal)
    with pytest.raises(ValueError):
        noisy.otoc(("Z", 0), ("X", 2), fold=2)
