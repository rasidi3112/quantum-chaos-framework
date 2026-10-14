"""Compatibility with the current PennyLane API (PLAN P0-1)."""

import importlib
import sys
import warnings

import numpy as np
import pennylane as qml
import pytest
from scipy.linalg import expm

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}
TERMS = [(0.7, "XY"), (0.3, "ZI"), (0.4, "YX"), (0.2, "IZ"), (-0.5, "ZZ")]


def pauli_matrix(terms):
    out = 0
    for c, s in terms:
        m = np.array([[1.0 + 0j]])
        for p in s:
            m = np.kron(m, PAULI[p])
        out = out + c * m
    return out


@pytest.mark.xfail(sys.version_info < (3, 14), strict=True,
                   reason="P0-1: annotation qml.Device is evaluated at import on Python < 3.14")
@pytest.mark.parametrize("name", ["hamiltonians", "circuits", "noise", "qml", "utils",
                                  "visualization"])
def test_subpackage_imports(name):
    importlib.import_module(f"quantum_chaos.{name}")


@pytest.mark.xfail(reason="P0-1: qml.operation.Tensor was removed", strict=True)
def test_hamiltonian_to_pennylane_reproduces_matrix():
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution

    H = TrotterEvolution(2).hamiltonian_to_pennylane(TERMS)
    np.testing.assert_allclose(qml.matrix(H, wire_order=[0, 1]), pauli_matrix(TERMS), atol=1e-12)


@pytest.mark.xfail(reason="P0-1: qml.QubitStateVector was removed", strict=True)
def test_evolve_state_from_given_initial_state():
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution

    rng = np.random.default_rng(0)
    psi0 = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi0 /= np.linalg.norm(psi0)
    exact = expm(-1j * pauli_matrix(TERMS) * 0.8) @ psi0
    got = TrotterEvolution(2).evolve_state(psi0, TERMS, 0.8, n_steps=400, order=2)
    assert abs(np.vdot(exact, got)) ** 2 == pytest.approx(1.0, abs=1e-8)


@pytest.mark.xfail(reason="P0-1: shots set on the device is deprecated", strict=True)
def test_sampled_expectation_without_deprecation_warning():
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution

    with warnings.catch_warnings():
        warnings.simplefilter("error", qml.exceptions.PennyLaneDeprecationWarning)
        te = TrotterEvolution(1, shots=2000)
        circuit = te.create_expectation_circuit([(0.0, "Z")], "Z", total_time=1.0, n_steps=1)
        assert float(circuit()) == pytest.approx(1.0)


@pytest.mark.parametrize("order, expected_slope", [(1, 1.0), (2, 2.0)])
def test_trotter_error_converges_with_expected_order(order, expected_slope):
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution

    te = TrotterEvolution(2)
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    exact = expm(-1j * pauli_matrix(TERMS)) @ psi0
    steps = np.array([8, 16, 32])
    errors = []
    for n in steps:
        state = np.array(te.create_evolution_circuit(TERMS, 1.0, int(n), order)())
        errors.append(np.linalg.norm(state - exact))
    slope = -np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert slope == pytest.approx(expected_slope, abs=0.15)
