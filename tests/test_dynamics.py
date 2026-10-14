import numpy as np
import pytest
from scipy.linalg import expm

from quantum_chaos.circuits.otoc_calculator import OTOCCalculator
from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian
from quantum_chaos.hamiltonians.syk_hamiltonian import SYKHamiltonian


def test_driven_hubbard_static_limit_propagator_is_exact():
    model = DrivenHubbardHamiltonian(n_sites=2, hopping_amplitude=0.0, onsite_interaction=1.7)
    np.testing.assert_allclose(
        model.time_evolution_operator(1.3, n_steps=3),
        expm(-1j * model.interaction_matrix * 1.3),
        atol=1e-12,
    )


def test_stroboscopic_evolution_applies_the_floquet_operator():
    model = DrivenHubbardHamiltonian(n_sites=2, driving_frequency=3.0)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    states = model.stroboscopic_evolution(psi, n_periods=3, n_steps_per_period=40)
    F = model.floquet_operator(n_steps=40)
    np.testing.assert_allclose(states[3], np.linalg.matrix_power(F, 3) @ psi, atol=1e-12)
    np.testing.assert_allclose(
        model.time_evolution_operator(2 * np.pi / 3.0, n_steps=40), F, atol=1e-12
    )


def test_driven_hubbard_conserves_particle_number_and_sz():
    model = DrivenHubbardHamiltonian(n_sites=3, periodic_boundary=True)
    N, Sz = model.get_total_number_operator(), model.get_total_spin_z()
    for t in (0.0, 0.4, 1.1):
        H = model.hamiltonian(t)
        np.testing.assert_allclose(H @ N - N @ H, 0, atol=1e-12)
        np.testing.assert_allclose(H @ Sz - Sz @ H, 0, atol=1e-12)
    summary = model.summary()
    assert summary["hilbert_dim"] == 64
    assert summary["driving_period"] == pytest.approx(2 * np.pi / model.driving_frequency)


def _calc():
    syk = SYKHamiltonian(8, seed=3)
    return OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)


def test_otoc_at_zero_time_and_commutator_relation():
    calc = _calc()
    W, V = calc.local_pauli("Z", 0), calc.local_pauli("X", 2)
    times = np.array([0.0, 0.7])
    F = calc.compute_otoc_exact(W, V, times)
    assert F[0].real == pytest.approx(1.0)
    anti = calc.compute_otoc_exact(W, calc.local_pauli("X", 0), np.array([0.0]))
    assert anti[0].real == pytest.approx(-1.0)
    np.testing.assert_allclose(
        calc.compute_commutator_squared(W, V, times), 2 * (1 - F.real), atol=1e-12
    )


def test_high_temperature_limits_of_thermal_otocs():
    calc = _calc()
    W, V = calc.local_pauli("Z", 0), calc.local_pauli("Z", 1)
    times = np.linspace(0, 2, 5)
    infinite = calc.compute_otoc_exact(W, V, times)
    np.testing.assert_allclose(
        calc.compute_otoc_exact(W, V, times, temperature=1e8), infinite, atol=1e-7
    )
    np.testing.assert_allclose(
        calc.compute_regularized_otoc(W, V, times, temperature=1e8), infinite, atol=1e-7
    )


def test_operator_size_of_local_operator():
    calc = _calc()
    sizes = calc.operator_size_growth(calc.local_pauli("Z", 0), np.array([0.0, 0.5, 2.0]))
    assert sizes[0] == pytest.approx(1.0)
    assert np.all(sizes >= 1.0 - 1e-12) and np.all(sizes <= calc.n_qubits + 1e-12)
    assert sizes[-1] > sizes[0]


def test_otoc_requires_a_hamiltonian():
    calc = OTOCCalculator(2)
    with pytest.raises(ValueError):
        calc.compute_otoc_exact(calc.local_pauli("Z", 0), calc.local_pauli("Z", 1), [0.0])
    calc.set_hamiltonian(np.diag([0.0, 1.0, 2.0, 3.0]))
    assert calc.compute_otoc_exact(calc.local_pauli("Z", 0), calc.local_pauli("Z", 1), [0.3])[
        0
    ].real == pytest.approx(1.0)


def test_trotter_diagnostics():
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution, matrix_to_pauli_terms

    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    H = 0.6 * np.kron(X, X) + 0.4 * np.kron(Z, np.eye(2))
    terms = matrix_to_pauli_terms(H, 2)
    assert {s for _, s in terms} == {"XX", "ZI"}
    te = TrotterEvolution(2)
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    _, fid = te.compute_fidelity_vs_time(psi0, terms, np.array([0.5, 1.0]), H, 40, order=2)
    assert np.all(fid > 1 - 1e-4)
    analysis = te.trotter_error_analysis(terms, H, 1.0, max_steps=20, rng=0)
    assert analysis["error_order1"][-1] < analysis["error_order1"][0]
    assert analysis["error_order2"][-1] < analysis["error_order1"][-1]
