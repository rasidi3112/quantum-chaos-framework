import numpy as np
import pytest

from quantum_chaos.chaos.otoc import (
    lyapunov_fit,
    lyapunov_with_systematics,
    normalized_decay,
    thermal_otoc,
)
from quantum_chaos.circuits.otoc_calculator import OTOCCalculator
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.hamiltonians.syk_hamiltonian import syk2_hamiltonian

TIMES = np.linspace(0, 3, 7)


def majoranas(N, half=True):
    syk = SYKHamiltonian(N, seed=0)
    scale = np.sqrt(0.5) if half else 1.0
    return [syk.get_majorana_operator(a) * scale for a in range(N)]


def test_infinite_temperature_value_and_regularisation_limit():
    syk = SYKHamiltonian(10, seed=2, chi_squared=0.5)
    chi = majoranas(10)
    reg = thermal_otoc(syk.hamiltonian_matrix, chi[0], chi[3], TIMES, beta=0.0)
    plain = thermal_otoc(syk.hamiltonian_matrix, chi[0], chi[3], TIMES, beta=0.0, regularized=False)
    assert reg[0].real == pytest.approx(-0.25)
    np.testing.assert_allclose(reg, plain, atol=1e-12)


@pytest.mark.parametrize("beta", [0.5, 3.0])
def test_agrees_with_generic_otoc_calculator(beta):
    syk = SYKHamiltonian(8, seed=5)
    calc = OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)
    W, V = calc.local_pauli("X", 0), calc.local_pauli("Z", 2)
    np.testing.assert_allclose(
        thermal_otoc(syk.hamiltonian_matrix, W, V, TIMES, beta=beta),
        calc.compute_regularized_otoc(W, V, TIMES, temperature=1 / beta),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        thermal_otoc(syk.hamiltonian_matrix, W, V, TIMES, beta=beta, regularized=False),
        calc.compute_otoc_exact(W, V, TIMES, temperature=1 / beta),
        atol=1e-10,
    )


def test_syk2_is_hermitian_and_quadratic():
    H = syk2_hamiltonian(10, seed=1).toarray()
    np.testing.assert_allclose(H, H.conj().T, atol=1e-14)
    chi = majoranas(10, half=False)
    commutator = H @ chi[0] - chi[0] @ H
    coefficients = [np.trace(c.conj().T @ commutator) / H.shape[0] for c in chi]
    rebuilt = sum(c * x for c, x in zip(coefficients, chi, strict=True))
    np.testing.assert_allclose(commutator, rebuilt, atol=1e-12)


def test_interacting_syk_scrambles_while_free_syk_does_not():
    N, times = 14, np.array([0.0, 10.0, 14.0])
    chi = majoranas(N)
    syk4 = np.mean(
        [
            thermal_otoc(
                SYKHamiltonian(N, seed=s, chi_squared=0.5).hamiltonian_matrix, chi[0], chi[1], times
            )
            for s in range(4)
        ],
        axis=0,
    )
    syk2 = np.mean(
        [
            thermal_otoc(
                syk2_hamiltonian(N, seed=s, chi_squared=0.5).toarray(), chi[0], chi[1], times
            )
            for s in range(4)
        ],
        axis=0,
    )
    assert np.all(normalized_decay(syk4)[1:] > 0.9)
    assert np.all(normalized_decay(syk2)[1:] < 0.5)


def test_lyapunov_fit_recovers_synthetic_rate():
    t = np.linspace(0, 5, 101)
    decay = 1e-3 * np.exp(1.7 * t)
    res = lyapunov_fit(t, decay, window=(0.01, 0.5))
    assert res["rate"] == pytest.approx(1.7, rel=1e-10)
    assert res["n_points"] > 10
    syst = lyapunov_with_systematics(t, decay)
    assert syst["systematic"] == pytest.approx(0.0, abs=1e-10)
    assert np.isnan(lyapunov_fit(t, np.zeros_like(t))["rate"])
