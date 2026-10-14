import numpy as np
import pytest
from scipy.linalg import expm

from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian


def fold(energies, omega):
    return ((np.asarray(energies) + omega / 2) % omega) - omega / 2


@pytest.mark.parametrize("U, omega", [(1.3, 10.0), (7.0, 10.0)])
def test_static_hamiltonian_quasi_energies_are_folded_eigenvalues(U, omega):
    model = DrivenHubbardHamiltonian(
        n_sites=2, hopping_amplitude=0.0, driving_frequency=omega, onsite_interaction=U
    )
    expected = np.sort(fold(np.linalg.eigvalsh(model.interaction_matrix), omega))
    np.testing.assert_allclose(model.quasi_energies(n_steps=8), expected, atol=1e-10)


def test_static_floquet_operator_is_exact_exponential():
    model = DrivenHubbardHamiltonian(
        n_sites=2, hopping_amplitude=0.0, driving_frequency=3.0, onsite_interaction=1.1
    )
    T = 2 * np.pi / model.driving_frequency
    np.testing.assert_allclose(
        model.floquet_operator(n_steps=5), expm(-1j * model.interaction_matrix * T), atol=1e-12
    )


def test_floquet_operator_is_unitary():
    F = DrivenHubbardHamiltonian(n_sites=2, driving_frequency=2.0).floquet_operator(n_steps=50)
    np.testing.assert_allclose(F.conj().T @ F, np.eye(F.shape[0]), atol=1e-12)


def test_floquet_operator_converges_at_second_order():
    model = DrivenHubbardHamiltonian(
        n_sites=2, hopping_amplitude=1.0, driving_frequency=2.0, onsite_interaction=1.0
    )
    reference = model.floquet_operator(n_steps=3200)
    steps = np.array([50, 100, 200])
    errors = [np.linalg.norm(model.floquet_operator(n_steps=n) - reference, 2) for n in steps]
    slope = -np.polyfit(np.log(steps), np.log(errors), 1)[0]
    assert slope == pytest.approx(2.0, abs=0.15)
