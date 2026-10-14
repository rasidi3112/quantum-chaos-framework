import numpy as np
import pytest

from quantum_chaos.utils.helpers import fidelity


def random_pure(dim, rng):
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return psi / np.linalg.norm(psi)


def random_mixed(dim, rank, rng):
    a = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def test_noiseless_compare_ideal_vs_noisy_has_unit_fidelity():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator
    from quantum_chaos.noise.noise_channels import NoiseModel

    noiseless = NoiseModel(depolarizing_rate=0.0, t1=np.inf, t2=np.inf)
    sim = NISQSimulator(n_qubits=1, noise_model=noiseless)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    out = sim.compare_ideal_vs_noisy(0.7 * X + 0.3 * Z, plus, np.array([0.0, 0.5, 1.3]), Z)
    np.testing.assert_allclose(out["fidelity"], 1.0, atol=1e-10)


@pytest.mark.parametrize("dim", [2, 4, 8])
def test_fidelity_of_state_with_itself_is_one(dim):
    rng = np.random.default_rng(dim)
    psi = random_pure(dim, rng)
    pure = np.outer(psi, psi.conj())
    for rho in (
        pure,
        random_mixed(dim, 1, rng),
        random_mixed(dim, 2, rng),
        random_mixed(dim, dim, rng),
    ):
        assert fidelity(rho, rho) == pytest.approx(1.0, abs=1e-10)


def test_pure_states_reduce_to_squared_overlap():
    rng = np.random.default_rng(7)
    for _ in range(5):
        psi, phi = random_pure(4, rng), random_pure(4, rng)
        expected = abs(np.vdot(psi, phi)) ** 2
        got = fidelity(np.outer(psi, psi.conj()), np.outer(phi, phi.conj()))
        assert got == pytest.approx(expected, abs=1e-10)


def test_symmetric_and_bounded():
    rng = np.random.default_rng(8)
    for _ in range(5):
        rho, sigma = random_mixed(4, 2, rng), random_mixed(4, 3, rng)
        f = fidelity(rho, sigma)
        assert 0.0 <= f <= 1.0 + 1e-12
        assert f == pytest.approx(fidelity(sigma, rho), abs=1e-10)


def test_commuting_states_give_classical_fidelity():
    p = np.array([0.5, 0.3, 0.2, 0.0])
    q = np.array([0.1, 0.1, 0.4, 0.4])
    assert fidelity(np.diag(p), np.diag(q)) == pytest.approx(np.sum(np.sqrt(p * q)) ** 2, abs=1e-12)


def test_orthogonal_states_have_zero_fidelity():
    assert fidelity(np.diag([1.0, 0.0]), np.diag([0.0, 1.0])) == pytest.approx(0.0, abs=1e-12)
