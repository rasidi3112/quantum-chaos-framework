import numpy as np
import pytest

from quantum_chaos.utils.helpers import entanglement_entropy, partial_trace


def random_density_matrix(dim, rng):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    rho = a @ a.conj().T
    return rho / np.trace(rho)


def kron_all(*ops):
    out = np.array([[1.0 + 0j]])
    for op in ops:
        out = np.kron(out, op)
    return out


@pytest.mark.parametrize("dims", [(2, 2), (2, 3), (3, 2)])
def test_trace_out_second_factor_of_product_state(dims):
    rng = np.random.default_rng(1)
    rho_a = random_density_matrix(dims[0], rng)
    rho_b = random_density_matrix(dims[1], rng)
    rho = np.kron(rho_a, rho_b)
    np.testing.assert_allclose(partial_trace(rho, [0], list(dims)), rho_a, atol=1e-12)
    np.testing.assert_allclose(partial_trace(rho, [1], list(dims)), rho_b, atol=1e-12)


def test_three_subsystems_keep_non_adjacent():
    rng = np.random.default_rng(2)
    rhos = [random_density_matrix(d, rng) for d in (2, 3, 2)]
    rho = kron_all(*rhos)
    np.testing.assert_allclose(
        partial_trace(rho, [0, 2], [2, 3, 2]), np.kron(rhos[0], rhos[2]), atol=1e-12
    )
    np.testing.assert_allclose(partial_trace(rho, [1], [2, 3, 2]), rhos[1], atol=1e-12)


def test_reduced_state_has_unit_trace():
    rng = np.random.default_rng(3)
    rho = random_density_matrix(8, rng)
    for keep in ([0], [1], [2], [0, 1], [1, 2], [0, 2]):
        assert np.trace(partial_trace(rho, keep, [2, 2, 2])) == pytest.approx(1.0, abs=1e-12)


def test_pure_product_state_reduces_to_projector():
    psi = np.kron([1.0, 0.0], [0.0, 1.0]).astype(complex)
    rho_a = partial_trace(np.outer(psi, psi.conj()), [0], [2, 2])
    np.testing.assert_allclose(rho_a, np.diag([1.0, 0.0]), atol=1e-12)


def test_entropy_of_product_state_is_zero():
    psi = np.kron([1.0, 0.0], [0.0, 1.0]).astype(complex)
    assert entanglement_entropy(psi, [0], 2) == pytest.approx(0.0, abs=1e-12)


def test_entropy_of_bell_state_is_one_bit():
    bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    assert entanglement_entropy(bell, [0], 2) == pytest.approx(1.0, abs=1e-12)


def test_entropy_ghz_and_complementary_subsystems():
    ghz = np.zeros(8, dtype=complex)
    ghz[0] = ghz[-1] = 1 / np.sqrt(2)
    assert entanglement_entropy(ghz, [1], 3) == pytest.approx(1.0, abs=1e-12)
    rng = np.random.default_rng(4)
    psi = rng.normal(size=16) + 1j * rng.normal(size=16)
    psi /= np.linalg.norm(psi)
    s_a = entanglement_entropy(psi, [0], 4)
    s_b = entanglement_entropy(psi, [1, 2, 3], 4)
    assert s_a > 0.1
    assert s_a == pytest.approx(s_b, abs=1e-10)
