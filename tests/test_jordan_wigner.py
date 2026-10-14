import numpy as np
import pytest

from quantum_chaos.utils.jordan_wigner import (
    JordanWignerMapper,
    get_pauli_string_representation,
    jordan_wigner_transform,
    pauli_operators,
)


def anticommutator(a, b):
    return a @ b + b @ a


@pytest.mark.parametrize("n", [1, 2, 3])
def test_canonical_anticommutation_relations(n):
    jw = JordanWignerMapper(n)
    c = [jw.annihilation_operator(j) for j in range(n)]
    cd = [jw.creation_operator(j) for j in range(n)]
    dim = 2**n
    for i in range(n):
        np.testing.assert_allclose(cd[i], c[i].conj().T, atol=0)
        for j in range(n):
            expected = np.eye(dim) if i == j else np.zeros((dim, dim))
            np.testing.assert_allclose(anticommutator(c[i], cd[j]), expected, atol=1e-14)
            np.testing.assert_allclose(anticommutator(c[i], c[j]), 0, atol=1e-14)


@pytest.mark.parametrize("n", [2, 3])
def test_majorana_algebra_and_relation_to_fermions(n):
    jw = JordanWignerMapper(n)
    chi = [jw.majorana_operator(a) for a in range(2 * n)]
    dim = 2**n
    for a in range(2 * n):
        np.testing.assert_allclose(chi[a], chi[a].conj().T, atol=0)
        for b in range(2 * n):
            expected = 2 * np.eye(dim) if a == b else np.zeros((dim, dim))
            np.testing.assert_allclose(anticommutator(chi[a], chi[b]), expected, atol=1e-14)
    for j in range(n):
        c = jw.annihilation_operator(j)
        np.testing.assert_allclose(c, (chi[2 * j] + 1j * chi[2 * j + 1]) / 2, atol=1e-14)
        np.testing.assert_allclose(jw.number_operator(j), jw.creation_operator(j) @ c, atol=1e-14)


def test_transform_dispatch_and_errors():
    jw = JordanWignerMapper(3)
    np.testing.assert_allclose(jordan_wigner_transform("creation", 1, 3), jw.creation_operator(1))
    np.testing.assert_allclose(jordan_wigner_transform("number", 2, 3), jw.number_operator(2))
    np.testing.assert_allclose(
        jordan_wigner_transform("hopping", (0, 2), 3), jw.hopping_operator(0, 2)
    )
    np.testing.assert_allclose(
        jordan_wigner_transform("interaction", (0, 1), 3), jw.interaction_operator(0, 1)
    )
    with pytest.raises(TypeError):
        jordan_wigner_transform("hopping", 1, 3)
    with pytest.raises(TypeError):
        jordan_wigner_transform("creation", (0, 1), 3)
    with pytest.raises(ValueError):
        jordan_wigner_transform("bogus", 0, 3)


def test_pauli_decomposition_round_trip():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    paulis = pauli_operators()
    rebuilt = np.zeros((8, 8), dtype=complex)
    for coeff, label in get_pauli_string_representation(a, 3):
        m = np.array([[1.0 + 0j]])
        for p in label:
            m = np.kron(m, paulis[p])
        rebuilt += coeff * m
    np.testing.assert_allclose(rebuilt, a, atol=1e-12)
