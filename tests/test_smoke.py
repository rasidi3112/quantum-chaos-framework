"""Smoke tests for the src-layout package (no physics assertions)."""

import numpy as np

import quantum_chaos


def test_version_attribute():
    assert isinstance(quantum_chaos.__version__, str)


def test_light_subpackages_import_without_pennylane_side_effects():
    from quantum_chaos import hamiltonians, utils  # noqa: F401


def test_syk_hamiltonian_is_hermitian():
    from quantum_chaos.hamiltonians import SYKHamiltonian

    H = SYKHamiltonian(n_majorana=8, seed=0).hamiltonian_matrix
    assert H.shape == (16, 16)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)


def test_hubbard_hamiltonian_is_hermitian():
    from quantum_chaos.hamiltonians import DrivenHubbardHamiltonian

    H = DrivenHubbardHamiltonian(n_sites=2).static_hamiltonian()
    assert H.shape == (16, 16)
    np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
