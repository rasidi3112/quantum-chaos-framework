"""Fidelity quantum kernels with Hamiltonian encoding (PLAN P0-7)."""

import numpy as np
import pytest

from quantum_chaos.qml.quantum_kernel import HubbardKernel, SYKKernel


def data(n, d, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d))


@pytest.mark.parametrize(
    "make", [lambda: SYKKernel(n_qubits=3, seed=1), lambda: HubbardKernel(n_sites=2)]
)
def test_kernel_matrix_is_symmetric_psd_with_unit_diagonal(make):
    kernel = make()
    K = kernel.kernel_matrix(data(6, 4))
    np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-12)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(K)) > -1e-10


def test_hubbard_kernel_is_not_constant():
    K = HubbardKernel(n_sites=2).kernel_matrix(data(8, 2))
    off_diagonal = K[np.triu_indices_from(K, k=1)]
    assert np.var(off_diagonal) > 1e-4
    assert np.min(off_diagonal) < 0.99


def test_syk_kernel_resolves_inputs_with_equal_mean():
    kernel = SYKKernel(n_qubits=3, seed=1)
    x1 = np.array([1.0, -1.0, 2.0, 0.0])
    x2 = np.array([0.5, 0.5, 0.5, 0.5])
    assert kernel.kernel_value(x1, x2) < 1 - 1e-3


def test_every_feature_changes_the_feature_state():
    for kernel in (SYKKernel(n_qubits=3, seed=3), HubbardKernel(n_sites=2)):
        x = np.array([0.3, -0.7, 1.1, 0.4])
        for j in range(len(x)):
            y = x.copy()
            y[j] += 0.5
            assert kernel.kernel_value(x, y) < 1 - 1e-4, (type(kernel).__name__, j)


def test_pennylane_kernel_matches_statevector_kernel():
    for kernel in (SYKKernel(n_qubits=3, seed=2), HubbardKernel(n_sites=2)):
        circuit_kernel = kernel.create_pennylane_kernel()
        X = data(3, 3, seed=5)
        for x1 in X:
            assert circuit_kernel(x1, x1) == pytest.approx(1.0, abs=1e-8)
            for x2 in X:
                assert circuit_kernel(x1, x2) == pytest.approx(
                    kernel.kernel_value(x1, x2), abs=1e-8
                )


def test_syk_kernel_rejects_trivial_n4_model():
    # N = 4: H_SYK = -J Z0 Z1 commutes with all Z_j fields -> K == 1
    with pytest.raises(ValueError):
        SYKKernel(n_qubits=2)
