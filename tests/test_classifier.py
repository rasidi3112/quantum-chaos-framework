"""Kernel SVM wrapper and the (descriptive) SYK-vs-Hubbard kernel benchmark."""

import numpy as np
import pytest

from quantum_chaos.qml.quantum_classifier import (
    QuantumClassifier,
    QuantumKernelClassificationBenchmark,
    generate_quantum_embedded_features,
)
from quantum_chaos.qml.quantum_kernel import QuantumKernel


class GaussianKernel(QuantumKernel):
    """Classical RBF kernel through the same interface (a sanity baseline)."""

    def __init__(self, gamma=0.5):
        super().__init__(n_qubits=1)
        self.gamma = gamma

    def kernel_matrix(self, X1, X2=None):
        X2 = X1 if X2 is None else X2
        d2 = ((X1[:, None, :] - X2[None, :, :]) ** 2).sum(-1)
        return np.exp(-self.gamma * d2)


def blobs(rng, n=40):
    X = np.vstack([rng.normal(-2, 0.3, size=(n // 2, 2)), rng.normal(2, 0.3, size=(n // 2, 2))])
    y = np.repeat([0, 1], n // 2)
    return X, y


def test_svm_wrapper_separates_well_separated_blobs():
    rng = np.random.default_rng(0)
    X, y = blobs(rng)
    clf = QuantumClassifier(GaussianKernel())
    with pytest.raises(RuntimeError):
        clf.predict(X)
    clf.fit(X, y)
    assert clf.score(X, y) == pytest.approx(1.0)
    report = clf.evaluate(X, y)
    assert report["accuracy"] == pytest.approx(1.0)
    assert set(report["predictions"]) == {0, 1}


def test_benchmark_runs_and_reports_descriptive_statistics_only():
    bench = QuantumKernelClassificationBenchmark(n_qubits=3, evolution_time=0.5)
    out = bench.run_full_comparison(n_trials=2, n_samples=16)
    assert out["n_trials"] == 2 and len(out["syk_accuracies"]) == 2
    for key in ("syk_mean", "syk_std", "hubbard_mean", "hubbard_std"):
        assert 0.0 <= out[key] <= 1.0
    assert not any("advantage" in k or "conclusion" in k or "wins" in k for k in out)
    single = bench.run_benchmark(*_split(bench))
    assert {"syk", "hubbard", "expressivity", "accuracy_difference"} <= set(single)


def _split(bench):
    X_train, X_test, y_train, y_test = bench.generate_classification_data(
        n_samples=16, n_features=3, random_state=1
    )
    return X_train, y_train, X_test, y_test


def test_kernel_pca_features():
    rng = np.random.default_rng(1)
    X, _ = blobs(rng, n=12)
    feats = generate_quantum_embedded_features(GaussianKernel(), X, n_components=3)
    assert feats.shape == (12, 3)
    # centred kernel -> embedded features have zero mean
    np.testing.assert_allclose(feats.mean(axis=0), 0, atol=1e-10)
