import numpy as np
import pytest
from sklearn.datasets import make_blobs

from quantum_chaos.qml.benchmark import KernelModel, compare_with_rbf_baseline
from quantum_chaos.qml.concentration import (
    effective_dimension_of_states,
    haar_fidelity_moments,
    kernel_offdiagonal_statistics,
)
from quantum_chaos.qml.quantum_kernel import QuantumKernel, SYKKernel


def test_haar_fidelity_moments_match_sampling():
    rng = np.random.default_rng(0)
    D = 16
    v = rng.normal(size=(6000, D)) + 1j * rng.normal(size=(6000, D))
    v /= np.linalg.norm(v, axis=1)[:, None]
    f = np.abs(np.sum(v[:3000].conj() * v[3000:], axis=1)) ** 2
    exact = haar_fidelity_moments(D)
    assert f.mean() == pytest.approx(exact["mean"], rel=0.05)
    assert f.var() == pytest.approx(exact["variance"], rel=0.1)


def test_syk_kernel_concentrates_exponentially_at_long_times():
    small = kernel_offdiagonal_statistics(SYKKernel(3, seed=0, evolution_time=10.0), 30, rng=1)
    large = kernel_offdiagonal_statistics(SYKKernel(6, seed=0, evolution_time=10.0), 30, rng=1)
    assert large["variance"] < small["variance"] / 10
    short = kernel_offdiagonal_statistics(SYKKernel(6, seed=0, evolution_time=0.1), 30, rng=1)
    assert short["mean"] > 0.99


def test_effective_dimension_of_identical_states_is_one():
    kernel = QuantumKernel(2, hamiltonian_constructor=lambda x: np.zeros((4, 4)))
    X = np.random.default_rng(0).normal(size=(5, 2))
    assert effective_dimension_of_states(kernel, X) == pytest.approx(1.0)


def test_rbf_comparison_reports_paired_statistics():
    def dataset(seed):
        return make_blobs(40, centers=2, cluster_std=0.5, random_state=seed)

    model = KernelModel("syk", lambda t: SYKKernel(3, seed=0, evolution_time=t), (0.5,))
    out = compare_with_rbf_baseline(
        dataset, [model], seeds=range(3), C_grid=(1.0,), gamma_grid=(0.5,)
    )
    assert out["n_seeds"] == 3
    assert out["results"]["rbf"]["mean"] == pytest.approx(1.0)
    row = out["results"]["syk"]
    assert {"mean", "std", "mean_difference_vs_rbf", "wilcoxon_p"} <= set(row)
    assert 0.0 <= row["wilcoxon_p"] <= 1.0
