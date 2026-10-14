from __future__ import annotations

from functools import partial
from math import comb
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.datasets import make_circles, make_moons

from ..qml.benchmark import KernelModel, compare_with_rbf_baseline
from ..qml.concentration import haar_fidelity_moments, kernel_offdiagonal_statistics
from ..qml.quantum_kernel import HubbardKernel, QuantumKernel, SYKKernel
from .config import FigureConfig


def _syk_kernel(t: float, n: int) -> QuantumKernel:
    return SYKKernel(n, seed=0, evolution_time=t)


def _hubbard_kernel(t: float, sites: int) -> QuantumKernel:
    return HubbardKernel(sites, evolution_time=t)


def _moons(seed: int, size: int):
    return make_moons(size, noise=0.25, random_state=seed)


def _circles(seed: int, size: int):
    return make_circles(size, noise=0.1, factor=0.5, random_state=seed)


def qml_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.qml is not None
    c = cfg.qml
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    data: dict[str, Any] = {"concentration": {"syk": {}, "hubbard": {}}, "classification": {}}

    for t in c.times:
        var = [
            kernel_offdiagonal_statistics(
                SYKKernel(n, seed=0, evolution_time=t), c.n_samples, rng=1
            )["variance"]
            for n in c.syk_qubits
        ]
        axes[0].semilogy(c.syk_qubits, var, "o-", label=f"SYK kernel, t={t:g}")
        data["concentration"]["syk"][str(t)] = var
    haar = [haar_fidelity_moments(2**n)["variance"] for n in c.syk_qubits]
    axes[0].semilogy(c.syk_qubits, haar, "k--", label="Haar, D = 2^n")
    axes[0].set_xlabel("qubits n")
    axes[0].set_ylabel(r"Var$[K(x,x')]$")
    axes[0].set_title("(a) SYK kernel concentration", fontweight="bold")
    axes[0].legend(fontsize=7)

    for t in c.times:
        var = [
            kernel_offdiagonal_statistics(
                HubbardKernel(L, evolution_time=t), c.n_samples, n_features=L, rng=1
            )["variance"]
            for L in c.hubbard_sites
        ]
        axes[1].semilogy(
            [2 * L for L in c.hubbard_sites], var, "s-", label=f"Hubbard kernel, t={t:g}"
        )
        data["concentration"]["hubbard"][str(t)] = var
    sector_haar = [
        haar_fidelity_moments(comb(L, (L + 1) // 2) * comb(L, L // 2))["variance"]
        for L in c.hubbard_sites
    ]
    axes[1].semilogy(
        [2 * L for L in c.hubbard_sites], sector_haar, "k--", label="Haar in the (N, S_z) sector"
    )
    axes[1].set_xlabel("qubits 2L")
    axes[1].set_ylabel(r"Var$[K(x,x')]$")
    axes[1].set_title("(b) Hubbard kernel concentration", fontweight="bold")
    axes[1].legend(fontsize=7)

    models = [
        KernelModel("SYK kernel", partial(_syk_kernel, n=3), c.evolution_grid),
        KernelModel("Hubbard kernel", partial(_hubbard_kernel, sites=2), c.evolution_grid),
    ]
    width = 0.25
    for i, (name, ds) in enumerate(
        (
            ("moons", partial(_moons, size=c.dataset_size)),
            ("circles", partial(_circles, size=c.dataset_size)),
        )
    ):
        res = compare_with_rbf_baseline(ds, models, range(c.seeds))["results"]
        data["classification"][name] = res
        for j, model in enumerate(("rbf", "SYK kernel", "Hubbard kernel")):
            axes[2].bar(
                i + (j - 1) * width,
                res[model]["mean"],
                width,
                yerr=res[model]["std"],
                capsize=3,
                color=("#666666", "#d95f02", "#3498db")[j],
                label=model if i == 0 else None,
            )
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["moons", "circles"])
    axes[2].set_ylim(0.5, 1.05)
    axes[2].set_ylabel(f"test accuracy (mean ± std, {c.seeds} seeds)")
    axes[2].set_title("(c) Kernel SVMs vs tuned RBF baseline", fontweight="bold")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    return fig, data
