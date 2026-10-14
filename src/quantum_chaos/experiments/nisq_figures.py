from __future__ import annotations

import warnings
from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..circuits.noisy_otoc import NoisyTrotterOTOC, zero_noise_extrapolated
from ..circuits.otoc_calculator import OTOCCalculator
from ..circuits.resources import syk_pauli_terms, trotter_step_cost
from ..circuits.trotter_bounds import first_order_commutator_bound, trotter_error
from ..circuits.trotter_evolution import matrix_to_pauli_terms
from ..hamiltonians.syk_hamiltonian import syk2_hamiltonian, syk_symmetry_class
from ..hamiltonians.syk_sparse import sparse_syk_couplings, syk_couplings, syk_sector_hamiltonian
from ..noise.noise_channels import NoiseModel
from ..spectral import R_MEAN, spacing_ratios
from ..utils.helpers import hermitian_eigvalsh
from ..utils.jordan_wigner import pauli_operators
from .config import FigureConfig


def _pauli_matrix(label: str) -> np.ndarray:
    P = pauli_operators()
    m = np.array([[1.0 + 0j]])
    for c in label:
        m = np.kron(m, P[c])
    return m


def nisq_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.nisq is not None
    c = cfg.nisq
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    data: dict[str, Any] = {}

    terms = syk_pauli_terms(c.bound_n_majorana, syk_couplings(c.bound_n_majorana, 1))
    mats = [coef * _pauli_matrix(s) for coef, s in terms]
    steps = np.array(c.bound_steps)
    bound = [first_order_commutator_bound(mats, c.bound_time, int(r)) for r in steps]
    err1 = [trotter_error(mats, c.bound_time, int(r), 1) for r in steps]
    err2 = [trotter_error(mats, c.bound_time, int(r), 2) for r in steps]
    ax = axes[0, 0]
    ax.loglog(steps, bound, "k--", label="1st-order commutator bound (Childs et al.)")
    ax.loglog(steps, err1, "o-", label="1st order, actual")
    ax.loglog(steps, err2, "s-", label="2nd order, actual")
    ax.set_xlabel("Trotter steps r")
    ax.set_ylabel(r"$\|U_{\rm Trotter}-e^{-iHt}\|$")
    ax.set_title(
        f"(a) Trotter error, SYK N={c.bound_n_majorana}, t={c.bound_time:g}", fontweight="bold"
    )
    ax.legend(fontsize=8)
    data["trotter"] = {"steps": steps.tolist(), "bound": bound, "order1": err1, "order2": err2}

    dense_cost, sparse_cost = [], []
    for N in c.cost_sizes:
        dense_cost.append(trotter_step_cost(syk_pauli_terms(N, syk_couplings(N, 0)))["cnot"])
        sparse_cost.append(
            trotter_step_cost(syk_pauli_terms(N, sparse_syk_couplings(N, c.sparse_k, 0)))["cnot"]
        )
    ax = axes[0, 1]
    ax.semilogy(c.cost_sizes, dense_cost, "o-", label="dense SYK")
    ax.semilogy(c.cost_sizes, sparse_cost, "s-", label=f"sparse SYK, k={c.sparse_k:g}")
    ax.set_xlabel("N")
    ax.set_ylabel("CNOTs per first-order Trotter step")
    ax.set_title("(b) Circuit cost (CNOT-ladder decomposition)", fontweight="bold")
    ax.legend(fontsize=8)
    data["cnot_per_step"] = {"N": list(c.cost_sizes), "dense": dense_cost, "sparse": sparse_cost}

    ax = axes[1, 0]
    data["sparse_statistics"] = {}
    for N in c.sparse_stat_sizes:
        cls = syk_symmetry_class(N)
        sectors = ("even",) if cls == "GUE" else ("even", "odd")
        means, errs = [], []
        for k in c.sparse_k_values:
            vals = []
            for seed in range(c.sparse_realisations):
                J = sparse_syk_couplings(N, k, seed)
                for sec in sectors:
                    e = hermitian_eigvalsh(
                        syk_sector_hamiltonian(N, couplings=J, sector=sec).toarray()
                    )
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        vals.append(spacing_ratios(e, kramers=cls == "GSE").mean())
            means.append(float(np.mean(vals)))
            errs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        ax.errorbar(
            c.sparse_k_values, means, yerr=errs, fmt="o-", capsize=3, label=f"N={N} ({cls})"
        )
        ax.axhline(R_MEAN[cls.lower()], ls="--", lw=1, color=ax.lines[-1].get_color())
        data["sparse_statistics"][str(N)] = {
            "class": cls,
            "k": list(c.sparse_k_values),
            "mean": means,
            "stderr": errs,
        }
    ax.axhline(R_MEAN["poisson"], color="gray", ls=":", label="Poisson")
    ax.set_xscale("log")
    ax.set_xlabel("k (terms per Majorana)")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_title("(c) Sparse SYK level statistics", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    N = c.otoc_n_majorana
    n = N // 2
    times = list(c.otoc_times)
    noise = NoiseModel(depolarizing_rate=c.depolarizing, t1=np.inf, t2=np.inf)
    W, V = ("Z", 0), ("X", n - 1)
    data["noisy_otoc"] = {"times": times, "depolarizing": c.depolarizing}
    models = {
        "SYK4": syk_pauli_terms(N, syk_couplings(N, 1)),
        "SYK2": matrix_to_pauli_terms(syk2_hamiltonian(N, seed=1).toarray(), n),
    }
    for (name, terms_m), color in zip(models.items(), ("#d95f02", "#7570b3"), strict=True):
        H = np.sum([coef * _pauli_matrix(s) for coef, s in terms_m], axis=0)
        calc = OTOCCalculator(n, H)
        exact = np.real(
            calc.compute_otoc_exact(calc.local_pauli(*W), calc.local_pauli(*V), np.array(times))
        )
        noisy = NoisyTrotterOTOC(terms_m, n, n_steps=c.otoc_steps, noise=noise)
        raw = np.array([noisy.otoc(W, V)(t) for t in times])
        echo = np.array([noisy.echo(V)(t) for t in times])

        def folded(fold: int, t: float, protocol: NoisyTrotterOTOC = noisy) -> float:
            return protocol.otoc(W, V, fold=fold)(t)

        zne = np.array(
            [zero_noise_extrapolated(partial(folded, t=t), c.folds)["extrapolated"] for t in times]
        )
        ax.plot(times, exact, "-", color=color, label=f"{name} exact")
        ax.plot(times, raw, "x", color=color, label=f"{name} noisy")
        ax.plot(times, raw / echo, "o", mfc="none", color=color, label=f"{name} noisy / echo")
        ax.plot(times, zne, "^", color=color, label=f"{name} ZNE")
        data["noisy_otoc"][name] = {
            "exact": exact.tolist(),
            "raw": raw.tolist(),
            "echo": echo.tolist(),
            "zne": zne.tolist(),
        }
    ax.set_xlabel("t")
    ax.set_ylabel(r"Re $F(t)$, $T=\infty$")
    ax.set_title(f"(d) Noisy interferometric OTOC, N={N}, p={c.depolarizing:g}", fontweight="bold")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig, data
