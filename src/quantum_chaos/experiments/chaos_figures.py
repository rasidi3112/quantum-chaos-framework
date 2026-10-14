from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..chaos.eth import (
    diagonal_fluctuation,
    eigenbasis_matrix_elements,
    eigenstate_entanglement,
    offdiagonal_moment_ratio,
    page_entropy,
)
from ..chaos.krylov import lanczos_coefficients
from ..chaos.operator_size import mean_size_from_anticommutators
from ..chaos.otoc import lyapunov_with_systematics, normalized_decay, thermal_otoc
from ..chaos.sff import (
    plateau_value,
    rmt_form_factor,
    spectral_form_factor,
    thouless_time,
    unfolded_form_factor,
)
from ..hamiltonians.syk_hamiltonian import SYKHamiltonian, syk2_hamiltonian
from ..utils.helpers import hermitian_eigh
from .config import FigureConfig
from .style import CLASS_COLORS, GOE, GUE, HUBBARD, POISSON, SYK, suptitle  # noqa: F401


def _sector_spectra(syk: SYKHamiltonian) -> list[np.ndarray]:
    if syk.symmetry_class == "GUE":
        return [syk.sector_eigenvalues("even")]
    return [syk.sector_eigenvalues("even"), syk.sector_eigenvalues("odd")]


def sff_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.sff is not None
    c = cfg.sff
    times = np.logspace(np.log10(c.t_min), np.log10(c.t_max), c.n_times)
    taus = np.linspace(c.tau_max / c.n_taus, c.tau_max, c.n_taus)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    suptitle(fig, "Spectral Form Factor of SYK")
    data: dict[str, Any] = {}
    for N, n_real in sorted(c.realisations.items()):
        syks = [SYKHamiltonian(N, seed=s) for s in range(n_real)]
        cls = syks[0].symmetry_class
        color = CLASS_COLORS[cls]
        full = [np.concatenate(_sector_spectra(s) * (2 if cls == "GUE" else 1)) for s in syks]
        raw = spectral_form_factor(full, times, beta=c.beta)
        L = 2 ** (N // 2)
        plateau = float(np.mean([plateau_value(e, beta=c.beta) for e in full]))
        axes[0].loglog(times, raw["g"], color=color, label=f"N={N} ({cls})")
        axes[0].axhline(plateau, color=color, ls=":", lw=1)
        sectors = [e for s in syks for e in _sector_spectra(s)]
        unf = unfolded_form_factor(sectors, taus, kramers=cls == "GSE")
        axes[1].errorbar(
            taus,
            unf["K"],
            yerr=unf["stderr"],
            fmt=".",
            color=color,
            label=f"N={N} ({cls})",
            alpha=0.8,
        )
        axes[1].plot(taus, rmt_form_factor(taus, cls), color=color, lw=1.5)
        dip = int(np.argmin(raw["g"]))
        data[str(N)] = {
            "class": cls,
            "realisations": n_real,
            "plateau_times_L": plateau * L,
            "late_time_g_times_L": float(np.mean(raw["g"][-10:]) * L),
            "dip_time": float(times[dip]),
            "dip_value_times_L": float(raw["g"][dip] * L),
            "thouless_tau": thouless_time(taus, unf["K"], cls, unf["stderr"]),
        }
    axes[0].set_xlabel("t")
    axes[0].set_ylabel(rf"$g(\beta={c.beta:g},t)$")
    axes[0].set_title(
        "(a) SFF: dip, ramp, plateau (dotted: exact plateau)", fontweight="bold", fontsize=10
    )
    axes[1].set_xlabel(r"$\tau = t/t_H$ (unfolded)")
    axes[1].set_ylabel(r"$K(\tau)$")
    axes[1].set_ylim(0, 1.6)
    axes[1].set_title(
        "(b) Unfolded connected form factor vs RMT (lines)", fontweight="bold", fontsize=10
    )
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, data


def _majoranas(N: int) -> list[np.ndarray]:
    syk = SYKHamiltonian(N, seed=0)
    return [syk.get_majorana_operator(a) for a in range(N)]


def dynamics_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.dynamics is not None
    c = cfg.dynamics
    N = c.n_majorana
    chi = _majoranas(N)
    half = [x * np.sqrt(0.5) for x in chi]
    times = np.linspace(0, c.t_max, c.n_times)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    suptitle(fig, "Scrambling and Operator Growth: SYK4 vs Free SYK2")
    data: dict[str, Any] = {"otoc": {}, "lanczos": {}, "size": {}, "eth": {}, "page": {}}

    ax = axes[0, 0]
    for model, style in (("SYK4", "-"), ("SYK2", "--")):
        for beta, color in zip(c.betas, (SYK, HUBBARD, GOE), strict=False):
            curves = []
            for s in range(c.realisations):
                if model == "SYK4":
                    H = SYKHamiltonian(N, seed=s, chi_squared=0.5).hamiltonian_matrix
                else:
                    H = syk2_hamiltonian(N, seed=s, chi_squared=0.5).toarray()
                curves.append(thermal_otoc(H, half[0], half[1], times, beta=beta))
            decay = normalized_decay(np.mean(curves, axis=0))
            fit = lyapunov_with_systematics(times, decay)
            ax.plot(times, 1 - decay, ls=style, color=color, label=f"{model}, beta={beta:g}")
            data["otoc"][f"{model}_beta{beta:g}"] = {
                "late_time_f": float(1 - decay[-1]),
                "fitted_rate": fit["rate"],
                "rate_statistical": fit["statistical"],
                "rate_systematic": fit["systematic"],
                "two_pi_T": float(2 * np.pi / beta) if beta > 0 else float("inf"),
            }
    ax.axhline(0, color="gray", lw=0.6)
    ax.set_xlabel("t")
    ax.set_ylabel(r"$F(t)/F(0)$, regularised")
    ax.set_title(f"(a) Majorana OTOC, N={N}, " + r"$\chi^2=1/2$", fontweight="bold")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    Nl = c.lanczos_n_majorana
    chi_l = _majoranas(Nl)
    b4 = lanczos_coefficients(
        SYKHamiltonian(Nl, seed=0).hamiltonian_matrix, chi_l[0], n_max=c.lanczos_steps
    )
    b2 = lanczos_coefficients(syk2_hamiltonian(Nl, seed=0), chi_l[0], n_max=c.lanczos_steps)
    ax.semilogy(np.arange(1, b4.size + 1), b4, "o-", color=SYK, label="SYK4")
    ax.semilogy(
        np.arange(1, b2.size + 1),
        b2,
        "s--",
        color=HUBBARD,
        label=f"SYK2 (Krylov space ends after {b2.size} steps)",
    )
    ax.set_xlabel("n")
    ax.set_ylabel(r"$b_n$")
    ax.set_title(rf"(b) Lanczos coefficients of $\chi_0$, N={Nl} ($\chi^2=1$)", fontweight="bold")
    ax.legend(fontsize=8)
    data["lanczos"] = {"n_majorana": Nl, "syk4": b4.tolist(), "syk2": b2.tolist()}

    ax = axes[1, 0]
    E4, V4 = hermitian_eigh(SYKHamiltonian(N, seed=0).hamiltonian_matrix)
    E2, V2 = hermitian_eigh(syk2_hamiltonian(N, seed=0).toarray())
    size_times = np.linspace(0, c.t_max, 25)
    for label, (E, V), color in (("SYK4", (E4, V4), SYK), ("SYK2", (E2, V2), HUBBARD)):
        sizes = []
        for t in size_times:
            U = (V * np.exp(-1j * E * t)) @ V.conj().T
            sizes.append(mean_size_from_anticommutators(U.conj().T @ chi[0] @ U, chi))
        ax.plot(size_times, sizes, color=color, label=label)
        data["size"][label] = sizes
    ax.axhline(N / 2, color="gray", ls=":", label="N/2")
    ax.set_xlabel("t")
    ax.set_ylabel(r"mean Majorana size of $\chi_0(t)$")
    ax.set_title("(c) Operator growth", fontweight="bold")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    Np = c.page_n_majorana
    n = Np // 2
    syk = SYKHamiltonian(Np, seed=0)
    idx = syk.parity_sector_indices("even")
    E, V = hermitian_eigh(syk.parity_projection("even"))
    half_window = max(1, min(10, E.size // 4))
    mid = np.arange(E.size // 2 - half_window, E.size // 2 + half_window)
    full = np.zeros((2**n, mid.size), dtype=complex)
    full[idx] = V[:, mid]
    ls = list(range(1, n + 1))
    ent = eigenstate_entanglement(full, n, ls, range(mid.size))
    ax.plot(ls, [ent[l].mean() for l in ls], "o", color=SYK, label=f"SYK N={Np} mid-spectrum")
    ax.plot(ls, [page_entropy(2**l, 2 ** (n - l)) for l in ls], "k-", label="Page")
    ax.plot(ls, [l * np.log(2) for l in ls], "k:", label="maximal")
    ax.set_xlabel("qubits in subsystem A")
    ax.set_ylabel("entanglement entropy")
    ax.set_title("(d) Eigenstate entanglement", fontweight="bold")
    ax.legend(fontsize=8)
    data["page"] = {
        str(l): {"eigenstates": float(ent[l].mean()), "page": page_entropy(2**l, 2 ** (n - l))}
        for l in ls
    }
    eth_rows = {}
    for Ne in c.eth_sizes:
        eth_model = SYKHamiltonian(Ne, seed=0)
        sidx = eth_model.parity_sector_indices("even")
        op = np.eye(2 ** (Ne // 2), dtype=complex)
        for a in range(4):
            op = op @ eth_model.get_majorana_operator(a)
        Ee, Om = eigenbasis_matrix_elements(
            eth_model.parity_projection("even"), op[np.ix_(sidx, sidx)]
        )
        eth_rows[str(Ne)] = {
            "class": eth_model.symmetry_class,
            "dim": int(Ee.size),
            "diagonal_fluctuation": diagonal_fluctuation(Ee, Om),
            "offdiagonal_ratio": offdiagonal_moment_ratio(Ee, Om),
        }
    data["eth"] = eth_rows
    fig.tight_layout()
    return fig, data
