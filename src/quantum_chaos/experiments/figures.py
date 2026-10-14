from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import warnings
from collections.abc import Callable
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from ..circuits.otoc_calculator import OTOCCalculator  # noqa: E402
from ..hamiltonians.hubbard_ed import HubbardRingED, hubbard_level_statistics  # noqa: E402
from ..hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian  # noqa: E402
from ..hamiltonians.syk_hamiltonian import (  # noqa: E402
    SYKHamiltonian,
    disorder_average_lsr,
    syk_rmt_reference,
)
from ..spectral import R_MEAN, ratio_density, spacing_ratios  # noqa: E402
from .config import (  # noqa: E402
    FigureConfig,
    HubbardStatsConfig,
    OTOCConfig,
    SYKStatsConfig,
    config_to_dict,
)

CLASS_COLORS = {"GOE": "#1b9e77", "GUE": "#7570b3", "GSE": "#d95f02", "Poisson": "#666666"}


def syk_statistics(cfg: SYKStatsConfig) -> list[dict[str, Any]]:
    rows = []
    for N, n_real in sorted(cfg.realisations.items()):
        data = disorder_average_lsr(N, n_realizations=n_real, trim=cfg.trim, rng=cfg.bootstrap_seed)
        ref = syk_rmt_reference(N, n_samples=cfg.rmt_samples, trim=cfg.trim, rng=cfg.rmt_seed)
        rows.append(
            {
                "N": N,
                "class": data["symmetry_class"],
                "levels_per_sector": data["levels_per_sector"],
                "realisations": n_real,
                "mean": data["mean"],
                "stderr": data["stderr"],
                "rmt_same_size": ref["mean"],
                "rmt_stderr": ref["stderr"],
                "atas": data["reference"],
            }
        )
    return rows


def hubbard_statistics(cfg: HubbardStatsConfig) -> dict[float, dict[str, Any]]:
    return {
        tp: hubbard_level_statistics(cfg.n_sites, cfg.n_up, cfg.n_down, t=cfg.t, tp=tp, U=cfg.U)
        for tp in cfg.tp_values
    }


def infinite_temperature_otocs(cfg: OTOCConfig) -> dict[str, np.ndarray]:
    times = np.linspace(0, cfg.t_max, cfg.n_times)
    syk = SYKHamiltonian(cfg.syk_n, seed=cfg.syk_seed)
    calc = OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)
    syk_otoc = np.real(
        calc.compute_otoc_exact(calc.local_pauli("Z", 0), calc.local_pauli("Z", 1), times)
    )
    hub = DrivenHubbardHamiltonian(
        n_sites=cfg.hubbard_sites,
        hopping_amplitude=1.0,
        onsite_interaction=cfg.hubbard_U,
        periodic_boundary=True,
    )
    hcalc = OTOCCalculator(hub.n_qubits, hub.static_hamiltonian())
    hub_otoc = np.real(
        hcalc.compute_otoc_exact(hcalc.local_pauli("Z", 0), hcalc.local_pauli("Z", 2), times)
    )
    return {"times": times, "syk": syk_otoc, "hubbard": hub_otoc}


def overview_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.syk is not None and cfg.hubbard is not None and cfg.otoc is not None
    syk_rows = syk_statistics(cfg.syk)
    hub = hubbard_statistics(cfg.hubbard)
    otoc = infinite_temperature_otocs(cfg.otoc)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    ax = axes[0]
    for cls in ("GOE", "GUE", "GSE"):
        ax.axhline(
            R_MEAN[cls.lower()],
            color=CLASS_COLORS[cls],
            ls="--",
            lw=1,
            label=f"{cls} (Atas et al. 2013)",
        )
    ax.axhline(R_MEAN["poisson"], color=CLASS_COLORS["Poisson"], ls=":", lw=1, label="Poisson")
    for row in syk_rows:
        color = CLASS_COLORS[row["class"]]
        ax.errorbar(row["N"], row["mean"], yerr=row["stderr"], fmt="o", color=color, capsize=3)
        ax.errorbar(
            row["N"] + 0.3,
            row["rmt_same_size"],
            yerr=row["rmt_stderr"],
            fmt="s",
            mfc="none",
            color=color,
            capsize=3,
        )
    ax.plot([], [], "ko", label="SYK (disorder avg.)")
    ax.plot([], [], "ks", mfc="none", label="RMT, same no. of levels")
    ax.set_xlabel("N (Majorana fermions)")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_title("(a) SYK: class set by N mod 8", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    labels = [f"t'={tp}" for tp in hub]
    ax.bar(
        labels,
        [r["mean"] for r in hub.values()],
        yerr=[r["stderr"] for r in hub.values()],
        color=["#999999", "#1b9e77", "#7570b3", "#d95f02"][: len(hub)],
        capsize=4,
    )
    ax.axhline(R_MEAN["poisson"], color=CLASS_COLORS["Poisson"], ls=":", label="Poisson")
    ax.axhline(R_MEAN["goe"], color=CLASS_COLORS["GOE"], ls="--", label="GOE")
    ax.set_ylim(0.3, 0.6)
    ax.set_ylabel(r"$\langle r\rangle$")
    h = cfg.hubbard
    ax.set_title(
        f"(b) Hubbard ring L={h.n_sites}, N={h.n_up}+{h.n_down}, U={h.U:g}\n"
        "(K, reflection, S resolved; eta for t'=0)",
        fontweight="bold",
        fontsize=10,
    )
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(otoc["times"], otoc["syk"], color="#d95f02", lw=2, label=f"SYK N={cfg.otoc.syk_n}")
    ax.plot(
        otoc["times"],
        otoc["hubbard"],
        color="#3498db",
        lw=2,
        label=f"Hubbard L={cfg.otoc.hubbard_sites} (t'=0)",
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"Re $F(t)$, $T=\infty$")
    ax.set_title("(c) OTOC, single realisation", fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()

    data = {
        "syk": syk_rows,
        "hubbard": {
            str(tp): {k: r[k] for k in ("mean", "stderr", "n_ratios", "n_blocks")}
            for tp, r in hub.items()
        },
        "otoc": {k: np.asarray(v).tolist() for k, v in otoc.items()},
    }
    return fig, data


def ratio_distribution_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.syk is not None and cfg.hubbard is not None
    grid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    data: dict[str, Any] = {"syk": {}, "hubbard": {}}

    ax = axes[0]
    for N, n_real in sorted(cfg.syk.realisations.items()):
        ratios = np.concatenate(
            [SYKHamiltonian(N, seed=s).spacing_ratios(trim=cfg.syk.trim) for s in range(n_real)]
        )
        cls = SYKHamiltonian(N, seed=0).symmetry_class
        ax.hist(
            ratios,
            bins=cfg.bins,
            range=(0, 1),
            density=True,
            histtype="step",
            lw=2,
            color=CLASS_COLORS[cls],
            label=f"SYK N={N} ({cls}, n={ratios.size})",
        )
        ax.plot(grid, ratio_density(grid, cls), color=CLASS_COLORS[cls], ls="--", lw=1)
        data["syk"][str(N)] = {
            "class": cls,
            "n_ratios": int(ratios.size),
            "mean": float(ratios.mean()),
        }
    ax.set_title("(a) SYK sectors vs surmise (dashed)", fontweight="bold")

    ax = axes[1]
    h = cfg.hubbard
    for tp, color in zip(h.tp_values, ["#666666", "#1b9e77", "#7570b3"], strict=False):
        blocks = HubbardRingED(h.n_sites, h.n_up, h.n_down, t=h.t, tp=tp, U=h.U).sector_blocks()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ratios = np.concatenate(
                [spacing_ratios(b["energies"]) for b in blocks if b["energies"].size >= 10]
            )
        ax.hist(
            ratios,
            bins=cfg.bins,
            range=(0, 1),
            density=True,
            histtype="step",
            lw=2,
            color=color,
            label=f"t'={tp} (n={ratios.size})",
        )
        data["hubbard"][str(tp)] = {"n_ratios": int(ratios.size), "mean": float(ratios.mean())}
    ax.plot(grid, ratio_density(grid, "poisson"), "k:", label="Poisson")
    ax.plot(grid, ratio_density(grid, "goe"), "k--", label="GOE (surmise)")
    ax.set_title(
        f"(b) Hubbard ring L={h.n_sites}, N={h.n_up}+{h.n_down}, U={h.U:g}", fontweight="bold"
    )
    for ax in axes:
        ax.set_xlabel(r"$\tilde r$")
        ax.set_ylabel(r"$P(\tilde r)$")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, data


from .chaos_figures import dynamics_figure, sff_figure  # noqa: E402
from .floquet_figures import floquet_figure  # noqa: E402
from .nisq_figures import nisq_figure  # noqa: E402
from .qml_figures import qml_figure  # noqa: E402

BUILDERS: dict[str, Callable[[FigureConfig], tuple[Figure, dict[str, Any]]]] = {
    "overview": overview_figure,
    "ratio_distributions": ratio_distribution_figure,
    "sff": sff_figure,
    "chaos_dynamics": dynamics_figure,
    "floquet_heating": floquet_figure,
    "nisq": nisq_figure,
    "qml": qml_figure,
}


def _git(args: list[str], cwd: pathlib.Path) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def provenance(root: pathlib.Path) -> dict[str, Any]:
    import pennylane
    import scipy

    import quantum_chaos

    return {
        "git_commit": _git(["rev-parse", "--short", "HEAD"], root),
        "src_tree": _git(["rev-parse", "--short", "HEAD:src"], root),
        "dirty": bool(_git(["status", "--porcelain", "--", "src"], root)),
        "package_version": quantum_chaos.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pennylane": pennylane.__version__,
        "platform": platform.platform(),
    }


def make_figure(cfg: FigureConfig, root: str | pathlib.Path = ".") -> pathlib.Path:
    root = pathlib.Path(root)
    if cfg.kind not in BUILDERS:
        raise ValueError(f"unknown figure kind {cfg.kind!r}; expected one of {sorted(BUILDERS)}")
    fig, data = BUILDERS[cfg.kind](cfg)
    out = root / cfg.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    sidecar = {"config": config_to_dict(cfg), "provenance": provenance(root), "data": data}
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=float) + "\n")
    return out
