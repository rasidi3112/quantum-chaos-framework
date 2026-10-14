#!/usr/bin/env python3
"""Quick-start demonstration: symmetry-resolved level statistics and OTOCs.

Regenerates ``output.png`` (all random seeds fixed). Runtime ~1-2 minutes.
For the validation tables see ``scripts/validate_level_statistics.py``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from quantum_chaos.circuits.otoc_calculator import OTOCCalculator  # noqa: E402
from quantum_chaos.hamiltonians.hubbard_ed import hubbard_level_statistics  # noqa: E402
from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian  # noqa: E402
from quantum_chaos.hamiltonians.syk_hamiltonian import (  # noqa: E402
    SYKHamiltonian,
    disorder_average_lsr,
    syk_rmt_reference,
)
from quantum_chaos.spectral import R_MEAN  # noqa: E402

CLASS_COLORS = {"GOE": "#1b9e77", "GUE": "#7570b3", "GSE": "#d95f02", "Poisson": "#666666"}

SYK_SIZES = {10: 300, 12: 200, 14: 150, 16: 100}   # N -> disorder realisations
HUBBARD = dict(n_sites=8, n_up=4, n_down=4, t=1.0, U=4.0)


def banner(text):
    print(f"\n{text}\n" + "-" * len(text))


def main():
    print("=" * 64)
    print("  Quantum chaos framework - symmetry-resolved diagnostics")
    print("=" * 64)

    banner("[1] SYK level statistics (sector resolved, class by N mod 8)")
    syk_rows = []
    for N, n_real in SYK_SIZES.items():
        data = disorder_average_lsr(N, n_realizations=n_real, rng=0)
        ref = syk_rmt_reference(N, n_samples=600, rng=1)
        syk_rows.append((N, data, ref))
        print(f"  N={N:2d} ({data['symmetry_class']}, {data['levels_per_sector']:4d} levels/sector, "
              f"{n_real} realisations): <r> = {data['mean']:.4f} +/- {data['stderr']:.4f} | "
              f"RMT same size {ref['mean']:.4f} +/- {ref['stderr']:.4f} | "
              f"large-N {data['reference']:.4f}")

    banner("[2] Hubbard ring, L=8, N=4+4, U=4: integrable (t'=0) vs t'=0.5")
    hub_rows = {}
    for tp in (0.0, 0.5):
        res = hubbard_level_statistics(tp=tp, **HUBBARD)
        hub_rows[tp] = res
        print(f"  t'={tp}: <r> = {res['mean']:.4f} +/- {res['stderr']:.4f} "
              f"({res['n_ratios']} ratios, {res['n_blocks']} symmetry blocks, "
              f"eta resolved: {res['eta_resolved']}, exact degeneracies left: "
              f"{100 * res['degenerate_fraction']:.1f}%)")
    print(f"  references: Poisson {R_MEAN['poisson']:.4f}, GOE {R_MEAN['goe']:.4f}")

    banner("[3] Infinite-temperature OTOC F(t) = Tr[Z0(t) Z1 Z0(t) Z1] / D")
    times = np.linspace(0, 6, 61)
    syk = SYKHamiltonian(12, seed=7)
    syk_calc = OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)
    syk_otoc = np.real(syk_calc.compute_otoc_exact(syk_calc.local_pauli("Z", 0),
                                                   syk_calc.local_pauli("Z", 1), times))
    hub = DrivenHubbardHamiltonian(n_sites=4, hopping_amplitude=1.0, onsite_interaction=4.0,
                                   periodic_boundary=True)
    hub_calc = OTOCCalculator(hub.n_qubits, hub.static_hamiltonian())
    hub_otoc = np.real(hub_calc.compute_otoc_exact(hub_calc.local_pauli("Z", 0),
                                                   hub_calc.local_pauli("Z", 2), times))
    print(f"  SYK N=12 (seed 7): F(6) = {syk_otoc[-1]:+.4f}; "
          f"Hubbard L=4 (Z on site 0 up / site 1 up): F(6) = {hub_otoc[-1]:+.4f}")
    print("  (single realisation, small systems: qualitative only)")

    make_figure(syk_rows, hub_rows, times, syk_otoc, hub_otoc)
    print("\nSaved output.png")


def make_figure(syk_rows, hub_rows, times, syk_otoc, hub_otoc):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for cls in ("GOE", "GUE", "GSE"):
        ax.axhline(R_MEAN[cls.lower()], color=CLASS_COLORS[cls], ls="--", lw=1,
                   label=f"{cls} (Atas et al. 2013)")
    ax.axhline(R_MEAN["poisson"], color=CLASS_COLORS["Poisson"], ls=":", lw=1, label="Poisson")
    for N, data, ref in syk_rows:
        color = CLASS_COLORS[data["symmetry_class"]]
        ax.errorbar(N, data["mean"], yerr=data["stderr"], fmt="o", color=color, capsize=3)
        ax.errorbar(N + 0.3, ref["mean"], yerr=ref["stderr"], fmt="s", mfc="none", color=color,
                    capsize=3)
    ax.plot([], [], "ko", label="SYK (disorder avg.)")
    ax.plot([], [], "ks", mfc="none", label="RMT, same no. of levels")
    ax.set_xlabel("N (Majorana fermions)")
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_title("(a) SYK: class set by N mod 8", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

    ax = axes[1]
    labels = [f"t'={tp}" for tp in hub_rows]
    means = [r["mean"] for r in hub_rows.values()]
    errs = [r["stderr"] for r in hub_rows.values()]
    ax.bar(labels, means, yerr=errs, color=["#999999", "#1b9e77"], capsize=4)
    ax.axhline(R_MEAN["poisson"], color=CLASS_COLORS["Poisson"], ls=":", label="Poisson")
    ax.axhline(R_MEAN["goe"], color=CLASS_COLORS["GOE"], ls="--", label="GOE")
    ax.set_ylim(0.3, 0.6)
    ax.set_ylabel(r"$\langle r\rangle$")
    ax.set_title("(b) Hubbard ring L=8, N=4+4, U=4\n(K, reflection, S resolved; eta for t'=0)",
                 fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.plot(times, syk_otoc, color="#d95f02", lw=2, label="SYK N=12")
    ax.plot(times, hub_otoc, color="#3498db", lw=2, label="Hubbard L=4 (t'=0)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel(r"Re $F(t)$, $T=\infty$")
    ax.set_title("(c) OTOC, single realisation", fontweight="bold")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("output.png", dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    main()
