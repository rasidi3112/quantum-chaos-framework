from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..floquet import (
    driven_ring_block,
    floquet_operator,
    heating_scan,
    quasi_energies,
    quasi_energy_distance,
    van_vleck_hamiltonian,
)
from ..hamiltonians.hubbard_ed import HubbardRingED
from .config import FigureConfig


def floquet_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.floquet is not None
    c = cfg.floquet
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    data: dict[str, Any] = {"heating": {}, "expansion": {}}
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for size in c.sizes:
        block = driven_ring_block(size, size // 2, size // 2, momentum=0)
        res = heating_scan(
            block,
            c.omegas,
            J0=c.J0,
            J1=c.J1,
            tp=c.tp,
            U=c.U,
            n_periods=c.n_periods,
            n_steps=c.n_steps,
        )
        H0 = block.static(c.J0, c.tp, c.U)
        evals = np.linalg.eigvalsh(H0)
        bandwidth = float(evals[-1] - evals[0])
        averaged = [r["time_averaged_fraction"] for r in res["omegas"].values()]
        axes[1].semilogy(
            c.omegas, np.maximum(averaged, 1e-6), "o-", label=f"L={size} (W={bandwidth:.1f})"
        )
        data["heating"][str(size)] = {
            "block_dim": res["dim"],
            "bandwidth": bandwidth,
            "omegas": list(c.omegas),
            "time_averaged_fraction": averaged,
            "heating_time": [r["heating_time"] for r in res["omegas"].values()],
        }
        if size == max(c.sizes):
            n = np.arange(c.n_periods + 1)
            for k, (omega, r) in enumerate(res["omegas"].items()):
                q = (r["energies"] - res["e_ground"]) / (res["e_infinite"] - res["e_ground"])
                axes[0].semilogx(
                    n[1:], q[1:], color=colors[k % len(colors)], lw=0.8, label=f"w={omega:g}"
                )
            axes[0].set_title(f"(a) Absorbed energy, L={size}, K=0 block", fontweight="bold")
    axes[0].set_xlabel("periods n")
    axes[0].set_ylabel(r"$(E_n-E_0)/(E_\infty-E_0)$")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].set_xlabel(r"$\omega$")
    axes[1].set_ylabel("time-averaged absorbed fraction")
    axes[1].set_title("(b) Heating vs drive frequency", fontweight="bold")
    axes[1].legend(fontsize=8)

    ed = HubbardRingED(5, 2, 2, t=1.0, U=0.0)
    hop = ed.hopping_operator(1).toarray()
    H0 = c.J0 * hop + c.U * ed.double_occupancy().toarray()
    harmonics = {0: H0, 1: c.J1 / 2 * hop, -1: c.J1 / 2 * hop}
    errors: dict[int, list[float]] = {0: [], 2: []}
    for omega in c.expansion_omegas:
        T = 2 * np.pi / omega

        def drive(t: float, w: float = omega) -> np.ndarray:
            return H0 + c.J1 * np.cos(w * t) * hop

        q = quasi_energies(floquet_operator(drive, T, n_steps=160), T)
        for order in errors:
            eff = np.linalg.eigvalsh(van_vleck_hamiltonian(harmonics, omega, order))
            errors[order].append(quasi_energy_distance(eff, q, omega))
    for order, marker in ((0, "o-"), (2, "s-")):
        axes[2].loglog(c.expansion_omegas, errors[order], marker, label=f"van Vleck order {order}")
    axes[2].set_xlabel(r"$\omega$")
    axes[2].set_ylabel("max quasi-energy error")
    axes[2].set_title("(c) High-frequency expansion vs exact (L=5)", fontweight="bold")
    axes[2].legend(fontsize=8)
    data["expansion"] = {
        "omegas": list(c.expansion_omegas),
        "order0": errors[0],
        "order2": errors[2],
        "slope_order0": float(np.polyfit(np.log(c.expansion_omegas), np.log(errors[0]), 1)[0]),
        "slope_order2": float(np.polyfit(np.log(c.expansion_omegas), np.log(errors[2]), 1)[0]),
    }
    fig.tight_layout()
    return fig, data
