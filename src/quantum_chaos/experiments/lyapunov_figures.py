from __future__ import annotations

import json
import pathlib
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..chaos.otoc import crossing_time, power_law_fit, rescaling_exponents, rescaling_fit
from .config import FigureConfig
from .style import CYCLE, SYK, summary_box, suptitle

REPO = pathlib.Path(__file__).resolve().parents[3]


def _otoc_arrays(results: dict[str, Any], beta: float) -> dict[int, np.ndarray]:
    by_size: dict[int, list[list[float]]] = {}
    for row in results["runs"]:
        by_size.setdefault(int(row["N"]), []).append(row["otoc"][str(float(beta))][0])
    return {N: np.array(v) for N, v in sorted(by_size.items())}


def _decay(F: np.ndarray) -> np.ndarray:
    m = F.mean(axis=0)
    return 1.0 - m / m[0]


def _estimates(times, arrays, sizes, threshold):
    tc = np.array([crossing_time(times, _decay(arrays[N]), threshold) for N in sizes])
    pairs = rescaling_exponents(sizes, tc)
    lam = pairs["exponent"]
    x = 1.0 / pairs["size"]
    ok = np.isfinite(lam)
    lam0 = float(np.polyfit(x[ok], lam[ok], 1)[1]) if ok.sum() >= 2 else float("nan")
    good = np.isfinite(tc)
    slope = np.polyfit(np.log(np.asarray(sizes)[good]), tc[good], 1)[0] if good.sum() >= 2 else 0
    lam_global = float(1.0 / slope) if slope > 0 else float("nan")
    corrected = float("nan")
    if good.sum() >= 4:
        corrected = rescaling_fit(np.asarray(sizes)[good], tc[good], np.ones(good.sum()))[
            "exponent"
        ]
    return tc, lam, lam0, lam_global, corrected


def lyapunov_analysis(
    results: dict[str, Any],
    thresholds: tuple[float, ...] = (0.25, 0.16),
    n_bootstrap: int = 400,
    seed: int = 0,
    min_sizes: dict[float, int] | None = None,
) -> dict[str, Any]:
    cuts = {float(b): int(n) for b, n in (min_sizes or {}).items()}
    times = np.asarray(results["times"], dtype=float)
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    for beta in results["betas"]:
        cut = cuts.get(float(beta), 0)
        arrays = {N: F for N, F in _otoc_arrays(results, beta).items() if N >= cut}
        sizes = np.array(sorted(arrays))
        per_beta: dict[str, Any] = {
            "min_size": cut,
            "sizes": sizes.tolist(),
            "realisations": [len(arrays[N]) for N in sizes],
            "decay": {int(N): _decay(arrays[N]).tolist() for N in sizes},
        }
        for thr in thresholds:
            tc, lam, lam0, lam_g, lam_c = _estimates(times, arrays, sizes, thr)
            boot = []
            for _ in range(n_bootstrap):
                resampled = {
                    N: arrays[N][rng.integers(0, len(arrays[N]), len(arrays[N]))] for N in sizes
                }
                b_tc, b_lam, b_lam0, b_lam_g, b_lam_c = _estimates(times, resampled, sizes, thr)
                boot.append(np.concatenate([b_tc, b_lam, [b_lam0, b_lam_g, b_lam_c]]))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                spread = np.nanstd(np.array(boot), axis=0, ddof=1)
            n = len(sizes)
            tc_err = spread[:n]
            ok = np.isfinite(tc) & (tc_err > 0)
            models: dict[str, Any] = {}
            if ok.sum() >= 4:
                models = {
                    "log_fit": rescaling_fit(sizes[ok], tc[ok], tc_err[ok], correction=False),
                    "log_fit_corrected": rescaling_fit(sizes[ok], tc[ok], tc_err[ok]),
                    "power_law": power_law_fit(sizes[ok], tc[ok], tc_err[ok]),
                }
            per_beta[f"threshold_{thr}"] = {
                "crossing_time": tc.tolist(),
                "crossing_time_err": spread[:n].tolist(),
                "pair_size": np.sqrt(sizes[1:] * sizes[:-1]).tolist(),
                "exponent": lam.tolist(),
                "exponent_err": spread[n : 2 * n - 1].tolist(),
                "extrapolated": lam0,
                "extrapolated_err": float(spread[2 * n - 1]),
                "global": lam_g,
                "global_err": float(spread[2 * n]),
                "corrected": lam_c,
                "corrected_err": float(spread[2 * n + 1]),
                "weighted_models": models,
            }
        out[str(float(beta))] = per_beta
    return {"times": times.tolist(), "thresholds": list(thresholds), "betas": out}


def _load(path: str) -> dict[str, Any]:
    p = pathlib.Path(path)
    if not p.is_absolute() and not p.exists():
        p = REPO / p
    return json.loads(p.read_text())


def lyapunov_figure(cfg: FigureConfig) -> tuple[Figure, dict[str, Any]]:
    assert cfg.lyapunov is not None
    c = cfg.lyapunov
    results = _load(c.results)
    data = lyapunov_analysis(results, c.thresholds, c.bootstrap, c.seed, c.min_sizes)
    times = np.asarray(data["times"])
    main = f"threshold_{c.thresholds[0]}"
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
    suptitle(fig, "SYK Scrambling: Finite-Size Rescaling of the OTOC")
    ax = axes[0, 0]
    first = data["betas"][str(float(results["betas"][0]))]
    for k, N in enumerate(first["sizes"]):
        ax.semilogy(times[1:], first["decay"][N][1:], color=CYCLE[k % len(CYCLE)], label=f"N={N}")
    for thr in c.thresholds:
        ax.axhline(thr, color="gray", ls=":", lw=1.5)
    ax.set_xlabel("t J")
    ax.set_ylabel(r"$1 - F(t)/F(0)$")
    ax.set_title(f"(a) Regularised OTOC, βJ = {results['betas'][0]:g}")
    ax.legend(ncol=2)
    colors = [SYK, "steelblue", "green", "darkorange"]
    lines = []
    for k, (beta, d) in enumerate(data["betas"].items()):
        col = colors[k % len(colors)]
        b = float(beta)
        sizes = np.asarray(d["sizes"], dtype=float)
        m = d[main]
        axes[0, 1].errorbar(
            np.log(sizes),
            m["crossing_time"],
            yerr=m["crossing_time_err"],
            fmt="o",
            color=col,
            label=f"βJ = {b:g}",
        )
        good = np.isfinite(m["crossing_time"])
        if good.sum() >= 2:
            fit = np.polyfit(np.log(sizes)[good], np.asarray(m["crossing_time"])[good], 1)
            xs = np.linspace(np.log(sizes).min(), np.log(sizes).max(), 10)
            axes[0, 1].plot(xs, np.polyval(fit, xs), color=col, lw=1.2)
        x = 1.0 / np.asarray(m["pair_size"])
        axes[1, 0].errorbar(
            x, m["exponent"], yerr=m["exponent_err"], fmt="o", color=col, label=f"βJ = {b:g}"
        )
        axes[1, 0].errorbar(
            [0.0],
            [m["corrected"]],
            yerr=[m["corrected_err"]],
            fmt="s",
            color=col,
            ms=9,
            label=f"ln N + 1/N fit (βJ = {b:g})",
        )
        if b > 0:
            axes[1, 0].axhline(2 * np.pi / b, color=col, ls=":", lw=2, label=f"2π/β (βJ = {b:g})")
        lines.append(f"βJ = {b:g}  (N = {d['sizes'][0]}-{d['sizes'][-1]}):")
        for thr in c.thresholds:
            r = d[f"threshold_{thr}"]
            lines.append(
                f"  F0={thr:.2f}  pairs   = {r['extrapolated']:.2f} ± {r['extrapolated_err']:.2f}"
            )
            lines.append(f"            ln N    = {r['global']:.2f} ± {r['global_err']:.2f}")
            lines.append(f"            +1/N    = {r['corrected']:.2f} ± {r['corrected_err']:.2f}")
            wm = r["weighted_models"]
            if wm:
                lines.append(
                    f"            t*~N^{wm['power_law']['power']:.2f}"
                    f"  chi2: {wm['power_law']['chi2_per_dof']:.2f}"
                    f" vs {wm['log_fit_corrected']['chi2_per_dof']:.2f}"
                )
        if b > 0:
            lines.append(f"  chaos bound 2π/β = {2 * np.pi / b:.3f}")
    axes[0, 1].set_xlabel("ln N")
    axes[0, 1].set_ylabel(f"t* J  (1 − F/F(0) = {c.thresholds[0]})")
    axes[0, 1].set_title("(b) Crossing time vs ln N")
    axes[0, 1].legend()
    axes[1, 0].set_xlabel(r"$1/\sqrt{N_1 N_2}$")
    axes[1, 0].set_ylabel(r"$\lambda(N) = \Delta \ln N / \Delta t^*$")
    axes[1, 0].set_title("(c) Pairwise exponents; squares: corrected global fit")
    axes[1, 0].set_xlim(left=-0.003)
    axes[1, 0].set_ylim(0, 3)
    axes[1, 0].legend(fontsize=8)
    sizes_all = data["betas"][next(iter(data["betas"]))]["sizes"]
    reals = data["betas"][next(iter(data["betas"]))]["realisations"]
    lines.append("")
    lines.append("N: " + ", ".join(map(str, sizes_all)))
    lines.append("realisations: " + ", ".join(map(str, reals)))
    lines.append("method: Kobrin et al., PRL 126, 030602")
    summary_box(axes[1, 1], "LYAPUNOV ESTIMATES (J = 1)", lines)
    fig.tight_layout()
    return fig, data
