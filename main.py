#!/usr/bin/env python3
"""Quick-start demonstration: symmetry-resolved level statistics and OTOCs.

Builds the overview figure from ``configs/figures/overview.yaml`` (all seeds
fixed) and prints the numbers it shows. Runtime ~30 s. ``make figures``
regenerates every figure; ``scripts/validate_level_statistics.py`` the
validation tables.
"""

import json
import pathlib

from quantum_chaos.experiments import load_figure_config, make_figure
from quantum_chaos.spectral import R_MEAN

ROOT = pathlib.Path(__file__).resolve().parent


def main() -> None:
    cfg = load_figure_config(ROOT / "configs" / "figures" / "overview.yaml")
    out = make_figure(cfg, root=ROOT)
    data = json.loads(out.with_suffix(".json").read_text())["data"]

    print("SYK level statistics (sector resolved, class from N mod 8)")
    for r in data["syk"]:
        print(
            f"  N={r['N']:2d} ({r['class']}, {r['levels_per_sector']:4d} levels/sector, "
            f"{r['realisations']} realisations): <r> = {r['mean']:.4f} +/- {r['stderr']:.4f}"
            f" | RMT same size {r['rmt_same_size']:.4f} +/- {r['rmt_stderr']:.4f}"
            f" | large-N {r['atas']:.4f}"
        )
    h = cfg.hubbard
    assert h is not None
    print(f"\nHubbard ring L={h.n_sites}, N={h.n_up}+{h.n_down}, U={h.U:g} (fully resolved)")
    for tp, r in data["hubbard"].items():
        print(f"  t'={tp}: <r> = {r['mean']:.4f} +/- {r['stderr']:.4f} ({r['n_ratios']} ratios)")
    print(f"  references: Poisson {R_MEAN['poisson']:.4f}, GOE {R_MEAN['goe']:.4f}")

    otoc = data["otoc"]
    print(
        f"\nT=inf OTOC at t={otoc['times'][-1]:g}: SYK {otoc['syk'][-1]:+.4f}, "
        f"Hubbard {otoc['hubbard'][-1]:+.4f} (single realisation, qualitative)"
    )
    print(f"\nWrote {out.relative_to(ROOT)} and {out.with_suffix('.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
