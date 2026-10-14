#!/usr/bin/env python3

import argparse
import json
import pathlib
import platform
import subprocess
import time
import warnings

import numpy as np

import quantum_chaos
from quantum_chaos.hamiltonians.hubbard_ed import HubbardRingED
from quantum_chaos.hamiltonians.syk_hamiltonian import disorder_average_lsr, syk_rmt_reference
from quantum_chaos.spectral import R_MEAN, spacing_ratios

ROOT = pathlib.Path(__file__).resolve().parents[1]

SYK_FULL = [(10, 400), (12, 400), (14, 300), (16, 200), (18, 100), (20, 40), (22, 20)]
SYK_QUICK = [(10, 200), (12, 200), (14, 100), (16, 60), (18, 30)]
HUB_FULL = [(8, 4, 4), (10, 3, 3), (10, 4, 4), (10, 5, 5)]
HUB_QUICK = [(8, 4, 4), (8, 3, 3)]
TP_VALUES = (0.0, 0.5)
U = 4.0


def git_revision(spec="HEAD"):
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", spec],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def syk_table(sizes):
    rows = []
    for N, n_real in sizes:
        t0 = time.time()
        data = disorder_average_lsr(N, n_realizations=n_real, rng=0)
        ref = syk_rmt_reference(N, n_samples=600 if N <= 16 else 200, rng=1)
        z = (data["mean"] - ref["mean"]) / np.hypot(data["stderr"], ref["stderr"])
        rows.append(
            {
                "N": N,
                "class": data["symmetry_class"],
                "levels_per_sector": data["levels_per_sector"],
                "realisations": n_real,
                "mean": data["mean"],
                "stderr": data["stderr"],
                "ci95": data["ci"],
                "rmt_same_size": ref["mean"],
                "rmt_stderr": ref["stderr"],
                "z_vs_rmt": z,
                "atas": data["reference"],
                "seconds": time.time() - t0,
            }
        )
        print(f"  SYK N={N} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def pooled(blocks, trim, min_levels):
    r = [
        spacing_ratios(b["energies"], trim=trim) for b in blocks if b["energies"].size >= min_levels
    ]
    if not r:
        return None
    r = np.concatenate(r)
    return {
        "mean": float(r.mean()),
        "stderr": float(r.std(ddof=1) / np.sqrt(r.size)),
        "n_ratios": int(r.size),
    }


def hubbard_table(sectors):
    rows = []
    for L, nu, nd in sectors:
        for tp in TP_VALUES:
            t0 = time.time()
            ed = HubbardRingED(L, nu, nd, t=1.0, tp=tp, U=U)
            blocks = ed.sector_blocks()
            spacings = [np.diff(b["energies"]) for b in blocks if b["energies"].size >= 10]
            n_deg = sum(int(np.sum(s <= 1e-10 * s.mean())) for s in spacings)
            n_sp = sum(s.size for s in spacings)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                variants = {
                    f"trim={trim},min={ml}": pooled(blocks, trim, ml)
                    for trim in (0.0, 0.2)
                    for ml in (10, 400)
                }
            rows.append(
                {
                    "L": L,
                    "n_up": nu,
                    "n_down": nd,
                    "tp": tp,
                    "U": U,
                    "sector_dim": ed.dim,
                    "eta_resolved": ed.has_eta_symmetry,
                    "n_blocks": len(blocks),
                    "largest_block": max(b["energies"].size for b in blocks),
                    "degenerate_fraction": n_deg / n_sp,
                    "variants": variants,
                    "seconds": time.time() - t0,
                }
            )
            print(f"  Hubbard L={L} N={nu}+{nd} t'={tp} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def fmt(x, e):
    return f"{x:.4f} ± {e:.4f}"


def print_markdown(result):
    print("\n### SYK (sector resolved; class from N mod 8)\n")
    print(
        "| N | class | levels/sector | realisations | ⟨r⟩ SYK | ⟨r⟩ RMT same size "
        "| z | Atas (N→∞) |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in result["syk"]:
        print(
            f"| {r['N']} | {r['class']} | {r['levels_per_sector']} | {r['realisations']} | "
            f"{fmt(r['mean'], r['stderr'])} | {fmt(r['rmt_same_size'], r['rmt_stderr'])} | "
            f"{r['z_vs_rmt']:+.2f} | {r['atas']:.4f} |"
        )
    print("\n### Hubbard ring (U = 4, t = 1; K, reflection, S and — for t'=0 — η resolved)\n")
    print(
        "| L | N↑+N↓ | t' | sector dim | blocks | exact degeneracies | ⟨r⟩ all blocks | "
        "⟨r⟩ blocks ≥ 400, trim 0.2 |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for r in result["hubbard"]:
        a = r["variants"]["trim=0.0,min=10"]
        b = r["variants"]["trim=0.2,min=400"]
        b_txt = f"{fmt(b['mean'], b['stderr'])} (n={b['n_ratios']})" if b else "—"
        print(
            f"| {r['L']} | {r['n_up']}+{r['n_down']} | {r['tp']} | {r['sector_dim']} | "
            f"{r['n_blocks']} | {100 * r['degenerate_fraction']:.1f}% | "
            f"{fmt(a['mean'], a['stderr'])} (n={a['n_ratios']}) | {b_txt} |"
        )
    print(
        f"\nReferences: Poisson {R_MEAN['poisson']:.4f}, GOE {R_MEAN['goe']:.4f}, "
        f"GUE {R_MEAN['gue']:.4f}, GSE {R_MEAN['gse']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default=str(ROOT / "results" / "level_statistics.json"))
    args = parser.parse_args()

    result = {
        "meta": {
            "git": git_revision(),
            "src_tree": git_revision("HEAD:src"),
            "package_version": quantum_chaos.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "quick": args.quick,
        },
        "syk": syk_table(SYK_QUICK if args.quick else SYK_FULL),
        "hubbard": hubbard_table(HUB_QUICK if args.quick else HUB_FULL),
    }
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=float) + "\n")
    print_markdown(result)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
