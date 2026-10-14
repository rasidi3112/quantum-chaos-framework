import argparse
import json
import pathlib
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from quantum_chaos.chaos.otoc import lyapunov_with_systematics
from quantum_chaos.chaos.typicality import SectorSYK, typical_otoc
from quantum_chaos.hamiltonians.syk_hamiltonian import syk_symmetry_class
from quantum_chaos.hamiltonians.syk_sparse import syk_sector_hamiltonian

ROOT = pathlib.Path(__file__).resolve().parents[1]
GROUND_STATE = {12: 20, 16: 20, 20: 10, 24: 8, 28: 6, 30: 4, 32: 3, 34: 3}
OTOC = {20: 6, 24: 3, 28: 2}


def ground_states() -> dict:
    rows = {}
    for N, n_real in GROUND_STATE.items():
        t0 = time.time()
        sectors = ("even",) if syk_symmetry_class(N) == "GUE" else ("even", "odd")
        e0 = []
        for seed in range(n_real):
            e0.append(
                min(
                    eigsh(
                        syk_sector_hamiltonian(N, seed, s, chi_squared=0.5),
                        k=1,
                        which="SA",
                        tol=1e-9,
                        return_eigenvectors=False,
                    )[0]
                    for s in sectors
                )
            )
        e = np.array(e0)
        rows[N] = {
            "realisations": n_real,
            "e0_mean": float(e.mean()),
            "e0_stderr": float(e.std(ddof=1) / np.sqrt(n_real)),
            "e0_per_n": float(e.mean() / N),
            "garcia_garcia_verbaarschot_fit": -0.055 - 0.029 * N,
            "seconds": time.time() - t0,
        }
        print(N, rows[N], flush=True)
    return rows


def otocs(times: np.ndarray) -> dict:
    rows = {}
    for N, n_real in OTOC.items():
        t0 = time.time()
        curves = []
        for seed in range(n_real):
            model = SectorSYK(N, seed=seed, chi_squared=0.5)
            curves.append(typical_otoc(model, 0, 1, times, n_vectors=1, rng=seed)["mean"])
        F = np.mean(curves, axis=0)
        decay = 1 - np.real(F / F[0])
        fit = lyapunov_with_systematics(times, decay)
        rows[N] = {
            "realisations": n_real,
            "times": times.tolist(),
            "decay": decay.tolist(),
            "fitted_rate": fit["rate"],
            "rate_statistical": fit["statistical"],
            "rate_systematic": fit["systematic"],
            "seconds": time.time() - t0,
        }
        print(N, {k: v for k, v in rows[N].items() if k not in ("times", "decay")}, flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "large_n_syk.json"))
    args = parser.parse_args()
    times = np.linspace(0, 8, 33)
    result = {"chi_squared": 0.5, "ground_state": ground_states(), "otoc_infinite_T": otocs(times)}
    pathlib.Path(args.output).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
