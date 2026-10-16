import argparse
import json
import os
import pathlib
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from quantum_chaos.chaos.typicality import SectorSYK, typical_otoc

ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache" / "syk_lyapunov"
REALISATIONS = {20: 64, 22: 48, 24: 32, 26: 24, 28: 16, 30: 10, 32: 6, 34: 4}
WORKERS = {30: 4, 32: 2, 34: 1}
BETAS = (0.0, 10.0)
TIMES = np.round(np.arange(0.0, 4.0001, 0.2), 10)


def one_realisation(job: tuple[int, int]) -> dict:
    N, seed = job
    path = CACHE / f"N{N}_s{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    t0 = time.time()
    model = SectorSYK(N, seed=seed, chi_squared=0.5)
    curves = {}
    for beta in BETAS:
        F = typical_otoc(model, 0, 1, TIMES, n_vectors=1, beta=beta, rng=10_000 + seed)["mean"]
        curves[str(beta)] = [F.real.tolist(), F.imag.tolist()]
    row = {"N": N, "seed": seed, "otoc": curves, "seconds": time.time() - t0}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row))
    print(N, seed, f"{row['seconds']:.1f}s", flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default=",".join(map(str, REALISATIONS)))
    parser.add_argument("--output", default=str(ROOT / "results" / "syk_lyapunov.json"))
    args = parser.parse_args()
    sizes = [int(n) for n in args.sizes.split(",")]
    rows = []
    for N in sizes:
        jobs = [(N, s) for s in range(REALISATIONS[N])]
        workers = WORKERS.get(N, max(1, (os.cpu_count() or 2) - 2))
        with ProcessPoolExecutor(workers) as pool:
            rows.extend(pool.map(one_realisation, jobs))
    result = {
        "model": "SYK4, chi_squared=0.5, <J^2>=3!J^2/N^3, J=1",
        "observable": (
            "regularised OTOC Tr(y W(t) y V y W(t) y V)/Tr(y^4), y=rho^(1/4), W=chi_0, V=chi_1"
        ),
        "estimator": "dynamical typicality, one Haar-random vector per realisation",
        "betas": list(BETAS),
        "times": TIMES.tolist(),
        "realisations": {str(N): REALISATIONS[N] for N in sizes},
        "runs": rows,
    }
    pathlib.Path(args.output).write_text(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
