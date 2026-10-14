import argparse
import json
import pathlib
import resource
import sys
import time

import numpy as np
from scipy.sparse.linalg import eigsh

from quantum_chaos.hamiltonians.syk_sparse import (
    n_flip_masks,
    sector_memory_bytes,
    syk_sector_hamiltonian,
)


def peak_rss_gb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e9 if sys.platform == "darwin" else rss / 1e6


def run(N: int, dtype: str) -> dict:
    t0 = time.time()
    H = syk_sector_hamiltonian(N, seed=0, chi_squared=0.5, dtype=np.dtype(dtype).type)
    build = time.time() - t0
    v = np.random.default_rng(0).normal(size=H.shape[0]).astype(H.dtype)
    t0 = time.time()
    for _ in range(5):
        H @ v
    matvec = (time.time() - t0) / 5
    t0 = time.time()
    e0 = eigsh(H, k=1, which="SA", tol=1e-8, return_eigenvectors=False)[0]
    lanczos = time.time() - t0
    return {
        "N": N,
        "sector_dim": int(H.shape[0]),
        "flip_masks": n_flip_masks(N),
        "nnz": int(H.nnz),
        "matrix_gb": (H.data.nbytes + H.indices.nbytes + H.indptr.nbytes) / 1e9,
        "predicted_gb": sector_memory_bytes(N, np.dtype(dtype).itemsize) / 1e9,
        "build_s": build,
        "matvec_s": matvec,
        "lanczos_ground_state_s": lanczos,
        "e0_even_sector": float(e0),
        "peak_rss_gb": peak_rss_gb(),
        "dtype": dtype,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sizes", nargs="+", type=int)
    parser.add_argument("--dtype", default="complex128")
    parser.add_argument("--output", default="benchmarks/results")
    args = parser.parse_args()
    out = pathlib.Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for N in args.sizes:
        row = run(N, args.dtype)
        print(json.dumps(row), flush=True)
        (out / f"syk_scaling_N{N}_{args.dtype}.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    main()
