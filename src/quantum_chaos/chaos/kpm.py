import numpy as np
from scipy.sparse.linalg import eigsh

from ..utils.helpers import RngLike


def spectral_bounds(hamiltonian, margin: float = 0.01) -> tuple[float, float]:
    lo = eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False, tol=1e-6)[0]
    hi = eigsh(hamiltonian, k=1, which="LA", return_eigenvectors=False, tol=1e-6)[0]
    width = hi - lo
    return float(lo - margin * width), float(hi + margin * width)


def chebyshev_moments(
    hamiltonian,
    n_moments: int,
    n_vectors: int = 10,
    bounds: tuple[float, float] | None = None,
    rng: RngLike = None,
) -> dict[str, np.ndarray | float]:
    lo, hi = spectral_bounds(hamiltonian) if bounds is None else bounds
    a = (hi - lo) / 2
    b = (hi + lo) / 2
    dim = hamiltonian.shape[0]
    rng = np.random.default_rng(rng)
    half = (n_moments + 1) // 2
    samples = np.zeros((n_vectors, 2 * half))
    for k in range(n_vectors):
        r = rng.normal(size=dim) + 1j * rng.normal(size=dim)
        r /= np.linalg.norm(r)
        alpha_prev = r
        alpha = (hamiltonian @ r - b * r) / a
        mu = samples[k]
        mu[0] = 1.0
        mu[1] = np.vdot(r, alpha).real
        for n in range(1, half):
            alpha_next = 2 * (hamiltonian @ alpha - b * alpha) / a - alpha_prev
            mu[2 * n] = 2 * np.vdot(alpha, alpha).real - mu[0]
            mu[2 * n + 1] = 2 * np.vdot(alpha_next, alpha).real - mu[1]
            alpha_prev, alpha = alpha, alpha_next
    mean = samples.mean(axis=0)[:n_moments]
    err = (
        samples.std(axis=0, ddof=1)[:n_moments] / np.sqrt(n_vectors) if n_vectors > 1 else 0 * mean
    )
    return {"moments": mean, "stderr": err, "scale": a, "shift": b}


def jackson_kernel(n_moments: int) -> np.ndarray:
    n = np.arange(n_moments)
    N = n_moments + 1
    return ((N - n) * np.cos(np.pi * n / N) + np.sin(np.pi * n / N) / np.tan(np.pi / N)) / N


def kpm_density(moments: dict, energies: np.ndarray) -> np.ndarray:
    mu = np.asarray(moments["moments"]) * jackson_kernel(len(moments["moments"]))
    a, b = moments["scale"], moments["shift"]
    x = (np.asarray(energies, dtype=float) - b) / a
    inside = np.abs(x) < 1
    xi = x[inside]
    T = np.polynomial.chebyshev.chebval(xi, np.concatenate([[mu[0]], 2 * mu[1:]]))
    out = np.zeros_like(x)
    out[inside] = T / (np.pi * np.sqrt(1 - xi**2)) / a
    return out


def moments_to_power_moments(moments: dict) -> tuple[float, float]:
    mu = np.asarray(moments["moments"])
    a, b = moments["scale"], moments["shift"]
    m1x = mu[1]
    m2x = (mu[2] + mu[0]) / 2
    mean = a * m1x + b
    second = a**2 * m2x + 2 * a * b * m1x + b**2
    return float(mean), float(second)
