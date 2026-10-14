"""Random-matrix reference values and finite-size ensemble samplers.

Reference values of the mean spacing ratio <r~> (min/max convention) from
Y. Y. Atas, E. Bogomolny, O. Giraud, G. Roux, "Distribution of the ratio of
consecutive level spacings in random matrix ensembles", Phys. Rev. Lett. 110,
084101 (2013), Table I (verified against arXiv:1212.5611, 2026-10-14):

* Poisson: 2 ln 2 - 1 = 0.38629... (exact)
* GOE: 0.5307(1), GUE: 0.5996(1), GSE: 0.6744(1) - numerics on 1000 x 1000
  Gaussian matrices; the Wigner-like surmise gives 4 - 2 sqrt(3) = 0.53590,
  2 sqrt(3)/pi - 1/2 = 0.60266 and (32/15) sqrt(3)/pi - 1/2 = 0.67617.

Finite spectra (e.g. SYK parity sectors of dimension 2^(N/2-1)) deviate from
these large-N values; :func:`ensemble_mean_r` produces a reference with the
same dimension and edge trimming, so that finite-size effects are not
mistaken for physics.
"""

from typing import Dict, Optional, Union

import numpy as np

from .statistics import spacing_ratios

R_MEAN: Dict[str, float] = {
    "poisson": 2 * np.log(2) - 1,
    "goe": 0.5307,
    "gue": 0.5996,
    "gse": 0.6744,
}
"""Large-N <r~> (Atas et al. 2013, Table I, numerical column; Poisson exact)."""

R_MEAN_UNCERTAINTY: Dict[str, float] = {"poisson": 0.0, "goe": 1e-4, "gue": 1e-4, "gse": 1e-4}

R_SURMISE: Dict[str, float] = {
    "poisson": 2 * np.log(2) - 1,
    "goe": 4 - 2 * np.sqrt(3),
    "gue": 2 * np.sqrt(3) / np.pi - 0.5,
    "gse": 32 / 15 * np.sqrt(3) / np.pi - 0.5,
}
"""Wigner-like surmise values (Atas et al. 2013, Eq. (7) and Table I)."""

DYSON_INDEX = {"goe": 1, "gue": 2, "gse": 4}

RngLike = Union[None, int, np.random.Generator]


def _normalise(ensemble: str) -> str:
    key = ensemble.lower()
    if key not in R_MEAN:
        raise ValueError(f"unknown ensemble {ensemble!r}; expected one of {sorted(R_MEAN)}")
    return key


def sample_spectrum(ensemble: str, dim: int, rng: RngLike = None) -> np.ndarray:
    """Eigenvalues of one random matrix with ``dim`` distinct levels.

    GOE: real symmetric; GUE: complex Hermitian; GSE: 2*dim x 2*dim quaternion
    self-dual Hermitian matrix [[A, B], [-B*, A*]] (A Hermitian, B complex
    antisymmetric) whose Kramers pairs are reduced to ``dim`` levels; Poisson:
    cumulative sum of i.i.d. exponential spacings.
    """
    key = _normalise(ensemble)
    rng = np.random.default_rng(rng)
    if key == "poisson":
        return np.cumsum(rng.exponential(size=dim))
    if key == "goe":
        x = rng.standard_normal((dim, dim))
        return np.linalg.eigvalsh((x + x.T) / 2)
    if key == "gue":
        x = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
        return np.linalg.eigvalsh((x + x.conj().T) / 2)
    x = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    y = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    a = (x + x.conj().T) / 2
    b = (y - y.T) / 2
    h = np.block([[a, b], [-b.conj(), a.conj()]])
    levels = np.linalg.eigvalsh(h)
    return (levels[0::2] + levels[1::2]) / 2


def ensemble_mean_r(
    ensemble: str,
    dim: int,
    n_samples: int,
    trim: float = 0.0,
    rng: RngLike = None,
) -> Dict[str, float]:
    """Finite-size <r~> of an ensemble: same dimension and trimming as the data.

    Returns the mean over samples of the per-matrix mean ratio and its
    standard error (samples are independent).
    """
    rng = np.random.default_rng(rng)
    per_sample = np.array([
        np.mean(spacing_ratios(sample_spectrum(ensemble, dim, rng), trim=trim))
        for _ in range(n_samples)
    ])
    return {
        "mean": float(per_sample.mean()),
        "stderr": float(per_sample.std(ddof=1) / np.sqrt(n_samples)),
        "n_samples": int(n_samples),
        "dim": int(dim),
        "trim": float(trim),
    }


def closest_ensemble(mean_r: float, candidates: Optional[list] = None) -> str:
    """Name of the reference ensemble whose large-N <r~> is closest (descriptive only)."""
    candidates = candidates or list(R_MEAN)
    return min(candidates, key=lambda k: abs(R_MEAN[_normalise(k)] - mean_r))


_Z_BETA = {1: 8 / 27, 2: 4 / 81 * np.pi / np.sqrt(3), 4: 4 / 729 * np.pi / np.sqrt(3)}


def ratio_density(r: np.ndarray, ensemble: str) -> np.ndarray:
    """Density of r~ = min(r, 1/r) on [0, 1].

    Poisson: 2/(1+r)^2. GOE/GUE/GSE: Wigner-like surmise of Atas et al.
    (2013), Eq. (7), P_W(r) = (r + r^2)^beta / Z_beta / (1 + r + r^2)^(1 + 3 beta/2)
    with Z_1 = 8/27, Z_2 = 4 pi / (81 sqrt 3), Z_4 = 4 pi / (729 sqrt 3), and
    P(r~) = 2 P(r) Theta(1 - r).
    """
    key = _normalise(ensemble)
    r = np.asarray(r, dtype=float)
    inside = (r >= 0) & (r <= 1)
    if key == "poisson":
        p = 2.0 / (1.0 + r) ** 2
    else:
        beta = DYSON_INDEX[key]
        p = 2.0 * (r + r * r) ** beta / _Z_BETA[beta] / (1.0 + r + r * r) ** (1.0 + 1.5 * beta)
    return np.where(inside, p, 0.0)
