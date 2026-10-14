"""Level-spacing-ratio statistics.

The ratio of consecutive spacings r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1}),
s_n = E_{n+1} - E_n, needs no unfolding (Oganesyan & Huse 2007; Atas et al.,
PRL 110, 084101 (2013)). It must be computed inside a single symmetry sector:
mixing sectors drives <r> towards the Poisson value, and exact degeneracies
signal a symmetry that has not been resolved. Degenerate levels are therefore
never discarded silently.
"""

import warnings
from typing import Any

import numpy as np


class DegenerateSpectrumWarning(UserWarning):
    """Exact (machine-precision) degeneracies were found in a spectrum."""


def remove_kramers_pairs(levels: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
    """Keep one level of every Kramers pair after checking that pairs are degenerate.

    ``rtol`` is relative to the spectral width. Raises ``ValueError`` if the
    sorted spectrum does not consist of degenerate pairs.
    """
    e = np.sort(np.asarray(levels, dtype=float))
    if e.size % 2:
        raise ValueError("a Kramers-degenerate spectrum must have an even number of levels")
    width = max(e[-1] - e[0], np.finfo(float).tiny)
    splitting = np.abs(e[1::2] - e[0::2])
    if np.any(splitting > rtol * width):
        raise ValueError(
            f"levels are not Kramers pairs (max splitting {splitting.max():.3e}, "
            f"tolerance {rtol * width:.3e})"
        )
    return (e[0::2] + e[1::2]) / 2


def spacing_ratios(
    levels: np.ndarray,
    kramers: bool = False,
    trim: float = 0.0,
    degeneracy_rtol: float = 1e-10,
) -> np.ndarray:
    """Ratios r_n = min(s_n, s_{n-1}) / max(s_n, s_{n-1}) of one symmetry sector.

    Parameters
    ----------
    levels : eigenvalues of one symmetry sector (any order).
    kramers : if True, the spectrum must consist of degenerate pairs
        (e.g. GSE); one level per pair is kept.
    trim : fraction of levels discarded at *each* edge of the spectrum
        (0 <= trim < 0.5), to avoid edge effects.
    degeneracy_rtol : spacings below ``degeneracy_rtol`` times the mean spacing
        count as exact degeneracies; they are kept (r = 0) and trigger a
        :class:`DegenerateSpectrumWarning`. Two consecutive zero spacings give an
        undefined ratio, which is excluded.
    """
    if not 0 <= trim < 0.5:
        raise ValueError("trim must be in [0, 0.5)")
    e = remove_kramers_pairs(levels) if kramers else np.sort(np.asarray(levels, dtype=float))
    cut = int(np.floor(trim * e.size))
    e = e[cut : e.size - cut]
    if e.size < 3:
        return np.array([], dtype=float)
    s = np.diff(e)
    mean_spacing = np.mean(s)
    n_degenerate = int(np.sum(s <= degeneracy_rtol * mean_spacing))
    if n_degenerate:
        warnings.warn(
            f"{n_degenerate} of {s.size} spacings are exact degeneracies; an unresolved "
            "symmetry (or Kramers pairs, use kramers=True) is likely",
            DegenerateSpectrumWarning,
            stacklevel=2,
        )
    lo = np.minimum(s[1:], s[:-1])
    hi = np.maximum(s[1:], s[:-1])
    valid = hi > 0
    return lo[valid] / hi[valid]


def mean_spacing_ratio(
    levels: np.ndarray,
    kramers: bool = False,
    trim: float = 0.0,
    degeneracy_rtol: float = 1e-10,
) -> float:
    """Mean of :func:`spacing_ratios`; ``nan`` if fewer than three levels remain."""
    r = spacing_ratios(levels, kramers=kramers, trim=trim, degeneracy_rtol=degeneracy_rtol)
    return float(np.mean(r)) if r.size else float("nan")


def bootstrap_mean(
    samples: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> dict[str, Any]:
    """Mean, standard error and percentile-bootstrap confidence interval of samples."""
    x = np.asarray(samples, dtype=float)
    rng = np.random.default_rng(rng)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    alpha = (1 - confidence) / 2
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else float("nan"),
        "stderr": float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan"),
        "ci": (float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))),
        "n_samples": int(x.size),
    }
