import warnings
from typing import Any

import numpy as np


class DegenerateSpectrumWarning(UserWarning):
    pass


def remove_kramers_pairs(levels: np.ndarray, rtol: float = 1e-8) -> np.ndarray:
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
    r = spacing_ratios(levels, kramers=kramers, trim=trim, degeneracy_rtol=degeneracy_rtol)
    return float(np.mean(r)) if r.size else float("nan")


def bootstrap_mean(
    samples: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    rng: int | np.random.Generator | None = None,
) -> dict[str, Any]:
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
