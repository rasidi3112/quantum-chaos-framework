import numpy as np

from .statistics import spacing_ratios

R_MEAN: dict[str, float] = {
    "poisson": 2 * np.log(2) - 1,
    "goe": 0.5307,
    "gue": 0.5996,
    "gse": 0.6744,
}

R_MEAN_UNCERTAINTY: dict[str, float] = {"poisson": 0.0, "goe": 1e-4, "gue": 1e-4, "gse": 1e-4}

R_SURMISE: dict[str, float] = {
    "poisson": 2 * np.log(2) - 1,
    "goe": 4 - 2 * np.sqrt(3),
    "gue": 2 * np.sqrt(3) / np.pi - 0.5,
    "gse": 32 / 15 * np.sqrt(3) / np.pi - 0.5,
}

DYSON_INDEX = {"goe": 1, "gue": 2, "gse": 4}

RngLike = None | int | np.random.Generator


def _normalise(ensemble: str) -> str:
    key = ensemble.lower()
    if key not in R_MEAN:
        raise ValueError(f"unknown ensemble {ensemble!r}; expected one of {sorted(R_MEAN)}")
    return key


def sample_spectrum(ensemble: str, dim: int, rng: RngLike = None) -> np.ndarray:
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
) -> dict[str, float]:
    rng = np.random.default_rng(rng)
    per_sample = np.array(
        [
            np.mean(spacing_ratios(sample_spectrum(ensemble, dim, rng), trim=trim))
            for _ in range(n_samples)
        ]
    )
    return {
        "mean": float(per_sample.mean()),
        "stderr": float(per_sample.std(ddof=1) / np.sqrt(n_samples)),
        "n_samples": int(n_samples),
        "dim": int(dim),
        "trim": float(trim),
    }


def closest_ensemble(mean_r: float, candidates: list | None = None) -> str:
    candidates = candidates or list(R_MEAN)
    return min(candidates, key=lambda k: abs(R_MEAN[_normalise(k)] - mean_r))


_Z_BETA = {1: 8 / 27, 2: 4 / 81 * np.pi / np.sqrt(3), 4: 4 / 729 * np.pi / np.sqrt(3)}


def ratio_density(r: np.ndarray, ensemble: str) -> np.ndarray:
    key = _normalise(ensemble)
    r = np.asarray(r, dtype=float)
    inside = (r >= 0) & (r <= 1)
    if key == "poisson":
        p = 2.0 / (1.0 + r) ** 2
    else:
        beta = DYSON_INDEX[key]
        p = 2.0 * (r + r * r) ** beta / _Z_BETA[beta] / (1.0 + r + r * r) ** (1.0 + 1.5 * beta)
    return np.where(inside, p, 0.0)
