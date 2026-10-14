from collections.abc import Sequence

import numpy as np

from ..spectral.statistics import remove_kramers_pairs


def partition_function(levels: np.ndarray, times: np.ndarray, beta: float = 0.0) -> np.ndarray:
    e = np.asarray(levels, dtype=float)
    shifted = e - e.min()
    weights = np.exp(-beta * shifted)
    phases = np.exp(-1j * np.outer(np.asarray(times, dtype=float), shifted))
    return phases @ weights


def spectral_form_factor(
    spectra: Sequence[np.ndarray], times: np.ndarray, beta: float = 0.0
) -> dict[str, np.ndarray]:
    Z = np.array([partition_function(e, times, beta) for e in spectra])
    Z0 = np.array([partition_function(e, np.zeros(1), beta)[0].real for e in spectra])
    norm = np.mean(Z0) ** 2
    abs2 = np.abs(Z) ** 2
    g = abs2.mean(axis=0) / norm
    connected = (abs2.mean(axis=0) - np.abs(Z.mean(axis=0)) ** 2) / norm
    stderr = abs2.std(axis=0, ddof=1) / np.sqrt(len(spectra)) / norm if len(spectra) > 1 else 0 * g
    return {"times": np.asarray(times), "g": g, "g_connected": connected, "stderr": stderr}


def plateau_value(levels: np.ndarray, beta: float = 0.0, rtol: float = 1e-9) -> float:
    e = np.sort(np.asarray(levels, dtype=float))
    w = np.exp(-beta * (e - e[0]))
    tol = rtol * max(e[-1] - e[0], 1e-300)
    breaks = np.flatnonzero(np.diff(e) > tol) + 1
    groups = np.split(w, breaks)
    return float(sum(np.sum(g) ** 2 for g in groups) / np.sum(w) ** 2)


def unfold(levels: np.ndarray, degree: int = 9) -> np.ndarray:
    e = np.sort(np.asarray(levels, dtype=float))
    counting = np.arange(1, e.size + 1, dtype=float)
    x = (e - e.mean()) / e.std()
    coeffs = np.polynomial.polynomial.polyfit(x, counting, degree)
    return np.polynomial.polynomial.polyval(x, coeffs)


def unfolded_form_factor(
    sector_spectra: Sequence[np.ndarray],
    taus: np.ndarray,
    kramers: bool = False,
    window: float = 0.25,
    degree: int = 9,
) -> dict[str, np.ndarray]:
    taus = np.asarray(taus, dtype=float)
    values = []
    means = []
    for levels in sector_spectra:
        e = remove_kramers_pairs(levels) if kramers else np.sort(np.asarray(levels, dtype=float))
        x = unfold(e, degree)
        centre = 0.5 * (x[0] + x[-1])
        width = window * (x[-1] - x[0])
        w = np.exp(-(((x - centre) / width) ** 2))
        w = w / np.sqrt(np.sum(w**2))
        amp = np.exp(-2j * np.pi * np.outer(taus, x)) @ w
        values.append(np.abs(amp) ** 2)
        means.append(amp)
    v = np.array(values)
    m = np.array(means)
    K = v.mean(axis=0) - np.abs(m.mean(axis=0)) ** 2
    stderr = v.std(axis=0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 * K
    return {"taus": taus, "K": K, "stderr": stderr}


def rmt_form_factor(taus: np.ndarray, ensemble: str) -> np.ndarray:
    t = np.abs(np.asarray(taus, dtype=float))
    key = ensemble.lower()
    out = np.ones_like(t)
    if key == "gue":
        inside = t < 1
        out[inside] = t[inside]
    elif key == "goe":
        inside = t <= 1
        ti = t[inside]
        out[inside] = 2 * ti - ti * np.log1p(2 * ti)
        to = t[~inside]
        out[~inside] = 2 - to * np.log((2 * to + 1) / (2 * to - 1))
    elif key == "gse":
        inside = t < 2
        ti = t[inside]
        with np.errstate(divide="ignore"):
            out[inside] = ti / 2 - ti / 4 * np.log(np.abs(1 - ti))
        out[np.isclose(t, 1.0)] = np.inf
    elif key == "poisson":
        pass
    else:
        raise ValueError(f"unknown ensemble {ensemble!r}")
    return out


def thouless_time(
    taus: np.ndarray,
    K: np.ndarray,
    ensemble: str,
    stderr: np.ndarray | None = None,
    rtol: float = 0.1,
    n_sigma: float = 2.0,
    tau_max: float = 0.8,
) -> float:
    taus = np.asarray(taus, dtype=float)
    ref = rmt_form_factor(taus, ensemble)
    err = np.zeros_like(taus) if stderr is None else np.asarray(stderr, dtype=float)
    tolerance = np.maximum(rtol * ref, n_sigma * err)
    ok = np.abs(np.asarray(K) - ref) <= tolerance
    considered = taus <= tau_max
    bad = np.flatnonzero(~ok & considered)
    start = int(np.max(bad)) + 1 if bad.size else 0
    if start >= taus.size or not considered[start]:
        return float("nan")
    return float(taus[start])
