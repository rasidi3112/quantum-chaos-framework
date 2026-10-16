from collections.abc import Sequence

import numpy as np

from ..utils.helpers import hermitian_eigh


def _eigenbasis(hamiltonian: np.ndarray, operators: Sequence[np.ndarray]):
    H = hamiltonian.toarray() if hasattr(hamiltonian, "toarray") else np.asarray(hamiltonian)
    E, V = hermitian_eigh(H)
    return E, [V.conj().T @ np.asarray(op) @ V for op in operators]


def thermal_otoc(
    hamiltonian: np.ndarray,
    W: np.ndarray,
    V: np.ndarray,
    times: np.ndarray,
    beta: float = 0.0,
    regularized: bool = True,
) -> np.ndarray:
    E, (A, B) = _eigenbasis(hamiltonian, [W, V])
    w = np.exp(-beta * (E - E.min()))
    rho = w / w.sum()
    y = rho**0.25
    omega = np.subtract.outer(E, E)
    out = np.empty(len(times), dtype=complex)
    for k, t in enumerate(times):
        At = A * np.exp(1j * omega * t)
        if regularized:
            M = y[:, None] * At
            N = y[:, None] * B
            out[k] = np.trace(M @ N @ M @ N)
        else:
            out[k] = np.trace((rho[:, None] * At) @ B @ At @ B)
    return out


def normalized_decay(otoc: np.ndarray) -> np.ndarray:
    f = np.real(np.asarray(otoc) / otoc[0])
    return 1.0 - f


def lyapunov_fit(
    times: np.ndarray, decay: np.ndarray, window: tuple[float, float] = (0.01, 0.2)
) -> dict[str, float]:
    t = np.asarray(times, dtype=float)
    d = np.asarray(decay, dtype=float)
    lo, hi = window
    above = np.flatnonzero(d > hi)
    stop = above[0] if above.size else d.size
    idx = np.arange(stop)
    sel = idx[(d[idx] > lo) & (d[idx] <= hi)]
    if sel.size < 3:
        return {
            "rate": float("nan"),
            "stderr": float("nan"),
            "n_points": int(sel.size),
            "t_start": float("nan"),
            "t_stop": float("nan"),
        }
    x, yv = t[sel], np.log(d[sel])
    coeffs, cov = np.polyfit(x, yv, 1, cov=True) if sel.size > 3 else (np.polyfit(x, yv, 1), None)
    stderr = float(np.sqrt(cov[0, 0])) if cov is not None else float("nan")
    return {
        "rate": float(coeffs[0]),
        "stderr": stderr,
        "n_points": int(sel.size),
        "t_start": float(x[0]),
        "t_stop": float(x[-1]),
    }


def lyapunov_with_systematics(
    times: np.ndarray,
    decay: np.ndarray,
    windows: Sequence[tuple[float, float]] = ((0.01, 0.2), (0.005, 0.1), (0.02, 0.3), (0.01, 0.1)),
) -> dict[str, float]:
    fits = [lyapunov_fit(times, decay, w) for w in windows]
    rates = np.array([f["rate"] for f in fits])
    valid = rates[np.isfinite(rates)]
    if valid.size == 0:
        return {"rate": float("nan"), "systematic": float("nan"), "statistical": float("nan")}
    stat = np.nanmax([f["stderr"] for f in fits])
    return {
        "rate": float(fits[0]["rate"]),
        "systematic": float(valid.max() - valid.min()) / 2,
        "statistical": float(stat),
    }


def finite_size_extrapolation(
    sizes: Sequence[int], curves: np.ndarray, errors: np.ndarray, order: int = 1
) -> dict[str, np.ndarray]:
    x = 1.0 / np.asarray(sizes, dtype=float)
    Y = np.asarray(curves, dtype=float)
    S = np.asarray(errors, dtype=float)
    A = np.vander(x, order + 1, increasing=True)
    n_t = Y.shape[1]
    value = np.empty(n_t)
    slope = np.empty(n_t)
    stderr = np.empty(n_t)
    chi2 = np.empty(n_t)
    dof = max(len(x) - order - 1, 1)
    for k in range(n_t):
        w = 1.0 / S[:, k]
        coef, *_ = np.linalg.lstsq(A * w[:, None], Y[:, k] * w, rcond=None)
        cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
        value[k], slope[k] = coef[0], coef[1]
        stderr[k] = np.sqrt(cov[0, 0])
        chi2[k] = np.sum(((A @ coef - Y[:, k]) * w) ** 2) / dof
    return {"value": value, "slope": slope, "stderr": stderr, "chi2_per_dof": chi2}


def local_exponent(times: np.ndarray, growth: np.ndarray) -> np.ndarray:
    return np.gradient(
        np.log(np.asarray(growth, dtype=float)), np.asarray(times, dtype=float), edge_order=2
    )


def crossing_time(times: np.ndarray, decay: np.ndarray, threshold: float) -> float:
    t = np.asarray(times, dtype=float)
    d = np.asarray(decay, dtype=float)
    above = np.flatnonzero(d >= threshold)
    if above.size == 0 or above[0] == 0:
        return float("nan")
    k = above[0]
    lo, hi = np.log(d[k - 1]), np.log(d[k])
    frac = (np.log(threshold) - lo) / (hi - lo)
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def rescaling_exponents(sizes: Sequence[int], crossing_times: Sequence[float]) -> dict:
    n = np.asarray(sizes, dtype=float)
    tc = np.asarray(crossing_times, dtype=float)
    exponent = np.diff(np.log(n)) / np.diff(tc)
    return {"size": np.sqrt(n[1:] * n[:-1]), "exponent": exponent}


def rescaling_fit(
    sizes: Sequence[int],
    crossing_times: Sequence[float],
    errors: Sequence[float],
    correction: bool = True,
) -> dict[str, float]:
    n = np.asarray(sizes, dtype=float)
    tc = np.asarray(crossing_times, dtype=float)
    w = 1.0 / np.asarray(errors, dtype=float)
    columns = [np.ones_like(n), np.log(n)] + ([1.0 / n] if correction else [])
    A = np.column_stack(columns)
    coef, *_ = np.linalg.lstsq(A * w[:, None], tc * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    dof = max(n.size - A.shape[1], 1)
    chi2 = float(np.sum(((A @ coef - tc) * w) ** 2) / dof)
    slope, slope_err = coef[1], np.sqrt(cov[1, 1])
    return {
        "exponent": float(1.0 / slope),
        "exponent_err": float(slope_err / slope**2),
        "chi2_per_dof": chi2,
    }


def power_law_fit(
    sizes: Sequence[int], crossing_times: Sequence[float], errors: Sequence[float]
) -> dict[str, float]:
    n = np.asarray(sizes, dtype=float)
    tc = np.asarray(crossing_times, dtype=float)
    w = tc / np.asarray(errors, dtype=float)
    A = np.column_stack([np.ones_like(n), np.log(n)])
    coef, *_ = np.linalg.lstsq(A * w[:, None], np.log(tc) * w, rcond=None)
    cov = np.linalg.inv((A * w[:, None]).T @ (A * w[:, None]))
    chi2 = float(np.sum(((A @ coef - np.log(tc)) * w) ** 2) / max(n.size - 2, 1))
    return {"power": float(coef[1]), "power_err": float(np.sqrt(cov[1, 1])), "chi2_per_dof": chi2}
