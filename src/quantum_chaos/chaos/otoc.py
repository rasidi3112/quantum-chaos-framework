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
