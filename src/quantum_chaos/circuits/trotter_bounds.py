from collections.abc import Sequence

import numpy as np
from scipy.linalg import expm


def _norm(a: np.ndarray) -> float:
    return float(np.linalg.norm(a, 2))


def first_order_commutator_bound(terms: Sequence[np.ndarray], time: float, steps: int) -> float:
    tail = np.zeros_like(terms[0], dtype=complex)
    total = 0.0
    for H in reversed(terms):
        total += _norm(tail @ H - H @ tail)
        tail = tail + H
    dt = time / steps
    return steps * dt**2 / 2 * total


def product_formula(terms: Sequence[np.ndarray], time: float, steps: int, order: int = 1):
    dt = time / steps
    if order == 1:
        layer = np.eye(terms[0].shape[0], dtype=complex)
        for H in terms:
            layer = expm(-1j * dt * H) @ layer
    elif order == 2:
        half = [expm(-0.5j * dt * H) for H in terms]
        layer = np.eye(terms[0].shape[0], dtype=complex)
        for U in half:
            layer = U @ layer
        for U in reversed(half):
            layer = U @ layer
    else:
        raise ValueError("order must be 1 or 2")
    return np.linalg.matrix_power(layer, steps)


def trotter_error(terms: Sequence[np.ndarray], time: float, steps: int, order: int = 1) -> float:
    exact = expm(-1j * time * sum(terms))
    return _norm(product_formula(terms, time, steps, order) - exact)
