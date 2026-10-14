from collections.abc import Mapping

import numpy as np


def _comm(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b - b @ a


def van_vleck_hamiltonian(
    harmonics: Mapping[int, np.ndarray], omega: float, order: int = 2
) -> np.ndarray:
    if order not in (0, 1, 2):
        raise NotImplementedError("van Vleck expansion implemented up to order 2")
    H = {m: np.asarray(h) for m, h in harmonics.items()}
    H0 = H.get(0, np.zeros_like(next(iter(H.values()))))
    out = H0.astype(complex).copy()
    ms = [m for m in H if m != 0]
    if order >= 1:
        for m in ms:
            if m > 0 and -m in H:
                out += _comm(H[m], H[-m]) / (m * omega)
    if order >= 2:
        for m in ms:
            if -m in H:
                out += _comm(_comm(H[-m], H0), H[m]) / (2 * m**2 * omega**2)
            for mp in ms:
                if mp == m or -mp not in H or (mp - m) not in H or (mp - m) == 0:
                    continue
                out += _comm(_comm(H[-mp], H[mp - m]), H[m]) / (3 * m * mp * omega**2)
    return out


def magnus_hamiltonian(
    harmonics: Mapping[int, np.ndarray], omega: float, order: int = 1
) -> np.ndarray:
    if order not in (0, 1):
        raise NotImplementedError("Floquet-Magnus expansion implemented up to order 1")
    H = {m: np.asarray(h) for m, h in harmonics.items()}
    H0 = H.get(0, np.zeros_like(next(iter(H.values()))))
    out = H0.astype(complex).copy()
    if order >= 1:
        for m, Hm in H.items():
            if m == 0:
                continue
            if -m in H and m > 0:
                out += _comm(Hm, H[-m]) / (m * omega)
            out += _comm(H0, Hm) / (m * omega)
    return out
