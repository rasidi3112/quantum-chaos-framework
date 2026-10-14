import numpy as np


def _walsh_hadamard(rows: np.ndarray) -> np.ndarray:
    out = np.array(rows, dtype=complex)
    n = out.shape[-1]
    h = 1
    while h < n:
        out = out.reshape(out.shape[0], -1, 2, h)
        a = out[:, :, 0, :] + out[:, :, 1, :]
        b = out[:, :, 0, :] - out[:, :, 1, :]
        out = np.stack([a, b], axis=2).reshape(out.shape[0], n)
        h *= 2
    return out


def pauli_coefficients(operator: np.ndarray) -> np.ndarray:
    O = np.asarray(operator, dtype=complex)
    dim = O.shape[0]
    basis = np.arange(dim)
    rows = basis[None, :] ^ basis[:, None]
    shifted = O[rows, basis[None, :]]
    return _walsh_hadamard(shifted) / dim


def majorana_sizes(n_qubits: int) -> np.ndarray:
    dim = 2**n_qubits
    x = np.arange(dim)[:, None]
    z = np.arange(dim)[None, :]
    size = np.zeros((dim, dim), dtype=np.int64)
    parity = np.zeros((dim, dim), dtype=np.int64)
    for q in range(n_qubits - 1, -1, -1):
        bit = n_qubits - 1 - q
        xq = (x >> bit) & 1
        zq = ((z >> bit) & 1) ^ parity
        a = xq * (1 - zq) + (1 - xq) * zq
        bq = xq * zq + (1 - xq) * zq
        count = a + bq
        size += count
        parity = (parity + count) % 2
    return size


def size_distribution(operator: np.ndarray) -> np.ndarray:
    coeffs = pauli_coefficients(operator)
    n_qubits = int(np.log2(coeffs.shape[0]))
    sizes = majorana_sizes(n_qubits)
    weights = np.abs(coeffs) ** 2
    return np.bincount(sizes.ravel(), weights=weights.ravel(), minlength=2 * n_qubits + 1) / (
        weights.sum()
    )


def mean_size_from_anticommutators(operator: np.ndarray, majoranas) -> float:
    O = np.asarray(operator)
    norm = np.vdot(O, O).real
    total = sum(np.vdot(c @ O + O @ c, c @ O + O @ c).real for c in majoranas)
    return float(total / (4 * norm))
