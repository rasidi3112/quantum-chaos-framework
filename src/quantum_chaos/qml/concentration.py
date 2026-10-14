import numpy as np

from ..utils.helpers import RngLike
from .quantum_kernel import QuantumKernel


def haar_fidelity_moments(dim: int) -> dict[str, float]:
    return {"mean": 1.0 / dim, "variance": (dim - 1) / (dim**2 * (dim + 1))}


def kernel_offdiagonal_statistics(
    kernel: QuantumKernel, n_samples: int = 40, n_features: int | None = None, rng: RngLike = None
) -> dict[str, float]:
    rng = np.random.default_rng(rng)
    d = kernel.n_qubits if n_features is None else n_features
    X = rng.standard_normal((n_samples, d))
    K = kernel.kernel_matrix(X)
    off = K[np.triu_indices(n_samples, k=1)]
    return {
        "mean": float(off.mean()),
        "variance": float(off.var(ddof=1)),
        "n_pairs": int(off.size),
    }


def effective_dimension_of_states(kernel: QuantumKernel, X: np.ndarray) -> float:
    states = np.array([kernel.feature_state(x) for x in X])
    probs = np.abs(states) ** 2
    mean = probs.mean(axis=0)
    return float(1.0 / np.sum(mean**2))
