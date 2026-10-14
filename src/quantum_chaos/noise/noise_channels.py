from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    depolarizing_rate: float = 0.01
    t1: float = 1000.0
    t2: float = 500.0
    gate_time: float = 1.0
    readout_error: float = 0.01


def pennylane_depolarizing_parameter(p: float) -> float:
    return 3.0 * p / 4.0


def relaxation_parameters(duration: float, t1: float, t2: float) -> tuple[float, float]:
    if t1 <= 0 or t2 <= 0:
        raise ValueError("T1 and T2 must be positive")
    if t2 > 2 * t1 * (1 + 1e-12):
        raise ValueError(f"unphysical T2={t2} > 2*T1={2 * t1}")
    rate_phi = max(1.0 / t2 - 1.0 / (2.0 * t1), 0.0)
    gamma_1 = -np.expm1(-duration / t1)
    gamma_phi = -np.expm1(-2.0 * duration * rate_phi)
    return float(gamma_1), float(gamma_phi)


class NoiseChannels:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

        self.I = np.array([[1, 0], [0, 1]], dtype=complex)
        self.X = np.array([[0, 1], [1, 0]], dtype=complex)
        self.Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.Z = np.array([[1, 0], [0, -1]], dtype=complex)

    def _embed_operator(self, op: np.ndarray, qubit: int) -> np.ndarray:
        result = np.array([[1]], dtype=complex)

        for i in range(self.n_qubits):
            if i == qubit:
                result = np.kron(result, op)
            else:
                result = np.kron(result, self.I)

        return result

    def depolarizing_kraus(self, p: float) -> list[np.ndarray]:
        sqrt_1mp = np.sqrt(1 - 3 * p / 4)
        sqrt_p = np.sqrt(p / 4)

        K0 = sqrt_1mp * self.I
        K1 = sqrt_p * self.X
        K2 = sqrt_p * self.Y
        K3 = sqrt_p * self.Z

        return [K0, K1, K2, K3]

    def amplitude_damping_kraus(self, gamma: float) -> list[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)

        K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)

        return [K0, K1]

    def phase_damping_kraus(self, gamma: float) -> list[np.ndarray]:
        K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)

        K1 = np.array([[0, 0], [0, np.sqrt(gamma)]], dtype=complex)

        return [K0, K1]

    def bit_flip_kraus(self, p: float) -> list[np.ndarray]:
        K0 = np.sqrt(1 - p) * self.I
        K1 = np.sqrt(p) * self.X

        return [K0, K1]

    def phase_flip_kraus(self, p: float) -> list[np.ndarray]:
        K0 = np.sqrt(1 - p) * self.I
        K1 = np.sqrt(p) * self.Z

        return [K0, K1]

    def apply_channel(self, rho: np.ndarray, kraus_ops: list[np.ndarray], qubit: int) -> np.ndarray:
        rho = np.asarray(rho, dtype=complex)
        result = np.zeros_like(rho)

        for K in kraus_ops:
            K_full = self._embed_operator(K, qubit)
            result += K_full @ rho @ K_full.conj().T

        return result

    def apply_depolarizing(
        self, rho: np.ndarray, p: float, qubits: list[int] | None = None
    ) -> np.ndarray:
        if qubits is None:
            qubits = list(range(self.n_qubits))

        kraus = self.depolarizing_kraus(p)

        for q in qubits:
            rho = self.apply_channel(rho, kraus, q)

        return rho

    def apply_amplitude_damping(
        self, rho: np.ndarray, gamma: float, qubits: list[int] | None = None
    ) -> np.ndarray:
        if qubits is None:
            qubits = list(range(self.n_qubits))

        kraus = self.amplitude_damping_kraus(gamma)

        for q in qubits:
            rho = self.apply_channel(rho, kraus, q)

        return rho

    def apply_phase_damping(
        self, rho: np.ndarray, gamma: float, qubits: list[int] | None = None
    ) -> np.ndarray:
        if qubits is None:
            qubits = list(range(self.n_qubits))

        kraus = self.phase_damping_kraus(gamma)

        for q in qubits:
            rho = self.apply_channel(rho, kraus, q)

        return rho

    def apply_readout_error(
        self,
        measurement_results: np.ndarray,
        error_rate: float,
        rng: None | int | np.random.Generator = None,
    ) -> np.ndarray:
        noise = np.random.default_rng(rng).random(np.shape(measurement_results)) < error_rate
        return np.logical_xor(measurement_results, noise).astype(int)

    def full_noise_model(
        self, rho: np.ndarray, noise_model: NoiseModel, duration: float | None = None
    ) -> np.ndarray:
        if duration is None:
            duration = noise_model.gate_time
        gamma_1, gamma_phi = relaxation_parameters(duration, noise_model.t1, noise_model.t2)
        rho = self.apply_amplitude_damping(rho, gamma_1)
        rho = self.apply_phase_damping(rho, gamma_phi)
        if noise_model.depolarizing_rate > 0:
            rho = self.apply_depolarizing(rho, noise_model.depolarizing_rate)
        return rho

    def verify_cptp(self, kraus_ops: list[np.ndarray], tol: float = 1e-10) -> bool:
        dim = kraus_ops[0].shape[0]
        identity = np.eye(dim)

        sum_KdagK = sum(K.conj().T @ K for K in kraus_ops)

        return np.allclose(sum_KdagK, identity, atol=tol)

    def channel_fidelity(self, kraus_ops: list[np.ndarray]) -> float:
        dim = kraus_ops[0].shape[0]

        F_e = sum(np.abs(np.trace(K)) ** 2 for K in kraus_ops) / dim**2

        F_avg = (dim * F_e + 1) / (dim + 1)

        return float(F_avg)
