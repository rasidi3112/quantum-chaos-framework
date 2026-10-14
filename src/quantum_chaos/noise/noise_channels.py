

import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class NoiseModel:
    depolarizing_rate: float = 0.01
    amplitude_damping_rate: float = 0.01
    phase_damping_rate: float = 0.01
    t1: float = 1000.0
    t2: float = 500.0
    gate_time: float = 1.0
    readout_error: float = 0.01


class NoiseChannels:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        
        # Pauli matrices
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
    
    def depolarizing_kraus(self, p: float) -> List[np.ndarray]:
        # Note: Different normalization conventions exist
        # This uses: E_0 = √(1-3p/4)I, E_1 = √(p/4)X, E_2 = √(p/4)Y, E_3 = √(p/4)Z
        sqrt_1mp = np.sqrt(1 - 3*p/4)
        sqrt_p = np.sqrt(p/4)
        
        K0 = sqrt_1mp * self.I
        K1 = sqrt_p * self.X
        K2 = sqrt_p * self.Y
        K3 = sqrt_p * self.Z
        
        return [K0, K1, K2, K3]
    
    def amplitude_damping_kraus(self, gamma: float) -> List[np.ndarray]:
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, np.sqrt(gamma)],
            [0, 0]
        ], dtype=complex)
        
        return [K0, K1]
    
    def phase_damping_kraus(self, gamma: float) -> List[np.ndarray]:
        K0 = np.array([
            [1, 0],
            [0, np.sqrt(1 - gamma)]
        ], dtype=complex)
        
        K1 = np.array([
            [0, 0],
            [0, np.sqrt(gamma)]
        ], dtype=complex)
        
        return [K0, K1]
    
    def bit_flip_kraus(self, p: float) -> List[np.ndarray]:
        K0 = np.sqrt(1 - p) * self.I
        K1 = np.sqrt(p) * self.X
        
        return [K0, K1]
    
    def phase_flip_kraus(self, p: float) -> List[np.ndarray]:
        K0 = np.sqrt(1 - p) * self.I
        K1 = np.sqrt(p) * self.Z
        
        return [K0, K1]
    
    def apply_channel(
        self,
        rho: np.ndarray,
        kraus_ops: List[np.ndarray],
        qubit: int
    ) -> np.ndarray:
        result = np.zeros_like(rho)
        
        for K in kraus_ops:
            K_full = self._embed_operator(K, qubit)
            result += K_full @ rho @ K_full.conj().T
        
        return result
    
    def apply_depolarizing(
        self,
        rho: np.ndarray,
        p: float,
        qubits: Optional[List[int]] = None
    ) -> np.ndarray:
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        kraus = self.depolarizing_kraus(p)
        
        for q in qubits:
            rho = self.apply_channel(rho, kraus, q)
        
        return rho
    
    def apply_amplitude_damping(
        self,
        rho: np.ndarray,
        gamma: float,
        qubits: Optional[List[int]] = None
    ) -> np.ndarray:
        if qubits is None:
            qubits = list(range(self.n_qubits))
        
        kraus = self.amplitude_damping_kraus(gamma)
        
        for q in qubits:
            rho = self.apply_channel(rho, kraus, q)
        
        return rho
    
    def apply_phase_damping(
        self,
        rho: np.ndarray,
        gamma: float,
        qubits: Optional[List[int]] = None
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
        error_rate: float
    ) -> np.ndarray:
        noise = np.random.random(measurement_results.shape) < error_rate
        return np.logical_xor(measurement_results, noise).astype(int)
    
    def full_noise_model(
        self,
        rho: np.ndarray,
        noise_model: NoiseModel
    ) -> np.ndarray:
        # Convert T1, T2 to damping parameters
        # γ = 1 - exp(-t_gate/T)
        gamma_t1 = 1 - np.exp(-noise_model.gate_time / noise_model.t1)
        gamma_t2 = 1 - np.exp(-noise_model.gate_time / noise_model.t2)
        
        # Apply amplitude damping (T1)
        rho = self.apply_amplitude_damping(rho, gamma_t1)
        
        # Apply phase damping (T2)
        # Note: Pure dephasing rate is related to T1 and T2
        # 1/T2 = 1/(2T1) + 1/T_phi
        rho = self.apply_phase_damping(rho, gamma_t2)
        
        # Apply depolarizing error
        rho = self.apply_depolarizing(rho, noise_model.depolarizing_rate)
        
        return rho
    
    def verify_cptp(
        self,
        kraus_ops: List[np.ndarray],
        tol: float = 1e-10
    ) -> bool:
        dim = kraus_ops[0].shape[0]
        identity = np.eye(dim)
        
        sum_KdagK = sum(K.conj().T @ K for K in kraus_ops)
        
        return np.allclose(sum_KdagK, identity, atol=tol)
    
    def channel_fidelity(
        self,
        kraus_ops: List[np.ndarray]
    ) -> float:
        dim = kraus_ops[0].shape[0]
        
        # Entanglement fidelity
        F_e = sum(np.abs(np.trace(K))**2 for K in kraus_ops) / dim**2
        
        # Average fidelity
        F_avg = (dim * F_e + 1) / (dim + 1)
        
        return float(F_avg)
