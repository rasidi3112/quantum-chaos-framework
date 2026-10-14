

import numpy as np
import pennylane as qml
from typing import List, Tuple, Dict, Optional, Callable, Union
from scipy.linalg import expm


class QuantumKernel:
    def __init__(
        self,
        n_qubits: int,
        hamiltonian_constructor: Callable[[np.ndarray], np.ndarray],
        evolution_time: float = 1.0,
        n_trotter_steps: int = 10
    ):
        self.n_qubits = n_qubits
        self.hamiltonian_constructor = hamiltonian_constructor
        self.evolution_time = evolution_time
        self.n_trotter_steps = n_trotter_steps
        
        self._device = None
    
    @property
    def device(self) -> qml.devices.Device:
        if self._device is None:
            self._device = qml.device('default.qubit', wires=self.n_qubits)
        return self._device
    
    def feature_map_matrix(self, x: np.ndarray) -> np.ndarray:
        H = self.hamiltonian_constructor(x)
        return expm(-1j * H * self.evolution_time)
    
    def feature_state(
        self,
        x: np.ndarray,
        initial_state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        dim = 2 ** self.n_qubits
        
        if initial_state is None:
            initial_state = np.zeros(dim, dtype=complex)
            initial_state[0] = 1.0
        
        U = self.feature_map_matrix(x)
        return U @ initial_state
    
    def kernel_value(self, x1: np.ndarray, x2: np.ndarray) -> float:
        state1 = self.feature_state(x1)
        state2 = self.feature_state(x2)
        
        overlap = np.abs(np.vdot(state1, state2)) ** 2
        return float(overlap)
    
    def kernel_matrix(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if X2 is None:
            X2 = X1
            symmetric = True
        else:
            symmetric = False
        
        n1, n2 = len(X1), len(X2)
        K = np.zeros((n1, n2))
        
        for i in range(n1):
            j_start = i if symmetric else 0
            for j in range(j_start, n2):
                K[i, j] = self.kernel_value(X1[i], X2[j])
                if symmetric and i != j:
                    K[j, i] = K[i, j]
        
        return K
    
    def expressivity_metric(
        self,
        n_samples: int = 100
    ) -> Dict:
        # Generate random data
        data_dim = self.n_qubits  # Assume input dimension = n_qubits
        X = np.random.randn(n_samples, data_dim)
        
        # Compute kernel matrix
        K = self.kernel_matrix(X)
        
        # Extract off-diagonal elements
        off_diag = K[np.triu_indices(n_samples, k=1)]
        
        # Compute metrics
        return {
            'mean_kernel': float(np.mean(off_diag)),
            'std_kernel': float(np.std(off_diag)),
            'min_kernel': float(np.min(off_diag)),
            'max_kernel': float(np.max(off_diag)),
            # Ideal: mean close to 0 (well-separated), std close to 0 (uniform)
            # High expressivity: low mean, low std
            'expressivity_score': float(1 - np.mean(off_diag)),
            'kernel_rank': int(np.linalg.matrix_rank(K, tol=1e-6)),
            'effective_dimension': self._effective_dimension(K)
        }
    
    def _effective_dimension(self, K: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        if len(eigenvalues) == 0:
            return 0.0
        eigenvalues /= eigenvalues.sum()
        return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues + 1e-12))))
    
    def create_pennylane_kernel(self) -> Callable:
        @qml.qnode(self.device)
        def kernel_circuit(x1, x2):
            # This is a simplified version
            # Full implementation would use swap test
            
            # Encode x1 using parametrized rotations
            for i in range(min(len(x1), self.n_qubits)):
                qml.RY(x1[i], wires=i)
                qml.RZ(x1[i], wires=i)
            
            # Apply entangling layer
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            
            # Encode difference x1 - x2
            for i in range(min(len(x2), self.n_qubits)):
                qml.RY(-x2[i], wires=i)
                qml.RZ(-x2[i], wires=i)
            
            return qml.probs(wires=range(self.n_qubits))
        
        def kernel_fn(x1, x2):
            probs = kernel_circuit(x1, x2)
            return float(probs[0])  # Probability of returning to |0...0⟩
        
        return kernel_fn


class SYKKernel(QuantumKernel):
    def __init__(
        self,
        n_qubits: int,
        coupling_strength: float = 1.0,
        evolution_time: float = 1.0,
        seed: Optional[int] = None
    ):
        from ..hamiltonians.syk_hamiltonian import SYKHamiltonian
        
        self.coupling_strength = coupling_strength
        self.seed = seed
        
        # Create base SYK Hamiltonian
        self._syk = SYKHamiltonian(
            n_majorana=2 * n_qubits,
            coupling_strength=coupling_strength,
            seed=seed
        )
        
        def syk_constructor(x: np.ndarray) -> np.ndarray:
            # Modulate coupling strength with data
            scale = 1.0 + 0.1 * np.mean(x)
            return scale * self._syk.hamiltonian_matrix
        
        super().__init__(
            n_qubits=n_qubits,
            hamiltonian_constructor=syk_constructor,
            evolution_time=evolution_time
        )


class HubbardKernel(QuantumKernel):
    def __init__(
        self,
        n_sites: int,
        hopping_amplitude: float = 1.0,
        onsite_interaction: float = 1.0,
        evolution_time: float = 1.0
    ):
        from ..hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian
        
        self.n_sites = n_sites
        self.hopping_amplitude = hopping_amplitude
        self.onsite_interaction = onsite_interaction
        
        n_qubits = 2 * n_sites  # Spin-up and spin-down
        
        def hubbard_constructor(x: np.ndarray) -> np.ndarray:
            # Modulate parameters with data
            J = hopping_amplitude * (1 + 0.1 * x[0] if len(x) > 0 else 1)
            U = onsite_interaction * (1 + 0.1 * x[1] if len(x) > 1 else 1)
            
            hubbard = DrivenHubbardHamiltonian(
                n_sites=n_sites,
                hopping_amplitude=J,
                driving_frequency=1.0,
                onsite_interaction=U
            )
            return hubbard.hamiltonian(t=0)
        
        super().__init__(
            n_qubits=n_qubits,
            hamiltonian_constructor=hubbard_constructor,
            evolution_time=evolution_time
        )


def compare_kernel_expressivity(
    syk_kernel: QuantumKernel,
    hubbard_kernel: QuantumKernel,
    n_samples: int = 100
) -> Dict:
    syk_metrics = syk_kernel.expressivity_metric(n_samples)
    hubbard_metrics = hubbard_kernel.expressivity_metric(n_samples)
    
    return {
        'syk': syk_metrics,
        'hubbard': hubbard_metrics,
        'expressivity_comparison': {
            'syk_higher': syk_metrics['expressivity_score'] > hubbard_metrics['expressivity_score'],
            'score_difference': syk_metrics['expressivity_score'] - hubbard_metrics['expressivity_score'],
            'kernel_separation_ratio': syk_metrics['std_kernel'] / (hubbard_metrics['std_kernel'] + 1e-10)
        }
    }
