

import numpy as np
import pennylane as qml
from typing import Tuple, List, Optional, Dict, Union
from scipy.linalg import expm


class OTOCCalculator:
    def __init__(
        self,
        n_qubits: int,
        hamiltonian: Optional[np.ndarray] = None
    ):
        self.n_qubits = n_qubits
        self.hamiltonian = hamiltonian
        self._eigenvalues = None
        self._eigenvectors = None
    
    def set_hamiltonian(self, H: np.ndarray):
        self.hamiltonian = H
        self._eigenvalues = None
        self._eigenvectors = None
    
    def _diagonalize(self):
        if self._eigenvalues is None:
            self._eigenvalues, self._eigenvectors = np.linalg.eigh(self.hamiltonian)
    
    def _time_evolution_operator(self, t: float) -> np.ndarray:
        self._diagonalize()
        phases = np.exp(-1j * self._eigenvalues * t)
        return self._eigenvectors @ np.diag(phases) @ self._eigenvectors.conj().T
    
    def _heisenberg_evolution(self, op: np.ndarray, t: float) -> np.ndarray:
        U = self._time_evolution_operator(t)
        return U.conj().T @ op @ U
    
    def compute_otoc_exact(
        self,
        W: np.ndarray,
        V: np.ndarray,
        times: np.ndarray,
        state: Optional[np.ndarray] = None,
        temperature: Optional[float] = None
    ) -> np.ndarray:
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not set")
        
        dim = 2 ** self.n_qubits
        otoc_values = []
        
        # Prepare density matrix
        if state is not None:
            rho = np.outer(state, state.conj())
        elif temperature is not None:
            self._diagonalize()
            beta = 1.0 / temperature
            boltzmann = np.exp(-beta * self._eigenvalues)
            Z = np.sum(boltzmann)
            rho = self._eigenvectors @ np.diag(boltzmann / Z) @ self._eigenvectors.conj().T
        else:
            # Infinite temperature (identity)
            rho = np.eye(dim) / dim
        
        V_dag = V.conj().T
        
        for t in times:
            W_t = self._heisenberg_evolution(W, t)
            W_t_dag = W_t.conj().T
            
            # C(t) = Tr[ρ W†(t) V† W(t) V]
            product = rho @ W_t_dag @ V_dag @ W_t @ V
            C_t = np.trace(product)
            
            otoc_values.append(C_t)
        
        return np.array(otoc_values)
    
    def compute_regularized_otoc(
        self,
        W: np.ndarray,
        V: np.ndarray,
        times: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        self._diagonalize()
        beta = 1.0 / temperature
        
        # ρ^(1/4)
        boltzmann_quarter = np.exp(-beta * self._eigenvalues / 4)
        rho_quarter = self._eigenvectors @ np.diag(boltzmann_quarter) @ self._eigenvectors.conj().T
        
        # Normalization
        Z = np.sum(np.exp(-beta * self._eigenvalues))
        rho_quarter /= (Z ** 0.25)
        
        otoc_values = []
        
        for t in times:
            W_t = self._heisenberg_evolution(W, t)
            
            # Regularized correlator
            product = rho_quarter @ W_t @ rho_quarter @ V @ rho_quarter @ W_t @ rho_quarter @ V
            C_t = np.trace(product)
            otoc_values.append(C_t)
        
        return np.array(otoc_values)
    
    def compute_commutator_squared(
        self,
        W: np.ndarray,
        V: np.ndarray,
        times: np.ndarray,
        state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        otoc = self.compute_otoc_exact(W, V, times, state=state)
        return 2 * (1 - np.real(otoc))
    
    def local_pauli(self, pauli: str, site: int) -> np.ndarray:
        I = np.eye(2)
        paulis = {
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)
        }
        
        ops = []
        for i in range(self.n_qubits):
            if i == site:
                ops.append(paulis[pauli])
            else:
                ops.append(I)
        
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        
        return result
    
    def butterfly_velocity(
        self,
        site_separation: int,
        times: np.ndarray,
        threshold: float = 0.5
    ) -> float:
        # Use Z operators at sites 0 and site_separation
        W = self.local_pauli('Z', 0)
        V = self.local_pauli('Z', min(site_separation, self.n_qubits - 1))
        
        comm_sq = self.compute_commutator_squared(W, V, times)
        
        # Find scrambling time (when |[W(t), V]|² > threshold)
        scrambled_idx = np.where(comm_sq > threshold)[0]
        
        if len(scrambled_idx) == 0:
            return 0.0  # No scrambling observed
        
        t_scramble = times[scrambled_idx[0]]
        
        if t_scramble > 0:
            return site_separation / t_scramble
        return np.inf
    
    def lyapunov_exponent(
        self,
        times: np.ndarray,
        temperature: Optional[float] = None,
        fit_window: Tuple[float, float] = (0.1, 0.9)
    ) -> Tuple[float, float]:
        # Use nearby sites for local operators
        W = self.local_pauli('Z', 0)
        V = self.local_pauli('Z', min(1, self.n_qubits - 1))
        
        otoc = self.compute_otoc_exact(W, V, times, temperature=temperature)
        
        # Compute 1 - Re[C(t)]
        deviation = 1 - np.real(otoc)
        deviation = np.maximum(deviation, 1e-15)  # Avoid log(0)
        
        # Fit exponential in log space
        log_dev = np.log(deviation)
        
        # Select fitting window
        n_points = len(times)
        start_idx = int(n_points * fit_window[0])
        end_idx = int(n_points * fit_window[1])
        
        if end_idx <= start_idx:
            return 0.0, np.inf
        
        times_fit = times[start_idx:end_idx]
        log_fit = log_dev[start_idx:end_idx]
        
        # Linear fit: log(1 - C) ≈ 2λt + const
        try:
            coeffs = np.polyfit(times_fit, log_fit, 1)
            lambda_L = coeffs[0] / 2  # Factor of 2 from definition
            
            # Estimate fit quality
            fit_vals = np.polyval(coeffs, times_fit)
            residuals = log_fit - fit_vals
            fit_error = np.std(residuals)
            
            return float(lambda_L), float(fit_error)
        except:
            return 0.0, np.inf
    
    def scrambling_time(
        self,
        threshold: float = 0.5,
        max_time: float = 10.0,
        n_time_points: int = 100
    ) -> float:
        times = np.linspace(0, max_time, n_time_points)
        
        W = self.local_pauli('Z', 0)
        V = self.local_pauli('Z', min(1, self.n_qubits - 1))
        
        otoc = self.compute_otoc_exact(W, V, times)
        otoc_real = np.real(otoc)
        
        # Find first time C(t) < threshold
        below_threshold = np.where(otoc_real < threshold)[0]
        
        if len(below_threshold) == 0:
            return max_time  # Not scrambled within time window
        
        return float(times[below_threshold[0]])
    
    def operator_size_growth(
        self,
        initial_op: np.ndarray,
        times: np.ndarray
    ) -> np.ndarray:
        from ..utils.jordan_wigner import get_pauli_string_representation
        
        sizes = []
        
        for t in times:
            op_t = self._heisenberg_evolution(initial_op, t)
            
            # Decompose into Pauli strings
            pauli_decomp = get_pauli_string_representation(op_t, self.n_qubits)
            
            # Compute average weight
            total_weight = 0
            total_magnitude = 0
            
            for coeff, pauli_str in pauli_decomp:
                weight = sum(1 for p in pauli_str if p != 'I')
                magnitude = np.abs(coeff) ** 2
                total_weight += weight * magnitude
                total_magnitude += magnitude
            
            if total_magnitude > 0:
                avg_size = total_weight / total_magnitude
            else:
                avg_size = 0
            
            sizes.append(avg_size)
        
        return np.array(sizes)
    
    def create_otoc_circuit(
        self,
        W_pauli: str,
        W_site: int,
        V_pauli: str,
        V_site: int
    ):
        n_wires = self.n_qubits + 1  # Extra ancilla
        dev = qml.device('default.qubit', wires=n_wires)
        
        pauli_gate = {
            'X': qml.PauliX,
            'Y': qml.PauliY,
            'Z': qml.PauliZ
        }
        
        @qml.qnode(dev)
        def otoc_circuit(time: float, n_trotter_steps: int = 10):
            ancilla = self.n_qubits
            
            # Prepare ancilla in |+⟩
            qml.Hadamard(wires=ancilla)
            
            # Controlled-V
            qml.ctrl(pauli_gate[V_pauli], control=ancilla)(wires=V_site)
            
            # Forward evolution (controlled)
            # Simplified: would need full Trotter implementation
            # This is a placeholder for demonstration
            
            # Controlled-W
            qml.ctrl(pauli_gate[W_pauli], control=ancilla)(wires=W_site)
            
            # Backward evolution (controlled)
            
            # Controlled-V†
            qml.ctrl(pauli_gate[V_pauli], control=ancilla)(wires=V_site)
            
            # Controlled-W†
            qml.ctrl(pauli_gate[W_pauli], control=ancilla)(wires=W_site)
            
            # Measure ancilla
            qml.Hadamard(wires=ancilla)
            
            return qml.expval(qml.PauliZ(ancilla))
        
        return otoc_circuit
    
    def compare_models(
        self,
        hamiltonian_2: np.ndarray,
        times: np.ndarray,
        model_names: Tuple[str, str] = ('Model 1', 'Model 2')
    ) -> Dict:
        W = self.local_pauli('Z', 0)
        V = self.local_pauli('Z', min(1, self.n_qubits - 1))
        
        # Model 1
        otoc_1 = self.compute_otoc_exact(W, V, times)
        lambda_1, err_1 = self.lyapunov_exponent(times)
        
        # Model 2
        orig_H = self.hamiltonian
        self.set_hamiltonian(hamiltonian_2)
        otoc_2 = self.compute_otoc_exact(W, V, times)
        lambda_2, err_2 = self.lyapunov_exponent(times)
        
        # Restore
        self.set_hamiltonian(orig_H)
        
        return {
            'times': times,
            model_names[0]: {
                'otoc': otoc_1,
                'lyapunov': lambda_1,
                'lyapunov_error': err_1,
                'scrambling_time': self.scrambling_time()
            },
            model_names[1]: {
                'otoc': otoc_2,
                'lyapunov': lambda_2,
                'lyapunov_error': err_2
            }
        }
