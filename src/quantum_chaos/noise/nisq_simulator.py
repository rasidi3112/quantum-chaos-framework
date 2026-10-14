

import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional, Dict, Callable
from scipy.linalg import expm

from .noise_channels import NoiseChannels, NoiseModel
from ..utils.helpers import fidelity


class NISQSimulator:
    def __init__(
        self,
        n_qubits: int,
        noise_model: Optional[NoiseModel] = None
    ):
        self.n_qubits = n_qubits
        self.noise_model = noise_model or NoiseModel()
        
        # Initialize noise channels
        self.noise_channels = NoiseChannels(n_qubits)
        
        # Create devices
        self._ideal_device = None
        self._noisy_device = None
    
    @property
    def ideal_device(self) -> qml.devices.Device:
        if self._ideal_device is None:
            self._ideal_device = qml.device('default.qubit', wires=self.n_qubits)
        return self._ideal_device
    
    @property
    def noisy_device(self) -> qml.devices.Device:
        if self._noisy_device is None:
            self._noisy_device = qml.device('default.mixed', wires=self.n_qubits)
        return self._noisy_device
    
    def create_noisy_circuit(
        self,
        circuit_fn: Callable,
        noise_after_each_gate: bool = True
    ) -> Callable:
        # This is a simplified implementation
        # Full implementation would use qml.transforms
        
        @qml.qnode(self.noisy_device)
        def noisy_circuit(*args, **kwargs):
            # Apply original circuit
            circuit_fn(*args, **kwargs)
            
            # Add depolarizing noise to all qubits
            for q in range(self.n_qubits):
                qml.DepolarizingChannel(
                    self.noise_model.depolarizing_rate,
                    wires=q
                )
            
            return qml.state()
        
        return noisy_circuit
    
    def evolve_with_noise(
        self,
        initial_rho: np.ndarray,
        hamiltonian: np.ndarray,
        total_time: float,
        n_steps: int = 10
    ) -> np.ndarray:
        """Split-step noisy evolution: exp(-i H dt), then noise, repeated n_steps times.

        T1/T2 relaxation acts for the physical step duration dt = total_time/n_steps
        (same time unit as H), so for H = 0 the result does not depend on n_steps.
        The depolarizing error is a per-step (gate-layer) error and therefore does
        scale with n_steps.
        """
        dt = total_time / n_steps
        rho = np.asarray(initial_rho, dtype=complex).copy()
        U = expm(-1j * hamiltonian * dt)
        
        for _ in range(n_steps):
            rho = U @ rho @ U.conj().T
            rho = self.noise_channels.full_noise_model(rho, self.noise_model, duration=dt)
        
        return rho
    
    def compute_noisy_eigenspectrum(
        self,
        hamiltonian: np.ndarray,
        n_samples: int = 100,
        measurement_noise: bool = True
    ) -> Dict:
        dim = 2 ** self.n_qubits
        
        # Ideal eigenvalues for reference
        ideal_eigenvalues = np.linalg.eigvalsh(hamiltonian)
        
        # Estimate ground state energy with noise
        noisy_energies = []
        
        for _ in range(n_samples):
            # Create random state
            psi = np.random.randn(dim) + 1j * np.random.randn(dim)
            psi /= np.linalg.norm(psi)
            rho = np.outer(psi, psi.conj())
            
            # Apply noise to state
            rho_noisy = self.noise_channels.full_noise_model(rho, self.noise_model)
            
            # Measure energy
            energy = np.real(np.trace(hamiltonian @ rho_noisy))
            noisy_energies.append(energy)
        
        noisy_energies = np.array(noisy_energies)
        
        return {
            'ideal_eigenvalues': ideal_eigenvalues,
            'noisy_energy_samples': noisy_energies,
            'noisy_mean': np.mean(noisy_energies),
            'noisy_std': np.std(noisy_energies),
            'ground_state_energy': ideal_eigenvalues[0],
            'energy_gap': ideal_eigenvalues[1] - ideal_eigenvalues[0]
        }
    
    def benchmark_spectral_stability(
        self,
        hamiltonian: np.ndarray,
        noise_levels: List[float],
        n_samples: int = 50
    ) -> Dict:
        dim = 2 ** self.n_qubits
        
        # Ideal spectrum
        ideal_eigenvalues = np.linalg.eigvalsh(hamiltonian)
        
        results = {
            'noise_levels': noise_levels,
            'ideal_spectrum': ideal_eigenvalues,
            'ground_state_error': [],
            'spectral_width_error': [],
            'energy_variance': []
        }
        
        original_noise = self.noise_model.depolarizing_rate
        
        for noise_rate in noise_levels:
            self.noise_model.depolarizing_rate = noise_rate
            
            ground_errors = []
            width_errors = []
            variances = []
            
            for _ in range(n_samples):
                # Random initial state
                psi = np.random.randn(dim) + 1j * np.random.randn(dim)
                psi /= np.linalg.norm(psi)
                rho = np.outer(psi, psi.conj())
                
                # Apply noise
                rho_noisy = self.noise_channels.full_noise_model(rho, self.noise_model)
                
                # Energy measurement
                E_mean = np.real(np.trace(hamiltonian @ rho_noisy))
                E2_mean = np.real(np.trace(hamiltonian @ hamiltonian @ rho_noisy))
                variance = E2_mean - E_mean ** 2
                
                variances.append(variance)
            
            results['energy_variance'].append(np.mean(variances))
            
            # Compute error relative to ideal ground state
            # Using variational principle: any state gives E ≥ E_0
            min_noisy_energy = ideal_eigenvalues[0] + np.sqrt(np.mean(variances))
            results['ground_state_error'].append(
                min_noisy_energy - ideal_eigenvalues[0]
            )
            
            # Spectral width error (thermal broadening from noise)
            spectral_width_ideal = ideal_eigenvalues[-1] - ideal_eigenvalues[0]
            thermal_broadening = np.sqrt(np.mean(variances))
            results['spectral_width_error'].append(thermal_broadening)
        
        # Restore original noise
        self.noise_model.depolarizing_rate = original_noise
        
        return results
    
    def compare_ideal_vs_noisy(
        self,
        hamiltonian: np.ndarray,
        initial_state: np.ndarray,
        times: np.ndarray,
        observable: np.ndarray,
        steps_per_unit_time: int = 10
    ) -> Dict:
        """Ideal vs noisy expectation values and Uhlmann fidelity versus time.

        The noisy run uses ``max(1, int(steps_per_unit_time * t))`` split steps.
        """
        # Initial density matrix
        rho_0 = np.outer(initial_state, initial_state.conj())
        
        ideal_values = []
        noisy_values = []
        fidelities = []
        
        for t in times:
            # Ideal evolution
            U = expm(-1j * hamiltonian * t)
            rho_ideal = U @ rho_0 @ U.conj().T
            ideal_exp = np.real(np.trace(observable @ rho_ideal))
            ideal_values.append(ideal_exp)
            
            # Noisy evolution
            n_steps = max(1, int(steps_per_unit_time * t))
            rho_noisy = self.evolve_with_noise(rho_0, hamiltonian, t, n_steps)
            noisy_exp = np.real(np.trace(observable @ rho_noisy))
            noisy_values.append(noisy_exp)
            
            # State fidelity
            fidelities.append(fidelity(rho_ideal, rho_noisy))
        
        return {
            'times': times,
            'ideal_expectation': np.array(ideal_values),
            'noisy_expectation': np.array(noisy_values),
            'fidelity': np.array(fidelities),
            'expectation_error': np.abs(
                np.array(ideal_values) - np.array(noisy_values)
            )
        }
    
    def error_mitigation_zne(
        self,
        circuit_fn: Callable,
        observable: str,
        scale_factors: List[float] = [1.0, 2.0, 3.0]
    ) -> float:
        results = []
        
        original_rate = self.noise_model.depolarizing_rate
        
        for scale in scale_factors:
            # Scale noise
            self.noise_model.depolarizing_rate = original_rate * scale
            
            # Run circuit and measure
            # (Simplified - would need full circuit integration)
            result = 0.0  # Placeholder
            results.append(result)
        
        # Restore noise
        self.noise_model.depolarizing_rate = original_rate
        
        # Richardson extrapolation to zero noise
        # For linear extrapolation with 2 points:
        # f(0) ≈ f(c_1) + [f(c_1) - f(c_2)] * c_1 / (c_2 - c_1)
        if len(results) >= 2:
            extrapolated = results[0] + (results[0] - results[1]) * scale_factors[0] / (scale_factors[1] - scale_factors[0])
            return extrapolated
        
        return results[0] if results else 0.0
    
    def estimate_circuit_depth_limit(
        self,
        target_fidelity: float = 0.9
    ) -> float:
        """Heuristic depth limit from depolarizing noise only.

        Largest depth d such that the probability that no qubit is depolarized,
        (1 - p)^(n_qubits * d), stays above ``target_fidelity``. Ignores T1/T2
        and two-qubit gate errors; returns ``inf`` for p = 0.
        """
        p = self.noise_model.depolarizing_rate
        if p <= 0:
            return float('inf')
        if p >= 1:
            return 0.0
        return float(np.floor(np.log(target_fidelity) / (self.n_qubits * np.log1p(-p))))
    
    def create_pennylane_noisy_circuit(
        self,
        pauli_terms: List[Tuple[float, str]],
        total_time: float,
        n_steps: int
    ) -> Callable:
        dt = total_time / n_steps
        p_depol = self.noise_model.depolarizing_rate
        gamma_ad = self.noise_model.amplitude_damping_rate
        
        @qml.qnode(self.noisy_device)
        def noisy_evolution_circuit(initial_state):
            # Prepare initial state
            qml.StatePrep(initial_state, wires=range(self.n_qubits))
            
            for _ in range(n_steps):
                # Trotter step (simplified)
                for coeff, pauli_str in pauli_terms:
                    if pauli_str == 'Z' * self.n_qubits:
                        for q in range(self.n_qubits):
                            qml.RZ(2 * coeff * dt, wires=q)
                
                # Apply noise after each Trotter step
                for q in range(self.n_qubits):
                    qml.DepolarizingChannel(p_depol, wires=q)
                    qml.AmplitudeDamping(gamma_ad, wires=q)
            
            return qml.density_matrix(wires=range(self.n_qubits))
        
        return noisy_evolution_circuit
    
    def summary(self) -> Dict:
        return {
            'n_qubits': self.n_qubits,
            'depolarizing_rate': self.noise_model.depolarizing_rate,
            'amplitude_damping_rate': self.noise_model.amplitude_damping_rate,
            'T1': self.noise_model.t1,
            'T2': self.noise_model.t2,
            'estimated_max_depth': self.estimate_circuit_depth_limit(),
            'average_gate_fidelity': self.noise_channels.channel_fidelity(
                self.noise_channels.depolarizing_kraus(self.noise_model.depolarizing_rate)
            )
        }
