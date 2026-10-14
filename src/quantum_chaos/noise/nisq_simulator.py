import numpy as np
import pennylane as qml
from typing import List, Tuple, Optional, Dict, Callable
from scipy.linalg import expm

from .noise_channels import NoiseChannels, NoiseModel
from ..utils.helpers import fidelity


class NISQSimulator:
    def __init__(self, n_qubits: int, noise_model: Optional[NoiseModel] = None):
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
            self._ideal_device = qml.device("default.qubit", wires=self.n_qubits)
        return self._ideal_device

    @property
    def noisy_device(self) -> qml.devices.Device:
        if self._noisy_device is None:
            self._noisy_device = qml.device("default.mixed", wires=self.n_qubits)
        return self._noisy_device

    def evolve_with_noise(
        self, initial_rho: np.ndarray, hamiltonian: np.ndarray, total_time: float, n_steps: int = 10
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

    def compare_ideal_vs_noisy(
        self,
        hamiltonian: np.ndarray,
        initial_state: np.ndarray,
        times: np.ndarray,
        observable: np.ndarray,
        steps_per_unit_time: int = 10,
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
            "times": times,
            "ideal_expectation": np.array(ideal_values),
            "noisy_expectation": np.array(noisy_values),
            "fidelity": np.array(fidelities),
            "expectation_error": np.abs(np.array(ideal_values) - np.array(noisy_values)),
        }

    def estimate_circuit_depth_limit(self, target_fidelity: float = 0.9) -> float:
        """Heuristic depth limit from depolarizing noise only.

        Largest depth d such that the probability that no qubit is depolarized,
        (1 - p)^(n_qubits * d), stays above ``target_fidelity``. Ignores T1/T2
        and two-qubit gate errors; returns ``inf`` for p = 0.
        """
        p = self.noise_model.depolarizing_rate
        if p <= 0:
            return float("inf")
        if p >= 1:
            return 0.0
        return float(np.floor(np.log(target_fidelity) / (self.n_qubits * np.log1p(-p))))

    def create_pennylane_noisy_circuit(
        self, pauli_terms: List[Tuple[float, str]], total_time: float, n_steps: int, order: int = 1
    ) -> Callable:
        """Noisy Trotter circuit on ``default.mixed``.

        Each of the ``n_steps`` Trotter layers (order 1 or 2, same gate
        decomposition as :class:`TrotterEvolution`) is followed, on every
        qubit, by T1/T2 relaxation for ``noise_model.gate_time`` and one
        depolarizing error - the same channel sequence as
        :meth:`NoiseChannels.full_noise_model`. Returns a function
        ``initial_state -> density matrix``.
        """
        from ..circuits.trotter_evolution import TrotterEvolution
        from .noise_channels import pennylane_depolarizing_parameter, relaxation_parameters

        if order not in (1, 2):
            raise ValueError(f"Trotter order {order} not implemented")
        dt = total_time / n_steps
        model = self.noise_model
        gamma_1, gamma_phi = relaxation_parameters(model.gate_time, model.t1, model.t2)
        q_depol = pennylane_depolarizing_parameter(model.depolarizing_rate)
        trotter = TrotterEvolution(self.n_qubits)
        layer = (
            trotter.first_order_trotter_layer if order == 1 else trotter.second_order_trotter_layer
        )
        wires = range(self.n_qubits)

        @qml.qnode(self.noisy_device)
        def noisy_evolution_circuit(initial_state):
            qml.StatePrep(initial_state, wires=wires)
            for _ in range(n_steps):
                layer(pauli_terms, dt)
                for q in wires:
                    if gamma_1 > 0:
                        qml.AmplitudeDamping(gamma_1, wires=q)
                    if gamma_phi > 0:
                        qml.PhaseDamping(gamma_phi, wires=q)
                    if q_depol > 0:
                        qml.DepolarizingChannel(q_depol, wires=q)
            return qml.density_matrix(wires=wires)

        return noisy_evolution_circuit

    def summary(self) -> Dict:
        return {
            "n_qubits": self.n_qubits,
            "depolarizing_rate": self.noise_model.depolarizing_rate,
            "T1": self.noise_model.t1,
            "T2": self.noise_model.t2,
            "estimated_max_depth": self.estimate_circuit_depth_limit(),
            "average_gate_fidelity": self.noise_channels.channel_fidelity(
                self.noise_channels.depolarizing_kraus(self.noise_model.depolarizing_rate)
            ),
        }
