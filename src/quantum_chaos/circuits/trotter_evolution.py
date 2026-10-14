import warnings

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from typing import List, Tuple, Dict, Optional, Union, Callable
from scipy.linalg import expm


class TrotterEvolution:
    def __init__(
        self, n_qubits: int, device_name: str = "default.qubit", shots: Optional[int] = None
    ):
        self.n_qubits = n_qubits
        self.device_name = device_name
        self.shots = shots
        self._device = None

        # Pauli mapping for PennyLane
        self.pauli_map = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    @property
    def device(self) -> qml.devices.Device:
        """Analytic device; finite ``shots`` are applied per QNode with ``qml.set_shots``."""
        if self._device is None:
            self._device = qml.device(self.device_name, wires=self.n_qubits)
        return self._device

    def hamiltonian_to_pennylane(
        self, pauli_terms: List[Tuple[complex, str]]
    ) -> qml.ops.LinearCombination:
        coeffs = []
        observables = []

        for coeff, pauli_str in pauli_terms:
            if np.abs(coeff) < 1e-12:
                continue

            # Build tensor product of Paulis
            ops = []
            for i, p in enumerate(pauli_str):
                if p != "I":
                    ops.append(self.pauli_map[p](i))

            if len(ops) == 0:
                # All identity - contributes constant
                coeffs.append(np.real(coeff))
                observables.append(qml.Identity(0))
            elif len(ops) == 1:
                coeffs.append(np.real(coeff))
                observables.append(ops[0])
            else:
                coeffs.append(np.real(coeff))
                observables.append(qml.prod(*ops))

        return qml.Hamiltonian(coeffs, observables)

    def _pauli_rotation(self, pauli_string: str, angle: float):
        # Find non-identity qubits and their Pauli types
        active_qubits = []
        pauli_types = []

        for i, p in enumerate(pauli_string):
            if p != "I":
                active_qubits.append(i)
                pauli_types.append(p)

        if len(active_qubits) == 0:
            # All identity - global phase, skip
            return

        if len(active_qubits) == 1:
            # Single Pauli rotation
            q = active_qubits[0]
            p = pauli_types[0]
            if p == "X":
                qml.RX(2 * angle, wires=q)
            elif p == "Y":
                qml.RY(2 * angle, wires=q)
            elif p == "Z":
                qml.RZ(2 * angle, wires=q)
            return

        # Multi-qubit Pauli rotation using CNOT ladder
        # 1. Change basis for X and Y
        for q, p in zip(active_qubits, pauli_types):
            if p == "X":
                qml.Hadamard(wires=q)
            elif p == "Y":
                qml.RX(np.pi / 2, wires=q)

        # 2. CNOT ladder to compute parity
        for i in range(len(active_qubits) - 1):
            qml.CNOT(wires=[active_qubits[i], active_qubits[i + 1]])

        # 3. Z rotation on last qubit
        qml.RZ(2 * angle, wires=active_qubits[-1])

        # 4. Reverse CNOT ladder
        for i in range(len(active_qubits) - 2, -1, -1):
            qml.CNOT(wires=[active_qubits[i], active_qubits[i + 1]])

        # 5. Undo basis change
        for q, p in zip(active_qubits, pauli_types):
            if p == "X":
                qml.Hadamard(wires=q)
            elif p == "Y":
                qml.RX(-np.pi / 2, wires=q)

    def first_order_trotter_layer(self, pauli_terms: List[Tuple[float, str]], dt: float):
        for coeff, pauli_str in pauli_terms:
            angle = coeff * dt
            self._pauli_rotation(pauli_str, angle)

    def second_order_trotter_layer(self, pauli_terms: List[Tuple[float, str]], dt: float):
        # Forward sweep with dt/2
        for coeff, pauli_str in pauli_terms:
            angle = coeff * dt / 2
            self._pauli_rotation(pauli_str, angle)

        # Backward sweep with dt/2
        for coeff, pauli_str in reversed(pauli_terms):
            angle = coeff * dt / 2
            self._pauli_rotation(pauli_str, angle)

    def create_evolution_circuit(
        self,
        pauli_terms: List[Tuple[float, str]],
        total_time: float,
        n_steps: int,
        order: int = 1,
        initial_state: Optional[np.ndarray] = None,
    ) -> Callable:
        dt = total_time / n_steps

        @qml.qnode(self.device)
        def circuit():
            # Prepare initial state if provided
            if initial_state is not None:
                qml.StatePrep(initial_state, wires=range(self.n_qubits))

            # Apply Trotter layers
            for _ in range(n_steps):
                if order == 1:
                    self.first_order_trotter_layer(pauli_terms, dt)
                elif order == 2:
                    self.second_order_trotter_layer(pauli_terms, dt)
                else:
                    raise ValueError(f"Order {order} not implemented")

            return qml.state()

        return circuit

    def create_expectation_circuit(
        self,
        pauli_terms: List[Tuple[float, str]],
        observable: str,
        total_time: float,
        n_steps: int,
        order: int = 1,
        initial_state: Optional[np.ndarray] = None,
    ) -> Callable:
        dt = total_time / n_steps

        # Build observable
        obs_ops = []
        for i, p in enumerate(observable):
            if p != "I":
                obs_ops.append(self.pauli_map[p](i))

        if len(obs_ops) == 0:
            obs = qml.Identity(0)
        elif len(obs_ops) == 1:
            obs = obs_ops[0]
        else:
            obs = qml.prod(*obs_ops)

        @qml.qnode(self.device)
        def circuit():
            if initial_state is not None:
                qml.StatePrep(initial_state, wires=range(self.n_qubits))

            for _ in range(n_steps):
                if order == 1:
                    self.first_order_trotter_layer(pauli_terms, dt)
                else:
                    self.second_order_trotter_layer(pauli_terms, dt)

            return qml.expval(obs)

        if self.shots is not None:
            circuit = qml.set_shots(circuit, shots=self.shots)
        return circuit

    def evolve_state(
        self,
        initial_state: np.ndarray,
        pauli_terms: List[Tuple[float, str]],
        total_time: float,
        n_steps: int = 10,
        order: int = 1,
    ) -> np.ndarray:
        circuit = self.create_evolution_circuit(
            pauli_terms, total_time, n_steps, order, initial_state
        )
        return np.array(circuit())

    def compute_fidelity_vs_time(
        self,
        initial_state: np.ndarray,
        pauli_terms: List[Tuple[float, str]],
        times: np.ndarray,
        hamiltonian_matrix: np.ndarray,
        n_steps_per_unit_time: int = 10,
        order: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        fidelities = []

        for t in times:
            n_steps = max(1, int(n_steps_per_unit_time * t))

            # Trotter evolution
            trotter_state = self.evolve_state(initial_state, pauli_terms, t, n_steps, order)

            # Exact evolution
            U_exact = expm(-1j * hamiltonian_matrix * t)
            exact_state = U_exact @ initial_state

            # Fidelity
            fid = np.abs(np.vdot(trotter_state, exact_state)) ** 2
            fidelities.append(fid)

        return times, np.array(fidelities)

    def trotter_error_analysis(
        self,
        pauli_terms: List[Tuple[float, str]],
        hamiltonian_matrix: np.ndarray,
        total_time: float,
        max_steps: int = 100,
        rng: Union[None, int, np.random.Generator] = None,
    ) -> Dict:
        """Infidelity of order-1/2 Trotter evolution vs exact, from a random state."""
        rng = np.random.default_rng(rng)
        dim = 2**self.n_qubits
        initial_state = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
        initial_state /= np.linalg.norm(initial_state)

        # Exact evolution
        U_exact = expm(-1j * hamiltonian_matrix * total_time)
        exact_state = U_exact @ initial_state

        steps_list = [1, 2, 5, 10, 20, 50, 100]
        steps_list = [s for s in steps_list if s <= max_steps]

        errors_order1 = []
        errors_order2 = []

        for n_steps in steps_list:
            # First order
            state1 = self.evolve_state(initial_state, pauli_terms, total_time, n_steps, order=1)
            err1 = 1 - np.abs(np.vdot(state1, exact_state)) ** 2
            errors_order1.append(err1)

            # Second order
            state2 = self.evolve_state(initial_state, pauli_terms, total_time, n_steps, order=2)
            err2 = 1 - np.abs(np.vdot(state2, exact_state)) ** 2
            errors_order2.append(err2)

        return {
            "steps": steps_list,
            "error_order1": errors_order1,
            "error_order2": errors_order2,
            "total_time": total_time,
        }


def matrix_to_pauli_terms(
    H: np.ndarray, n_qubits: int, threshold: float = 1e-10
) -> List[Tuple[float, str]]:
    from ..utils.jordan_wigner import get_pauli_string_representation

    pauli_decomp = get_pauli_string_representation(H, n_qubits, threshold)

    # Filter to real coefficients (Hamiltonian is Hermitian)
    result = []
    for coeff, pauli_str in pauli_decomp:
        if np.abs(np.imag(coeff)) > 1e-10:
            warnings.warn(
                f"non-Hermitian term {pauli_str} with coefficient {coeff}; "
                "keeping only the real part",
                UserWarning,
                stacklevel=2,
            )
        result.append((np.real(coeff), pauli_str))

    return result
