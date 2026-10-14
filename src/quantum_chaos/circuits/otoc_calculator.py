import numpy as np
import pennylane as qml
from typing import Tuple, List, Optional, Dict, Union
from scipy.linalg import expm


class OTOCCalculator:
    def __init__(self, n_qubits: int, hamiltonian: Optional[np.ndarray] = None):
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
        temperature: Optional[float] = None,
    ) -> np.ndarray:
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not set")

        dim = 2**self.n_qubits
        otoc_values = []

        # Prepare density matrix
        if state is not None:
            rho = np.outer(state, state.conj())
        elif temperature is not None:
            self._diagonalize()
            beta = 1.0 / temperature
            boltzmann = np.exp(-beta * (self._eigenvalues - np.min(self._eigenvalues)))
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
        self, W: np.ndarray, V: np.ndarray, times: np.ndarray, temperature: float
    ) -> np.ndarray:
        self._diagonalize()
        beta = 1.0 / temperature

        # y = rho^(1/4) with rho = exp(-beta H)/Z; energies shifted by E_0 (no overflow)
        shifted = self._eigenvalues - np.min(self._eigenvalues)
        boltzmann_quarter = np.exp(-beta * shifted / 4)
        rho_quarter = self._eigenvectors @ np.diag(boltzmann_quarter) @ self._eigenvectors.conj().T

        Z = np.sum(np.exp(-beta * shifted))
        rho_quarter /= Z**0.25

        otoc_values = []

        for t in times:
            W_t = self._heisenberg_evolution(W, t)

            # Regularized correlator
            product = rho_quarter @ W_t @ rho_quarter @ V @ rho_quarter @ W_t @ rho_quarter @ V
            C_t = np.trace(product)
            otoc_values.append(C_t)

        return np.array(otoc_values)

    def compute_commutator_squared(
        self, W: np.ndarray, V: np.ndarray, times: np.ndarray, state: Optional[np.ndarray] = None
    ) -> np.ndarray:
        otoc = self.compute_otoc_exact(W, V, times, state=state)
        return 2 * (1 - np.real(otoc))

    def local_pauli(self, pauli: str, site: int) -> np.ndarray:
        I = np.eye(2)
        paulis = {
            "X": np.array([[0, 1], [1, 0]], dtype=complex),
            "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
            "Z": np.array([[1, 0], [0, -1]], dtype=complex),
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

    def operator_size_growth(self, initial_op: np.ndarray, times: np.ndarray) -> np.ndarray:
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
                weight = sum(1 for p in pauli_str if p != "I")
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
        V_site: int,
        initial_state: Optional[np.ndarray] = None,
    ):
        """Interferometric (Hadamard-test) circuit for Re F(t), F(t) = Tr[rho W(t)^dag V^dag W(t) V].

        Uses controlled-(U^dag X U) = U^dag (controlled-X) U, so the time
        evolution U = exp(-iHt) itself is never controlled (Swingle et al.,
        PRA 94, 040302 (2016)). Here U is applied exactly as a matrix
        (``qml.QubitUnitary``); a Trotterised version is planned (PLAN 3D).

        ``initial_state=None`` gives infinite temperature, rho = I/D, realised
        by purification with n reference qubits in Bell pairs; otherwise
        rho = |psi><psi|. Returns a function ``t -> Re F(t)`` equal to
        ``Re compute_otoc_exact(W, V, [t], state=initial_state)``.
        """
        if self.hamiltonian is None:
            raise ValueError("Hamiltonian not set")
        n = self.n_qubits
        system = list(range(n))
        ancilla = n
        reference = list(range(n + 1, 2 * n + 1)) if initial_state is None else []
        dev = qml.device("default.qubit", wires=n + 1 + len(reference))
        gate = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

        @qml.qnode(dev)
        def hadamard_test(U):
            if initial_state is None:
                for r, q in zip(reference, system):
                    qml.Hadamard(wires=r)
                    qml.CNOT(wires=[r, q])
            else:
                qml.StatePrep(initial_state, wires=system)
            qml.Hadamard(wires=ancilla)
            # controlled-A, A = U^dag W U V^dag U^dag W U V (Paulis are Hermitian),
            # applied right to left
            qml.ctrl(gate[V_pauli](wires=V_site), control=ancilla)
            qml.QubitUnitary(U, wires=system)
            qml.ctrl(gate[W_pauli](wires=W_site), control=ancilla)
            qml.adjoint(qml.QubitUnitary(U, wires=system))
            qml.ctrl(gate[V_pauli](wires=V_site), control=ancilla)
            qml.QubitUnitary(U, wires=system)
            qml.ctrl(gate[W_pauli](wires=W_site), control=ancilla)
            qml.adjoint(qml.QubitUnitary(U, wires=system))
            qml.Hadamard(wires=ancilla)
            return qml.expval(qml.PauliZ(ancilla))

        def otoc(t: float) -> float:
            return float(hadamard_test(self._time_evolution_operator(t)))

        return otoc
