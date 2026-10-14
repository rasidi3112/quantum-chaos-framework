import numpy as np
import pennylane as qml
from typing import List, Tuple, Dict, Optional, Callable, Union, Sequence
from scipy.linalg import expm


def _basis_state(n_qubits: int, index: int = 0) -> np.ndarray:
    psi = np.zeros(2 ** n_qubits, dtype=complex)
    psi[index] = 1.0
    return psi


def _preparation_unitary(psi: np.ndarray) -> np.ndarray:
    """Unitary V with V|0...0> = psi (up to a global phase), via a Householder reflection."""
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)
    phase = np.exp(1j * np.angle(psi[0])) if abs(psi[0]) > 1e-15 else 1.0
    v = psi / phase  # v[0] is real and >= 0
    e0 = _basis_state(int(np.log2(len(psi))))
    w = e0 - v
    norm = np.linalg.norm(w)
    if norm < 1e-14:
        return np.eye(len(psi), dtype=complex)
    w = w / norm
    return np.eye(len(psi), dtype=complex) - 2.0 * np.outer(w, w.conj())


class QuantumKernel:
    """Fidelity kernel K(x, x') = |<psi(x)|psi(x')>|^2 with |psi(x)> = U(x)|psi_0>.

    U(x) = exp(-i H(x) t) for a user-supplied ``hamiltonian_constructor``.
    The reference state ``initial_state`` (default |0...0>) must not be an
    eigenstate of every H(x), otherwise the kernel is identically 1.
    """

    def __init__(
        self,
        n_qubits: int,
        hamiltonian_constructor: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        evolution_time: float = 1.0,
        initial_state: Optional[np.ndarray] = None
    ):
        self.n_qubits = n_qubits
        self.hamiltonian_constructor = hamiltonian_constructor
        self.evolution_time = evolution_time
        if initial_state is None:
            initial_state = _basis_state(n_qubits)
        self.initial_state = np.asarray(initial_state, dtype=complex)
        
        self._device = None
    
    @property
    def device(self) -> qml.devices.Device:
        if self._device is None:
            self._device = qml.device('default.qubit', wires=self.n_qubits)
        return self._device
    
    def feature_map_matrix(self, x: np.ndarray) -> np.ndarray:
        H = self.hamiltonian_constructor(np.asarray(x, dtype=float))
        return expm(-1j * H * self.evolution_time)
    
    def feature_state(self, x: np.ndarray) -> np.ndarray:
        return self.feature_map_matrix(x) @ self.initial_state
    
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
        symmetric = X2 is None
        states1 = [self.feature_state(x) for x in X1]
        states2 = states1 if symmetric else [self.feature_state(x) for x in X2]
        A = np.array(states1)
        B = np.array(states2)
        K = np.abs(A.conj() @ B.T) ** 2
        if symmetric:
            K = (K + K.T) / 2
            np.fill_diagonal(K, 1.0)
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
        """Circuit estimate of K(x1, x2) = |<psi_0| U(x2)^dagger U(x1) |psi_0>|^2.

        Prepares |psi_0> with a unitary V (V|0> = psi_0), applies U(x1), then
        U(x2)^dagger and V^dagger, and returns the probability of |0...0>.
        """
        wires = list(range(self.n_qubits))
        V = _preparation_unitary(self.initial_state)
        
        @qml.qnode(self.device)
        def kernel_circuit(U1, U2):
            qml.QubitUnitary(V, wires=wires)
            qml.QubitUnitary(U1, wires=wires)
            qml.adjoint(qml.QubitUnitary(U2, wires=wires))
            qml.adjoint(qml.QubitUnitary(V, wires=wires))
            return qml.probs(wires=wires)
        
        def kernel_fn(x1, x2):
            probs = kernel_circuit(self.feature_map_matrix(x1), self.feature_map_matrix(x2))
            return float(probs[0])
        
        return kernel_fn


class HamiltonianEncodingKernel(QuantumKernel):
    """Hamiltonian ("local-field") encoding of classical data.

    U(x) = prod_l exp(-i t [H_model + alpha * sum_j x_{l,j} O_j]), where the
    features are split into consecutive blocks of ``len(local_operators)``
    (zero-padded); each block is one layer (data re-uploading). Every feature
    therefore enters the feature map.
    """

    def __init__(
        self,
        model_hamiltonian: np.ndarray,
        local_operators: Sequence[np.ndarray],
        evolution_time: float = 1.0,
        initial_state: Optional[np.ndarray] = None,
        feature_scale: float = 1.0
    ):
        n_qubits = int(np.log2(model_hamiltonian.shape[0]))
        super().__init__(n_qubits=n_qubits, evolution_time=evolution_time,
                         initial_state=initial_state)
        self.model_hamiltonian = np.asarray(model_hamiltonian, dtype=complex)
        self.local_operators = [np.asarray(O, dtype=complex) for O in local_operators]
        self.feature_scale = feature_scale
    
    def layer_hamiltonian(self, x_block: np.ndarray) -> np.ndarray:
        H = self.model_hamiltonian.copy()
        for xj, O in zip(x_block, self.local_operators):
            H = H + self.feature_scale * xj * O
        return H
    
    def _blocks(self, x: np.ndarray) -> List[np.ndarray]:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        m = len(self.local_operators)
        n_layers = max(1, int(np.ceil(len(x) / m)))
        padded = np.zeros(n_layers * m)
        padded[:len(x)] = x
        return list(padded.reshape(n_layers, m))
    
    def feature_map_matrix(self, x: np.ndarray) -> np.ndarray:
        U = np.eye(2 ** self.n_qubits, dtype=complex)
        for block in self._blocks(x):
            U = expm(-1j * self.layer_hamiltonian(block) * self.evolution_time) @ U
        return U


class SYKKernel(HamiltonianEncodingKernel):
    """SYK dynamics with local fields: O_j = Z_j = -i chi_{2j} chi_{2j+1}.

    Reference state |0...0> (not an eigenstate of H_SYK for N >= 6). For N = 4
    the SYK Hamiltonian is a single term -J Z_0 Z_1, which commutes with every
    Z_j, so the kernel would be identically 1; n_qubits >= 3 is required.
    """

    def __init__(
        self,
        n_qubits: int,
        coupling_strength: float = 1.0,
        evolution_time: float = 1.0,
        seed: Optional[int] = None,
        feature_scale: float = 1.0
    ):
        from ..hamiltonians.syk_hamiltonian import SYKHamiltonian
        
        if n_qubits < 3:
            raise ValueError("SYKKernel needs n_qubits >= 3 (N=4 SYK commutes with the encoding)")
        self.coupling_strength = coupling_strength
        self.seed = seed
        self._syk = SYKHamiltonian(
            n_majorana=2 * n_qubits,
            coupling_strength=coupling_strength,
            seed=seed
        )
        majorana = self._syk.get_majorana_operator
        local_z = [-1j * majorana(2 * j) @ majorana(2 * j + 1) for j in range(n_qubits)]
        super().__init__(
            model_hamiltonian=self._syk.hamiltonian_matrix,
            local_operators=local_z,
            evolution_time=evolution_time,
            feature_scale=feature_scale
        )


class HubbardKernel(HamiltonianEncodingKernel):
    """Static Hubbard dynamics with site potentials: O_j = n_{j,up} + n_{j,down}.

    Reference state: half-filled Neel Fock state (up on even sites, down on odd
    sites). Because the particle number is conserved, a uniform shift of all
    features within one layer only adds a global phase (K is invariant).
    """

    def __init__(
        self,
        n_sites: int,
        hopping_amplitude: float = 1.0,
        onsite_interaction: float = 1.0,
        evolution_time: float = 1.0,
        feature_scale: float = 1.0
    ):
        from ..hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian
        
        self.n_sites = n_sites
        self.hopping_amplitude = hopping_amplitude
        self.onsite_interaction = onsite_interaction
        hubbard = DrivenHubbardHamiltonian(
            n_sites=n_sites,
            hopping_amplitude=hopping_amplitude,
            onsite_interaction=onsite_interaction
        )
        densities = [
            hubbard.get_number_operator(j, 'up') + hubbard.get_number_operator(j, 'down')
            for j in range(n_sites)
        ]
        super().__init__(
            model_hamiltonian=hubbard.static_hamiltonian(),
            local_operators=densities,
            evolution_time=evolution_time,
            initial_state=neel_state(hubbard),
            feature_scale=feature_scale
        )


def neel_state(hubbard) -> np.ndarray:
    """Half-filled Neel Fock state of a ``DrivenHubbardHamiltonian`` (qubit 0 most significant)."""
    n_qubits = hubbard.n_qubits
    occupied = [hubbard._site_spin_to_qubit(j, 'up' if j % 2 == 0 else 'down')
                for j in range(hubbard.n_sites)]
    index = sum(1 << (n_qubits - 1 - q) for q in occupied)
    return _basis_state(n_qubits, index)


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
