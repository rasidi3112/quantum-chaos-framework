import numpy as np

from ..utils.helpers import spectral_density
from ..utils.jordan_wigner import JordanWignerMapper


class DrivenHubbardHamiltonian:
    """Periodically driven Hubbard chain in the full Fock space (Jordan-Wigner).

    H(t) = J0 cos(omega t) H_hop + U sum_j n_{j,up} n_{j,down}. For level
    statistics use :class:`quantum_chaos.hamiltonians.hubbard_ed.HubbardRingED`,
    which resolves the symmetry sectors; pooling the whole Fock space is not
    meaningful.
    """

    def __init__(
        self,
        n_sites: int,
        hopping_amplitude: float = 1.0,
        driving_frequency: float = 1.0,
        onsite_interaction: float = 1.0,
        periodic_boundary: bool = False,
    ):
        if n_sites < 2:
            raise ValueError("Need at least 2 lattice sites")

        self.n_sites = n_sites
        self.n_qubits = 2 * n_sites
        self.hopping_amplitude = hopping_amplitude
        self.driving_frequency = driving_frequency
        self.onsite_interaction = onsite_interaction
        self.periodic_boundary = periodic_boundary

        self._jw_mapper = JordanWignerMapper(self.n_qubits)
        self._hopping_matrix = None
        self._interaction_matrix = None
        self._static_eigenvalues = None
        self._static_eigenvectors = None

    def hopping_strength(self, t: float) -> float:
        return self.hopping_amplitude * np.cos(self.driving_frequency * t)

    @property
    def hopping_matrix(self) -> np.ndarray:
        if self._hopping_matrix is None:
            self._hopping_matrix = self._build_hopping_hamiltonian()
        return self._hopping_matrix

    @property
    def interaction_matrix(self) -> np.ndarray:
        if self._interaction_matrix is None:
            self._interaction_matrix = self._build_interaction_hamiltonian()
        return self._interaction_matrix

    def _site_spin_to_qubit(self, site: int, spin: str) -> int:
        if spin == "up":
            return 2 * site
        elif spin == "down":
            return 2 * site + 1
        else:
            raise ValueError(f"Invalid spin: {spin}")

    def _build_hopping_hamiltonian(self) -> np.ndarray:
        dim = 2**self.n_qubits
        H_hop = np.zeros((dim, dim), dtype=complex)

        for i in range(self.n_sites - 1):
            for spin in ["up", "down"]:
                q_i = self._site_spin_to_qubit(i, spin)
                q_j = self._site_spin_to_qubit(i + 1, spin)
                hop_term = self._jw_mapper.creation_operator(
                    q_i
                ) @ self._jw_mapper.annihilation_operator(q_j) + self._jw_mapper.creation_operator(
                    q_j
                ) @ self._jw_mapper.annihilation_operator(q_i)
                H_hop -= hop_term

        if self.periodic_boundary and self.n_sites > 2:
            for spin in ["up", "down"]:
                q_0 = self._site_spin_to_qubit(0, spin)
                q_L = self._site_spin_to_qubit(self.n_sites - 1, spin)
                hop_term = self._jw_mapper.creation_operator(
                    q_0
                ) @ self._jw_mapper.annihilation_operator(q_L) + self._jw_mapper.creation_operator(
                    q_L
                ) @ self._jw_mapper.annihilation_operator(q_0)
                H_hop -= hop_term

        return H_hop

    def _build_interaction_hamiltonian(self) -> np.ndarray:
        dim = 2**self.n_qubits
        H_int = np.zeros((dim, dim), dtype=complex)

        for i in range(self.n_sites):
            q_up = self._site_spin_to_qubit(i, "up")
            q_down = self._site_spin_to_qubit(i, "down")
            n_up = self._jw_mapper.number_operator(q_up)
            n_down = self._jw_mapper.number_operator(q_down)
            H_int += self.onsite_interaction * (n_up @ n_down)

        return H_int

    def hamiltonian(self, t: float) -> np.ndarray:
        J_t = self.hopping_strength(t)
        return J_t * self.hopping_matrix + self.interaction_matrix

    def static_hamiltonian(self) -> np.ndarray:
        return self.hamiltonian(t=0)

    def _diagonalize_static(self):
        H_static = self.static_hamiltonian()
        self._static_eigenvalues, self._static_eigenvectors = np.linalg.eigh(H_static)

    @property
    def static_eigenvalues(self) -> np.ndarray:
        if self._static_eigenvalues is None:
            self._diagonalize_static()
        return self._static_eigenvalues

    @property
    def static_eigenvectors(self) -> np.ndarray:
        if self._static_eigenvectors is None:
            self._diagonalize_static()
        return self._static_eigenvectors

    def floquet_operator(self, n_steps: int = 100) -> np.ndarray:
        """One-period propagator U(T, 0), T = 2*pi/omega.

        Product of midpoint exponentials (second-order accurate in T/n_steps).
        """
        from scipy.linalg import expm

        T = 2 * np.pi / self.driving_frequency
        dt = T / n_steps
        dim = 2**self.n_qubits
        F = np.eye(dim, dtype=complex)
        for step in range(n_steps):
            t_mid = (step + 0.5) * dt
            F = expm(-1j * self.hamiltonian(t_mid) * dt) @ F
        return F

    def quasi_energies(self, n_steps: int = 100) -> np.ndarray:
        """Sorted quasi-energies in the zone [-omega/2, omega/2).

        With F = exp(-i H_F T), the eigenvalues are exp(-i eps T), hence
        eps = -angle(lambda) / T.
        """
        F = self.floquet_operator(n_steps)
        T = 2 * np.pi / self.driving_frequency
        return np.sort(-np.angle(np.linalg.eigvals(F)) / T)

    def time_evolution_operator(self, t: float, n_steps: int = 50) -> np.ndarray:
        from scipy.linalg import expm

        dt = t / n_steps
        dim = 2**self.n_qubits
        U = np.eye(dim, dtype=complex)
        for step in range(n_steps):
            t_mid = (step + 0.5) * dt
            H_mid = self.hamiltonian(t_mid)
            U = expm(-1j * H_mid * dt) @ U
        return U

    def evolve_state(self, initial_state: np.ndarray, t: float, n_steps: int = 50) -> np.ndarray:
        U = self.time_evolution_operator(t, n_steps)
        return U @ initial_state

    def stroboscopic_evolution(
        self, initial_state: np.ndarray, n_periods: int, n_steps_per_period: int = 100
    ) -> list[np.ndarray]:
        F = self.floquet_operator(n_steps_per_period)
        states = [initial_state.copy()]
        current_state = initial_state.copy()
        for _ in range(n_periods):
            current_state = F @ current_state
            states.append(current_state.copy())
        return states

    def get_spectral_density(self, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
        return spectral_density(self.static_eigenvalues, bins=bins)

    def get_number_operator(self, site: int, spin: str) -> np.ndarray:
        q = self._site_spin_to_qubit(site, spin)
        return self._jw_mapper.number_operator(q)

    def get_total_number_operator(self) -> np.ndarray:
        dim = 2**self.n_qubits
        N_total = np.zeros((dim, dim), dtype=complex)
        for i in range(self.n_sites):
            for spin in ["up", "down"]:
                N_total += self.get_number_operator(i, spin)
        return N_total

    def get_total_spin_z(self) -> np.ndarray:
        dim = 2**self.n_qubits
        S_z = np.zeros((dim, dim), dtype=complex)
        for i in range(self.n_sites):
            S_z += 0.5 * (self.get_number_operator(i, "up") - self.get_number_operator(i, "down"))
        return S_z

    def effective_hamiltonian(self, order: int = 0) -> np.ndarray:
        """High-frequency effective Hamiltonian truncated at O(omega^-order).

        Only the leading term is implemented: the period average
        (1/T) int_0^T H(t) dt. For J(t) = J0 cos(omega t) the hopping averages
        to zero, so this equals the interaction term. Higher orders (van Vleck /
        Floquet-Magnus) are planned (PLAN 3C) and raise NotImplementedError.
        """
        if order != 0:
            raise NotImplementedError("only the order-0 (period-averaged) term is implemented")
        T = 2 * np.pi / self.driving_frequency
        n = 64  # midpoint rule, exact for trigonometric polynomials of degree < 64
        mean_hopping = np.mean([self.hopping_strength((k + 0.5) * T / n) for k in range(n)])
        return mean_hopping * self.hopping_matrix + self.interaction_matrix

    def summary(self) -> dict:
        return {
            "n_sites": self.n_sites,
            "n_qubits": self.n_qubits,
            "hilbert_dim": 2**self.n_qubits,
            "hopping_amplitude": self.hopping_amplitude,
            "driving_frequency": self.driving_frequency,
            "onsite_interaction": self.onsite_interaction,
            "periodic_boundary": self.periodic_boundary,
            "driving_period": 2 * np.pi / self.driving_frequency,
            "ground_state_energy": float(self.static_eigenvalues[0]),
        }

    def __repr__(self) -> str:
        return (
            f"DrivenHubbardHamiltonian(L={self.n_sites}, J₀={self.hopping_amplitude}, "
            f"ω={self.driving_frequency}, U={self.onsite_interaction})"
        )
