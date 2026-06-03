import numpy as np
from scipy.sparse import csr_matrix, kron as sparse_kron, eye as sparse_eye
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from typing import Optional, Tuple, Dict, List
from itertools import combinations

try:
    from ..utils.jordan_wigner import JordanWignerMapper
    from ..utils.helpers import generate_random_coupling, spectral_density, level_spacing_ratio
except ImportError:
    from utils.jordan_wigner import JordanWignerMapper
    from utils.helpers import generate_random_coupling, spectral_density, level_spacing_ratio


class SYKHamiltonian:
    """
    SYK (Sachdev-Ye-Kitaev) Model Implementation
    
    Supports both dense and sparse matrix representations for scalability.
    Implements disorder averaging and finite-size scaling analysis.
    
    Reference: Maldacena & Stanford, JHEP 2016
    """
    
    def __init__(
        self,
        n_majorana: int,
        coupling_strength: float = 1.0,
        seed: Optional[int] = None,
        use_sparse: bool = False
    ):
        if n_majorana < 4:
            raise ValueError("SYK model requires at least 4 Majorana fermions")
        if n_majorana % 2 != 0:
            raise ValueError("Number of Majorana fermions must be even")
        
        self.n_majorana = n_majorana
        self.n_qubits = n_majorana // 2
        self.coupling_strength = coupling_strength
        self.seed = seed
        self.use_sparse = use_sparse
        
        self._jw_mapper = JordanWignerMapper(self.n_qubits)
        self._couplings = None
        self._hamiltonian_matrix = None
        self._eigenvalues = None
        self._eigenvectors = None
        
    @property
    def couplings(self) -> np.ndarray:
        if self._couplings is None:
            self._couplings = generate_random_coupling(
                self.n_majorana,
                variance=self.coupling_strength ** 2,
                seed=self.seed,
                antisymmetric=True
            )
        return self._couplings
    
    @property
    def hamiltonian_matrix(self) -> np.ndarray:
        if self._hamiltonian_matrix is None:
            self._hamiltonian_matrix = self._build_hamiltonian()
        return self._hamiltonian_matrix
    
    @property
    def eigenvalues(self) -> np.ndarray:
        if self._eigenvalues is None:
            self._diagonalize()
        return self._eigenvalues
    
    @property
    def eigenvectors(self) -> np.ndarray:
        if self._eigenvectors is None:
            self._diagonalize()
        return self._eigenvectors
    
    def _build_hamiltonian(self) -> np.ndarray:
        dim = 2 ** self.n_qubits
        H = np.zeros((dim, dim), dtype=complex)
        
        majorana_ops = [
            self._jw_mapper.majorana_operator(i) 
            for i in range(self.n_majorana)
        ]
        
        for i, j, k, l in combinations(range(self.n_majorana), 4):
            J_ijkl = self.couplings[i, j, k, l]
            if np.abs(J_ijkl) > 1e-15:
                term = majorana_ops[i] @ majorana_ops[j] @ majorana_ops[k] @ majorana_ops[l]
                H += J_ijkl * term
        
        H = (H + H.conj().T) / 2
        
        if self.use_sparse:
            return csr_matrix(H)
        return H
    
    def _diagonalize(self, n_eigenvalues: Optional[int] = None):
        if self.use_sparse and n_eigenvalues is not None:
            self._eigenvalues, self._eigenvectors = eigsh(
                self.hamiltonian_matrix, k=n_eigenvalues, which='SA'
            )
        else:
            H = self.hamiltonian_matrix
            if hasattr(H, 'toarray'):
                H = H.toarray()
            self._eigenvalues, self._eigenvectors = eigh(H)
    
    def get_ground_state(self) -> Tuple[float, np.ndarray]:
        return self.eigenvalues[0], self.eigenvectors[:, 0]
    
    def get_spectral_density(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        return spectral_density(self.eigenvalues, bins=bins)
    
    def get_level_spacing_ratio(self) -> float:
        return level_spacing_ratio(self.eigenvalues)
    
    def get_thermal_state(self, temperature: float) -> np.ndarray:
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        beta = 1.0 / temperature
        boltzmann_factors = np.exp(-beta * self.eigenvalues)
        Z = np.sum(boltzmann_factors)
        thermal_diag = np.diag(boltzmann_factors / Z)
        rho = self.eigenvectors @ thermal_diag @ self.eigenvectors.conj().T
        return rho
    
    def time_evolution_operator(self, t: float) -> np.ndarray:
        phases = np.exp(-1j * self.eigenvalues * t)
        U = self.eigenvectors @ np.diag(phases) @ self.eigenvectors.conj().T
        return U
    
    def evolve_state(self, initial_state: np.ndarray, t: float) -> np.ndarray:
        U = self.time_evolution_operator(t)
        return U @ initial_state
    
    def get_majorana_operator(self, index: int) -> np.ndarray:
        return self._jw_mapper.majorana_operator(index)
    
    def parity_projection(self, sector: str = 'even') -> np.ndarray:
        """Project Hamiltonian to fermion parity sector."""
        dim = 2 ** self.n_qubits
        
        # Build parity operator P = Z⊗Z⊗...⊗Z
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)
        P = np.eye(dim, dtype=complex)
        
        for q in range(self.n_qubits):
            Z_full = np.eye(1)
            for i in range(self.n_qubits):
                Z_full = np.kron(Z_full, Z if i == q else I)
            P = P @ Z_full
        
        proj = (np.eye(dim) + P) / 2 if sector == 'even' else (np.eye(dim) - P) / 2
        indices = np.where(np.abs(np.diag(proj)) > 0.5)[0]
        
        H = self.hamiltonian_matrix
        if hasattr(H, 'toarray'):
            H = H.toarray()
        H_reduced = H[np.ix_(indices, indices)]
        
        return H_reduced
    
    def get_level_spacing_ratio_parity(self, sector: str = 'even') -> float:
        """Get level spacing ratio in specific parity sector."""
        H_sector = self.parity_projection(sector)
        eigenvalues = np.linalg.eigvalsh(H_sector)
        return level_spacing_ratio(eigenvalues)
    
    def summary(self) -> Dict:
        return {
            'n_majorana': self.n_majorana,
            'n_qubits': self.n_qubits,
            'hilbert_dim': 2 ** self.n_qubits,
            'coupling_strength': self.coupling_strength,
            'ground_state_energy': float(self.eigenvalues[0]),
            'spectral_width': float(self.eigenvalues[-1] - self.eigenvalues[0]),
            'level_spacing_ratio': self.get_level_spacing_ratio(),
            'level_spacing_ratio_even': self.get_level_spacing_ratio_parity('even'),
            'chaos_indicator': 'GOE-like' if self.get_level_spacing_ratio_parity('even') > 0.45 else 'Poisson-like'
        }
    
    def __repr__(self) -> str:
        return f"SYKHamiltonian(N={self.n_majorana}, n_qubits={self.n_qubits}, J={self.coupling_strength})"


def disorder_average_lsr(n_majorana: int, n_realizations: int = 20, seed_start: int = 0) -> Dict:
    """
    Compute disorder-averaged level spacing ratio.
    
    Returns mean, std, and stderr for statistical analysis.
    """
    lsr_samples = []
    
    for seed in range(seed_start, seed_start + n_realizations):
        syk = SYKHamiltonian(n_majorana=n_majorana, seed=seed)
        lsr = syk.get_level_spacing_ratio_parity('even')
        if not np.isnan(lsr) and lsr > 0:
            lsr_samples.append(lsr)
    
    if len(lsr_samples) == 0:
        return {'mean': np.nan, 'std': np.nan, 'stderr': np.nan, 'n_samples': 0}
    
    return {
        'mean': np.mean(lsr_samples),
        'std': np.std(lsr_samples),
        'stderr': np.std(lsr_samples) / np.sqrt(len(lsr_samples)),
        'n_samples': len(lsr_samples)
    }


def finite_size_scaling(N_values: List[int], n_realizations: int = 20) -> Dict:
    """
    Perform finite-size scaling analysis of level spacing ratio.
    
    Extrapolates to thermodynamic limit using 1/N scaling.
    """
    results = {}
    N_arr = []
    lsr_arr = []
    err_arr = []
    
    for N in N_values:
        stats = disorder_average_lsr(N, n_realizations)
        if not np.isnan(stats['mean']):
            results[N] = stats
            N_arr.append(N)
            lsr_arr.append(stats['mean'])
            err_arr.append(stats['stderr'])
    
    # Fit 1/N extrapolation
    if len(N_arr) >= 2:
        try:
            from scipy.optimize import curve_fit
            def model(N, r_inf, a):
                return r_inf + a / N
            
            popt, pcov = curve_fit(model, N_arr, lsr_arr, sigma=err_arr, p0=[0.53, 0.1])
            results['extrapolation'] = {
                'r_infinity': popt[0],
                'r_infinity_err': np.sqrt(pcov[0, 0]),
                'goe_deviation': abs(popt[0] - 0.530) / 0.530 * 100
            }
        except (RuntimeError, ValueError):
            pass
    
    return results
