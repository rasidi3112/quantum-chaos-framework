import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from functools import reduce


def pauli_operators() -> Dict[str, np.ndarray]:
    return {
        'I': np.array([[1, 0], [0, 1]], dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.array([[1, 0], [0, -1]], dtype=complex)
    }


def tensor_product(operators: List[np.ndarray]) -> np.ndarray:
    return reduce(np.kron, operators)


class JordanWignerMapper:
    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.paulis = pauli_operators()
        
    def _z_string(self, site: int) -> np.ndarray:
        operators = []
        for i in range(self.n_qubits):
            if i < site:
                operators.append(self.paulis['Z'])
            else:
                operators.append(self.paulis['I'])
        return tensor_product(operators)
    
    def creation_operator(self, site: int) -> np.ndarray:
        operators = []
        for i in range(self.n_qubits):
            if i < site:
                operators.append(self.paulis['Z'])
            elif i == site:
                operators.append(0.5 * (self.paulis['X'] - 1j * self.paulis['Y']))
            else:
                operators.append(self.paulis['I'])
        return tensor_product(operators)
    
    def annihilation_operator(self, site: int) -> np.ndarray:
        operators = []
        for i in range(self.n_qubits):
            if i < site:
                operators.append(self.paulis['Z'])
            elif i == site:
                operators.append(0.5 * (self.paulis['X'] + 1j * self.paulis['Y']))
            else:
                operators.append(self.paulis['I'])
        return tensor_product(operators)
    
    def majorana_operator(self, index: int) -> np.ndarray:
        qubit_site = index // 2
        is_even = (index % 2 == 0)
        
        operators = []
        for i in range(self.n_qubits):
            if i < qubit_site:
                operators.append(self.paulis['Z'])
            elif i == qubit_site:
                operators.append(self.paulis['X'] if is_even else self.paulis['Y'])
            else:
                operators.append(self.paulis['I'])
        return tensor_product(operators)
    
    def number_operator(self, site: int) -> np.ndarray:
        operators = []
        for i in range(self.n_qubits):
            if i == site:
                operators.append(0.5 * (self.paulis['I'] - self.paulis['Z']))
            else:
                operators.append(self.paulis['I'])
        return tensor_product(operators)
    
    def hopping_operator(self, site_i: int, site_j: int) -> np.ndarray:
        return (self.creation_operator(site_i) @ self.annihilation_operator(site_j) +
                self.creation_operator(site_j) @ self.annihilation_operator(site_i))
    
    def interaction_operator(self, site_i: int, site_j: int) -> np.ndarray:
        return self.number_operator(site_i) @ self.number_operator(site_j)


def jordan_wigner_transform(
    operator_type: str,
    indices: Union[int, Tuple[int, ...]],
    n_qubits: int
) -> np.ndarray:
    mapper = JordanWignerMapper(n_qubits)
    
    if operator_type == 'creation':
        return mapper.creation_operator(indices)
    elif operator_type == 'annihilation':
        return mapper.annihilation_operator(indices)
    elif operator_type == 'majorana':
        return mapper.majorana_operator(indices)
    elif operator_type == 'number':
        return mapper.number_operator(indices)
    elif operator_type == 'hopping':
        return mapper.hopping_operator(indices[0], indices[1])
    elif operator_type == 'interaction':
        return mapper.interaction_operator(indices[0], indices[1])
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")


def get_pauli_string_representation(
    operator: np.ndarray,
    n_qubits: int,
    threshold: float = 1e-10
) -> List[Tuple[complex, str]]:
    paulis = pauli_operators()
    pauli_labels = ['I', 'X', 'Y', 'Z']
    dim = 2 ** n_qubits
    
    result = []
    
    for idx in range(4 ** n_qubits):
        pauli_indices = []
        temp = idx
        for _ in range(n_qubits):
            pauli_indices.append(temp % 4)
            temp //= 4
        pauli_indices = pauli_indices[::-1]
        
        pauli_ops = [paulis[pauli_labels[i]] for i in pauli_indices]
        pauli_string = ''.join([pauli_labels[i] for i in pauli_indices])
        pauli_matrix = tensor_product(pauli_ops)
        
        coeff = np.trace(pauli_matrix.conj().T @ operator) / dim
        
        if np.abs(coeff) > threshold:
            result.append((coeff, pauli_string))
    
    return result
