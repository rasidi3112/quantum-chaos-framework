from itertools import combinations

import numpy as np

from ..hamiltonians.syk_hamiltonian import _pauli_product, majorana_pauli


def syk_pauli_terms(n_majorana: int, couplings: np.ndarray) -> list[tuple[float, str]]:
    n = n_majorana // 2
    chis = [majorana_pauli(a, n) for a in range(n_majorana)]
    terms = []
    for J, (i, j, k, l) in zip(couplings, combinations(range(n_majorana), 4), strict=True):
        if J == 0:
            continue
        x, z, phase = _pauli_product(
            _pauli_product(chis[i], chis[j]), _pauli_product(chis[k], chis[l])
        )
        label = []
        n_y = 0
        for q in range(n):
            bit = 1 << (n - 1 - q)
            xq, zq = bool(x & bit), bool(z & bit)
            label.append("Y" if xq and zq else "X" if xq else "Z" if zq else "I")
            n_y += xq and zq
        coeff = J * phase * (1j) ** (-n_y)
        terms.append((float(np.real(coeff)), "".join(label)))
    return terms


def pauli_rotation_cost(label: str) -> dict[str, int]:
    weight = sum(p != "I" for p in label)
    basis_changes = sum(p in "XY" for p in label)
    return {
        "cnot": 2 * max(weight - 1, 0),
        "single_qubit": 1 + 2 * basis_changes if weight else 0,
    }


def trotter_step_cost(terms: list[tuple[float, str]], order: int = 1) -> dict[str, int]:
    total = {"cnot": 0, "single_qubit": 0, "rotations": 0}
    for _, label in terms:
        cost = pauli_rotation_cost(label)
        if not any(p != "I" for p in label):
            continue
        total["cnot"] += cost["cnot"]
        total["single_qubit"] += cost["single_qubit"]
        total["rotations"] += 1
    if order == 2:
        total = {key: 2 * value for key, value in total.items()}
    return total
