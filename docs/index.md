# Quantum Chaos Framework

Exact-diagonalisation and small-circuit tools for quantum chaos and information
scrambling in the Sachdev–Ye–Kitaev (SYK) model and the Hubbard chain.

- **Spectral statistics** inside symmetry sectors, compared with random-matrix
  theory: SYK with its N mod 8 classes (GOE/GUE/GSE, Kramers pairs handled
  explicitly) and the Hubbard ring resolved by particle numbers, momentum,
  reflection, total spin and η-spin.
- **OTOCs** at infinite and finite temperature (also regularised), and as an
  interferometric Hadamard-test circuit.
- **Floquet** quasi-energies of the driven Hubbard chain.
- **Trotter circuits** (PennyLane) with a physical T1/T2 + depolarizing noise model.
- **Quantum kernels** with Hamiltonian encoding (tools only; no advantage is claimed).

Every result shown in the documentation is reproducible: figures are generated
from YAML configurations with explicit seeds (`make figures`), and the
[validation report](validation.md) lists what has been tested and what has not.

## Installation

```bash
git clone https://github.com/rasidi3112/quantum-chaos-framework.git
cd quantum-chaos-framework
pip install -e ".[test]"
pytest
```

## Quick start

```python
from quantum_chaos.hamiltonians import SYKHamiltonian, hubbard_level_statistics
from quantum_chaos.hamiltonians.syk_hamiltonian import disorder_average_lsr

syk = SYKHamiltonian(n_majorana=14, seed=1)
print(syk.symmetry_class)  # GUE (N mod 8 = 6)

stats = disorder_average_lsr(14, n_realizations=100, rng=0)
print(stats["mean"], stats["stderr"], stats["reference"])

res = hubbard_level_statistics(n_sites=8, n_up=4, n_down=4, tp=0.5, U=4.0)
print(res["mean"], res["stderr"])
```

![Overview](https://raw.githubusercontent.com/rasidi3112/quantum-chaos-framework/main/figures/overview.png)
