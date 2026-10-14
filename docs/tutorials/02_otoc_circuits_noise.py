# %% [markdown]
# # OTOCs, circuits and noise
#
# 1. The infinite-temperature OTOC of an SYK realisation, computed exactly and
#    with the interferometric (Hadamard-test) circuit.
# 2. The T1/T2 noise model: coherence decays as $e^{-t/T_2}$, independent of
#    the number of simulation steps.
# 3. A noisy Trotter circuit on PennyLane's mixed-state simulator, checked
#    against the NumPy noise model.

# %%
import matplotlib.pyplot as plt
import numpy as np

from quantum_chaos.circuits.otoc_calculator import OTOCCalculator
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.noise.nisq_simulator import NISQSimulator
from quantum_chaos.noise.noise_channels import NoiseModel

# %% [markdown]
# ## 1. OTOC: exact vs circuit

# %%
syk = SYKHamiltonian(8, seed=3)
calc = OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)
times = np.linspace(0, 4, 21)
W, V = calc.local_pauli("Z", 0), calc.local_pauli("X", 2)
exact = np.real(calc.compute_otoc_exact(W, V, times))
circuit = calc.create_otoc_circuit("Z", 0, "X", 2)  # T = inf via Bell-pair purification
measured = np.array([circuit(t) for t in times[::4]])
print("max |circuit - exact| =", np.max(np.abs(measured - exact[::4])))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(times, exact, label="exact")
ax.plot(times[::4], measured, "o", label="Hadamard-test circuit")
ax.set_xlabel("t")
ax.set_ylabel(r"Re $F(t)$")
ax.legend()
plt.show()

# %% [markdown]
# ## 2. T1/T2 relaxation

# %%
model = NoiseModel(depolarizing_rate=0.0, t1=40.0, t2=30.0)
sim = NISQSimulator(1, noise_model=model)
plus = np.full((2, 2), 0.5, dtype=complex)
for n_steps in (1, 5, 50):
    rho = sim.evolve_with_noise(plus, np.zeros((2, 2)), total_time=6.0, n_steps=n_steps)
    print(f"n_steps={n_steps:3d}: |rho01| = {abs(rho[0, 1]):.6f}")
print(f"expected 0.5 exp(-t/T2) = {0.5 * np.exp(-6.0 / 30.0):.6f}")

# %% [markdown]
# ## 3. Noisy Trotter circuit

# %%
terms = [(0.6, "XX"), (0.4, "ZI"), (-0.3, "YZ")]
model = NoiseModel(depolarizing_rate=0.01, t1=50.0, t2=40.0, gate_time=0.5)
sim = NISQSimulator(2, noise_model=model)
psi0 = np.array([1, 0, 0, 0], dtype=complex)
for n_steps in (2, 8):
    rho = sim.create_pennylane_noisy_circuit(terms, total_time=1.0, n_steps=n_steps)(psi0)
    purity = np.real(np.trace(rho @ rho))
    print(f"{n_steps} Trotter layers: purity = {purity:.4f}")
