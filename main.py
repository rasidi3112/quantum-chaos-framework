#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm

from hamiltonians.syk_hamiltonian import SYKHamiltonian, disorder_average_lsr, finite_size_scaling
from hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian

print("=" * 60)
print("  QUANTUM CHAOS FRAMEWORK")
print("  SYK Model Analysis with Statistical Methods")
print("=" * 60)

# 1. Single SYK instance
print("\n[1] SYK Model (N=8 Majorana)")
print("-" * 40)
syk = SYKHamiltonian(n_majorana=8, coupling_strength=1.0, seed=42)
summary = syk.summary()
print(f"    Hilbert dimension: {summary['hilbert_dim']}")
print(f"    Ground state energy: {summary['ground_state_energy']:.4f}")
print(f"    Level spacing ⟨r⟩ (full): {summary['level_spacing_ratio']:.4f}")
print(f"    Level spacing ⟨r⟩ (even parity): {summary['level_spacing_ratio_even']:.4f}")
print(f"    Chaos indicator: {summary['chaos_indicator']}")

# 2. Disorder averaging
print("\n[2] Disorder Averaging (20 realizations)")
print("-" * 40)
for N in [6, 8, 10]:
    stats = disorder_average_lsr(N, n_realizations=20)
    goe_dev = abs(stats['mean'] - 0.530) / 0.530 * 100
    print(f"    N={N:2d}: ⟨r⟩ = {stats['mean']:.4f} ± {stats['stderr']:.4f} (GOE deviation: {goe_dev:.1f}%)")

# 3. Hubbard model comparison
print("\n[3] Hubbard Model (L=2 sites)")
print("-" * 40)
hub = DrivenHubbardHamiltonian(n_sites=2)
print(f"    Hilbert dimension: {2**hub.n_qubits}")
print(f"    Ground state energy: {hub.static_eigenvalues[0]:.4f}")
print(f"    Level spacing ⟨r⟩: {hub.get_level_spacing_ratio():.4f}")

# 4. OTOC calculation
print("\n[4] OTOC Scrambling Analysis")
print("-" * 40)

def local_Z(site, n_qubits):
    I, Z = np.eye(2), np.diag([1,-1]).astype(complex)
    ops = [Z if i==site else I for i in range(n_qubits)]
    result = ops[0]
    for op in ops[1:]: result = np.kron(result, op)
    return result

def compute_otoc(H, n_qubits, times):
    W, V = local_Z(0, n_qubits), local_Z(min(1,n_qubits-1), n_qubits)
    dim = H.shape[0]
    otoc = []
    for t in times:
        U = expm(-1j * H * t)
        W_t = U.conj().T @ W @ U
        C = np.real(np.trace(W_t.conj().T @ V.conj().T @ W_t @ V) / dim)
        otoc.append(C)
    return np.array(otoc)

times = np.linspace(0, 5, 50)
syk_otoc = compute_otoc(syk.hamiltonian_matrix, syk.n_qubits, times)
hub_otoc = compute_otoc(hub.static_hamiltonian(), hub.n_qubits, times)

# Find scrambling time
scramble_idx = np.where(syk_otoc < 0.5)[0]
t_scramble = times[scramble_idx[0]] if len(scramble_idx) > 0 else times[-1]
print(f"    SYK scrambling time (C < 0.5): t* = {t_scramble:.2f}")
print(f"    SYK OTOC(t=5): {syk_otoc[-1]:.4f}")
print(f"    Hubbard OTOC(t=5): {hub_otoc[-1]:.4f}")

# 5. Generate visualization
print("\n[5] Generating Publication Figure...")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Quantum Chaos: SYK vs Hubbard Model', fontsize=14, fontweight='bold')

# Panel A: Level Spacing with error bars
ax1 = axes[0, 0]
N_list = [6, 8, 10]
means = []
errs = []
for N in N_list:
    stats = disorder_average_lsr(N, n_realizations=20)
    means.append(stats['mean'])
    errs.append(stats['stderr'])

ax1.errorbar(N_list, means, yerr=errs, fmt='ko', markersize=8, capsize=5, capthick=2, linewidth=2)
ax1.axhline(0.530, color='green', ls='--', lw=2, label='GOE (0.530)')
ax1.axhline(0.386, color='gray', ls=':', lw=2, label='Poisson (0.386)')
ax1.fill_between([5, 11], 0.48, 0.58, alpha=0.15, color='green')
ax1.set_xlabel('N (Majorana fermions)', fontsize=11)
ax1.set_ylabel('Level Spacing Ratio ⟨r⟩', fontsize=11)
ax1.set_title('(a) Chaos Diagnostic with Error Bars', fontweight='bold')
ax1.legend(fontsize=9)
ax1.set_xlim(5, 11)
ax1.grid(alpha=0.3)

# Panel B: Spectral Density
ax2 = axes[0, 1]
ax2.hist(syk.eigenvalues, bins=15, alpha=0.7, color='crimson', 
         label=f'SYK (N={syk.n_majorana})', density=True, edgecolor='black')
ax2.hist(hub.static_eigenvalues, bins=15, alpha=0.7, color='steelblue',
         label=f'Hubbard (L={hub.n_sites})', density=True, edgecolor='black')
ax2.set_xlabel('Energy', fontsize=11)
ax2.set_ylabel('Density of States', fontsize=11)
ax2.set_title('(b) Spectral Density Comparison', fontweight='bold')
ax2.legend(fontsize=9)

# Panel C: OTOC
ax3 = axes[1, 0]
ax3.plot(times, syk_otoc, 'r-', lw=2.5, label='SYK')
ax3.plot(times, hub_otoc, 'b-', lw=2.5, label='Hubbard')
ax3.axhline(0, color='gray', ls=':', alpha=0.5)
ax3.axvline(t_scramble, color='red', ls='--', alpha=0.7, label=f't* = {t_scramble:.2f}')
ax3.fill_between(times, syk_otoc, alpha=0.2, color='crimson')
ax3.set_xlabel('Time t', fontsize=11)
ax3.set_ylabel('OTOC C(t)', fontsize=11)
ax3.set_title('(c) Information Scrambling (OTOC)', fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Panel D: Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
┌──────────────────────────────────────────┐
│         QUANTITATIVE RESULTS             │
├──────────────────────────────────────────┤
│                                          │
│  SYK MODEL (N={syk.n_majorana} Majorana):               │
│    Hilbert dim: {summary['hilbert_dim']}                       │
│    E₀: {summary['ground_state_energy']:.4f}                          │
│    ⟨r⟩ (even parity): {summary['level_spacing_ratio_even']:.4f}          │
│    Scrambling time: t* = {t_scramble:.2f}            │
│                                          │
│  HUBBARD MODEL (L={hub.n_sites} sites):             │
│    Hilbert dim: {2**hub.n_qubits}                       │
│    E₀: {hub.static_eigenvalues[0]:.4f}                          │
│    ⟨r⟩: {hub.get_level_spacing_ratio():.4f}                          │
│                                          │
│  REFERENCE VALUES:                       │
│    GOE (chaotic): ⟨r⟩ ≈ 0.530            │
│    Poisson (integrable): ⟨r⟩ ≈ 0.386     │
│                                          │
│  METHODOLOGY:                            │
│    20 disorder realizations              │
│    Even parity sector projection         │
│    Error bars from std/√n                │
│                                          │
└──────────────────────────────────────────┘
"""
ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=9,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig('output.png', dpi=150, bbox_inches='tight', facecolor='white')
print("    Saved: output.png")

print("\n" + "=" * 60)
print("  Analysis Complete!")
print("=" * 60)
