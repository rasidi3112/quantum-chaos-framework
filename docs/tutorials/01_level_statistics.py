# %% [markdown]
# # Level statistics and symmetry classes
#
# This tutorial shows why spacing statistics must be computed inside one
# symmetry sector, how the SYK class follows from $N \bmod 8$, and how the
# fully symmetry-resolved Hubbard ring separates integrable from
# non-integrable dynamics. Sizes are kept small so that it runs in seconds.

# %%
import warnings

import matplotlib.pyplot as plt
import numpy as np

from quantum_chaos.hamiltonians import SYKHamiltonian, hubbard_level_statistics
from quantum_chaos.hamiltonians.syk_hamiltonian import disorder_average_lsr, syk_rmt_reference
from quantum_chaos.spectral import R_MEAN, DegenerateSpectrumWarning, ratio_density, spacing_ratios

# %% [markdown]
# ## 1. The SYK parity sectors
#
# For $N \bmod 8 = 2, 6$ the particle-hole symmetry maps the even sector onto
# the odd one, so the two spectra coincide. For $N \bmod 8 = 4$ each sector
# consists of Kramers pairs.

# %%
for N in (10, 12, 16):
    syk = SYKHamiltonian(N, seed=0)
    even, odd = syk.sector_eigenvalues("even"), syk.sector_eigenvalues("odd")
    print(
        f"N={N} ({syk.symmetry_class}): sectors identical: {np.allclose(even, odd)}, "
        f"Kramers pairs in even sector: {np.allclose(even[0::2], even[1::2])}"
    )

# %% [markdown]
# ## 2. Why sectors must not be merged
#
# Pooling the full spectrum mixes independent sectors (towards Poisson) and,
# for $N \bmod 8 \neq 0$, introduces exact degeneracies. The library keeps such
# degeneracies (they give $r = 0$) and warns, instead of silently filtering them.

# %%
syk = SYKHamiltonian(12, seed=0)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    pooled = spacing_ratios(syk.eigenvalues).mean()
print(
    f"pooled full spectrum: <r> = {pooled:.3f}; warnings: {[w.category.__name__ for w in caught]}"
)
print(f"sector resolved (Kramers-reduced): <r> = {syk.get_level_spacing_ratio():.3f}")
print(f"GSE reference: {R_MEAN['gse']:.4f}")

# %% [markdown]
# ## 3. Disorder average against random-matrix theory of the same size

# %%
for N in (12, 14, 16):
    data = disorder_average_lsr(N, n_realizations=40, rng=0)
    ref = syk_rmt_reference(N, n_samples=200, rng=1)
    print(
        f"N={N} {data['symmetry_class']}: SYK {data['mean']:.4f} ± {data['stderr']:.4f}, "
        f"RMT same size {ref['mean']:.4f} ± {ref['stderr']:.4f}, large N {data['reference']:.4f}"
    )

# %% [markdown]
# ## 4. Distributions of $\tilde r$

# %%
grid = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6, 4))
for N, color in ((16, "C0"), (14, "C1"), (12, "C2")):
    r = np.concatenate([SYKHamiltonian(N, seed=s).spacing_ratios() for s in range(30)])
    cls = SYKHamiltonian(N, seed=0).symmetry_class
    ax.hist(
        r, bins=20, range=(0, 1), density=True, histtype="step", color=color, label=f"N={N} ({cls})"
    )
    ax.plot(grid, ratio_density(grid, cls), color=color, ls="--")
ax.set_xlabel(r"$\tilde r$")
ax.set_ylabel(r"$P(\tilde r)$")
ax.legend()
plt.show()

# %% [markdown]
# ## 5. Hubbard ring: integrable vs non-integrable
#
# With momentum, reflection, total spin and (for $t'=0$) η-spin resolved, the
# integrable chain is close to Poisson while next-nearest-neighbour hopping
# moves the statistics towards GOE. At $L = 6$ the blocks are small, so the
# numbers are only indicative; see the validation report for $L = 10$.

# %%
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DegenerateSpectrumWarning)
    for tp in (0.0, 0.5):
        res = hubbard_level_statistics(n_sites=6, n_up=3, n_down=3, tp=tp, U=4.0, min_levels=8)
        print(
            f"t'={tp}: <r> = {res['mean']:.3f} ± {res['stderr']:.3f} "
            f"({res['n_ratios']} ratios in {res['n_blocks']} blocks)"
        )
print(f"Poisson {R_MEAN['poisson']:.4f}, GOE {R_MEAN['goe']:.4f}")
