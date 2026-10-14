# VALIDATION

What has been tested, with which tolerance, what came out, and what is **not**
yet verified. Status after phase 1 of [PLAN.md](https://github.com/rasidi3112/quantum-chaos-framework/blob/main/PLAN.md) (2026-10-14).

Environment of the recorded runs: macOS (Apple M1, 16 GB), Python 3.13.5 and
3.14.6, NumPy 2.5.4, SciPy 1.18.1, PennyLane 0.45.1. The test suite has **not**
yet run on CI or on Linux (phase 2).

Reproduce:

```bash
pytest                                        # fast suite
pytest -m slow                                # statistical tests (~1-2 min)
python scripts/validate_level_statistics.py   # section 2 tables (~40 min)
make figures                                  # figures/*.png (+ JSON sidecars)
```

## 1. Exact and deterministic checks (unit tests)

Each bug fix of phase 0 was preceded by a test that failed on the original
code (`git log` shows the failing-first commits with the original output).

| Area | Check | Tolerance | Test file |
|---|---|---|---|
| Partial trace | Tr_B(ρ_A⊗ρ_B)=ρ_A for 2×2, 2×3, 3×2 and three subsystems; Tr=1; product state → projector (old code: zero matrix) | 1e-12 | `test_partial_trace.py` |
| Entropy | product 0, Bell 1 bit, GHZ 1 bit, S_A=S_B for pure states | 1e-12 / 1e-10 | `test_partial_trace.py` |
| Fidelity | F(ρ,ρ)=1 (pure, rank 1, 2, full; old NISQ code gave 4), pure overlap, symmetry, bounds, commuting states | 1e-10 | `test_fidelity.py` |
| Floquet | static H: quasi-energies = folded eigenvalues (old code: sign inverted); F = exp(−iHT); unitarity; midpoint propagator converges as n⁻² | 1e-10 / 1e-12 / slope ±0.15 | `test_floquet.py` |
| Noise | Kraus completeness; \|ρ01\| = ½e^{−t/T2} and P1 = e^{−t/T1} (old: rate 1/(2T1)+1/(2T2)); T2>2T1 rejected; H=0 result independent of step count; NumPy ≡ PennyLane `default.mixed` | 1e-10 – 1e-14 | `test_noise_channels.py`, `test_no_fake_features.py` |
| PennyLane 0.45 | all subpackages import (old: 3 of 6 failed on Python ≤3.13); Pauli Hamiltonian matrix; StatePrep evolution; `set_shots`; Trotter error slopes 1 and 2 | 1e-12 / 1e-8 / ±0.15 | `test_pennylane_compat.py` |
| OTOC circuit | Hadamard-test circuit = exact OTOC, T=∞ (Bell-pair purification) and pure state (old circuit: 1 for all t) | 1e-8 | `test_no_fake_features.py` |
| Kernels | K(x,x)=1, symmetric PSD; every feature changes the state (old SYK kernel: mean(x) only; old Hubbard kernel: K≡1); circuit = state-vector kernel | 1e-12 / 1e-8 | `test_quantum_kernel.py` |
| Effective Hamiltonian | order 0 = period average; higher orders raise (old: silently ignored) | 1e-12 | `test_no_fake_features.py` |
| SYK construction | Pauli-string build = Jordan–Wigner product build (N=4…12); χ²=½ gives H/4; Tr H²/D = Σ J² | 1e-12 | `test_syk_symmetry.py` |
| SYK symmetry | P = (∏χ_{2j})K: [P,H]=0, P²=(−1)^{n(n−1)/2}, PΓP⁻¹=(−1)^nΓ (N=8…16); GUE sectors degenerate; GSE Kramers pairs; GOE no degeneracy | 1e-12 / 1e-10 | `test_syk_symmetry.py` |
| Hubbard ED | [H,T]=[H,R]=[H,S²]=0; [H,η²]=0 only for t'=0; union of blocks = full sector spectrum; U=0 free fermions with t'; t=0 doublon counting; full Fock spectrum = Jordan–Wigner model (L=3,4); k ≡ −k | 1e-10 – 1e-12 | `test_hubbard_ed.py` |
| RMT references | sampled GOE/GUE/GSE/Poisson reproduce Atas et al. (dim 64); surmise densities integrate to 1 and reproduce the tabulated surmise means | 3σ+1e-3 / 1e-10 | `test_spectral.py` |
| README | the quick-start block executes | — | `test_readme.py` |

## 2. Level statistics

Produced by `scripts/validate_level_statistics.py` (raw numbers and provenance
in [results/level_statistics.json](https://github.com/rasidi3112/quantum-chaos-framework/blob/main/results/level_statistics.json); package
source tree `03c1a9a`). Errors are one standard error. Same-size RMT: random
GOE/GUE/GSE matrices with the same number of distinct levels as one SYK
sector, analysed identically (`syk_rmt_reference`); z is the difference in
units of the combined standard error.

### 2.1 SYK (sector resolved; class from N mod 8)

| N | class | levels/sector | realisations | ⟨r⟩ SYK | ⟨r⟩ RMT same size | z | Atas (N→∞) |
|---|---|---|---|---|---|---|---|
| 10 | GUE | 16 | 400 | 0.6024 ± 0.0037 | 0.5972 ± 0.0030 | +1.10 | 0.5996 |
| 12 | GSE | 16 | 400 | 0.6731 ± 0.0023 | 0.6720 ± 0.0026 | +0.31 | 0.6744 |
| 14 | GUE | 64 | 300 | 0.5997 ± 0.0019 | 0.5990 ± 0.0015 | +0.31 | 0.5996 |
| 16 | GOE | 128 | 200 | 0.5305 ± 0.0015 | 0.5293 ± 0.0012 | +0.65 | 0.5307 |
| 18 | GUE | 256 | 100 | 0.5983 ± 0.0019 | 0.5996 ± 0.0012 | −0.58 | 0.5996 |
| 20 | GSE | 256 | 40 | 0.6743 ± 0.0016 | 0.6758 ± 0.0011 | −0.74 | 0.6744 |
| 22 | GUE | 1024 | 20 | 0.5997 ± 0.0023 | 0.6000 ± 0.0006 | −0.13 | 0.5996 |

All seven sizes, covering the three classes, agree with the same-size RMT
reference within |z| ≤ 1.1 and with the large-N values of Atas et al. within
the statistical error plus the finite-size offset (≤ 0.005 for 16 levels).
For comparison, the original code compared N = 6, 8, 10 with GOE; N = 6 and 10
are GUE.

Tolerance used in the tests: |z| < 3 against the same-size reference and
|⟨r⟩ − Atas| < 3σ + 0.005 (N = 12, 14, 16 in the fast suite; 18, 20 in
`-m slow`).

### 2.2 Hubbard ring (t = 1, U = 4; K, reflection, S and — for t' = 0 — η resolved)

| L | N↑+N↓ | t' | sector dim | blocks | exact degeneracies left | ⟨r⟩ all blocks (≥ 10 levels) | ⟨r⟩ blocks ≥ 400 levels, 20 % edge trim |
|---|---|---|---|---|---|---|---|
| 8 | 4+4 | 0.0 | 4900 | 89 | 1.7 % | 0.3863 ± 0.0053 (n=2840) | — |
| 8 | 4+4 | 0.5 | 4900 | 29 | 0.1 % | 0.5173 ± 0.0048 (n=2986) | — |
| 10 | 3+3 | 0.0 | 14400 | 71 | 1.1 % | 0.3794 ± 0.0030 (n=8443) | 0.3762 ± 0.0057 (n=2402) |
| 10 | 3+3 | 0.5 | 14400 | 32 | 0.2 % | 0.5148 ± 0.0028 (n=8564) | 0.5290 ± 0.0048 (n=2840) |
| 10 | 4+4 | 0.0 | 44100 | 110 | 0.4 % | 0.3840 ± 0.0017 (n=26172) | 0.3866 ± 0.0025 (n=12000) |
| 10 | 4+4 | 0.5 | 44100 | 39 | 0.0 % | 0.5291 ± 0.0016 (n=26369) | 0.5313 ± 0.0020 (n=15412) |
| 10 | 5+5 | 0.0 | 63504 | 150 | 0.2 % | 0.3857 ± 0.0014 (n=37720) | 0.3860 ± 0.0021 (n=17508) |
| 10 | 5+5 | 0.5 | 63504 | 41 | 0.0 % | 0.5287 ± 0.0013 (n=38012) | 0.5296 ± 0.0017 (n=22087) |

References: Poisson 0.3863, GOE 0.5307.

Interpretation:

- **Integrable chain (t' = 0):** consistent with Poisson at L = 10, N = 4+4 and
  5+5 (0.3840(17), 0.3857(14)); the low-density sector 3+3 lies 2.3σ below
  Poisson, partly because exactly degenerate spacings remain (1.1 %), which
  contribute r = 0.
- **t' = 0.5 (non-integrable):** at L = 10 with N = 4+4 and 5+5 the result is
  consistent with GOE (0.5291(16), 0.5287(13); −1.0σ and −1.5σ), and closer
  still for large blocks with edge trimming. At L = 8 and at the low density
  L = 10, N = 3+3, ⟨r⟩ is clearly below GOE (0.515–0.517), approaching it when
  only large blocks are used — a finite-size effect at these sizes.
- The standard errors treat the ratios as independent; neighbouring ratios
  are weakly correlated, so the errors are slight underestimates.
- The original code reported ⟨r⟩ = 0.7808 for L = 2 with all sectors pooled;
  that number has no meaning and the method was removed.

## 3. Literature values used

| Quantity | Value | Source | Status |
|---|---|---|---|
| ⟨r̃⟩ Poisson | 2 ln 2 − 1 = 0.38629 | Atas et al., PRL 110, 084101 (2013), Table I | verified (arXiv:1212.5611) |
| ⟨r̃⟩ GOE / GUE / GSE (numerics, 1000×1000) | 0.5307(1) / 0.5996(1) / 0.6744(1) | same | verified |
| ⟨r̃⟩ surmise | 0.53590 / 0.60266 / 0.67617 | same, Eq. (7) | verified; reproduced by numerical integration |
| SYK class per parity sector | N mod 8 = 0: GOE, 2/6: GUE, 4: GSE | Cotler et al., JHEP 05 (2017) 118, §3.1 | verified |
| SYK conventions | {ψ_a,ψ_b}=δ_ab, ⟨J²⟩=3!J²/N³ | Cotler et al. §2; Maldacena & Stanford, PRD 94, 106002 (2016) | verified (code: χ²=1 → H_code = 4 H_MS) |

## 4. Not yet verified / open

- **Hubbard t' ≠ 0:** consistent with GOE only for L = 10 at N = 4+4 and 5+5;
  smaller or more dilute sectors are still below GOE. Only t' = 0.5 and U = 4
  were studied; the dependence on t' and U (crossover scale) is not.
- **Residual exact degeneracies of the integrable chain** (0.2–1.7 % of the
  spacings after resolving K, reflection, S and η) are not explained.
- SYK ground-state energy and density of states vs García-García & Verbaarschot,
  PRD 94, 126010 (2016): not done (phase 3B).
- Spectral form factor, OTOC Lyapunov exponents, Krylov complexity, ETH, Page
  curve: not implemented (phase 3B).
- Higher-order Floquet expansions, heating: not implemented (phase 3C).
- Trotter commutator bounds, sparse SYK, teleportation protocol, ZNE: not
  implemented (phase 3D).
- Kernel concentration vs N and t with classical baselines: not run (phase 3E).
- References cited from memory and not yet checked: Essler et al. (2005),
  Yang PRL 63, 2144 (1989), and the items marked I/V in PLAN.md §9.
- Platform note: Apple Accelerate's `zheevd` fails for complex matrices with
  n ≳ 3000; dense Hermitian problems therefore use `*heevr`
  (`utils.helpers.hermitian_eigh`).
