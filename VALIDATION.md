# VALIDATION

What has been tested, with which tolerance, what came out, and what is **not**
yet verified. Status at version 0.3.0 (2026-10-15).

Environment of the recorded runs: macOS (Apple M1, 16 GB), Python 3.13.5 and
3.14.6, NumPy 2.5.4, SciPy 1.18.1, PennyLane 0.45.1. A GitHub Actions workflow
(Ubuntu and macOS, Python 3.11–3.13) is configured but has **not** run yet, so
nothing has been tested on Linux or on Python 3.11/3.12.

Code quality at 0.2.0: 170 fast tests + 4 slow tests pass, line coverage 95 %
(gate 85 %), `ruff` and `mypy` clean, `mkdocs build --strict` succeeds with
the tutorials executed.

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

## 3. Chaos diagnostics (phase 3B)

Figures `figures/sff.png` and `figures/chaos_dynamics.png` (numbers in the JSON
sidecars). Majorana operators with χ² = 1 unless stated; OTOCs use χ² = ½.

### 3.1 Spectral moments and ground-state energy

| Check | Result | Status |
|---|---|---|
| Disorder-averaged kurtosis M4/M2² − 3 from Wick contractions vs García-García & Verbaarschot, PRD 94, 126010 (2016), Eq. (8) | identical formula (tested N = 6…34); disorder averages agree (N = 8: −0.922 vs −0.914; N = 12: −1.029 vs −1.034; N = 16: −1.019 vs −1.020) | verified |
| Second moment M2 = C(N,4)·6J²/N³·(χ²)⁴ | per-realisation identity Tr H²/D = Σ J² and disorder average | verified |
| Ground-state energy vs the same paper's fit E0 = −0.055 − 0.029 N (Eq. 14, quoted verbatim) | our ensemble ⟨E_min⟩ (χ² = ½): −0.568(3), −0.783(3), −0.963(3), −1.149(3) at N = 12, 16, 20, 24, i.e. −2.14 σ at N = 24 with σ from the exact M2; the fit corresponds to −1.4 σ | **not reconciled — PERLU VERIFIKASI** (convention difference not identifiable from the text) |

### 3.2 Spectral form factor (β = 0)

| N (class) | realisations | plateau × L (exact degeneracies) | late-time g × L | dip time | Thouless τ (10 % or 2σ of RMT) |
|---|---|---|---|---|---|
| 16 (GOE) | 60 | 1.000 | 0.94 | 4.1 | 0.75 |
| 18 (GUE) | 40 | 2.000 | 1.89 | 3.8 | 0.51 |
| 20 (GSE) | 30 | 2.000 | 2.06 | 4.9 | not resolved |

The plateau heights 1/L (N mod 8 = 0) and 2/L (otherwise) are those stated by
Cotler et al. (JHEP 05 (2017) 118, sec. 3.1). The analytic GOE/GUE/GSE form
factors used for the ramp were checked against sampled random matrices. With
≤ 60 realisations the unfolded ramp agrees with RMT within errors, but the
Thouless time is not determined reliably at these sizes.

### 3.3 OTOCs and Lyapunov exponent (N = 14, 6 realisations, χ² = ½)

| model | β | late-time F/F(0) | fitted early-time rate | 2πT |
|---|---|---|---|---|
| SYK4 | 0 | 0.009 | 1.96 ± 0.21 (stat) ± 0.75 (window) | ∞ |
| SYK4 | 10 | 0.076 | 1.52 ± 0.14 ± 0.56 | 0.63 |
| SYK2 (free) | 0 | 0.90 | −0.01 ± 0.09 ± 0.41 | ∞ |
| SYK2 (free) | 10 | 1.29 | 1.73 ± 0.27 ± 1.41 | 0.63 |

- The late-time OTOC cleanly separates interacting SYK (decays to ≈ 0) from
  the free SYK2 control.
- **Negative result:** no Lyapunov exponent can be extracted at these sizes.
  The early-time rate exceeds 2πT at low temperature, depends strongly on the
  fit window, and the free model gives rates of the same size at β = 10. The
  chaos bound λ ≤ 2πT is therefore **not tested** here.
- Larger N with dynamical typicality (T = ∞, `results/large_n_syk.json`;
  N = 20, 24, 28 with 6, 3, 2 realisations): the local slope
  d ln(1 − F/F(0))/dt falls monotonically like ≈ 2/t (e.g. N = 28: 4.37,
  2.00, 1.30, 0.95, 0.74 at t = 0.5, 1, 1.5, 2, 2.5), i.e. 1 − F/F(0) ∝ t²
  over the whole early-time window, nearly independent of N (1 − F/F(0) at
  t = 1: 0.051, 0.045, 0.041). There is no regime of constant slope: the
  window fits (1.74, 1.74, 1.60 ± ~0.5) measure the curvature of a t² law,
  not a Lyapunov exponent.

### 3.4 Krylov complexity and operator size

- b₁² = 4 Σ_{α∋0} J_α² holds exactly per realisation; the Krylov amplitude
  φ₀(t) equals the exact autocorrelation function.
- SYK4 (N = 10–14): b_n rises over the first ~3 steps and then saturates
  (≈ 2.7 at N = 14); the linear-growth regime of Parker et al. is too short to
  be resolved at these sizes.
- Free SYK2: the Krylov space of χ₀ ends after N − 1 steps (N = 10: b₁₀ ≈ 2·10⁻⁸).
  **Numerical limitation:** the residual b_N grows with N (1·10⁻⁵ at N = 12,
  4·10⁻³ at N = 14) and termination is not detected at N = 16 in double
  precision (Lanczos error amplification); the figure therefore uses N = 10.
- Mean Majorana size of χ₀(t): stays exactly 1 for SYK2 and saturates at N/2
  for SYK4.

### 3.5 ETH and eigenstate entanglement (even parity sector)

| N | class | sector dim | diagonal fluctuation of χ₀χ₁χ₂χ₃ | ⟨\|O\|⁴⟩/⟨\|O\|²⟩² off-diagonal |
|---|---|---|---|---|
| 12 | GSE | 32 | 0.164 | 2.22 |
| 14 | GUE | 64 | 0.114 | 1.96 (Gaussian complex: 2) |
| 16 | GOE | 128 | 0.103 | 2.90 (Gaussian real: 3) |
| 18 | GUE | 256 | 0.061 | 1.98 |

- Diagonal fluctuations scale as D^−0.485 (N = 10–20; ETH: D^−1/2).
- The particle-hole-odd operator −iχ₀χ₁ has an exactly vanishing diagonal for
  N mod 8 = 0 — a symmetry effect, not ETH.
- N = 20 mid-spectrum eigenstates follow the Page curve within 0.002 for
  l = 1…5 qubits (Page formula checked against Haar-random states); at N = 16
  the half-system value is 2.22 vs Page 2.27.

## 4. Large-N numerics (phase 3A)

Benchmarks on the recording machine (`benchmarks/results/*.json`; one parity
sector, complex128, χ² = ½):

| N | sector dim | flip masks | matrix memory | build | mat-vec | Lanczos E0 | peak RSS |
|---|---|---|---|---|---|---|---|
| 24 | 2 048 | 562 | 0.023 GB | 0.13 s | 1 ms | 0.16 s | 0.10 GB |
| 28 | 8 192 | 1 093 | 0.18 GB | 0.84 s | 8 ms | 1.1 s | 0.28 GB |
| 30 | 16 384 | 1 471 | 0.48 GB | 2.8 s | 24 ms | 4.9 s | 0.61 GB |
| 32 | 32 768 | 1 941 | 1.27 GB | 8.2 s | 66 ms | 15 s | 1.51 GB |
| 34 | 65 536 | 2 517 | 3.30 GB | 24 s | 0.38 s | 66 s | 3.49 GB |

The memory prediction D/2·(C(n,4)+C(n,2)+1)·(itemsize+4) bytes agrees with the
measured matrix size to < 0.1 %. Dense diagonalisation of a sector is limited
to N ≈ 28 on 16 GB; shift-invert/POLFED for mid-spectrum statistics at N ≥ 30 is
**not implemented**.

Checks: sector matrices equal the dense construction exactly; Lanczos E0
equals dense ED (N = 16, 1e-10); the KPM second moment reproduces
Σ J²(χ²)⁴ (3 %, stochastic trace); typicality autocorrelations and
(regularised, β = 2) OTOCs agree with exact results within 5σ of the
random-vector error (N = 12).

Ensemble-averaged ground-state energy (χ² = ½, `results/large_n_syk.json`):

| N | 12 | 16 | 20 | 24 | 28 | 30 | 32 | 34 |
|---|---|---|---|---|---|---|---|---|
| realisations | 20 | 20 | 10 | 8 | 6 | 4 | 3 | 3 |
| ⟨E0⟩ | −0.566(6) | −0.783(4) | −0.961(4) | −1.149(3) | −1.308(3) | −1.391(5) | −1.478(5) | −1.563(3) |
| ⟨E0⟩/N | −0.0471 | −0.0490 | −0.0480 | −0.0479 | −0.0467 | −0.0464 | −0.0462 | −0.0460 |

A 1/N fit through N = 20 and 34 gives E0/N → ≈ −0.043; no verified literature
value is available here for comparison (the García-García & Verbaarschot fit
remains unreconciled, section 3.1).

## 5. Floquet dynamics (phase 3C)

- High-frequency expansions validated against exact Floquet operators
  (L = 5 Hubbard ring, J0 = J1 = 1, U = 4): the van Vleck spectral error scales
  as ω^−2.07 at order 0 and ω^−4.09 at order 2 (odd orders vanish for a cosine
  drive); the Floquet–Magnus operator error scales as ω^−1 / ω^−2 at orders
  0 / 1 for a sine drive. Propagators: midpoint (2nd order) and 4th-order
  Magnus with tested convergence orders.
- Heating (`figures/floquet_heating.png`; J(t) = 1 + cos ωt, t' = 0.5,
  U = 4, half filling, K = 0, 2·10⁴ periods from the ground state of H0):

| ω | 4 | 8 | 10 | 12 | 14 | 16 | 20 | 24 |
|---|---|---|---|---|---|---|---|---|
| L = 6 (dim 68, W = 19.5) | 0.986 | 0.894 | 0.284 | 0.581 | 0.026 | 0.008 | 0.0025 | 0.0010 |
| L = 8 (dim 618, W = 25.3) | 0.980 | 0.873 | 0.752 | 0.703 | 0.020 | 0.007 | 0.0022 | 0.0009 |

  (time-averaged absorbed energy fraction; W = many-body bandwidth.)
  Heating to infinite temperature within 2–50 periods for ω ≤ 12 and less than
  2 % absorption for ω ≥ 14, i.e. already below W. **Not established:** whether
  the suppression follows the exponential law of Abanin et al. / Mori et al.;
  at L ≤ 8 the residual absorption is not separable from perturbative
  dressing and finite-size resonances (L = 6 is non-monotonic at ω = 10–12).
- The choice of drive protocol is still open; the code supports
  J(t) = J0 + J1 cos ωt (J0 = 0 is the original model).

## 6. Circuits and NISQ (phase 3D)

- First-order commutator bound of Childs et al. (PRX 11, 011020, Prop. 15,
  verified) holds for SYK N = 8 at all r and is ~8.5× above the actual error;
  both scale as 1/r. The second-order bound was not implemented.
- CNOTs per first-order Trotter step (CNOT-ladder decomposition): dense SYK
  300 (N = 8) → 1.2·10⁵ (N = 24); sparse SYK with k = 4: 138 → 1.0·10³.
- Sparse SYK (Xu, Susskind, Su, Swingle; definition verified): ⟨r⟩ recovers
  the RMT value for k ≥ 4 (N = 16 GOE: 0.5306(33); N = 20 GSE: 0.6719(30)),
  deviates at k = 2 and fails for k ≤ 1.
- Interferometric OTOC with Trotter evolution reproduces the exact OTOC
  (N = 6, 1e-3 with 12 second-order steps). Under depolarizing noise
  (p = 0.005 per qubit and layer, `figures/nisq.png`), normalising by the
  echo with W = identity and zero-noise extrapolation (folds 1, 3, 5,
  Richardson) both move the signal towards the noiseless value for SYK4 and
  for the SYK2 control; in the unit test ZNE reduces the error ≥ 3×. These are
  small-N, single-noise-model statements, not hardware results.

## 7. Quantum kernels (phase 3E)

- Haar fidelity moments (mean 1/D, variance (D−1)/(D²(D+1))) checked by sampling.
- SYK Hamiltonian-encoding kernels concentrate exponentially with the number
  of qubits at long evolution times (Var[K] ∝ 2^(−2.06 n) at t = 10,
  n = 3–7; ~4.5× above Haar at n = 7), not at short times (t = 0.1: K ≈ 1).
- Hubbard kernels concentrate much less (L = 4: 1.2·10⁻² vs 7·10⁻⁴ for Haar
  in the (N, S_z) sector), because conserved quantities restrict the
  explored space.
- Classification (20 seeds, cross-validated hyperparameters for every model,
  paired Wilcoxon tests): both quantum kernels are **significantly worse**
  than an RBF-SVM — moons: RBF 0.921, SYK 0.775 (p = 1e-4), Hubbard 0.835
  (p = 2e-4); circles: RBF 0.988, SYK 0.929 (p = 0.013), Hubbard 0.7875
  (p = 1e-4). No quantum or "chaos" advantage is observed.

## 8. Literature values used

| Quantity | Value | Source | Status |
|---|---|---|---|
| ⟨r̃⟩ Poisson | 2 ln 2 − 1 = 0.38629 | Atas et al., PRL 110, 084101 (2013), Table I | verified (arXiv:1212.5611) |
| ⟨r̃⟩ GOE / GUE / GSE (numerics, 1000×1000) | 0.5307(1) / 0.5996(1) / 0.6744(1) | same | verified |
| ⟨r̃⟩ surmise | 0.53590 / 0.60266 / 0.67617 | same, Eq. (7) | verified; reproduced by numerical integration |
| SYK class per parity sector | N mod 8 = 0: GOE, 2/6: GUE, 4: GSE | Cotler et al., JHEP 05 (2017) 118, §3.1 | verified |
| SYK conventions | {ψ_a,ψ_b}=δ_ab, ⟨J²⟩=3!J²/N³ | Cotler et al. §2; Maldacena & Stanford, PRD 94, 106002 (2016) | verified (code: χ²=1 → H_code = 4 H_MS) |
| SYK kurtosis M4/M2² − 3 = −32(N−4)(N²−11N+36)/[N(N−1)(N−2)(N−3)] | exact | García-García & Verbaarschot, PRD 94, 126010 (2016), Eq. (8) | verified; re-derived from Wick contractions |
| SYK E0 fit −0.055 − 0.029 N | fit | same, Eq. (14) | quoted verbatim; **not reproduced** (section 3.1) |
| First-order Trotter commutator bound | (t²/2) Σ ‖[Σ_{γ2>γ1} H_γ2, H_γ1]‖ | Childs, Su, Tran, Wiebe, Zhu, PRX 11, 011020 (2021), Prop. 15, Eq. (145) | verified |
| Sparse SYK pruning | p = kN/C(N,4), ⟨J²⟩ = 3!J²/(N³p) | Xu, Susskind, Su, Swingle, arXiv:2008.02303 | verified |

## 9. Not yet verified / open

- **Not run on CI or Linux**; Python 3.11/3.12 untested.
- SYK ground-state energy vs García-García & Verbaarschot's fit: ~1.5×
  discrepancy not reconciled (section 3.1).
- Lyapunov exponent and the chaos bound: no exponential regime up to N = 28
  (T = ∞ early-time growth is ∝ t², section 3.3); finite-temperature OTOCs at
  large N, where an exponential window is expected, were not computed.
- Thouless time and the linear growth of Lanczos coefficients: not resolved.
- Hubbard ring: GOE only reached at L = 10 for N = 4+4, 5+5; residual
  degeneracies of the integrable chain unexplained.
- Floquet: exponential suppression of heating not established at L ≤ 8;
  drive protocol not decided.
- Mid-spectrum level statistics beyond N ≈ 28 (shift-invert / POLFED) not
  implemented; second-order Trotter commutator bound not implemented.
- NISQ results use a single depolarizing model on simulators; no hardware,
  no transpilation to a device topology.
- References from memory not yet checked: Essler et al. (2005), Yang PRL 63,
  2144 (1989), Kobrin et al. (2021), Roberts–Stanford–Streicher (2018),
  Parker et al. (2019), Abanin et al. (2015), Mori et al. (2016),
  Eckardt & Anisimovas (2015), Bukov et al. (2015), Bartsch & Gemmer (2009),
  Thanasilp et al. (2022) — cited for context only, no numbers taken from them.
- Platform note: Apple Accelerate's `zheevd` fails for complex matrices with
  n ≳ 3000; dense Hermitian problems use `*heevr` (`utils.helpers.hermitian_eigh`).
