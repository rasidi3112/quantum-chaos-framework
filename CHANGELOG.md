# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/). Before 1.0.0 the public API may
change between minor versions.

## [0.2.0] - 2026-10-15

Correctness release: the original code is audited, fixed and validated (see
`PLAN.md` phases 0–2 and `VALIDATION.md`). Every bug fix was preceded by a test
that failed on the old code.

### Breaking

- Package moved to `src/quantum_chaos` (`pip install -e .`); import as
  `quantum_chaos.<subpackage>`. Python ≥ 3.11, PennyLane 0.45.x.
- All randomness uses explicit `numpy.random.Generator`s; **SYK realisations
  for a given integer seed differ from 0.1**.
- `SYKHamiltonian.get_level_spacing_ratio()` and `disorder_average_lsr()` are
  now sector resolved and class aware; `disorder_average_lsr` returns a
  bootstrap interval, the symmetry class and the Atas et al. reference.
- Removed (fake or misleading): `error_mitigation_zne` (returned 0),
  `create_noisy_circuit`, `compute_noisy_eigenspectrum`,
  `benchmark_spectral_stability`, `plot_spectral_stability`,
  `lyapunov_exponent`, `compare_models`, `butterfly_velocity`,
  `scrambling_time`, `finite_size_scaling` (1/N fit across symmetry classes),
  `DrivenHubbardHamiltonian.get_level_spacing_ratio` (pooled the Fock space),
  `expressivity_score`, `chaos_advantage`, `_draw_conclusion`, the hard-coded
  "Chaos Advantage" dashboard text, `NoiseModel.amplitude_damping_rate` /
  `phase_damping_rate`, `NISQSimulator(shots=...)`, the unused
  `QuantumKernel(n_trotter_steps=...)`.
- `effective_hamiltonian(order)` only implements `order=0` and raises otherwise.
- `plot_level_spacing_statistics` now takes symmetry-resolved ratio arrays.

### Fixed

- PennyLane 0.45: removed `qml.Device` (import failure on Python ≤ 3.13),
  `qml.QubitStateVector`, `qml.operation.Tensor`, device-level shots.
- `partial_trace` (wrong axes; product states gave the zero matrix).
- Fidelity: element-wise square roots gave F(ρ,ρ)=4; `scipy.linalg.sqrtm` was
  inaccurate for rank-deficient states.
- Quasi-energy sign; Floquet propagator now second order (midpoint).
- Noise: complex dtype in `apply_channel`; pure dephasing
  1/T_φ = 1/T2 − 1/(2T1) so that coherences decay as exp(−t/T2); T2 ≤ 2T1
  enforced; relaxation acts for the physical step duration; consistent
  depolarizing convention with PennyLane; average gate fidelity 1 − p/2.
- Kernels: Hubbard kernel started from the vacuum (K ≡ 1); SYK kernel used only
  mean(x); the PennyLane kernel circuit was not U(x2)†U(x1) (K(x,x) ≠ 1).
- Level statistics: no silent "spacing < 1e-12" filter; SYK compared with the
  correct class (N mod 8); Kramers pairs handled explicitly.
- Overflow of Boltzmann weights at low temperature; `nan` instead of 0.0 for
  too few levels; Trotter-plot reference slopes; bare `except`; shebang.
- Dense complex eigenproblems use LAPACK `*heevr` (Apple Accelerate's `zheevd`
  fails for n ≳ 3000).

### Added

- `spectral`: spacing ratios with degeneracy warnings and Kramers reduction,
  Atas et al. reference values and surmise densities (verified), finite-size
  GOE/GUE/GSE/Poisson samplers, bootstrap means.
- SYK: sparse Pauli-string construction (N = 24 in < 1 s), symmetry class,
  particle-hole operator, parity-sector spectra, same-size RMT references,
  `chi_squared=0.5` (Maldacena–Stanford normalisation).
- `HubbardRingED`: Hubbard ring with NNN hopping resolved by N↑, N↓, momentum,
  reflection, total spin and η-spin; `hubbard_level_statistics`.
- Hadamard-test OTOC circuit (T = ∞ via purification or a pure state); noisy
  Trotter circuits on `default.mixed`; Hamiltonian-encoding fidelity kernels.
- `experiments`: YAML configs with explicit seeds, `make figures` with JSON
  provenance sidecars; `scripts/validate_level_statistics.py`.
- Tooling: pytest (coverage ≥ 85 %, currently 95 %), ruff, mypy, pre-commit,
  GitHub Actions workflow, MkDocs documentation with executed tutorials.

## [0.1.0]

Original code (commit `69396f5`), published with version string 1.0.0.
