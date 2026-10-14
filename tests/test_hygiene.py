"""Reproducibility, numerical robustness and repository hygiene (PLAN P0-10)."""

import pathlib
import re
import warnings

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "quantum_chaos"


def global_state_untouched(fn):
    before = np.random.get_state()[1].copy()
    fn()
    return np.array_equal(before, np.random.get_state()[1])


# --- explicit RNG -------------------------------------------------------------


def test_couplings_are_reproducible_without_global_rng():
    from quantum_chaos.utils.helpers import generate_random_coupling

    np.random.seed(123)
    assert global_state_untouched(lambda: generate_random_coupling(6, seed=7))
    np.testing.assert_array_equal(generate_random_coupling(6, seed=7),
                                  generate_random_coupling(6, seed=7))
    np.testing.assert_array_equal(generate_random_coupling(6, seed=np.random.default_rng(7)),
                                  generate_random_coupling(6, seed=7))


def test_stochastic_helpers_accept_rng_and_leave_global_state_alone():
    from quantum_chaos.noise.noise_channels import NoiseChannels
    from quantum_chaos.qml.quantum_classifier import QuantumKernelClassificationBenchmark
    from quantum_chaos.qml.quantum_kernel import SYKKernel

    bench = QuantumKernelClassificationBenchmark(n_qubits=3)
    assert global_state_untouched(lambda: bench.generate_classification_data(n_samples=20))
    a = bench.generate_classification_data(n_samples=20, random_state=4)
    b = bench.generate_classification_data(n_samples=20, random_state=4)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)

    kernel = SYKKernel(n_qubits=3, seed=0)
    m1 = kernel.expressivity_metric(n_samples=6, rng=1)
    m2 = kernel.expressivity_metric(n_samples=6, rng=1)
    assert m1 == m2

    bits = np.zeros(200, dtype=int)
    nc = NoiseChannels(1)
    np.testing.assert_array_equal(nc.apply_readout_error(bits, 0.3, rng=5),
                                  nc.apply_readout_error(bits, 0.3, rng=5))


def test_trotter_error_analysis_is_reproducible():
    from quantum_chaos.circuits.trotter_evolution import TrotterEvolution

    terms = [(0.5, "XY"), (0.3, "ZZ")]
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.diag([1.0, -1.0]).astype(complex)
    H = 0.5 * np.kron(X, Y) + 0.3 * np.kron(Z, Z)
    te = TrotterEvolution(2)
    r1 = te.trotter_error_analysis(terms, H, 1.0, max_steps=5, rng=3)
    r2 = te.trotter_error_analysis(terms, H, 1.0, max_steps=5, rng=3)
    assert r1["error_order1"] == r2["error_order1"]


# --- numerical robustness ---------------------------------------------------


def test_thermal_states_at_low_temperature_are_finite():
    from quantum_chaos.circuits.otoc_calculator import OTOCCalculator
    from quantum_chaos.hamiltonians.syk_hamiltonian import SYKHamiltonian

    syk = SYKHamiltonian(8, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        rho = syk.get_thermal_state(temperature=1e-3)
        calc = OTOCCalculator(syk.n_qubits, syk.hamiltonian_matrix)
        W, V = calc.local_pauli("Z", 0), calc.local_pauli("X", 1)
        c = calc.compute_otoc_exact(W, V, np.array([0.0, 0.5]), temperature=1e-3)
        c_reg = calc.compute_regularized_otoc(W, V, np.array([0.0, 0.5]), temperature=1e-3)
    assert np.all(np.isfinite(rho)) and np.trace(rho) == pytest.approx(1.0)
    # beta -> infinity: thermal state -> ground-state projector
    np.testing.assert_allclose(rho, np.outer(syk.eigenvectors[:, 0], syk.eigenvectors[:, 0].conj()),
                               atol=1e-8)
    assert np.all(np.isfinite(c)) and np.all(np.isfinite(c_reg))


def test_level_spacing_ratio_with_too_few_levels_is_nan():
    from quantum_chaos.utils.helpers import level_spacing_ratio

    assert np.isnan(level_spacing_ratio(np.array([0.0, 1.0])))


def test_matrix_to_pauli_terms_warns_for_non_hermitian_input():
    from quantum_chaos.circuits.trotter_evolution import matrix_to_pauli_terms

    with pytest.warns(UserWarning):
        matrix_to_pauli_terms(np.array([[0, 1], [0, 0]], dtype=complex), 1)


# --- hygiene ------------------------------------------------------------------


def test_no_bare_except():
    offenders = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py")
                 for i, line in enumerate(p.read_text().splitlines(), 1)
                 if re.match(r"\s*except\s*:", line)]
    assert not offenders


def test_no_global_numpy_rng_in_package():
    pattern = re.compile(r"np\.random\.(seed|rand|randn|random|normal|uniform|choice|randint)\(")
    offenders = [f"{p.relative_to(SRC)}:{i}" for p in SRC.rglob("*.py")
                 for i, line in enumerate(p.read_text().splitlines(), 1) if pattern.search(line)]
    assert not offenders


def test_entry_points_and_readme():
    assert (ROOT / "main.py").read_text().splitlines()[0] == "#!/usr/bin/env python3"
    assert "yourusername" not in (ROOT / "README.md").read_text()
