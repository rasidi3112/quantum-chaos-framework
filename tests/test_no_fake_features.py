"""No placeholder or misleading functionality (PLAN P0-9)."""

import pathlib
import re

import numpy as np
import pytest
from scipy.linalg import expm

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "quantum_chaos"

REMOVED = [
    ("quantum_chaos.noise.nisq_simulator", "NISQSimulator", "error_mitigation_zne"),
    ("quantum_chaos.noise.nisq_simulator", "NISQSimulator", "create_noisy_circuit"),
    ("quantum_chaos.noise.nisq_simulator", "NISQSimulator", "benchmark_spectral_stability"),
    ("quantum_chaos.noise.nisq_simulator", "NISQSimulator", "compute_noisy_eigenspectrum"),
    ("quantum_chaos.circuits.otoc_calculator", "OTOCCalculator", "lyapunov_exponent"),
    ("quantum_chaos.circuits.otoc_calculator", "OTOCCalculator", "compare_models"),
    ("quantum_chaos.circuits.otoc_calculator", "OTOCCalculator", "butterfly_velocity"),
    ("quantum_chaos.circuits.otoc_calculator", "OTOCCalculator", "scrambling_time"),
    ("quantum_chaos.qml.quantum_classifier", "QuantumKernelClassificationBenchmark",
     "_draw_conclusion"),
    ("quantum_chaos.visualization.quantum_visualizer", "QuantumVisualizer",
     "plot_spectral_stability"),
]


@pytest.mark.xfail(reason="P0-9: placeholder/misleading API still present", strict=True)
@pytest.mark.parametrize("module, cls, attr", REMOVED)
def test_placeholder_api_removed(module, cls, attr):
    import importlib

    assert not hasattr(getattr(importlib.import_module(module), cls), attr)


@pytest.mark.xfail(reason="P0-9: source still contains placeholders or advantage claims",
                   strict=True)
def test_source_has_no_placeholders_or_unsupported_claims():
    pattern = re.compile(r"placeholder|simplified|advantage|expressivity_score|Higher expressivity",
                         re.IGNORECASE)
    offenders = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


# --- create_otoc_circuit -------------------------------------------------


def random_hamiltonian(n_qubits, seed):
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(2 ** n_qubits,) * 2) + 1j * rng.normal(size=(2 ** n_qubits,) * 2)
    return (a + a.conj().T) / 2


@pytest.mark.xfail(reason="P0-9: create_otoc_circuit has no time evolution", strict=True)
@pytest.mark.parametrize("pure", [False, True])
def test_otoc_circuit_matches_exact_otoc(pure):
    from quantum_chaos.circuits.otoc_calculator import OTOCCalculator

    n = 3
    calc = OTOCCalculator(n, hamiltonian=random_hamiltonian(n, seed=11))
    times = np.array([0.0, 0.3, 1.0, 2.5])
    W, V = calc.local_pauli("X", 0), calc.local_pauli("Z", 2)
    psi = None
    if pure:
        rng = np.random.default_rng(3)
        psi = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
        psi /= np.linalg.norm(psi)
    exact = np.real(calc.compute_otoc_exact(W, V, times, state=psi))
    if pure:
        circuit = calc.create_otoc_circuit("X", 0, "Z", 2, initial_state=psi)
    else:  # infinite temperature; same call signature as the original code
        circuit = calc.create_otoc_circuit("X", 0, "Z", 2)
    got = np.array([float(circuit(t)) for t in times])
    np.testing.assert_allclose(got, exact, atol=1e-8)
    assert np.ptp(exact) > 0.1  # the check is not trivially satisfied by a constant


# --- create_pennylane_noisy_circuit ---------------------------------------

PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}
TERMS = [(0.6, "XX"), (0.4, "ZI"), (-0.3, "YZ"), (0.2, "IY")]


def pauli_string(s):
    m = np.array([[1.0 + 0j]])
    for p in s:
        m = np.kron(m, PAULI[p])
    return m


def first_order_layer(terms, dt):
    U = np.eye(4, dtype=complex)
    for c, s in terms:
        U = expm(-1j * c * dt * pauli_string(s)) @ U
    return U


@pytest.mark.xfail(reason="P0-9: noisy circuit only implements all-Z terms", strict=True)
def test_noisy_trotter_circuit_without_noise_matches_ideal_layers():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator
    from quantum_chaos.noise.noise_channels import NoiseModel

    model = NoiseModel(depolarizing_rate=0.0, t1=np.inf, t2=np.inf)
    sim = NISQSimulator(2, noise_model=model)
    psi0 = np.array([1, 0, 0, 0], dtype=complex)
    rho = sim.create_pennylane_noisy_circuit(TERMS, total_time=1.0, n_steps=4)(psi0)
    U = np.linalg.matrix_power(first_order_layer(TERMS, 0.25), 4)
    psi = U @ psi0
    np.testing.assert_allclose(rho, np.outer(psi, psi.conj()), atol=1e-10)


@pytest.mark.xfail(reason="P0-9: noisy circuit only implements all-Z terms", strict=True)
def test_noisy_trotter_circuit_matches_numpy_noise_model():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator
    from quantum_chaos.noise.noise_channels import NoiseModel

    model = NoiseModel(depolarizing_rate=0.02, t1=30.0, t2=20.0, gate_time=0.5)
    sim = NISQSimulator(2, noise_model=model)
    psi0 = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    rho_pl = sim.create_pennylane_noisy_circuit(TERMS, total_time=1.2, n_steps=3)(psi0)

    U = first_order_layer(TERMS, 0.4)
    rho = np.outer(psi0, psi0.conj())
    for _ in range(3):
        rho = U @ rho @ U.conj().T
        rho = sim.noise_channels.full_noise_model(rho, model)  # duration = gate_time
    np.testing.assert_allclose(rho_pl, rho, atol=1e-10)


# --- effective Hamiltonian --------------------------------------------------


def test_lowest_order_effective_hamiltonian_is_period_average():
    from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian

    model = DrivenHubbardHamiltonian(n_sites=2, hopping_amplitude=1.3, driving_frequency=2.0)
    T = 2 * np.pi / model.driving_frequency
    ts = (np.arange(400) + 0.5) * T / 400
    average = sum(model.hamiltonian(t) for t in ts) / len(ts)
    np.testing.assert_allclose(model.effective_hamiltonian(order=0), average, atol=1e-12)


@pytest.mark.xfail(reason="P0-9: effective_hamiltonian silently ignores `order`", strict=True)
def test_higher_order_effective_hamiltonian_is_not_faked():
    from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian

    model = DrivenHubbardHamiltonian(n_sites=2)
    for order in (1, 2, 5):
        with pytest.raises(NotImplementedError):
            model.effective_hamiltonian(order=order)
