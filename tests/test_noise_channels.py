"""Kraus channels and the T1/T2 noise model (PLAN P0-5, P0-6)."""

import numpy as np
import pytest

from quantum_chaos.noise.noise_channels import NoiseChannels, NoiseModel

PLUS = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)
EXCITED = np.diag([0.0, 1.0]).astype(complex)


def all_kraus_sets(nc):
    return {
        "depolarizing": nc.depolarizing_kraus(0.3),
        "amplitude_damping": nc.amplitude_damping_kraus(0.2),
        "phase_damping": nc.phase_damping_kraus(0.4),
        "bit_flip": nc.bit_flip_kraus(0.1),
        "phase_flip": nc.phase_flip_kraus(0.25),
    }


# --- P0-5 -------------------------------------------------------------------


def test_apply_channel_accepts_real_density_matrix():
    nc = NoiseChannels(1)
    rho_real = np.array([[0.75, 0.25], [0.25, 0.25]])
    for kraus in all_kraus_sets(nc).values():
        out = nc.apply_channel(rho_real, kraus, 0)
        np.testing.assert_allclose(out, nc.apply_channel(rho_real.astype(complex), kraus, 0),
                                   atol=1e-15)


@pytest.mark.parametrize("name", ["depolarizing", "amplitude_damping", "phase_damping",
                                  "bit_flip", "phase_flip"])
def test_kraus_sets_are_trace_preserving(name):
    nc = NoiseChannels(2)
    kraus = all_kraus_sets(nc)[name]
    assert nc.verify_cptp(kraus)
    rng = np.random.default_rng(0)
    a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    out = nc.apply_channel(rho, kraus, 1)
    assert np.trace(out) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(out, out.conj().T, atol=1e-12)


def test_depolarizing_convention_is_replacement_by_maximally_mixed_state():
    # rho -> (1 - p) rho + p I/2 ; average gate fidelity 1 - p/2
    nc = NoiseChannels(1)
    p = 0.3
    out = nc.apply_channel(EXCITED, nc.depolarizing_kraus(p), 0)
    np.testing.assert_allclose(out, (1 - p) * EXCITED + p * np.eye(2) / 2, atol=1e-15)
    assert nc.channel_fidelity(nc.depolarizing_kraus(p)) == pytest.approx(1 - p / 2, abs=1e-15)


# --- P0-6 -------------------------------------------------------------------


def relax(rho, model, n_calls=1):
    """Apply the noise model n_calls times, each for the default duration gate_time."""
    nc = NoiseChannels(1)
    for _ in range(n_calls):
        rho = nc.full_noise_model(rho, model)
    return rho


@pytest.mark.parametrize("t1, t2", [(100.0, 80.0), (50.0, 100.0), (30.0, 5.0)])
def test_coherence_decays_as_exp_minus_t_over_t2(t1, t2):
    t = 20.0
    model = NoiseModel(depolarizing_rate=0.0, t1=t1, t2=t2, gate_time=t / 10)
    rho = relax(PLUS, model, n_calls=10)
    assert abs(rho[0, 1]) == pytest.approx(0.5 * np.exp(-t / t2), rel=1e-10)


@pytest.mark.parametrize("t1, t2", [(100.0, 80.0), (50.0, 100.0)])
def test_population_decays_as_exp_minus_t_over_t1(t1, t2):
    model = NoiseModel(depolarizing_rate=0.0, t1=t1, t2=t2, gate_time=2.0)
    rho = relax(EXCITED, model, n_calls=10)
    assert rho[1, 1].real == pytest.approx(np.exp(-20.0 / t1), rel=1e-12)


def test_unphysical_t2_is_rejected():
    with pytest.raises(ValueError):
        relax(PLUS, NoiseModel(depolarizing_rate=0.0, t1=10.0, t2=25.0))


def test_evolve_with_noise_is_independent_of_step_count_for_h_zero():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator

    model = NoiseModel(depolarizing_rate=0.0, t1=40.0, t2=30.0, gate_time=1.0)
    sim = NISQSimulator(n_qubits=1, noise_model=model)
    H0 = np.zeros((2, 2))
    t = 6.0
    results = [sim.evolve_with_noise(PLUS, H0, t, n_steps=n) for n in (1, 3, 12)]
    for rho in results:
        assert abs(rho[0, 1]) == pytest.approx(0.5 * np.exp(-t / 30.0), rel=1e-10)
        assert rho[1, 1].real == pytest.approx(0.5 * np.exp(-t / 40.0), rel=1e-10)


def test_summary_reports_average_gate_fidelity():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator

    p = 0.02
    sim = NISQSimulator(n_qubits=2, noise_model=NoiseModel(depolarizing_rate=p))
    nc = NoiseChannels(1)
    assert sim.summary()["average_gate_fidelity"] == pytest.approx(
        nc.channel_fidelity(nc.depolarizing_kraus(p)), abs=1e-15)


def test_pennylane_depolarizing_parameter_matches_numpy_convention():
    import pennylane as qml

    from quantum_chaos.noise.noise_channels import pennylane_depolarizing_parameter

    nc = NoiseChannels(1)
    p = 0.3
    kraus_pl = qml.DepolarizingChannel(pennylane_depolarizing_parameter(p), wires=0).kraus_matrices()
    rng = np.random.default_rng(5)
    a = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    rho = a @ a.conj().T
    rho /= np.trace(rho)
    expected = nc.apply_channel(rho, nc.depolarizing_kraus(p), 0)
    got = sum(K @ rho @ K.conj().T for K in kraus_pl)
    np.testing.assert_allclose(got, expected, atol=1e-14)


def test_depth_limit_is_largest_depth_above_target():
    from quantum_chaos.noise.nisq_simulator import NISQSimulator

    p, n, target = 0.01, 3, 0.9
    d = NISQSimulator(n_qubits=n, noise_model=NoiseModel(depolarizing_rate=p)) \
        .estimate_circuit_depth_limit(target)
    assert (1 - p) ** (n * d) >= target > (1 - p) ** (n * (d + 1))
    noiseless = NISQSimulator(n_qubits=n, noise_model=NoiseModel(depolarizing_rate=0.0))
    assert noiseless.estimate_circuit_depth_limit(target) == float("inf")
