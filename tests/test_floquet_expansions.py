import numpy as np
import pytest
from scipy.linalg import logm

from quantum_chaos.floquet import (
    driven_ring_block,
    floquet_operator,
    heating_time,
    magnus_hamiltonian,
    quasi_energies,
    quasi_energy_distance,
    stroboscopic_energies,
    van_vleck_hamiltonian,
)
from quantum_chaos.hamiltonians.hubbard_ed import HubbardRingED

OMEGAS = np.array([20.0, 40.0, 80.0])


def ring_operators():
    ed = HubbardRingED(5, 2, 2, t=1.0, U=0.0)
    hop = ed.hopping_operator(1).toarray()
    return hop, hop + 1.0 * ed.double_occupancy().toarray()


def slope(errors):
    return np.polyfit(np.log(OMEGAS), np.log(errors), 1)[0]


def test_van_vleck_spectral_error_scales_with_truncation_order():
    hop, H0 = ring_operators()
    harmonics = {0: H0, 1: hop / 2, -1: hop / 2}
    errors = {0: [], 2: []}
    for omega in OMEGAS:
        T = 2 * np.pi / omega
        q = quasi_energies(
            floquet_operator(lambda t, w=omega: H0 + np.cos(w * t) * hop, T, n_steps=160), T
        )
        for order in errors:
            eff = np.linalg.eigvalsh(van_vleck_hamiltonian(harmonics, omega, order))
            errors[order].append(quasi_energy_distance(eff, q, omega))
    assert slope(errors[0]) == pytest.approx(-2, abs=0.3)
    assert slope(errors[2]) == pytest.approx(-4, abs=0.6)


def test_floquet_magnus_operator_error_scales_with_order():
    hop, H0 = ring_operators()
    harmonics = {0: H0, 1: hop / 2j, -1: -hop / 2j}
    errors = {0: [], 1: []}
    for omega in OMEGAS:
        T = 2 * np.pi / omega
        F = floquet_operator(lambda t, w=omega: H0 + np.sin(w * t) * hop, T, n_steps=160)
        HF = 1j * logm(F) / T
        for order in errors:
            errors[order].append(
                np.linalg.norm(HF - magnus_hamiltonian(harmonics, omega, order), 2)
            )
    assert slope(errors[0]) == pytest.approx(-1, abs=0.2)
    assert slope(errors[1]) == pytest.approx(-2, abs=0.3)


def test_effective_hamiltonians_are_hermitian_and_orders_are_guarded():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    H0 = a + a.conj().T
    H1 = rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6))
    harmonics = {0: H0, 1: H1, -1: H1.conj().T}
    for order in (0, 1, 2):
        H = van_vleck_hamiltonian(harmonics, 7.0, order)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)
    with pytest.raises(NotImplementedError):
        van_vleck_hamiltonian(harmonics, 7.0, 3)
    with pytest.raises(NotImplementedError):
        magnus_hamiltonian(harmonics, 7.0, 2)


@pytest.mark.parametrize("method, order", [("midpoint", 2), ("magnus4", 4)])
def test_propagator_convergence_order(method, order):
    hop, H0 = ring_operators()
    T = 2 * np.pi / 3.0
    ref = floquet_operator(lambda t: H0 + np.cos(3 * t) * hop, T, n_steps=640)
    steps = np.array([20, 40, 80])
    errs = [
        np.linalg.norm(floquet_operator(lambda t: H0 + np.cos(3 * t) * hop, T, n, method) - ref, 2)
        for n in steps
    ]
    assert -np.polyfit(np.log(steps), np.log(errs), 1)[0] == pytest.approx(order, abs=0.3)


def test_stroboscopic_energies_and_heating_time():
    hop, H0 = ring_operators()
    F = floquet_operator(lambda t: H0 + np.cos(5 * t) * hop, 2 * np.pi / 5, n_steps=40)
    psi = np.linalg.eigh(H0)[1][:, 0]
    energies = stroboscopic_energies(F, psi, H0, 5)
    direct = [
        np.vdot(np.linalg.matrix_power(F, n) @ psi, H0 @ np.linalg.matrix_power(F, n) @ psi)
        for n in range(6)
    ]
    np.testing.assert_allclose(energies, np.real(direct), atol=1e-10)
    assert heating_time(np.array([0.0, 0.2, 0.6, 1.0]), 0.0, 1.0) == 2.0
    assert heating_time(np.array([0.0, 0.1]), 0.0, 1.0) == float("inf")


def test_momentum_block_reproduces_resolved_spectrum():
    block = driven_ring_block(6, 2, 2, momentum=0)
    H = block.static(1.0, 0.4, 3.0)
    ed = HubbardRingED(6, 2, 2, t=1.0, tp=0.4, U=3.0)
    ref = np.sort(np.concatenate([b["energies"] for b in ed.sector_blocks(momenta=[0])]))
    np.testing.assert_allclose(np.linalg.eigvalsh(H), ref, atol=1e-10)
