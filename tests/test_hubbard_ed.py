"""Symmetry-resolved exact diagonalisation of the Hubbard ring (PLAN P1-4)."""

from itertools import combinations

import numpy as np
import pytest

from quantum_chaos.hamiltonians.hubbard_ed import HubbardRingED, hubbard_level_statistics


def commutator_norm(A, B):
    C = A @ B - B @ A
    return abs(C).max() if C.nnz else 0.0


@pytest.mark.parametrize(
    "L, nu, nd, tp", [(4, 2, 2, 0.0), (6, 3, 3, 0.0), (6, 2, 3, 0.4), (5, 2, 2, 0.3)]
)
def test_symmetry_operators_commute_with_h(L, nu, nd, tp):
    ed = HubbardRingED(L, nu, nd, t=1.0, tp=tp, U=2.7)
    H = ed.hamiltonian
    assert commutator_norm(H, H.conj().T) == 0.0
    for op in (ed.translation(), ed.reflection(), ed.spin_squared()):
        assert commutator_norm(H, op) < 1e-12
    if ed.has_eta_symmetry:
        assert commutator_norm(H, ed.eta_squared()) < 1e-12


def test_next_nearest_neighbour_hopping_breaks_eta_pairing():
    ed = HubbardRingED(6, 3, 3, t=1.0, tp=0.4, U=2.7)
    assert not ed.has_eta_symmetry
    assert commutator_norm(ed.hamiltonian, ed.eta_squared()) > 0.1


@pytest.mark.parametrize(
    "L, nu, nd, tp", [(5, 2, 2, 0.3), (6, 3, 3, 0.0), (6, 2, 3, 0.4), (6, 3, 3, 0.4)]
)
def test_resolved_blocks_reproduce_full_sector_spectrum(L, nu, nd, tp):
    ed = HubbardRingED(L, nu, nd, t=1.0, tp=tp, U=3.1)
    blocks = ed.sector_blocks(momenta=range(L))
    union = np.sort(np.concatenate([b["energies"] for b in blocks]))
    np.testing.assert_allclose(union, np.linalg.eigvalsh(ed.hamiltonian.toarray()), atol=1e-10)
    sz = (nu - nd) / 2
    for b in blocks:
        assert b["S"] >= abs(sz) - 1e-12 and float(b["S"] - sz).is_integer()


def test_momenta_k_and_minus_k_are_degenerate():
    ed = HubbardRingED(6, 2, 3, t=1.0, tp=0.4, U=3.1)
    spectrum = {}
    for b in ed.sector_blocks(momenta=range(6)):
        spectrum.setdefault(b["m"], []).extend(b["energies"])
    for m in (1, 2):
        np.testing.assert_allclose(np.sort(spectrum[m]), np.sort(spectrum[6 - m]), atol=1e-10)


@pytest.mark.parametrize("L, nu, nd, tp", [(6, 2, 3, 0.0), (6, 3, 3, 0.35), (7, 3, 2, 0.2)])
def test_free_fermion_limit(L, nu, nd, tp):
    k = 2 * np.pi * np.arange(L) / L
    eps = -2 * np.cos(k) - 2 * tp * np.cos(2 * k)
    energies = sorted(
        sum(eps[list(a)]) + sum(eps[list(b)])
        for a in combinations(range(L), nu)
        for b in combinations(range(L), nd)
    )
    ed = HubbardRingED(L, nu, nd, t=1.0, tp=tp, U=0.0)
    np.testing.assert_allclose(np.linalg.eigvalsh(ed.hamiltonian.toarray()), energies, atol=1e-10)


def test_atomic_limit_counts_doublons():
    L, nu, nd, U = 5, 3, 2, 1.7
    ed = HubbardRingED(L, nu, nd, t=0.0, tp=0.0, U=U)
    expected = sorted(
        U * len(set(a) & set(b))
        for a in combinations(range(L), nu)
        for b in combinations(range(L), nd)
    )
    np.testing.assert_allclose(np.linalg.eigvalsh(ed.hamiltonian.toarray()), expected, atol=1e-12)


@pytest.mark.parametrize("L", [3, 4])
def test_cross_check_with_jordan_wigner_fock_space_model(L):
    from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian

    jw = DrivenHubbardHamiltonian(
        n_sites=L, hopping_amplitude=1.0, onsite_interaction=2.3, periodic_boundary=True
    )
    reference = np.linalg.eigvalsh(jw.static_hamiltonian())
    levels = []
    for nu in range(L + 1):
        for nd in range(L + 1):
            ed = HubbardRingED(L, nu, nd, t=1.0, tp=0.0, U=2.3)
            levels.extend(np.linalg.eigvalsh(ed.hamiltonian.toarray()))
    np.testing.assert_allclose(np.sort(levels), reference, atol=1e-10)


def test_invalid_geometries_are_rejected():
    with pytest.raises(ValueError):
        HubbardRingED(2, 1, 1)
    with pytest.raises(ValueError):
        HubbardRingED(4, 2, 2, tp=0.5)  # NNN bonds would be double counted


def test_level_statistics_report_is_complete():
    res = hubbard_level_statistics(6, 3, 3, t=1.0, tp=0.5, U=4.0, min_levels=5)
    assert res["n_ratios"] > 20 and np.isfinite(res["mean"]) and not res["eta_resolved"]
    assert 0.0 <= res["degenerate_fraction"] <= 1.0


@pytest.mark.slow
def test_integrable_versus_nonintegrable_level_statistics():
    """L = 8, U = 4: t' = 0 is close to Poisson, t' = 0.5 is clearly not.

    Measured (2026-10-14): t'=0: 0.3865(53); t'=0.5: 0.5173(48), i.e. still below
    GOE 0.5307 at this size - reported in VALIDATION.md, not hidden by the bound.
    """
    integrable = hubbard_level_statistics(8, 4, 4, tp=0.0, U=4.0)
    chaotic = hubbard_level_statistics(8, 4, 4, tp=0.5, U=4.0)
    assert abs(integrable["mean"] - 0.3863) < 4 * integrable["stderr"]
    assert chaotic["mean"] - integrable["mean"] > 10 * np.hypot(
        chaotic["stderr"], integrable["stderr"]
    )


def test_unresolved_fock_space_level_statistics_removed():
    from quantum_chaos.hamiltonians.hubbard_hamiltonian import DrivenHubbardHamiltonian

    assert not hasattr(DrivenHubbardHamiltonian, "get_level_spacing_ratio")
    assert "level_spacing_ratio" not in DrivenHubbardHamiltonian(n_sites=2).summary()
