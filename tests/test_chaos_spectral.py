import numpy as np
import pytest

from quantum_chaos.chaos import (
    normalized_moments,
    plateau_value,
    rmt_form_factor,
    spectral_form_factor,
    syk_kurtosis_exact,
    syk_kurtosis_garcia_garcia_verbaarschot,
    syk_second_moment,
    thouless_time,
    unfolded_form_factor,
)
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.spectral.rmt import sample_spectrum


@pytest.mark.parametrize("N", [6, 8, 10, 12, 16, 20, 34])
def test_wick_kurtosis_equals_garcia_garcia_verbaarschot_formula(N):
    assert syk_kurtosis_exact(N) == pytest.approx(
        syk_kurtosis_garcia_garcia_verbaarschot(N), abs=1e-12
    )


@pytest.mark.parametrize("N, realisations", [(8, 1500), (12, 300)])
def test_disorder_averaged_moments_match_exact_values(N, realisations):
    m2, m4 = np.array(
        [
            normalized_moments(SYKHamiltonian(N, seed=s, chi_squared=0.5).sparse_hamiltonian)
            for s in range(realisations)
        ]
    ).T
    assert m2.mean() == pytest.approx(syk_second_moment(N, chi_squared=0.5), rel=0.05)
    kurtosis = m4.mean() / m2.mean() ** 2 - 3
    assert kurtosis == pytest.approx(syk_kurtosis_exact(N), abs=0.03)


@pytest.mark.parametrize("N, expected", [(8, 1), (10, 2), (12, 2), (14, 2), (16, 1)])
def test_sff_plateau_follows_protected_degeneracies(N, expected):
    syk = SYKHamiltonian(N, seed=0)
    L = 2 ** (N // 2)
    assert plateau_value(syk.eigenvalues) * L == pytest.approx(expected, rel=1e-9)


def test_late_time_sff_approaches_the_plateau():
    syk = SYKHamiltonian(10, seed=4)
    times = np.random.default_rng(0).uniform(1e4, 1e6, 4000)
    g = spectral_form_factor([syk.eigenvalues], times)["g"]
    assert g.mean() == pytest.approx(plateau_value(syk.eigenvalues), rel=0.08)


def test_sff_starts_at_one_and_connected_part_vanishes_for_one_spectrum():
    syk = SYKHamiltonian(10, seed=1)
    r = spectral_form_factor([syk.eigenvalues], np.array([0.0, 1.0]))
    assert r["g"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(r["g_connected"], 0, atol=1e-12)


@pytest.mark.parametrize(
    "ensemble, dim, samples, taus",
    [
        ("gue", 200, 200, [0.1, 0.3, 0.6, 1.5]),
        ("goe", 200, 200, [0.1, 0.3, 0.6, 1.5]),
        ("gse", 150, 150, [0.1, 0.3, 0.5, 2.5]),
    ],
)
def test_analytic_form_factors_match_sampled_random_matrices(ensemble, dim, samples, taus):
    rng = np.random.default_rng(3)
    spectra = [sample_spectrum(ensemble, dim, rng) for _ in range(samples)]
    taus = np.array(taus)
    res = unfolded_form_factor(spectra, taus)
    ref = rmt_form_factor(taus, ensemble)
    assert np.all(np.abs(res["K"] - ref) < 4 * res["stderr"] + 0.03)


def test_thouless_time_estimator():
    taus = np.linspace(0.05, 1.0, 20)
    ref = rmt_form_factor(taus, "gue")
    K = ref.copy()
    K[taus < 0.3] *= 3
    assert thouless_time(taus, K, "gue") == pytest.approx(taus[taus >= 0.3][0])
    assert np.isnan(thouless_time(taus, 3 * ref, "gue"))
