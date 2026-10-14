"""Spacing-ratio statistics and RMT references (PLAN P1-1, P1-2)."""

import numpy as np
import pytest

from quantum_chaos.spectral import (
    DegenerateSpectrumWarning, R_MEAN, ensemble_mean_r, remove_kramers_pairs, spacing_ratios,
)


def test_exact_degeneracies_are_kept_and_reported():
    with pytest.warns(DegenerateSpectrumWarning):
        r = spacing_ratios([0.0, 1.0, 1.0, 3.0])
    np.testing.assert_allclose(r, [0.0, 0.0])


def test_kramers_pairs_reduce_to_unique_levels():
    unique = np.array([0.0, 0.7, 1.9, 2.2])
    np.testing.assert_allclose(remove_kramers_pairs(np.repeat(unique, 2)), unique)
    np.testing.assert_allclose(spacing_ratios(np.repeat(unique, 2), kramers=True),
                               spacing_ratios(unique))
    with pytest.raises(ValueError):
        remove_kramers_pairs([0.0, 0.5, 1.0, 1.5])


@pytest.mark.parametrize("ensemble", ["poisson", "goe", "gue", "gse"])
def test_sampled_ensembles_reproduce_atas_values(ensemble):
    # dim 64, no trimming: measured finite-size offsets are below 5e-4 (PLAN §8)
    res = ensemble_mean_r(ensemble, dim=64, n_samples=600, rng=7)
    assert abs(res["mean"] - R_MEAN[ensemble]) < 3 * res["stderr"] + 1e-3


@pytest.mark.slow
def test_hermitian_eigh_handles_large_complex_matrices():
    # zheevd from Apple Accelerate fails for n >~ 3000 (see helpers.hermitian_eigh)
    from quantum_chaos.utils.helpers import hermitian_eigh

    rng = np.random.default_rng(0)
    a = rng.standard_normal((3200, 3200)) + 1j * rng.standard_normal((3200, 3200))
    a = (a + a.conj().T) / 2
    w, v = hermitian_eigh(a)
    np.testing.assert_allclose(a @ v[:, :5], v[:, :5] * w[:5], atol=1e-8)


@pytest.mark.parametrize("ensemble", ["poisson", "goe", "gue", "gse"])
def test_surmise_density_is_normalised_and_reproduces_table_means(ensemble):
    from scipy.integrate import quad

    from quantum_chaos.spectral import R_SURMISE, ratio_density

    norm = quad(lambda r: ratio_density(r, ensemble), 0, 1)[0]
    mean = quad(lambda r: r * ratio_density(r, ensemble), 0, 1)[0]
    assert norm == pytest.approx(1.0, abs=1e-10)
    assert mean == pytest.approx(R_SURMISE[ensemble], abs=1e-10)
