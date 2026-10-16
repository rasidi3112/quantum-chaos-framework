import numpy as np
import pytest
from scipy.linalg import expm
from scipy.sparse import csc_matrix

from quantum_chaos.chaos.kpm import chebyshev_evolve, spectral_bounds
from quantum_chaos.chaos.otoc import thermal_otoc
from quantum_chaos.chaos.typicality import SectorSYK, typical_otoc
from quantum_chaos.hamiltonians import SYKHamiltonian
from quantum_chaos.hamiltonians.syk_sparse import syk_sector_hamiltonian


@pytest.fixture(scope="module")
def sector():
    H = syk_sector_hamiltonian(14, seed=4, chi_squared=0.5)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=H.shape[0]) + 1j * rng.normal(size=H.shape[0])
    return H, psi / np.linalg.norm(psi), spectral_bounds(H)


@pytest.mark.parametrize("t", [0.0, 0.3, 2.0, 11.0])
def test_real_time_chebyshev_matches_dense_exponential(sector, t):
    H, psi, bounds = sector
    exact = expm(-1j * t * H.toarray()) @ psi
    np.testing.assert_allclose(chebyshev_evolve(H, psi, t, bounds), exact, atol=1e-11)


@pytest.mark.parametrize("tau", [0.0, 0.5, 5.0, 40.0])
def test_imaginary_time_chebyshev_matches_dense_exponential(sector, tau):
    H, psi, bounds = sector
    exact = expm(-tau * H.toarray()) @ psi
    got = chebyshev_evolve(H, psi, tau, bounds, imaginary=True)
    np.testing.assert_allclose(got, exact, rtol=1e-10, atol=1e-12 * np.linalg.norm(exact))


def test_chebyshev_rejects_bounds_that_miss_the_spectrum(sector):
    H, psi, (lo, hi) = sector
    w = hi - lo
    with pytest.raises(ValueError):
        chebyshev_evolve(H, psi, 11.0, (lo + 0.4 * w, hi - 0.4 * w))
    with pytest.raises(ValueError):
        chebyshev_evolve(H, psi, 5.0, (lo + 0.4 * w, hi - 0.4 * w), imaginary=True)
    with pytest.raises(ValueError):
        chebyshev_evolve(H, psi, -1.0, (lo, hi), imaginary=True)


def test_chebyshev_handles_a_dense_hermitian_matrix():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(30, 30)) + 1j * rng.normal(size=(30, 30))
    H = (A + A.conj().T) / 2
    psi = rng.normal(size=30) + 0j
    got = chebyshev_evolve(csc_matrix(H), psi, 1.7, spectral_bounds(csc_matrix(H)))
    np.testing.assert_allclose(got, expm(-1.7j * H) @ psi, atol=1e-11)


def test_otoc_averaged_over_several_partners_equals_the_mean_of_single_partners():
    model = SectorSYK(12, seed=1, chi_squared=0.5)
    times = np.array([0.0, 0.7, 1.9])
    for beta in (0.0, 3.0):
        joint = typical_otoc(model, 0, [1, 2, 5], times, n_vectors=2, beta=beta, rng=7)
        single = [
            typical_otoc(model, 0, v, times, n_vectors=2, beta=beta, rng=7)["mean"]
            for v in (1, 2, 5)
        ]
        np.testing.assert_allclose(joint["mean"], np.mean(single, axis=0), atol=1e-12)


def test_regularised_otoc_with_full_basis_trace_is_exact():
    N = 10
    model = SectorSYK(N, seed=2, chi_squared=0.5)
    syk = SYKHamiltonian(N, seed=2, chi_squared=0.5)
    half = [syk.get_majorana_operator(a) * np.sqrt(0.5) for a in (0, 3)]
    times = np.array([0.0, 0.8, 2.2])
    exact = thermal_otoc(syk.hamiltonian_matrix, half[0], half[1], times, beta=4.0)
    est = typical_otoc(model, 0, 3, times, n_vectors=200, beta=4.0, rng=3)
    assert np.all(np.abs(est["mean"] - exact) < 5 * est["stderr"] + 1e-3)


def test_finite_size_extrapolation_recovers_the_large_n_curve():
    from quantum_chaos.chaos.otoc import finite_size_extrapolation

    t = np.linspace(0, 3, 16)
    sizes = np.array([20, 24, 28, 32])
    g_inf = np.exp(0.8 * t) - 1
    curves = np.array([g_inf + np.sin(t) * 5 / n for n in sizes])
    out = finite_size_extrapolation(sizes, curves, 0.01 * np.ones_like(curves))
    np.testing.assert_allclose(out["value"], g_inf, atol=1e-10)
    np.testing.assert_allclose(out["slope"], np.sin(t) * 5, atol=1e-9)
    assert np.all(out["stderr"] > 0)
    assert np.allclose(out["chi2_per_dof"], 0, atol=1e-12)


def test_local_exponent_of_exponential_and_power_law():
    from quantum_chaos.chaos.otoc import local_exponent

    t = np.linspace(0.5, 3, 26)
    np.testing.assert_allclose(local_exponent(t, 3 * np.exp(0.7 * t)), 0.7, atol=1e-12)
    np.testing.assert_allclose(local_exponent(t, t**2), 2 / t, rtol=2e-2)


def test_crossing_time_interpolates_log_decay():
    from quantum_chaos.chaos.otoc import crossing_time

    t = np.linspace(0, 4, 21)
    decay = 0.01 * np.exp(1.1 * t)
    assert crossing_time(t, decay, 0.25) == pytest.approx(np.log(25) / 1.1, rel=1e-12)
    assert np.isnan(crossing_time(t, decay, 5e3))


def test_rescaling_exponent_is_exact_for_the_large_n_ansatz():
    from quantum_chaos.chaos.otoc import crossing_time, rescaling_exponents

    t = np.linspace(0, 6, 61)
    sizes = np.array([20, 24, 28, 32])
    lam = 0.9
    times = [crossing_time(t, 1.3 * np.exp(lam * t) / n, 0.25) for n in sizes]
    out = rescaling_exponents(sizes, times)
    np.testing.assert_allclose(out["exponent"], lam, rtol=1e-10)
    np.testing.assert_allclose(out["size"], np.sqrt(sizes[1:] * sizes[:-1]))


def test_rescaling_exponent_of_a_power_law_drifts_to_zero():
    from quantum_chaos.chaos.otoc import crossing_time, rescaling_exponents

    t = np.linspace(0.01, 12, 1200)
    sizes = np.array([20, 40, 80, 160])
    times = [crossing_time(t, 2 * t**2 / n, 0.25) for n in sizes]
    lam = rescaling_exponents(sizes, times)["exponent"]
    assert np.all(np.diff(lam) < 0)
    np.testing.assert_allclose(
        lam * np.sqrt(np.sqrt(sizes[1:] * sizes[:-1])),
        lam[0] * np.sqrt(np.sqrt(sizes[1] * sizes[0])),
        rtol=0.02,
    )


def _synthetic_results(lam, sizes, betas=(0.0, 4.0), n_real=5, noise=0.0):
    rng = np.random.default_rng(0)
    t = np.linspace(0, 6, 61)
    runs = []
    for N in sizes:
        for s in range(n_real):
            F = 1 - 1.4 * np.exp(lam * t) / N
            F = F * (1 + noise * rng.normal())
            otoc = {str(float(b)): [F.tolist(), (0 * F).tolist()] for b in betas}
            runs.append({"N": N, "seed": s, "otoc": otoc})
    return {"betas": list(betas), "times": t.tolist(), "runs": runs}


def test_lyapunov_analysis_recovers_the_exponent_of_the_ansatz():
    from quantum_chaos.experiments.lyapunov_figures import lyapunov_analysis

    out = lyapunov_analysis(_synthetic_results(0.7, list(range(20, 36, 2))), n_bootstrap=20)
    for beta in ("0.0", "4.0"):
        for thr in (0.25, 0.16):
            r = out["betas"][beta][f"threshold_{thr}"]
            n = np.array(r["pair_size"])
            bias = 1 + 1.4 * (1 - thr) / (thr * n)
            np.testing.assert_allclose(r["exponent"], 0.7 * bias, rtol=2e-3)
            assert r["extrapolated"] == pytest.approx(0.7, rel=1e-2)
            assert r["global"] > 0.7 * 1.05
            assert r["corrected"] == pytest.approx(0.7, rel=5e-2)
            assert r["extrapolated_err"] < 1e-6


def test_corrected_rescaling_fit_removes_the_leading_finite_size_bias():
    from quantum_chaos.chaos.otoc import rescaling_fit

    sizes = np.arange(20, 36, 2)
    lam, c, thr = 0.8, 1.4, 0.25
    tc = np.log(1 + thr * (sizes - c) / c) / lam
    out = rescaling_fit(sizes, tc, 0.01 * np.ones(sizes.size))
    assert out["exponent"] == pytest.approx(lam, rel=3e-2)
    assert out["exponent_err"] > 0
    plain = rescaling_fit(sizes, tc, 0.01 * np.ones(sizes.size), correction=False)
    assert plain["exponent"] > 1.1 * lam


def test_lyapunov_analysis_drops_sizes_below_the_validity_cut():
    from quantum_chaos.experiments.lyapunov_figures import lyapunov_analysis

    res = _synthetic_results(0.7, list(range(20, 36, 2)))
    out = lyapunov_analysis(res, n_bootstrap=5, min_sizes={4.0: 24})
    assert out["betas"]["4.0"]["sizes"] == list(range(24, 36, 2))
    assert out["betas"]["0.0"]["sizes"] == list(range(20, 36, 2))
    assert out["betas"]["4.0"]["min_size"] == 24


def test_min_sizes_are_parsed_from_yaml_mapping():
    from quantum_chaos.experiments.config import figure_config_from_dict

    cfg = figure_config_from_dict(
        {
            "name": "x",
            "kind": "syk_lyapunov",
            "output": "figures/x.png",
            "lyapunov": {"min_sizes": {"10": 24}},
        }
    )
    assert cfg.lyapunov.min_sizes == {10.0: 24}


def test_power_law_fit_of_crossing_times():
    from quantum_chaos.chaos.otoc import power_law_fit

    sizes = np.arange(20, 36, 2)
    tc = 0.7 * sizes ** (1 / 2.5)
    out = power_law_fit(sizes, tc, 0.01 * np.ones(sizes.size))
    assert out["power"] == pytest.approx(1 / 2.5, rel=1e-10)
    assert out["chi2_per_dof"] == pytest.approx(0, abs=1e-12)
