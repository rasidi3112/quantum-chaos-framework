"""Smoke tests: every plotting routine runs and saves a figure (Agg backend)."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from quantum_chaos.visualization.quantum_visualizer import QuantumVisualizer  # noqa: E402


@pytest.fixture
def viz(tmp_path):
    yield QuantumVisualizer(save_dir=str(tmp_path))
    plt.close("all")


def test_all_plots_render_and_save(viz, tmp_path):
    rng = np.random.default_rng(0)
    e1, e2 = np.sort(rng.normal(size=64)), np.sort(rng.normal(size=64))
    times = np.linspace(0, 2, 11)
    viz.plot_spectral_density_comparison(e1, e2, bins=10)
    viz.plot_level_spacing_statistics(e1, e2)
    viz.plot_fidelity_evolution(times, np.exp(-times), np.exp(-times / 2))
    viz.plot_otoc_comparison(times, np.cos(times), np.cos(times / 2))
    viz.plot_error_rate_comparison([0.001, 0.01], [0.1, 0.2], [0.05, 0.1])
    viz.plot_ideal_vs_noisy_evolution({
        "times": times, "ideal_expectation": np.cos(times), "noisy_expectation": 0.9 * np.cos(times),
        "fidelity": np.exp(-times), "expectation_error": 0.1 * np.abs(np.cos(times)),
    })
    accs = {"n_trials": 3, "syk_accuracies": [0.8, 0.7, 0.9], "hubbard_accuracies": [0.6, 0.7, 0.8],
            "syk_mean": 0.8, "syk_std": 0.08, "hubbard_mean": 0.7, "hubbard_std": 0.08}
    viz.plot_qml_comparison(accs)
    viz.plot_trotter_error({"steps": [1, 2, 5], "error_order1": [1e-1, 2.5e-2, 4e-3],
                            "error_order2": [1e-2, 6e-4, 1.6e-5], "total_time": 1.0})
    viz.create_summary_dashboard(
        {"eigenvalues": e1, "level_spacing_ratio": 0.53, "ground_state_energy": e1[0]},
        {"eigenvalues": e2, "level_spacing_ratio": 0.39, "ground_state_energy": e2[0]},
        nisq_data={"noise_levels": [0.0, 0.01], "energy_variance": [0.0, 0.1]},
        qml_data=accs,
    )
    saved = {p.stem for p in tmp_path.glob("*.png")}
    assert {"spectral_density_comparison", "level_spacing_statistics", "fidelity_evolution",
            "otoc_comparison", "error_rate_comparison", "ideal_vs_noisy_evolution",
            "qml_comparison", "trotter_error", "summary_dashboard"} <= saved
