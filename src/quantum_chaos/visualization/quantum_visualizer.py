import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Optional, Union, Sequence
from pathlib import Path


class QuantumVisualizer:
    def __init__(
        self,
        style: str = "seaborn-v0_8-darkgrid",
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        save_dir: Optional[str] = None,
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir) if save_dir else None

        # Try to use specified style, fall back to default
        try:
            plt.style.use(style)
        except OSError:  # unknown style name
            plt.style.use("default")

        # Color palette for SYK vs Hubbard comparison
        self.colors = {
            "syk": "#E74C3C",
            "hubbard": "#3498DB",
            "ideal": "#2ECC71",  # Green for ideal
            "noisy": "#9B59B6",  # Purple for noisy
            "error": "#E67E22",  # Orange for error
        }

    def _save_figure(self, fig: Figure, name: str):
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / f"{name}.png", dpi=self.dpi, bbox_inches="tight")

    def plot_spectral_density_comparison(
        self,
        syk_eigenvalues: np.ndarray,
        hubbard_eigenvalues: np.ndarray,
        bins: int = 50,
        title: str = "Spectral Density Comparison: SYK vs Hubbard",
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # SYK spectral density
        axes[0].hist(
            syk_eigenvalues,
            bins=bins,
            density=True,
            alpha=0.7,
            color=self.colors["syk"],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[0].set_xlabel("Energy", fontsize=12)
        axes[0].set_ylabel("Density of States", fontsize=12)
        axes[0].set_title("SYK model", fontsize=14, fontweight="bold")
        axes[0].axvline(
            np.mean(syk_eigenvalues),
            color="black",
            linestyle="--",
            label=f"Mean: {np.mean(syk_eigenvalues):.2f}",
        )
        axes[0].legend()

        # Hubbard spectral density
        axes[1].hist(
            hubbard_eigenvalues,
            bins=bins,
            density=True,
            alpha=0.7,
            color=self.colors["hubbard"],
            edgecolor="black",
            linewidth=0.5,
        )
        axes[1].set_xlabel("Energy", fontsize=12)
        axes[1].set_ylabel("Density of States", fontsize=12)
        axes[1].set_title("Hubbard model", fontsize=14, fontweight="bold")
        axes[1].axvline(
            np.mean(hubbard_eigenvalues),
            color="black",
            linestyle="--",
            label=f"Mean: {np.mean(hubbard_eigenvalues):.2f}",
        )
        axes[1].legend()

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, "spectral_density_comparison")
        return fig

    def plot_level_spacing_statistics(
        self,
        ratios: Dict[str, np.ndarray],
        references: Sequence[str] = ("poisson", "goe"),
        bins: int = 25,
        title: str = "Spacing-ratio distributions",
    ) -> Figure:
        """Histograms of r~ = min(s_n, s_{n-1})/max(s_n, s_{n-1}) with RMT surmises.

        ``ratios`` maps a label to the ratios of one or more *symmetry-resolved*
        sectors (e.g. ``SYKHamiltonian.spacing_ratios()``); no unfolding is needed.
        ``references`` are drawn with the Wigner-like surmise of Atas et al. (2013).
        """
        from ..spectral.rmt import ratio_density

        fig, ax = plt.subplots(figsize=self.figsize)
        palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, (label, r) in enumerate(ratios.items()):
            ax.hist(
                r,
                bins=bins,
                range=(0, 1),
                density=True,
                histtype="step",
                linewidth=2,
                color=palette[i % len(palette)],
                label=f"{label} (n={len(r)})",
            )
        grid = np.linspace(0, 1, 201)
        for ens in references:
            ax.plot(
                grid,
                ratio_density(grid, ens),
                "k",
                linestyle={"poisson": ":", "goe": "--", "gue": "-.", "gse": "-"}[ens.lower()],
                label=f"{ens.upper()} (surmise)",
            )
        ax.set_xlabel(r"$\tilde r$", fontsize=12)
        ax.set_ylabel(r"$P(\tilde r)$", fontsize=12)
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)

        plt.tight_layout()
        self._save_figure(fig, "level_spacing_statistics")
        return fig

    def plot_fidelity_evolution(
        self,
        times: np.ndarray,
        syk_fidelity: np.ndarray,
        hubbard_fidelity: np.ndarray,
        title: str = "Fidelity Evolution Comparison",
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)

        ax.plot(times, syk_fidelity, "-", color=self.colors["syk"], linewidth=2, label="SYK")
        ax.plot(
            times, hubbard_fidelity, "-", color=self.colors["hubbard"], linewidth=2, label="Hubbard"
        )

        ax.set_xlabel("Time", fontsize=12)
        ax.set_ylabel(r"Fidelity $F(t) = |\langle\psi(0)|\psi(t)\rangle|^2$", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, "fidelity_evolution")
        return fig

    def plot_otoc_comparison(
        self,
        times: np.ndarray,
        syk_otoc: np.ndarray,
        hubbard_otoc: np.ndarray,
        title: str = "OTOC Comparison: Scrambling Dynamics",
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Real part of OTOC
        axes[0].plot(
            times, np.real(syk_otoc), "-", color=self.colors["syk"], linewidth=2, label="SYK"
        )
        axes[0].plot(
            times,
            np.real(hubbard_otoc),
            "-",
            color=self.colors["hubbard"],
            linewidth=2,
            label="Hubbard",
        )
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_ylabel("Re[OTOC(t)]", fontsize=12)
        axes[0].set_title("OTOC Real Part", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 1 - Re[OTOC] on log scale (shows exponential scrambling)
        deviation_syk = 1 - np.real(syk_otoc)
        deviation_hub = 1 - np.real(hubbard_otoc)

        # Avoid log(0)
        deviation_syk = np.maximum(deviation_syk, 1e-15)
        deviation_hub = np.maximum(deviation_hub, 1e-15)

        axes[1].semilogy(
            times, deviation_syk, "-", color=self.colors["syk"], linewidth=2, label="SYK"
        )
        axes[1].semilogy(
            times, deviation_hub, "-", color=self.colors["hubbard"], linewidth=2, label="Hubbard"
        )
        axes[1].set_xlabel("Time", fontsize=12)
        axes[1].set_ylabel("1 - Re[OTOC(t)]", fontsize=12)
        axes[1].set_title("Scrambling (Log Scale)", fontsize=14, fontweight="bold")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, "otoc_comparison")
        return fig

    def plot_error_rate_comparison(
        self,
        noise_levels: List[float],
        syk_errors: List[float],
        hubbard_errors: List[float],
        title: str = "NISQ Error Rate: Ideal vs Noisy Simulation",
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)

        x = np.arange(len(noise_levels))
        width = 0.35

        bars1 = ax.bar(
            x - width / 2, syk_errors, width, label="SYK", color=self.colors["syk"], alpha=0.8
        )
        bars2 = ax.bar(
            x + width / 2,
            hubbard_errors,
            width,
            label="Hubbard",
            color=self.colors["hubbard"],
            alpha=0.8,
        )

        ax.set_xlabel("Depolarizing Rate", fontsize=12)
        ax.set_ylabel("Energy Error", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{p:.3f}" for p in noise_levels])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        self._save_figure(fig, "error_rate_comparison")
        return fig

    def plot_ideal_vs_noisy_evolution(
        self, comparison_results: Dict, title: str = "Ideal vs NISQ Evolution"
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        times = comparison_results["times"]

        # Expectation value comparison
        axes[0].plot(
            times,
            comparison_results["ideal_expectation"],
            "-",
            color=self.colors["ideal"],
            linewidth=2,
            label="Ideal",
        )
        axes[0].plot(
            times,
            comparison_results["noisy_expectation"],
            "--",
            color=self.colors["noisy"],
            linewidth=2,
            label="NISQ",
        )
        axes[0].set_xlabel("Time", fontsize=12)
        axes[0].set_ylabel(r"$\langle O\rangle$", fontsize=12)
        axes[0].set_title("Observable Expectation", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Fidelity decay
        axes[1].plot(
            times, comparison_results["fidelity"], "-", color=self.colors["error"], linewidth=2
        )
        axes[1].set_xlabel("Time", fontsize=12)
        axes[1].set_ylabel("Fidelity", fontsize=12)
        axes[1].set_title("State Fidelity Decay", fontsize=14, fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)

        # Expectation error
        axes[2].plot(
            times,
            comparison_results["expectation_error"],
            "-",
            color=self.colors["error"],
            linewidth=2,
        )
        axes[2].set_xlabel("Time", fontsize=12)
        axes[2].set_ylabel(
            r"$|\langle O\rangle_{\rm ideal} - \langle O\rangle_{\rm noisy}|$", fontsize=12
        )
        axes[2].set_title("Expectation Value Error", fontsize=14, fontweight="bold")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, "ideal_vs_noisy_evolution")
        return fig

    def plot_qml_comparison(
        self, benchmark_results: Dict, title: str = "Kernel SVM accuracy: SYK vs Hubbard kernels"
    ) -> Figure:
        """Per-trial accuracies and mean +/- std. Descriptive only: no significance test."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        trials = np.arange(1, benchmark_results["n_trials"] + 1)
        axes[0].plot(
            trials,
            benchmark_results["syk_accuracies"],
            "o-",
            color=self.colors["syk"],
            linewidth=2,
            markersize=8,
            label="SYK kernel",
        )
        axes[0].plot(
            trials,
            benchmark_results["hubbard_accuracies"],
            "s-",
            color=self.colors["hubbard"],
            linewidth=2,
            markersize=8,
            label="Hubbard kernel",
        )
        axes[0].set_xlabel("Trial", fontsize=12)
        axes[0].set_ylabel("Test accuracy", fontsize=12)
        axes[0].set_title("Accuracy per trial", fontsize=14, fontweight="bold")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)

        means = [benchmark_results["syk_mean"], benchmark_results["hubbard_mean"]]
        stds = [benchmark_results["syk_std"], benchmark_results["hubbard_std"]]
        axes[1].bar(
            [0, 1],
            means,
            yerr=stds,
            color=[self.colors["syk"], self.colors["hubbard"]],
            alpha=0.8,
            capsize=5,
            edgecolor="black",
            linewidth=1.5,
        )
        axes[1].set_xticks([0, 1])
        axes[1].set_xticklabels(["SYK", "Hubbard"], fontsize=11)
        axes[1].set_ylabel("Mean accuracy (± std)", fontsize=12)
        axes[1].set_title(f"{benchmark_results['n_trials']} trials", fontsize=14, fontweight="bold")
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3, axis="y")

        fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, "qml_comparison")
        return fig

    def plot_trotter_error(
        self, error_analysis: Dict, title: str = "Trotter Decomposition Error Analysis"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)

        steps = error_analysis["steps"]

        # State error is O(t^2/n) (order 1) and O(t^3/n^2) (order 2), so the
        # infidelity 1 - F ~ |error|^2 scales as n^-2 and n^-4.
        ax.loglog(
            steps,
            error_analysis["error_order1"],
            "o-",
            color=self.colors["syk"],
            linewidth=2,
            markersize=8,
            label="First order",
        )
        ax.loglog(
            steps,
            error_analysis["error_order2"],
            "s-",
            color=self.colors["hubbard"],
            linewidth=2,
            markersize=8,
            label="Second order",
        )

        # Reference slopes anchored at the first data point
        steps_arr = np.array(steps, dtype=float)
        e1, e2 = error_analysis["error_order1"][0], error_analysis["error_order2"][0]
        ax.loglog(
            steps_arr,
            e1 * (steps_arr / steps_arr[0]) ** -2,
            "k--",
            alpha=0.5,
            label=r"$\propto n^{-2}$",
        )
        ax.loglog(
            steps_arr,
            e2 * (steps_arr / steps_arr[0]) ** -4,
            "k:",
            alpha=0.5,
            label=r"$\propto n^{-4}$",
        )

        ax.set_xlabel("Number of Trotter Steps", fontsize=12)
        ax.set_ylabel("Infidelity (1 - F)", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self._save_figure(fig, "trotter_error")
        return fig

    def create_summary_dashboard(
        self,
        syk_data: Dict,
        hubbard_data: Dict,
        nisq_data: Optional[Dict] = None,
        qml_data: Optional[Dict] = None,
        title: str = "Quantum Dynamics Comparison Dashboard",
    ) -> Figure:
        n_cols = 3
        n_rows = 2 if (nisq_data or qml_data) else 1

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        # Row 1: Spectral and OTOC
        # Spectral density
        ax = axes[0, 0]
        if "eigenvalues" in syk_data:
            ax.hist(
                syk_data["eigenvalues"],
                bins=30,
                density=True,
                alpha=0.6,
                color=self.colors["syk"],
                label="SYK",
            )
        if "eigenvalues" in hubbard_data:
            ax.hist(
                hubbard_data["eigenvalues"],
                bins=30,
                density=True,
                alpha=0.6,
                color=self.colors["hubbard"],
                label="Hubbard",
            )
        ax.set_xlabel("Energy")
        ax.set_ylabel("Density")
        ax.set_title("Spectral Density", fontweight="bold")
        ax.legend()

        # Level spacing ratio
        ax = axes[0, 1]
        x = [0, 1]
        lsr_values = [
            syk_data.get("level_spacing_ratio", 0),
            hubbard_data.get("level_spacing_ratio", 0),
        ]
        ax.bar(x, lsr_values, color=[self.colors["syk"], self.colors["hubbard"]], alpha=0.8)
        from ..spectral.rmt import R_MEAN

        ax.axhline(
            R_MEAN["poisson"],
            color="gray",
            linestyle=":",
            label=f"Poisson ({R_MEAN['poisson']:.4f})",
        )
        for ens, ls in (("goe", "--"), ("gue", "-."), ("gse", "-")):
            ax.axhline(
                R_MEAN[ens],
                color="black",
                linestyle=ls,
                linewidth=0.8,
                label=f"{ens.upper()} ({R_MEAN[ens]:.4f})",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(["SYK", "Hubbard"])
        ax.set_ylabel(r"$\langle r\rangle$")
        ax.set_title("Level Spacing Ratio", fontweight="bold")
        ax.legend(fontsize=9)

        # Summary text
        ax = axes[0, 2]
        ax.axis("off")
        summary_text = "SUMMARY\n" + "=" * 40 + "\n\n"

        if "ground_state_energy" in syk_data:
            summary_text += f"SYK Ground State: {syk_data['ground_state_energy']:.4f}\n"
        if "ground_state_energy" in hubbard_data:
            summary_text += f"Hubbard Ground State: {hubbard_data['ground_state_energy']:.4f}\n"
        summary_text += f"\nSYK <r>: {lsr_values[0]:.4f}\n"
        summary_text += f"Hubbard <r>: {lsr_values[1]:.4f}\n"

        ax.text(
            0.1,
            0.9,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax.set_title("Analysis Summary", fontweight="bold")

        # Row 2: NISQ and QML (if available)
        if n_rows > 1 and (nisq_data or qml_data):
            if nisq_data:
                ax = axes[1, 0]
                if "noise_levels" in nisq_data:
                    ax.plot(
                        nisq_data["noise_levels"],
                        nisq_data.get("energy_variance", []),
                        "o-",
                        color=self.colors["noisy"],
                        linewidth=2,
                    )
                    ax.set_xlabel("Noise Level")
                    ax.set_ylabel("Energy Variance")
                    ax.set_title("NISQ Energy Uncertainty", fontweight="bold")

            if qml_data:
                ax = axes[1, 1]
                if "syk_mean" in qml_data:
                    means = [qml_data["syk_mean"], qml_data["hubbard_mean"]]
                    stds = [qml_data["syk_std"], qml_data["hubbard_std"]]
                    ax.bar(
                        [0, 1],
                        means,
                        yerr=stds,
                        color=[self.colors["syk"], self.colors["hubbard"]],
                        capsize=5,
                        alpha=0.8,
                    )
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(["SYK", "Hubbard"])
                    ax.set_ylabel("Accuracy")
                    ax.set_title("QML Classification", fontweight="bold")
                    ax.set_ylim(0, 1.1)

            axes[1, 2].axis("off")

        fig.suptitle(title, fontsize=18, fontweight="bold", y=1.02)
        plt.tight_layout()

        self._save_figure(fig, "summary_dashboard")
        return fig
