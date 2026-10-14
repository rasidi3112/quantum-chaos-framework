

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path


class QuantumVisualizer:
    def __init__(
        self,
        style: str = 'seaborn-v0_8-darkgrid',
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 150,
        save_dir: Optional[str] = None
    ):
        self.figsize = figsize
        self.dpi = dpi
        self.save_dir = Path(save_dir) if save_dir else None
        
        # Try to use specified style, fall back to default
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        # Color palette for SYK vs Hubbard comparison
        self.colors = {
            'syk': '#E74C3C',       # Red for chaotic
            'hubbard': '#3498DB',   # Blue for regular
            'ideal': '#2ECC71',     # Green for ideal
            'noisy': '#9B59B6',     # Purple for noisy
            'error': '#E67E22'      # Orange for error
        }
    
    def _save_figure(self, fig: Figure, name: str):
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.save_dir / f"{name}.png", dpi=self.dpi, bbox_inches='tight')
    
    def plot_spectral_density_comparison(
        self,
        syk_eigenvalues: np.ndarray,
        hubbard_eigenvalues: np.ndarray,
        bins: int = 50,
        title: str = "Spectral Density Comparison: SYK vs Hubbard"
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # SYK spectral density
        axes[0].hist(
            syk_eigenvalues, 
            bins=bins, 
            density=True, 
            alpha=0.7,
            color=self.colors['syk'],
            edgecolor='black',
            linewidth=0.5
        )
        axes[0].set_xlabel('Energy', fontsize=12)
        axes[0].set_ylabel('Density of States', fontsize=12)
        axes[0].set_title('SYK Model (Chaotic)', fontsize=14, fontweight='bold')
        axes[0].axvline(
            np.mean(syk_eigenvalues), 
            color='black', 
            linestyle='--', 
            label=f'Mean: {np.mean(syk_eigenvalues):.2f}'
        )
        axes[0].legend()
        
        # Hubbard spectral density
        axes[1].hist(
            hubbard_eigenvalues, 
            bins=bins, 
            density=True, 
            alpha=0.7,
            color=self.colors['hubbard'],
            edgecolor='black',
            linewidth=0.5
        )
        axes[1].set_xlabel('Energy', fontsize=12)
        axes[1].set_ylabel('Density of States', fontsize=12)
        axes[1].set_title('Hubbard Model (Regular)', fontsize=14, fontweight='bold')
        axes[1].axvline(
            np.mean(hubbard_eigenvalues), 
            color='black', 
            linestyle='--', 
            label=f'Mean: {np.mean(hubbard_eigenvalues):.2f}'
        )
        axes[1].legend()
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'spectral_density_comparison')
        return fig
    
    def plot_level_spacing_statistics(
        self,
        syk_eigenvalues: np.ndarray,
        hubbard_eigenvalues: np.ndarray,
        title: str = "Level Spacing Statistics"
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Compute level spacings
        syk_spacings = np.diff(np.sort(syk_eigenvalues))
        syk_spacings = syk_spacings / np.mean(syk_spacings)  # Unfold
        
        hubbard_spacings = np.diff(np.sort(hubbard_eigenvalues))
        hubbard_spacings = hubbard_spacings / np.mean(hubbard_spacings)
        
        # Reference distributions
        s = np.linspace(0, 4, 100)
        poisson = np.exp(-s)
        goe = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
        
        # SYK
        axes[0].hist(syk_spacings, bins=30, density=True, alpha=0.7, 
                     color=self.colors['syk'], label='SYK')
        axes[0].plot(s, poisson, 'k--', label='Poisson (Integrable)', linewidth=2)
        axes[0].plot(s, goe, 'k-', label='GOE (Chaotic)', linewidth=2)
        axes[0].set_xlabel('Normalized Spacing s', fontsize=12)
        axes[0].set_ylabel('P(s)', fontsize=12)
        axes[0].set_title('SYK Level Spacing', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].set_xlim(0, 4)
        
        # Hubbard
        axes[1].hist(hubbard_spacings, bins=30, density=True, alpha=0.7,
                     color=self.colors['hubbard'], label='Hubbard')
        axes[1].plot(s, poisson, 'k--', label='Poisson (Integrable)', linewidth=2)
        axes[1].plot(s, goe, 'k-', label='GOE (Chaotic)', linewidth=2)
        axes[1].set_xlabel('Normalized Spacing s', fontsize=12)
        axes[1].set_ylabel('P(s)', fontsize=12)
        axes[1].set_title('Hubbard Level Spacing', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].set_xlim(0, 4)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'level_spacing_statistics')
        return fig
    
    def plot_fidelity_evolution(
        self,
        times: np.ndarray,
        syk_fidelity: np.ndarray,
        hubbard_fidelity: np.ndarray,
        title: str = "Fidelity Evolution Comparison"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(times, syk_fidelity, '-', color=self.colors['syk'], 
                linewidth=2, label='SYK (Chaotic)')
        ax.plot(times, hubbard_fidelity, '-', color=self.colors['hubbard'], 
                linewidth=2, label='Hubbard (Regular)')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Fidelity F(t) = |⟨ψ(0)|ψ(t)⟩|²', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'fidelity_evolution')
        return fig
    
    def plot_otoc_comparison(
        self,
        times: np.ndarray,
        syk_otoc: np.ndarray,
        hubbard_otoc: np.ndarray,
        title: str = "OTOC Comparison: Scrambling Dynamics"
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Real part of OTOC
        axes[0].plot(times, np.real(syk_otoc), '-', color=self.colors['syk'], 
                     linewidth=2, label='SYK')
        axes[0].plot(times, np.real(hubbard_otoc), '-', color=self.colors['hubbard'], 
                     linewidth=2, label='Hubbard')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('Re[OTOC(t)]', fontsize=12)
        axes[0].set_title('OTOC Real Part', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 1 - Re[OTOC] on log scale (shows exponential scrambling)
        deviation_syk = 1 - np.real(syk_otoc)
        deviation_hub = 1 - np.real(hubbard_otoc)
        
        # Avoid log(0)
        deviation_syk = np.maximum(deviation_syk, 1e-15)
        deviation_hub = np.maximum(deviation_hub, 1e-15)
        
        axes[1].semilogy(times, deviation_syk, '-', color=self.colors['syk'], 
                         linewidth=2, label='SYK (expect exponential)')
        axes[1].semilogy(times, deviation_hub, '-', color=self.colors['hubbard'], 
                         linewidth=2, label='Hubbard')
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('1 - Re[OTOC(t)]', fontsize=12)
        axes[1].set_title('Scrambling (Log Scale)', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'otoc_comparison')
        return fig
    
    def plot_error_rate_comparison(
        self,
        noise_levels: List[float],
        syk_errors: List[float],
        hubbard_errors: List[float],
        title: str = "NISQ Error Rate: Ideal vs Noisy Simulation"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x = np.arange(len(noise_levels))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, syk_errors, width, label='SYK',
                       color=self.colors['syk'], alpha=0.8)
        bars2 = ax.bar(x + width/2, hubbard_errors, width, label='Hubbard',
                       color=self.colors['hubbard'], alpha=0.8)
        
        ax.set_xlabel('Depolarizing Rate', fontsize=12)
        ax.set_ylabel('Energy Error', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p:.3f}' for p in noise_levels])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        self._save_figure(fig, 'error_rate_comparison')
        return fig
    
    def plot_spectral_stability(
        self,
        benchmark_results: Dict,
        title: str = "Spectral Stability Under Noise"
    ) -> Figure:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        noise_levels = benchmark_results['noise_levels']
        
        # Energy variance vs noise
        axes[0].plot(noise_levels, benchmark_results['energy_variance'], 
                     'o-', color=self.colors['error'], linewidth=2, markersize=8)
        axes[0].set_xlabel('Depolarizing Rate', fontsize=12)
        axes[0].set_ylabel('Energy Variance', fontsize=12)
        axes[0].set_title('Energy Measurement Uncertainty', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Ground state error vs noise
        axes[1].plot(noise_levels, benchmark_results['ground_state_error'],
                     's-', color=self.colors['noisy'], linewidth=2, markersize=8)
        axes[1].set_xlabel('Depolarizing Rate', fontsize=12)
        axes[1].set_ylabel('Ground State Energy Error', fontsize=12)
        axes[1].set_title('Ground State Estimation Error', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'spectral_stability')
        return fig
    
    def plot_ideal_vs_noisy_evolution(
        self,
        comparison_results: Dict,
        title: str = "Ideal vs NISQ Evolution"
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        times = comparison_results['times']
        
        # Expectation value comparison
        axes[0].plot(times, comparison_results['ideal_expectation'], 
                     '-', color=self.colors['ideal'], linewidth=2, label='Ideal')
        axes[0].plot(times, comparison_results['noisy_expectation'],
                     '--', color=self.colors['noisy'], linewidth=2, label='NISQ')
        axes[0].set_xlabel('Time', fontsize=12)
        axes[0].set_ylabel('⟨O⟩', fontsize=12)
        axes[0].set_title('Observable Expectation', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Fidelity decay
        axes[1].plot(times, comparison_results['fidelity'],
                     '-', color=self.colors['error'], linewidth=2)
        axes[1].set_xlabel('Time', fontsize=12)
        axes[1].set_ylabel('Fidelity', fontsize=12)
        axes[1].set_title('State Fidelity Decay', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True, alpha=0.3)
        
        # Expectation error
        axes[2].plot(times, comparison_results['expectation_error'],
                     '-', color=self.colors['error'], linewidth=2)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].set_ylabel('|⟨O⟩_ideal - ⟨O⟩_noisy|', fontsize=12)
        axes[2].set_title('Expectation Value Error', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'ideal_vs_noisy_evolution')
        return fig
    
    def plot_qml_comparison(
        self,
        benchmark_results: Dict,
        title: str = "QML Performance: Chaos vs Regular Dynamics"
    ) -> Figure:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Accuracy comparison
        trials = np.arange(1, benchmark_results['n_trials'] + 1)
        axes[0].plot(trials, benchmark_results['syk_accuracies'], 
                     'o-', color=self.colors['syk'], linewidth=2, 
                     markersize=8, label='SYK (Chaotic)')
        axes[0].plot(trials, benchmark_results['hubbard_accuracies'],
                     's-', color=self.colors['hubbard'], linewidth=2,
                     markersize=8, label='Hubbard (Regular)')
        axes[0].axhline(benchmark_results['syk_mean'], color=self.colors['syk'], 
                        linestyle='--', alpha=0.5)
        axes[0].axhline(benchmark_results['hubbard_mean'], color=self.colors['hubbard'], 
                        linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Trial', fontsize=12)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Classification Accuracy per Trial', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(0, 1.05)
        
        # Bar plot of mean accuracy with error bars
        means = [benchmark_results['syk_mean'], benchmark_results['hubbard_mean']]
        stds = [benchmark_results['syk_std'], benchmark_results['hubbard_std']]
        x = [0, 1]
        colors_bar = [self.colors['syk'], self.colors['hubbard']]
        
        axes[1].bar(x, means, yerr=stds, color=colors_bar, alpha=0.8,
                    capsize=5, edgecolor='black', linewidth=1.5)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['SYK\n(Chaotic)', 'Hubbard\n(Regular)'], fontsize=11)
        axes[1].set_ylabel('Mean Accuracy', fontsize=12)
        axes[1].set_title('Average Performance', fontsize=14, fontweight='bold')
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Win distribution pie chart
        syk_wins = benchmark_results['syk_wins_count']
        hubbard_wins = benchmark_results['n_trials'] - syk_wins
        
        if syk_wins + hubbard_wins > 0:
            axes[2].pie([syk_wins, hubbard_wins], 
                       labels=['SYK Wins', 'Hubbard Wins'],
                       colors=[self.colors['syk'], self.colors['hubbard']],
                       autopct='%1.1f%%', startangle=90,
                       explode=(0.05, 0.05))
            axes[2].set_title('Win Distribution', fontsize=14, fontweight='bold')
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'qml_comparison')
        return fig
    
    def plot_trotter_error(
        self,
        error_analysis: Dict,
        title: str = "Trotter Decomposition Error Analysis"
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.figsize)
        
        steps = error_analysis['steps']
        
        ax.loglog(steps, error_analysis['error_order1'], 
                  'o-', color=self.colors['syk'], linewidth=2, 
                  markersize=8, label='First Order (O(t²/n))')
        ax.loglog(steps, error_analysis['error_order2'],
                  's-', color=self.colors['hubbard'], linewidth=2,
                  markersize=8, label='Second Order (O(t³/n²))')
        
        # Reference lines
        steps_arr = np.array(steps)
        ax.loglog(steps_arr, 0.1 / steps_arr, 'k--', alpha=0.5, label='1/n scaling')
        ax.loglog(steps_arr, 0.1 / steps_arr**2, 'k:', alpha=0.5, label='1/n² scaling')
        
        ax.set_xlabel('Number of Trotter Steps', fontsize=12)
        ax.set_ylabel('Infidelity (1 - F)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(fig, 'trotter_error')
        return fig
    
    def create_summary_dashboard(
        self,
        syk_data: Dict,
        hubbard_data: Dict,
        nisq_data: Optional[Dict] = None,
        qml_data: Optional[Dict] = None,
        title: str = "Quantum Dynamics Comparison Dashboard"
    ) -> Figure:
        n_cols = 3
        n_rows = 2 if (nisq_data or qml_data) else 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Row 1: Spectral and OTOC
        # Spectral density
        ax = axes[0, 0]
        if 'eigenvalues' in syk_data:
            ax.hist(syk_data['eigenvalues'], bins=30, density=True, alpha=0.6,
                    color=self.colors['syk'], label='SYK')
        if 'eigenvalues' in hubbard_data:
            ax.hist(hubbard_data['eigenvalues'], bins=30, density=True, alpha=0.6,
                    color=self.colors['hubbard'], label='Hubbard')
        ax.set_xlabel('Energy')
        ax.set_ylabel('Density')
        ax.set_title('Spectral Density', fontweight='bold')
        ax.legend()
        
        # Level spacing ratio
        ax = axes[0, 1]
        x = [0, 1]
        lsr_values = [
            syk_data.get('level_spacing_ratio', 0),
            hubbard_data.get('level_spacing_ratio', 0)
        ]
        ax.bar(x, lsr_values, color=[self.colors['syk'], self.colors['hubbard']], alpha=0.8)
        ax.axhline(0.386, color='gray', linestyle='--', label='Poisson (0.386)')
        ax.axhline(0.530, color='black', linestyle='-', label='GOE (0.530)')
        ax.set_xticks(x)
        ax.set_xticklabels(['SYK', 'Hubbard'])
        ax.set_ylabel('⟨r⟩')
        ax.set_title('Level Spacing Ratio', fontweight='bold')
        ax.legend(fontsize=9)
        
        # Summary text
        ax = axes[0, 2]
        ax.axis('off')
        summary_text = "SUMMARY\n" + "="*40 + "\n\n"
        
        if 'ground_state_energy' in syk_data:
            summary_text += f"SYK Ground State: {syk_data['ground_state_energy']:.4f}\n"
        if 'ground_state_energy' in hubbard_data:
            summary_text += f"Hubbard Ground State: {hubbard_data['ground_state_energy']:.4f}\n"
        if lsr_values[0] > 0.45:
            summary_text += "\nSYK: Chaotic (Random Matrix)\n"
        else:
            summary_text += "\nSYK: Less Chaotic\n"
        if lsr_values[1] > 0.45:
            summary_text += "Hubbard: Chaotic\n"
        else:
            summary_text += "Hubbard: More Integrable\n"
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Analysis Summary', fontweight='bold')
        
        # Row 2: NISQ and QML (if available)
        if n_rows > 1 and (nisq_data or qml_data):
            if nisq_data:
                ax = axes[1, 0]
                if 'noise_levels' in nisq_data:
                    ax.plot(nisq_data['noise_levels'], nisq_data.get('energy_variance', []),
                            'o-', color=self.colors['noisy'], linewidth=2)
                    ax.set_xlabel('Noise Level')
                    ax.set_ylabel('Energy Variance')
                    ax.set_title('NISQ Energy Uncertainty', fontweight='bold')
            
            if qml_data:
                ax = axes[1, 1]
                if 'syk_mean' in qml_data:
                    means = [qml_data['syk_mean'], qml_data['hubbard_mean']]
                    stds = [qml_data['syk_std'], qml_data['hubbard_std']]
                    ax.bar([0, 1], means, yerr=stds, 
                           color=[self.colors['syk'], self.colors['hubbard']],
                           capsize=5, alpha=0.8)
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels(['SYK', 'Hubbard'])
                    ax.set_ylabel('Accuracy')
                    ax.set_title('QML Classification', fontweight='bold')
                    ax.set_ylim(0, 1.1)
            
            # Conclusion
            ax = axes[1, 2]
            ax.axis('off')
            conclusion = "CONCLUSIONS\n" + "="*40 + "\n\n"
            
            if qml_data and 'conclusion' in qml_data:
                conclusion += f"QML: {qml_data['conclusion']}\n"
            
            conclusion += "\nChaos Advantage:\n"
            conclusion += "- SYK exhibits maximal scrambling\n"
            conclusion += "- Higher expressivity for QML\n"
            conclusion += "- But potentially higher noise sensitivity\n"
            
            ax.text(0.1, 0.9, conclusion, transform=ax.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
            ax.set_title('Conclusions', fontweight='bold')
        
        fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        self._save_figure(fig, 'summary_dashboard')
        return fig
