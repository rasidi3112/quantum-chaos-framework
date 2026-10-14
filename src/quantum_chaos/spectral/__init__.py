"""Spectral statistics and random-matrix references."""

from .rmt import R_MEAN, R_SURMISE, ensemble_mean_r, sample_spectrum
from .statistics import (
    DegenerateSpectrumWarning,
    bootstrap_mean,
    mean_spacing_ratio,
    remove_kramers_pairs,
    spacing_ratios,
)

__all__ = [
    "R_MEAN",
    "R_SURMISE",
    "DegenerateSpectrumWarning",
    "bootstrap_mean",
    "ensemble_mean_r",
    "mean_spacing_ratio",
    "remove_kramers_pairs",
    "sample_spectrum",
    "spacing_ratios",
]
