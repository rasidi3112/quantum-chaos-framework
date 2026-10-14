"""Experiment configurations: YAML files validated into frozen dataclasses.

Every figure is described by one YAML file under ``configs/figures/``; all
random seeds are explicit fields. Unknown keys are rejected so that a typo
cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class SYKStatsConfig:
    """Disorder-averaged SYK level statistics: N -> number of realisations."""

    realisations: dict[int, int]
    rmt_samples: int = 600
    trim: float = 0.0
    bootstrap_seed: int = 0
    rmt_seed: int = 1


@dataclass(frozen=True)
class HubbardStatsConfig:
    """Symmetry-resolved Hubbard-ring statistics for several t' values."""

    n_sites: int = 8
    n_up: int = 4
    n_down: int = 4
    t: float = 1.0
    U: float = 4.0
    tp_values: tuple[float, ...] = (0.0, 0.5)


@dataclass(frozen=True)
class OTOCConfig:
    """Infinite-temperature OTOCs of one SYK realisation and a small Hubbard ring."""

    syk_n: int = 12
    syk_seed: int = 7
    hubbard_sites: int = 4
    hubbard_U: float = 4.0
    t_max: float = 6.0
    n_times: int = 61


@dataclass(frozen=True)
class FigureConfig:
    """One figure: ``kind`` selects the builder in :mod:`quantum_chaos.experiments.figures`."""

    name: str
    kind: str
    output: str
    syk: SYKStatsConfig | None = None
    hubbard: HubbardStatsConfig | None = None
    otoc: OTOCConfig | None = None
    bins: int = 25
    extra: dict[str, Any] = field(default_factory=dict)


_SECTIONS = {"syk": SYKStatsConfig, "hubbard": HubbardStatsConfig, "otoc": OTOCConfig}


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(names)}")
    return cls(**data)


def figure_config_from_dict(data: dict[str, Any], where: str = "config") -> FigureConfig:
    data = dict(data)
    for key, cls in _SECTIONS.items():
        if key in data and data[key] is not None:
            section = dict(data[key])
            if key == "syk":
                section["realisations"] = {
                    int(n): int(r) for n, r in section["realisations"].items()
                }
            if key == "hubbard" and "tp_values" in section:
                section["tp_values"] = tuple(float(x) for x in section["tp_values"])
            data[key] = _build(cls, section, f"{where}.{key}")
    return _build(FigureConfig, data, where)


def load_figure_config(path: str | pathlib.Path) -> FigureConfig:
    path = pathlib.Path(path)
    with path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return figure_config_from_dict(data, where=str(path))


def config_to_dict(cfg: FigureConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
