from __future__ import annotations

import dataclasses
import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class SYKStatsConfig:
    realisations: dict[int, int]
    rmt_samples: int = 600
    trim: float = 0.0
    bootstrap_seed: int = 0
    rmt_seed: int = 1


@dataclass(frozen=True)
class HubbardStatsConfig:
    n_sites: int = 8
    n_up: int = 4
    n_down: int = 4
    t: float = 1.0
    U: float = 4.0
    tp_values: tuple[float, ...] = (0.0, 0.5)


@dataclass(frozen=True)
class OTOCConfig:
    syk_n: int = 12
    syk_seed: int = 7
    hubbard_sites: int = 4
    hubbard_U: float = 4.0
    t_max: float = 6.0
    n_times: int = 61


@dataclass(frozen=True)
class SFFConfig:
    realisations: dict[int, int]
    beta: float = 0.0
    t_min: float = 0.1
    t_max: float = 3000.0
    n_times: int = 120
    n_taus: int = 60
    tau_max: float = 1.6


@dataclass(frozen=True)
class DynamicsConfig:
    n_majorana: int = 14
    realisations: int = 6
    betas: tuple[float, ...] = (0.0, 10.0)
    t_max: float = 12.0
    n_times: int = 121
    lanczos_steps: int = 40
    lanczos_n_majorana: int = 10
    page_n_majorana: int = 20
    eth_sizes: tuple[int, ...] = (12, 14, 16, 18)


@dataclass(frozen=True)
class FloquetConfig:
    sizes: tuple[int, ...] = (6, 8)
    omegas: tuple[float, ...] = (4.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0)
    J0: float = 1.0
    J1: float = 1.0
    tp: float = 0.5
    U: float = 4.0
    n_periods: int = 20000
    n_steps: int = 48
    expansion_omegas: tuple[float, ...] = (20.0, 40.0, 80.0)


@dataclass(frozen=True)
class NISQConfig:
    bound_n_majorana: int = 8
    bound_time: float = 2.0
    bound_steps: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    cost_sizes: tuple[int, ...] = (8, 12, 16, 20, 24)
    sparse_k: float = 4.0
    sparse_stat_sizes: tuple[int, ...] = (16, 20)
    sparse_k_values: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0)
    sparse_realisations: int = 12
    otoc_n_majorana: int = 6
    otoc_times: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    otoc_steps: int = 6
    depolarizing: float = 0.005
    folds: tuple[int, ...] = (1, 3, 5)


@dataclass(frozen=True)
class QMLConfig:
    syk_qubits: tuple[int, ...] = (3, 4, 5, 6, 7)
    hubbard_sites: tuple[int, ...] = (2, 3, 4)
    times: tuple[float, ...] = (0.1, 1.0, 10.0)
    n_samples: int = 30
    seeds: int = 20
    dataset_size: int = 80
    evolution_grid: tuple[float, ...] = (0.3, 1.0, 3.0)


@dataclass(frozen=True)
class LyapunovConfig:
    results: str = "results/syk_lyapunov.json"
    thresholds: tuple[float, ...] = (0.25, 0.16)
    bootstrap: int = 400
    seed: int = 0


@dataclass(frozen=True)
class FigureConfig:
    name: str
    kind: str
    output: str
    syk: SYKStatsConfig | None = None
    hubbard: HubbardStatsConfig | None = None
    otoc: OTOCConfig | None = None
    sff: SFFConfig | None = None
    dynamics: DynamicsConfig | None = None
    floquet: FloquetConfig | None = None
    nisq: NISQConfig | None = None
    qml: QMLConfig | None = None
    lyapunov: LyapunovConfig | None = None
    bins: int = 25
    extra: dict[str, Any] = field(default_factory=dict)


_SECTIONS = {
    "syk": SYKStatsConfig,
    "hubbard": HubbardStatsConfig,
    "otoc": OTOCConfig,
    "sff": SFFConfig,
    "dynamics": DynamicsConfig,
    "floquet": FloquetConfig,
    "nisq": NISQConfig,
    "qml": QMLConfig,
    "lyapunov": LyapunovConfig,
}


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
            if key in ("syk", "sff"):
                section["realisations"] = {
                    int(n): int(r) for n, r in section["realisations"].items()
                }
            if key == "hubbard" and "tp_values" in section:
                section["tp_values"] = tuple(float(x) for x in section["tp_values"])
            if key == "dynamics" and "betas" in section:
                section["betas"] = tuple(float(x) for x in section["betas"])
            if key == "floquet":
                for name, cast in (("sizes", int), ("omegas", float), ("expansion_omegas", float)):
                    if name in section:
                        section[name] = tuple(cast(x) for x in section[name])
            if key == "nisq":
                for name, cast in (
                    ("bound_steps", int),
                    ("cost_sizes", int),
                    ("sparse_stat_sizes", int),
                    ("sparse_k_values", float),
                    ("otoc_times", float),
                    ("folds", int),
                ):
                    if name in section:
                        section[name] = tuple(cast(x) for x in section[name])
            if key == "qml":
                for name, cast in (
                    ("syk_qubits", int),
                    ("hubbard_sites", int),
                    ("times", float),
                    ("evolution_grid", float),
                ):
                    if name in section:
                        section[name] = tuple(cast(x) for x in section[name])
            if key == "lyapunov" and "thresholds" in section:
                section["thresholds"] = tuple(float(x) for x in section["thresholds"])
            if key == "dynamics" and "eth_sizes" in section:
                section["eth_sizes"] = tuple(int(x) for x in section["eth_sizes"])
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
