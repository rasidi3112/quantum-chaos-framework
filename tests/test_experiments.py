import json
import pathlib

import pytest

from quantum_chaos.experiments import BUILDERS, load_figure_config, make_figure
from quantum_chaos.experiments.config import figure_config_from_dict

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "figures").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_committed_config_is_valid(path):
    cfg = load_figure_config(path)
    assert cfg.kind in BUILDERS
    assert cfg.output.startswith("figures/") and cfg.output.endswith(".png")


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        figure_config_from_dict({"name": "x", "kind": "overview", "output": "a.png", "seed": 1})
    with pytest.raises(ValueError, match="unknown keys"):
        figure_config_from_dict(
            {
                "name": "x",
                "kind": "overview",
                "output": "a.png",
                "hubbard": {"n_sites": 8, "Uu": 3.0},
            }
        )


TINY = {
    "syk": {"realisations": {10: 3, 12: 3}, "rmt_samples": 5},
    "hubbard": {"n_sites": 5, "n_up": 2, "n_down": 2, "U": 3.0, "tp_values": [0.0, 0.4]},
    "otoc": {"syk_n": 8, "hubbard_sites": 3, "n_times": 5, "t_max": 1.0},
}


@pytest.mark.parametrize("kind", sorted(BUILDERS))
def test_tiny_figures_are_written_with_provenance(kind, tmp_path):
    cfg = figure_config_from_dict(
        {"name": kind, "kind": kind, "output": f"figures/{kind}.png", "bins": 10, **TINY}
    )
    out = make_figure(cfg, root=tmp_path)
    assert out.exists() and out.stat().st_size > 0
    sidecar = json.loads(out.with_suffix(".json").read_text())
    assert sidecar["config"]["kind"] == kind
    for key in ("git_commit", "src_tree", "package_version", "numpy", "pennylane"):
        assert key in sidecar["provenance"]
    assert sidecar["data"]


def test_unknown_kind_is_rejected(tmp_path):
    cfg = figure_config_from_dict({"name": "x", "kind": "nope", "output": "figures/x.png"})
    with pytest.raises(ValueError, match="unknown figure kind"):
        make_figure(cfg, root=tmp_path)
