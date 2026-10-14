from collections.abc import Sequence

import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..spectral import R_MEAN

SYK = "crimson"
HUBBARD = "steelblue"
GOE = "green"
GUE = "darkorange"
GSE = "purple"
POISSON = "gray"
CLASS_COLORS = {"GOE": GOE, "GUE": GUE, "GSE": GSE, "Poisson": POISSON}
CYCLE = [SYK, HUBBARD, GOE, GUE, GSE, "black", "saddlebrown", "teal"]
BOX = {"boxstyle": "round", "facecolor": "lightyellow", "alpha": 0.9}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titleweight": "bold",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
            "errorbar.capsize": 5,
            "axes.prop_cycle": cycler(color=CYCLE),
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def suptitle(fig: Figure, text: str) -> None:
    fig.suptitle(text, fontsize=14, fontweight="bold")


def reference_lines(ax: Axes, classes: Sequence[str] = ("GOE", "Poisson")) -> None:
    for cls in classes:
        value = R_MEAN[cls.lower()]
        style = ":" if cls == "Poisson" else "--"
        ax.axhline(value, color=CLASS_COLORS[cls], ls=style, lw=2, label=f"{cls} ({value:.3f})")


def summary_box(ax: Axes, title: str, lines: Sequence[str]) -> None:
    ax.axis("off")
    width = max(len(title), *(len(line) for line in lines)) + 4
    rule = "-" * width
    text = "\n".join([rule, title.center(width), rule, "", *lines, ""])
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=BOX,
    )
