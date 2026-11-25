"""Shared plot style for strux-rs benchmark figures.

Every figure in benchmarks/ uses this module so they read as one set: same palette,
monospace type, left-aligned title with a grey subtitle, no boxed legends, dotted
y-grid and no left spine. Import it rather than re-declaring colours in a new script.
"""
from __future__ import annotations

import math
import subprocess
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

# Surfaces and ink
PAPER = "#ffffff"
INK = "#2a2723"
INK2 = "#6b655c"
GRID = "#e2dacb"
REF = INK2

# Series colours. In hardware figures these are fixed per device:
# RUST = L4, COBALT = RTX A2000, JADE = H100.
RUST = "#c94f27"
COBALT = "#2f62b8"
JADE = "#069a8e"
OCHRE = "#d4960f"
PLUM = "#7a4a8c"

DASH = (0, (4, 3))
DOT = (0, (1, 2))


def setup() -> None:
    """Apply the house rcParams. Call once before creating a figure."""
    mpl.rcParams.update({
        "font.family": ["DejaVu Sans Mono", "monospace"],
        "mathtext.fontset": "cm",
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.major.size": 3,
        "ytick.major.size": 0,
        "ytick.minor.size": 0,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": DOT,
        "lines.linewidth": 1.5,
        "lines.markersize": 4.8,
        "font.size": 8,
        "axes.titlesize": 9.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "figure.dpi": 240,
        "pdf.fonttype": 42,
    })


def titles(ax, main: str, sub: str | None = None) -> None:
    """Left-aligned panel title with an optional grey subtitle above the axes."""
    ax.set_title(main, loc="left", pad=15 if sub else 6, fontweight="normal", color=INK)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6.6, color=INK2, va="bottom")


def end_label(ax, x, y, text, dx=5, dy=0, **kw) -> None:
    """Label a series at its last point, outside the axes if need be."""
    ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points",
        va="center", fontsize=kw.pop("fontsize", 6.8), color=kw.pop("color", INK),
        annotation_clip=False, **kw
    )


def plain_log_axis(ax, axis: str = "x") -> None:
    """Log ticks written as 1e6 rather than matplotlib's 10^6 superscripts."""
    def fmt(v, _):
        if v <= 0:
            return ""
        e = math.log10(v)
        if abs(e - round(e)) > 1e-9:
            return f"{v:g}"
        return f"{v:g}" if -2 <= e <= 3 else f"1e{int(round(e))}"

    a = getattr(ax, f"{axis}axis")
    a.set_major_formatter(FuncFormatter(fmt))
    a.set_minor_formatter(NullFormatter())


def git_commit() -> str:
    """Short commit of this checkout, or 'unknown' outside a git tree."""
    here = str(Path(__file__).resolve().parent)
    try:
        rev = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if rev.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "-C", here, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def provenance_footnote(fig, text: str) -> None:
    """One grey line under the figure saying which build and machine produced it."""
    fig.text(0.01, -0.02, text, fontsize=6.0, color=INK2, ha="left", va="top")


__all__ = [
    "PAPER", "INK", "INK2", "GRID", "REF", "RUST", "COBALT", "JADE", "OCHRE", "PLUM",
    "DASH", "DOT", "setup", "titles", "end_label", "plain_log_axis",
    "git_commit", "provenance_footnote", "plt", "mpl",
]
