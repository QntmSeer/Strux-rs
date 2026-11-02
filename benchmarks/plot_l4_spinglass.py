"""Regenerate hardware_l4_scaling_comparison.png matching the exact proportions,
breathing room, typography, and clean axes of benchmark_scaling.png."""
from __future__ import annotations

import math
from pathlib import Path
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

# Theme A: Surfaces and ink
PAPER = "#ffffff"
INK = "#2a2723"
INK2 = "#6b655c"
GRID = "#e2dacb"
REF = INK2

RUST = "#c94f27"
COBALT = "#2f62b8"
JADE = "#069a8e"
OCHRE = "#d4960f"
GRAYS = ["#4a463f", "#8c857a", "#b3ab9e"]
DASH = (0, (4, 3))
DOT = (0, (1, 2))

def setup():
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
        "errorbar.capsize": 0,
        "font.size": 8,
        "axes.titlesize": 9.0,
        "axes.labelsize": 7.8,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "figure.dpi": 220,
        "pdf.fonttype": 42,
    })

def titles(ax, main: str, sub: str | None = None):
    ax.set_title(main, loc="left", pad=15 if sub else 6, fontweight="normal", color=INK)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6.6, color=INK2, va="bottom")

def end_label(ax, x, y, text, dx=5, dy=0, **kw):
    ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points",
        va="center", fontsize=kw.pop("fontsize", 6.8), color=kw.pop("color", INK),
        annotation_clip=False, **kw
    )

def plain_log_axis(ax, axis="x"):
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

def main():
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.2))

    # =========================================================================
    # Panel 1: Trajectory Pairwise RMSD Throughput Scaling (L4 vs RTX A2000)
    # =========================================================================
    frames_l4 = np.array([1000, 2500, 5000, 10000, 15000, 20000], dtype=float)
    pairs_l4 = frames_l4 ** 2
    time_l4_sec = np.array([159.6, 221.9, 551.7, 1767.5, 3792.9, 6606.1]) / 1000.0
    rate_l4 = (pairs_l4 / 1e6) / time_l4_sec  # Million alignments / s

    frames_a2000 = np.array([1000, 2500, 5000, 10000], dtype=float)
    pairs_a2000 = frames_a2000 ** 2
    time_a2000_sec = np.array([211.4, 552.6, 1843.1, 6969.6]) / 1000.0
    rate_a2000 = (pairs_a2000 / 1e6) / time_a2000_sec

    ax1.plot(pairs_l4, rate_l4, "-o", color=RUST, mec=PAPER, mew=0.7, lw=1.6, ms=5.0)
    ax1.plot(pairs_a2000, rate_a2000, "--s", color=COBALT, mec=PAPER, mew=0.7, lw=1.3, ms=4.5)
    ax1.axhline(1.47, color=REF, lw=0.9, ls=DOT)

    ax1.set_xscale("log")
    plain_log_axis(ax1, "x")
    ax1.set_xlim(7e5, 1.8e9)
    ax1.set_ylim(-2.0, 72.0)
    ax1.set_xlabel("pairwise alignments $T^2$ ($N=250$ atoms)")
    ax1.set_ylabel("million alignments / s")

    # Clean inline annotations matching benchmark_scaling.png
    end_label(ax1, pairs_l4[-1], rate_l4[-1], "CUDA L4 (60.5M/s)", dx=6, dy=1)
    end_label(ax1, pairs_a2000[-1], rate_a2000[-1], "A2000 (14.3M/s)", dx=6, dy=-12)
    ax1.text(3.5e6, 3.2, "8T Cloud CPU (1.47M/s)", fontsize=6.5, color=INK2)

    # OOM marker
    ax1.scatter([2.25e8, 4.0e8], [14.34, 14.34], marker="x", color=INK2, s=40, zorder=5, lw=1.2)
    ax1.text(3.1e8, 17.5, "OOM (>4GB)", fontsize=6.2, color=INK2, ha="center")

    titles(
        ax1,
        "L4 reaches 60.5M alignments/s scaling to 400M pairs",
        "250-atom CA; L4 (24GB) vs RTX A2000 (4GB); 400M pairs in 6.60s (15.1B atom-pairs/s)",
    )

    # =========================================================================
    # Panel 2: Therapeutic & Biophysics Acceleration (Matching Right Panel of benchmark_scaling)
    # Vertical lollipop bars with crisp spacing and clean top speedup labels
    # =========================================================================
    benchmarks = [
        ("Pose RMSD\n(10k)", 326.3, RUST),
        ("Shrake BSA\n(2k at.)", 90.6, COBALT),
        ("pH ddG\n(6.0/7.4)", 55.0, COBALT),
        ("DCD Filter\n(100x RAM)", 43.6, JADE),
        ("AF-Cluster\n(5k x 256)", 38.0, JADE),
        ("Novelty DB\n(5k x 500)", 1793.8, RUST),
    ]
    xs = np.arange(len(benchmarks)) * 1.35  # Wider horizontal stride to avoid x-tick collision

    ax2.axhline(1.0, color=REF, lw=0.9, ls=DASH)
    ax2.text(
        xs[2] + 0.65,
        1.25,
        "Python baseline parity (1.0x)",
        fontsize=6.3,
        color=INK2,
        ha="center",
        bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.0),
    )

    for i, (name, sp, col) in enumerate(benchmarks):
        ax2.vlines(xs[i], 1.0, sp, color=col, lw=2.2)
        ax2.plot(xs[i], sp, "o", color=col, ms=6.0, mec=PAPER, mew=0.8, zorder=4)
        label_text = f"{sp:.0f}x" if sp >= 100 else f"{sp:.1f}x"
        ax2.text(xs[i], sp * 1.35, label_text, ha="center", va="bottom", fontsize=6.8, color=INK)

    ax2.set_yscale("log")
    plain_log_axis(ax2, "y")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([b[0] for b in benchmarks], fontsize=6.8)
    ax2.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax2.set_ylim(0.4, 7500.0)
    ax2.set_xlabel("strux-rs biophysics & design primitive")
    ax2.set_ylabel("speedup over standard Python stack")

    titles(
        ax2,
        "Zero-C primitives accelerate pipelines 38x to 1,794x",
        "NVIDIA L4 node; identical analytical outputs (max coord drift < 1.6e-6 A)",
    )

    fig.tight_layout(w_pad=3.8)

    out_benchmarks = "benchmarks/hardware_l4_scaling_comparison.png"
    out_artifact = r"C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f\hardware_l4_scaling_comparison.png"

    fig.savefig(out_benchmarks, bbox_inches="tight", dpi=220)
    try:
        fig.savefig(out_artifact, bbox_inches="tight", dpi=220)
    except Exception:
        pass
    plt.close(fig)
    print("Regenerated hardware_l4_scaling_comparison.png matching benchmark_scaling.png exactly.")

if __name__ == "__main__":
    main()
