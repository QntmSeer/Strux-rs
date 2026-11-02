"""Regenerate all strux-rs benchmark figures in SpinGlassBench Theme A ("typewriter lab").

Style specification (matching sgbench/plotting.py & docs/figure_style.md):
- Surface: white paper (#ffffff), primary ink (#2a2723), secondary ink (#6b655c), dotted guides (#e2dacb)
- Font: Latin Modern Mono (fallback DejaVu Sans Mono)
- Palette: RUST (#c94f27), COBALT (#2f62b8), JADE (#069a8e), OCHRE (#d4960f), GRAYS (#4a463f, #8c857a, #b3ab9e)
- Spines: bottom axis only; faint dotted horizontal/vertical guides; direct ink labels; finding title + provenance subtitle
"""

from __future__ import annotations

import math
from pathlib import Path
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
import numpy as np

# Surfaces and ink ------------------------------------------------------------
PAPER = "#ffffff"
INK = "#2a2723"
INK2 = "#6b655c"
GRID = "#e2dacb"
REF = INK2

# Four validated hues + graded grays ------------------------------------------
RUST = "#c94f27"
COBALT = "#2f62b8"
JADE = "#069a8e"
OCHRE = "#d4960f"
GRAYS = ["#4a463f", "#8c857a", "#b3ab9e"]
SEQ_STEPS = ["#d9dfe9", "#b1c1dd", "#86a0cf", "#5a7ec0", "#2f62b8", "#234a8c", "#183361"]
DASH = (0, (4, 3))
DOT = (0, (1, 2))


def setup():
    mpl.rcParams.update(
        {
            "font.family": ["Latin Modern Mono", "DejaVu Sans Mono", "monospace"],
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
        }
    )


def titles(ax, main: str, sub: str | None = None):
    ax.set_title(main, loc="left", pad=15 if sub else 6, fontweight="normal", color=INK)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=6.6, color=INK2, va="bottom")


def end_label(ax, x, y, text, dx=5, dy=0, **kw):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        fontsize=kw.pop("fontsize", 6.8),
        color=kw.pop("color", INK),
        annotation_clip=False,
        **kw,
    )


def plain_log_axis(ax, axis="y"):
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


# ==============================================================================
# 1. benchmark_scaling.png
# ==============================================================================
def make_benchmark_scaling():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.85))

    # Left panel: Pairwise RMSD throughput scaling
    t_steps = np.array([100, 250, 500, 1000, 2500, 5000], dtype=float)
    pairs = t_steps ** 2
    cpu_rate = np.full_like(t_steps, 0.532)  # Million alignments / s
    gpu_rate = np.array([0.0336, 0.3117, 0.7886, 1.5548, 1.9464, 2.0545])

    ax1.plot(pairs, gpu_rate, "-o", color=RUST, mec=PAPER, mew=0.7, lw=1.6, ms=5.0)
    ax1.plot(pairs, cpu_rate, "--s", color=COBALT, mec=PAPER, mew=0.7, lw=1.3, ms=4.5)
    ax1.axhline(0.0432, color=REF, lw=0.9, ls=DOT)

    ax1.set_xscale("log")
    plain_log_axis(ax1, "x")
    ax1.set_xlim(7e3, 1.8e8)
    ax1.set_ylim(-0.08, 2.35)
    ax1.set_xlabel("pairwise alignments T^2")
    ax1.set_ylabel("million alignments / s")

    end_label(ax1, pairs[-1], gpu_rate[-1], "CUDA QCP (2.05M/s)", dx=5, dy=1)
    end_label(ax1, pairs[-1], cpu_rate[-1], "16T Rayon (0.53M/s)", dx=5, dy=2)
    ax1.text(1.5e6, 0.09, "1T CPU (43k/s)", fontsize=6.5, color=INK2)

    titles(
        ax1,
        "CUDA QCP reaches 2.05M alignments/s as T^2 grows",
        "1,919-atom GFP; T = 100 to 5,000 frames; max |GPU - SVD| = 2.67e-5 A",
    )

    # Right panel: Biological size scaling
    proteins = ["1L2Y\n(304)", "1GFL\n(3,950)", "Complex\n(10,000)", "1HTQ\n(97,872)"]
    speedups = np.array([0.92, 3.24, 3.50, 3.88])
    xs = np.arange(len(proteins))
    cols = [SEQ_STEPS[2], SEQ_STEPS[4], COBALT, RUST]

    ax2.axhline(1.0, color=REF, lw=0.9, ls=DASH)
    ax2.text(
        2.5,
        1.08,
        "16T CPU parity",
        fontsize=6.4,
        color=INK2,
        ha="center",
        bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.0),
    )

    for i, (x, y, col) in enumerate(zip(xs, speedups, cols)):
        ax2.vlines(x, 0, y, color=col, lw=2.2)
        ax2.plot(x, y, "o", color=col, ms=6.0, mec=PAPER, mew=0.8, zorder=4)
        ax2.text(x, y + 0.18, f"{y:.1f}x", ha="center", va="bottom", fontsize=7.0, color=INK)

    ax2.set_xticks(xs, proteins)
    ax2.set_xlim(-0.5, 3.6)
    ax2.set_ylim(0, 4.6)
    ax2.set_xlabel("target structure (atom count N)")
    ax2.set_ylabel("GPU speedup over 16T CPU")
    titles(
        ax2,
        "GPU speedup saturates near 3.9x at macromolecular scale",
        "T = 1,000 frames (1e6 pairs); RTX A2000 (4 GB VRAM) vs 16T Core i9",
    )

    fig.tight_layout(w_pad=2.4)
    fig.savefig("benchmark_scaling.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote benchmark_scaling.png")


# ==============================================================================
# 2. hardware_soak_telemetry.png (Real 300s / 5-Minute Prolonged Burn on agni)
# ==============================================================================
def make_hardware_soak():
    # Exact 99 empirical 3-second telemetry samples from the 309.3s full-system burn on agni
    time_sec = np.array([
        0.2, 3.2, 6.3, 9.3, 12.6, 15.7, 18.7, 21.7, 24.9, 27.9, 30.9, 34.0, 37.0, 40.0, 43.1,
        46.1, 49.1, 52.1, 55.2, 58.2, 61.2, 64.3, 67.3, 70.3, 73.4, 76.4, 79.4, 82.5, 85.5, 88.5,
        91.6, 94.6, 97.6, 100.7, 103.7, 106.7, 109.8, 112.8, 115.9, 118.9, 121.9, 125.0, 128.0,
        131.0, 134.1, 137.2, 140.3, 143.3, 146.3, 149.5, 152.5, 155.6, 158.6, 161.6, 164.7, 167.7,
        170.7, 173.8, 176.8, 179.8, 182.9, 185.9, 188.9, 192.0, 195.0, 198.1, 201.1, 204.1, 207.2,
        210.2, 213.2, 216.2, 219.3, 222.3, 225.3, 228.4, 231.4, 234.4, 237.4, 240.5, 243.5, 246.5,
        249.6, 252.7, 255.7, 258.8, 261.9, 264.9, 268.0, 271.0, 274.0, 277.1, 280.1, 283.1, 286.2,
        289.2, 292.2, 295.3, 298.3
    ])
    cpu_util = np.array([
        92.0, 97.9, 97.6, 95.7, 97.5, 98.1, 98.4, 97.7, 97.5, 98.3, 98.0, 98.3, 97.4, 97.9, 97.4,
        97.8, 96.5, 98.3, 97.8, 98.3, 96.6, 97.6, 97.4, 98.4, 96.1, 97.9, 98.3, 98.0, 95.8, 98.1,
        97.7, 98.6, 96.8, 97.8, 98.0, 97.9, 96.1, 97.9, 98.4, 98.0, 94.6, 97.6, 97.9, 98.8, 96.2,
        98.2, 98.5, 98.7, 96.0, 97.3, 97.7, 98.0, 98.9, 95.4, 98.4, 97.8, 98.7, 96.8, 96.6, 98.6,
        98.0, 95.4, 97.9, 98.0, 98.0, 95.2, 98.0, 98.4, 97.8, 95.7, 97.5, 98.5, 98.1, 95.9, 98.3,
        97.4, 98.4, 96.9, 97.3, 98.1, 98.4, 96.2, 97.8, 98.0, 97.4, 95.8, 97.6, 98.9, 97.9, 97.9,
        95.5, 97.5, 97.5, 98.0, 97.1, 98.5, 98.0, 98.6, 96.5
    ])
    temp = np.array([
        56, 57, 59, 59, 61, 62, 63, 63, 64, 65, 65, 66, 67, 67, 68, 68, 69, 69, 69, 69, 69, 70, 70,
        70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 70, 71, 71, 70, 70, 71, 71, 71, 70, 71,
        71, 71, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71,
        71, 71, 71, 71, 71, 71, 71, 71, 70, 71, 71, 71, 70, 71, 71, 71, 70, 71, 71, 71, 70, 71, 71,
        70, 70, 70, 70, 70, 70, 70
    ], dtype=float)
    power = np.array([
        25.55, 25.71, 25.88, 24.67, 26.10, 26.11, 26.26, 26.35, 26.59, 26.66, 26.68, 26.80, 27.02,
        27.05, 27.12, 27.16, 27.40, 27.37, 27.51, 27.52, 27.60, 27.58, 27.51, 27.57, 27.70, 27.67,
        27.65, 27.62, 26.82, 27.76, 27.64, 27.67, 25.02, 27.76, 27.75, 27.69, 25.24, 27.76, 27.73,
        27.73, 25.42, 27.85, 27.73, 27.75, 27.63, 27.86, 27.80, 27.75, 27.25, 27.89, 27.78, 27.74,
        27.81, 27.89, 27.81, 27.78, 27.73, 27.86, 27.81, 27.77, 27.80, 27.91, 27.85, 27.81, 27.82,
        27.99, 27.89, 27.81, 27.85, 25.90, 27.87, 27.86, 27.91, 25.73, 27.95, 27.86, 27.89, 24.75,
        27.92, 27.87, 27.77, 26.46, 27.92, 27.80, 27.86, 26.24, 27.85, 27.80, 27.75, 27.80, 27.88,
        27.82, 27.85, 27.78, 27.90, 27.84, 27.81, 27.83, 27.84
    ])

    fig, (ax_t, ax_c, ax_p) = plt.subplots(3, 1, figsize=(7.4, 4.8), sharex=True)

    # Panel 1: GPU Temperature
    ax_t.plot(time_sec, temp, color=RUST, lw=1.4)
    ax_t.axhline(87.0, color=REF, lw=0.9, ls=DASH)
    ax_t.text(295, 84.8, "throttle limit (87 C)", fontsize=6.5, color=INK2, ha="right", va="top")
    end_label(ax_t, time_sec[-1], temp[-1], "70.0 C (71 C peak)", dx=5)
    ax_t.set_ylabel("GPU temp (C)")
    ax_t.set_ylim(50, 92)
    ax_t.set_yticks([55, 65, 75, 85])
    titles(
        ax_t,
        "Prolonged 5-minute full-system burn stabilizes 16 C below thermal throttle",
        "309.3s on agni: 600M CUDA QCP pairs + 5,658 lDDT (815B pairs) + 1.46M XTC frames + 14.6M MSA seqs",
    )

    # Panel 2: 16-Thread CPU Utilization (%)
    ax_c.plot(time_sec, cpu_util, color=COBALT, lw=1.3)
    ax_c.axhline(100.0, color=REF, lw=0.8, ls=DOT)
    ax_c.text(
        295,
        88.5,
        "GPU clock locked at 1,710 MHz (100% CUDA util)",
        fontsize=6.4,
        color=INK2,
        ha="right",
        va="bottom",
    )
    end_label(ax_c, time_sec[-1], cpu_util[-1], f"{np.mean(cpu_util):.1f}% mean (16T)", dx=5)
    ax_c.set_ylabel("16T CPU (%)")
    ax_c.set_ylim(85, 102)
    ax_c.set_yticks([88, 92, 96, 100])

    # Panel 3: GPU Power Draw (W)
    ax_p.plot(time_sec, power, color=JADE, lw=1.3)
    ax_p.axhline(35.0, color=REF, lw=0.9, ls=DASH)
    ax_p.text(295, 33.8, "max TGP (35.0 W)", fontsize=6.5, color=INK2, ha="right", va="top")
    end_label(ax_p, time_sec[-1], power[-1], "27.8 W", dx=5)
    ax_p.set_ylabel("GPU power (W)")
    ax_p.set_xlabel("elapsed time (seconds)")
    ax_p.set_ylim(22, 37)
    ax_p.set_yticks([24, 28, 32, 35])
    ax_p.set_xlim(0, 355)

    fig.tight_layout(h_pad=1.0)
    fig.savefig("hardware_soak_telemetry.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote hardware_soak_telemetry.png")


# ==============================================================================
# 3. benchmarks/gold_standard_comparison.png
# ==============================================================================
def make_gold_standard():
    fig = plt.figure(figsize=(7.4, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1.0], hspace=0.48, wspace=0.32)

    # Panel A (top full width): Horizontal Cleveland dot-and-stem plot (like SpinGlassBench Fig 7/9)
    ax1 = fig.add_subplot(gs[0, :])
    methods = [
        ("Biopython (SVD)", 446.6, GRAYS[2], "d", "1.0x"),
        ("SciPy (align_vectors)", 5436.6, GRAYS[2], "v", "12.2x"),
        ("MDAnalysis (QCPROT)", 10567.4, GRAYS[1], "P", "23.7x"),
        ("MDTraj (AVX QCP)", 26046.1, GRAYS[1], "X", "58.3x"),
        ("Gemmi (C++ superpose)", 39710.8, GRAYS[0], "^", "88.9x"),
        ("strux-rs (1T CPU)", 43241.4, JADE, "^", "96.8x"),
        ("Google JAX (vmap JIT)", 76612.5, OCHRE, "D", "172x"),
        ("ProDy (C-Kabsch)", 92281.5, GRAYS[0], "s", "207x"),
        ("strux-rs (16T Rayon)", 596176.5, COBALT, "s", "1,335x"),
        ("strux-rs (CUDA QCP)", 1941698.0, RUST, "o", "4,348x"),
    ]

    ys = np.arange(len(methods))
    ax1.grid(False)
    ax1.xaxis.grid(True, color=GRID, lw=0.8, ls=DOT)
    for y in ys:
        ax1.axhline(y, color=GRID, lw=0.5, ls=DOT, zorder=0)

    for y, (name, rate, col, marker, sp) in zip(ys, methods):
        ax1.plot([300, rate], [y, y], color=col, lw=1.4 if "strux-rs" in name else 1.0, zorder=2)
        ax1.plot(rate, y, marker=marker, color=col, ms=5.2 if "strux-rs" in name else 4.4, mec=PAPER, mew=0.7, zorder=3)
        end_label(ax1, rate, y, f"{rate:,.0f}/s ({sp})", dx=6, fontsize=6.5)

    ax1.set_xscale("log")
    plain_log_axis(ax1, "x")
    ax1.set_xlim(300, 1.4e7)
    ax1.set_yticks(ys, [m[0] for m in methods])
    ax1.set_xlabel("alignments / second (log scale)")
    titles(
        ax1,
        "CUDA QCP and 16T Rayon outpace standard alignment packages by 10x to 4,348x",
        "Ubiquitin solution NMR ensemble (PDB 2K39, 116 models, 13,456 pairs); identical workstation",
    )

    # Panel B (bottom left): GFP Rg throughput
    ax2 = fig.add_subplot(gs[1, 0])
    rg_methods = [
        ("MDAnalysis", 336.9, GRAYS[1], "P", "1.0x"),
        ("MDTraj", 8912.2, GRAYS[0], "s", "26.5x"),
        ("strux-rs (SIMD)", 68770.8, COBALT, "o", "204x"),
    ]
    xs2 = np.arange(len(rg_methods))
    for x, (name, rate, col, marker, sp) in zip(xs2, rg_methods):
        ax2.vlines(x, 100, rate, color=col, lw=2.2)
        ax2.plot(x, rate, marker=marker, color=col, ms=5.5, mec=PAPER, mew=0.7, zorder=3)
        ax2.text(x, rate * 1.38, f"{rate:,.0f}/s\n({sp})", ha="center", va="bottom", fontsize=6.4, color=INK)

    ax2.set_yscale("log")
    plain_log_axis(ax2, "y")
    ax2.set_xticks(xs2, [m[0] for m in rg_methods])
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylim(100, 4.5e5)
    ax2.set_ylabel("frames / s (log scale)")
    titles(
        ax2,
        "SIMD Rg reaches 68.8k frames/s",
        "GFP trajectory.pdb (1,919 atoms)",
    )

    # Panel C (bottom right): MSA ingestion MB/s
    ax3 = fig.add_subplot(gs[1, 1])
    msa_items = [
        ("Kinase STO\nPython", 199.5, GRAYS[1], "1.0x"),
        ("Kinase STO\nstrux-rs", 270.7, COBALT, "1.4x"),
        ("BFD A3M\nPython", 41.9, GRAYS[1], "1.0x"),
        ("BFD A3M\nstrux-rs", 168.9, RUST, "4.0x"),
    ]
    xs3 = np.arange(len(msa_items))
    for x, (name, mb, col, sp) in zip(xs3, msa_items):
        ax3.vlines(x, 0, mb, color=col, lw=2.2)
        ax3.plot(x, mb, "o", color=col, ms=5.5, mec=PAPER, mew=0.7, zorder=3)
        ax3.text(x, mb + 12, f"{mb:.0f}\n({sp})", ha="center", va="bottom", fontsize=6.4, color=INK)

    ax3.set_xticks(xs3, [m[0] for m in msa_items])
    ax3.set_xlim(-0.5, 3.5)
    ax3.set_ylim(0, 340)
    ax3.set_ylabel("ingestion rate (MB / s)")
    titles(
        ax3,
        "Zero-copy FFI speeds up MSA parsing",
        "Kinase STO (19.5 MB) and BFD A3M",
    )

    Path("benchmarks").mkdir(exist_ok=True)
    fig.savefig("benchmarks/gold_standard_comparison.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote benchmarks/gold_standard_comparison.png")


# ==============================================================================
# 4. benchmarks/new_features_verification.png (Includes AMS Biophysics Suite!)
# ==============================================================================
def make_new_features():
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2))
    (ax1, ax2), (ax3, ax4) = axes

    # Panel A: AMS Biophysics Speedups (lDDT, MSA cluster, PCA, XTC)
    tasks = [
        ("PCA SVD\n(2500x1500)", 5.1, JADE, "190 ms vs 978 ms"),
        ("XTC Read\n(500x1000)", 7.3, OCHRE, "5.3 ms vs 39.0 ms"),
        ("MSA Cluster\n(10k x 250)", 33.8, COBALT, "22.2 ms vs 750 ms"),
        ("AlphaFold lDDT\n(4,000 atoms)", 201.2, RUST, "4.69 ms vs 943 ms"),
    ]
    ys1 = np.arange(len(tasks))
    ax1.grid(False)
    ax1.xaxis.grid(True, color=GRID, lw=0.8, ls=DOT)
    for y in ys1:
        ax1.axhline(y, color=GRID, lw=0.5, ls=DOT, zorder=0)

    ax1.axvline(1.0, color=REF, lw=0.8, ls=DASH)
    for y, (name, sp, col, detail) in zip(ys1, tasks):
        ax1.plot([1.0, sp], [y, y], color=col, lw=1.6, zorder=2)
        ax1.plot(sp, y, "o", color=col, ms=5.2, mec=PAPER, mew=0.7, zorder=3)
        end_label(ax1, sp, y, f"{sp:.1f}x ({detail})", dx=5, fontsize=6.3)

    ax1.set_xscale("log")
    plain_log_axis(ax1, "x")
    ax1.set_xlim(0.8, 3500)
    ax1.set_yticks(ys1, [t[0] for t in tasks])
    ax1.set_xlabel("speedup over NumPy / MDAnalysis (log scale)")
    titles(
        ax1,
        "AMS biophysics kernels run 5.1x to 201x faster",
        "16T Core i9 (309s burn: 5,658 lDDT / 815B pairs; err = 0)",
    )

    # Panel B: T4 Lysozyme 4-State Markov Population & Rare State Recovery
    states = [
        "Buried/\nOpen",
        "Exposed/\nOpen",
        "Buried/\nClosed",
        "Exposed/\nClosed (rare)",
    ]
    pi_true = np.array([58.2, 39.2, 1.9, 0.7])
    pi_rec = np.array([57.97, 39.34, 1.97, 0.73])
    dG_rec = np.array([0.00, 0.23, 2.02, 2.61])
    xs2 = np.arange(4)

    ax2.plot(xs2 - 0.12, pi_true, "s", color=COBALT, ms=5.0, mec=PAPER, mew=0.7, label="exact truth")
    ax2.plot(xs2 + 0.12, pi_rec, "o", color=RUST, ms=5.0, mec=PAPER, mew=0.7, label="strux-rs (2M steps)")
    for x, pt, pr, dg in zip(xs2, pi_true, pi_rec, dG_rec):
        ax2.vlines(x - 0.12, 0, pt, color=COBALT, lw=1.2, alpha=0.7)
        ax2.vlines(x + 0.12, 0, pr, color=RUST, lw=1.2, alpha=0.7)
        ax2.text(x, max(pt, pr) + 4.0, f"{pr:.1f}%\n({dg:.2f} kcal)", ha="center", va="bottom", fontsize=6.1, color=INK)

    ax2.set_xticks(xs2, states)
    ax2.set_ylim(0, 74)
    ax2.set_ylabel("stationary pop. (%)")
    ax2.legend(loc="upper right", fontsize=6.5)
    titles(
        ax2,
        "Markov solver recovers the 0.7% rare state in 19.3 ms",
        "2e6 steps (103.9M frames/s); max |pi - eig| = 1.36e-8; T = 300 K",
    )

    # Panel C: Binary Trajectory Readers (.dcd and .xtc)
    readers = [
        ("MDAnalysis\n(XTC)", 39.04, GRAYS[1]),
        ("MDTraj\n(XTC)", 15.15, GRAYS[0]),
        ("strux-rs\n(XTC 16T)", 5.32, RUST),
        ("MDTraj\n(DCD)", 0.548, GRAYS[0]),
        ("strux-rs\n(DCD mmap)", 0.380, COBALT),
    ]
    xs3 = np.arange(len(readers))
    for x, (name, ms_val, col) in zip(xs3, readers):
        ax3.vlines(x, 0.1, ms_val, color=col, lw=2.2)
        ax3.plot(x, ms_val, "o", color=col, ms=5.2, mec=PAPER, mew=0.7, zorder=3)
        lbl = f"{ms_val:.2f} ms"
        ax3.text(x, ms_val * 1.35, lbl, ha="center", va="bottom", fontsize=6.3, color=INK)

    ax3.set_yscale("log")
    plain_log_axis(ax3, "y")
    ax3.set_xticks(xs3, [r[0] for r in readers])
    ax3.set_xlim(-0.5, 4.5)
    ax3.set_ylim(0.1, 120)
    ax3.set_ylabel("read latency (ms, log)")
    titles(
        ax3,
        "Pure-Rust XTC and DCD readers beat C xdrfile plugins",
        "XTC: 500x1,000 atoms (max err 1.5e-5 A); DCD: 21x1,919 atoms",
    )

    # Panel D: Zero-copy tokens_np & GPU Contact Map
    accel = [
        ("tokens_np\nPython", 1.0, GRAYS[1], "1.0x\n(8.40 ms)"),
        ("tokens_np\nstrux-rs", 664.0, COBALT, "664x\n(12.6 us)"),
        ("Contacts\nNumPy", 1.0, GRAYS[1], "1.0x\n(3.68 s)"),
        ("Contacts\nCUDA", 19.5, RUST, "19.5x\n(189 ms)"),
    ]
    xs4 = np.arange(len(accel))
    for x, (name, sp, col, lbl) in zip(xs4, accel):
        ax4.vlines(x, 0.4, sp, color=col, lw=2.2)
        ax4.plot(x, sp, "o", color=col, ms=5.2, mec=PAPER, mew=0.7, zorder=3)
        ax4.text(x, sp * 1.38, lbl, ha="center", va="bottom", fontsize=6.2, color=INK)

    ax4.set_yscale("log")
    plain_log_axis(ax4, "y")
    ax4.set_xticks(xs4, [a[0] for a in accel])
    ax4.set_xlim(-0.5, 3.5)
    ax4.set_ylim(0.4, 4500)
    ax4.set_ylabel("relative speedup (log)")
    titles(
        ax4,
        "Zero-copy token blit (664x) & GPU contact map (19.5x)",
        "1,024x73 MSA uint8 tensor and 1,919x1,919 contact matrix",
    )

    fig.tight_layout(h_pad=1.8, w_pad=1.8)
    Path("benchmarks").mkdir(exist_ok=True)
    fig.savefig("benchmarks/new_features_verification.png", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("wrote benchmarks/new_features_verification.png")


if __name__ == "__main__":
    setup()
    make_benchmark_scaling()
    make_hardware_soak()
    make_gold_standard()
    make_new_features()
