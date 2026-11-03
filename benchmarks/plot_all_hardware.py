"""Generate publication-grade benchmark plot for all hardware data:
NVIDIA H100 (80GB SXM), NVIDIA L4 (24GB Cloud), and NVIDIA RTX A2000 (4GB Laptop).
Reads dynamically from standardized_results_*.json files.
"""
from __future__ import annotations

import json
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

RUST = "#c94f27"      # L4 Cloud GPU
COBALT = "#2f62b8"    # RTX A2000 Laptop GPU
JADE = "#069a8e"      # H100 80GB SXM GPU
OCHRE = "#d4960f"
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.4))

    bdir = Path(__file__).resolve().parent

    # Load JSON files
    h100_path = bdir / "standardized_results_h100_80gb_hbm3.json"
    l4_path = bdir / "standardized_results_l4.json"
    a2000_path = bdir / "standardized_results_rtx_a2000_laptop_gpu.json"

    if not (h100_path.exists() and l4_path.exists() and a2000_path.exists()):
        raise SystemExit("Missing benchmark JSON files. Ensure all 3 standardized results exist.")

    h100_data = json.loads(h100_path.read_text(encoding="utf-8"))
    l4_data = json.loads(l4_path.read_text(encoding="utf-8"))
    a2000_data = json.loads(a2000_path.read_text(encoding="utf-8"))

    h100_b = {b["benchmark"]: b for b in h100_data.get("benchmarks", [])}
    l4_b = {b["benchmark"]: b for b in l4_data.get("benchmarks", [])}
    a2000_b = {b["benchmark"]: b for b in a2000_data.get("benchmarks", [])}

    # =========================================================================
    # Panel 1: Trajectory Pairwise RMSD Throughput Scaling across 3 GPUs
    # =========================================================================
    frames_full = np.array([1000, 2500, 5000, 10000, 15000, 20000], dtype=float)
    pairs_full = frames_full ** 2

    frames_a2000 = np.array([1000, 2500, 5000, 10000], dtype=float)
    pairs_a2000 = frames_a2000 ** 2

    # H100 timings
    time_h100_sec = np.array([
        h100_b["cuda_pairwise_rmsd_1000x1000_250atoms"]["strux_ms"],
        h100_b["cuda_pairwise_rmsd_2500x2500_250atoms"]["strux_ms"],
        h100_b["cuda_pairwise_rmsd_5000x5000_250atoms"]["strux_ms"],
        h100_b["cuda_pairwise_rmsd_10000x10000_250atoms"]["strux_ms"],
        h100_b["cuda_pairwise_rmsd_15000x15000_250atoms"]["strux_ms"],
        h100_b["cuda_pairwise_rmsd_20000x20000_250atoms"]["strux_ms"],
    ]) / 1000.0
    rate_h100 = (pairs_full / 1e6) / time_h100_sec

    # L4 timings
    time_l4_sec = np.array([
        l4_b["cuda_pairwise_rmsd_1000x1000_250atoms"]["strux_ms"],
        l4_b["cuda_pairwise_rmsd_2500x2500_250atoms"]["strux_ms"],
        l4_b["cuda_pairwise_rmsd_5000x5000_250atoms"]["strux_ms"],
        l4_b["cuda_pairwise_rmsd_10000x10000_250atoms"]["strux_ms"],
        l4_b["cuda_pairwise_rmsd_15000x15000_250atoms"]["strux_ms"],
        l4_b["cuda_pairwise_rmsd_20000x20000_250atoms"]["strux_ms"],
    ]) / 1000.0
    rate_l4 = (pairs_full / 1e6) / time_l4_sec

    # A2000 timings
    time_a2000_sec = np.array([
        a2000_b["cuda_pairwise_rmsd_1000x1000_250atoms"]["strux_ms"],
        a2000_b["cuda_pairwise_rmsd_2500x2500_250atoms"]["strux_ms"],
        a2000_b["cuda_pairwise_rmsd_5000x5000_250atoms"]["strux_ms"],
        a2000_b["cuda_pairwise_rmsd_10000x10000_250atoms"]["strux_ms"],
    ]) / 1000.0
    rate_a2000 = (pairs_a2000 / 1e6) / time_a2000_sec

    # CPU baseline
    cpu_baseline_sec = l4_b["cuda_pairwise_rmsd_1000x1000_250atoms"]["baseline_ms"] / 1000.0
    cpu_rate = 1.0 / cpu_baseline_sec

    # Plot curves
    ax1.plot(pairs_full, rate_h100, "-D", color=JADE, mec=PAPER, mew=0.7, lw=1.8, ms=5.2, label="H100 (80GB)")
    ax1.plot(pairs_full, rate_l4, "-o", color=RUST, mec=PAPER, mew=0.7, lw=1.5, ms=4.8, label="L4 (24GB)")
    ax1.plot(pairs_a2000, rate_a2000, "--s", color=COBALT, mec=PAPER, mew=0.7, lw=1.3, ms=4.5, label="RTX A2000 (4GB)")
    ax1.axhline(cpu_rate, color=REF, lw=0.9, ls=DOT)

    ax1.set_xscale("log")
    plain_log_axis(ax1, "x")
    ax1.set_xlim(7e5, 1.9e9)
    ax1.set_ylim(-5.0, 245.0)
    ax1.set_xlabel("pairwise alignments $T^2$ ($N=250$ atoms)")
    ax1.set_ylabel("million alignments / s")

    # End labels
    end_label(ax1, pairs_full[-1], rate_h100[-1], f"CUDA H100 ({rate_h100[-1]:.1f}M/s)", dx=6, dy=1, color=JADE)
    end_label(ax1, pairs_full[-1], rate_l4[-1], f"CUDA L4 ({rate_l4[-1]:.1f}M/s)", dx=6, dy=1, color=RUST)
    end_label(ax1, pairs_a2000[-1], rate_a2000[-1], f"A2000 ({rate_a2000[-1]:.1f}M/s)", dx=6, dy=-12, color=COBALT)
    ax1.text(3.5e6, 5.0, f"8T Cloud CPU ({cpu_rate:.2f}M/s)", fontsize=6.5, color=INK2)

    # OOM marker for A2000
    ax1.scatter([2.25e8, 4.0e8], [rate_a2000[-1], rate_a2000[-1]], marker="x", color=INK2, s=40, zorder=5, lw=1.2)
    ax1.text(3.1e8, 17.5, "OOM (>4GB)", fontsize=6.2, color=INK2, ha="center")

    total_pairs = pairs_full[-1]
    time_max_h100 = time_h100_sec[-1]
    atom_pairs_rate_h100 = (total_pairs * 250) / time_max_h100 / 1e9
    titles(
        ax1,
        f"H100 reaches {rate_h100[-1]:.1f}M alignments/s scaling to 400M pairs",
        f"250-atom CA; H100 (80GB) vs L4 (24GB) vs RTX A2000 (4GB); 400M pairs in {time_max_h100:.2f}s ({atom_pairs_rate_h100:.1f}B at-pairs/s)",
    )

    # =========================================================================
    # Panel 2: Multi-Hardware Acceleration Across Biophysics Primitives
    # Grouped lollipops showing Laptop A2000, Cloud L4, and Data Center H100
    # =========================================================================
    keys = [
        ("Pose RMSD\n(10k)", "batch_pose_and_sc_rmsd_10000"),
        ("Shrake BSA\n(2k at.)", "calculate_bsa_shrake_rupley_per_complex"),
        ("pH ddG\n(6.0/7.4)", "calculate_ph_switch_ddg"),
        ("DCD Filter\n(100x RAM)", "parse_dcd_solute_filtered_40k_to_400"),
        ("AF-Cluster\n(est. 5k)", "af_cluster_a3m_5000x256"),
        ("Novelty DB\n(5k x 500)", "sliding_kmer_novelty_5000x500_k15"),
    ]

    xs = np.arange(len(keys)) * 1.55  # Generous horizontal spacing
    offsets = [-0.22, 0.0, 0.22]
    colors = [COBALT, RUST, JADE]
    hw_labels = ["RTX A2000 (4GB)", "L4 (24GB)", "H100 (80GB)"]
    datasets = [a2000_b, l4_b, h100_b]

    ax2.axhline(1.0, color=REF, lw=0.9, ls=DASH)
    ax2.text(
        xs[2] + 0.35,
        1.25,
        "Python baseline parity (1.0x)",
        fontsize=6.3,
        color=INK2,
        ha="center",
        bbox=dict(facecolor=PAPER, edgecolor="none", pad=1.0),
    )

    all_measured_sp = []
    for i, (name, bench_key) in enumerate(keys):
        estimated = "est." in name
        sp_vals = []
        for j, (off, col, data) in enumerate(zip(offsets, colors, datasets)):
            sp = data.get(bench_key, {}).get("speedup", 1.0)
            sp_vals.append(sp)
            if not estimated:
                all_measured_sp.append(sp)

            x_pos = xs[i] + off
            ax2.vlines(x_pos, 1.0, sp, color=col, lw=1.6, linestyles=DOT if estimated else "solid")
            marker = "s" if j == 0 else ("o" if j == 1 else "D")
            ax2.plot(
                x_pos, sp, marker, color=col, ms=4.8,
                mfc=PAPER if estimated else col,
                mec=col if estimated else PAPER,
                mew=1.0 if estimated else 0.7,
                zorder=4,
            )

        # Label highest speedup or range on top
        max_sp = max(sp_vals)
        label_text = f"{max_sp:.0f}x" if max_sp >= 100 else f"{max_sp:.1f}x"
        if estimated:
            label_text += " (est.)"
        ax2.text(xs[i], max_sp * 1.35, label_text, ha="center", va="bottom", fontsize=6.8, color=INK)

    ax2.set_yscale("log")
    plain_log_axis(ax2, "y")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([k[0] for k in keys], fontsize=6.8)
    ax2.set_xlim(xs[0] - 0.7, xs[-1] + 0.7)
    ax2.set_ylim(0.4, 9500.0)
    ax2.set_xlabel("strux-rs biophysics & design primitive")
    ax2.set_ylabel("speedup over standard Python stack")

    # Legend for hardware tiers
    legend_handles = [
        mpl.lines.Line2D([0], [0], marker="s", color="w", markerfacecolor=COBALT, markeredgecolor=PAPER, markersize=5, label="RTX A2000 (Laptop)"),
        mpl.lines.Line2D([0], [0], marker="o", color="w", markerfacecolor=RUST, markeredgecolor=PAPER, markersize=5, label="NVIDIA L4 (Cloud)"),
        mpl.lines.Line2D([0], [0], marker="D", color="w", markerfacecolor=JADE, markeredgecolor=PAPER, markersize=5, label="NVIDIA H100 (SXM)"),
    ]
    ax2.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=6.4, handletextpad=0.3)

    min_sp, max_sp = min(all_measured_sp), max(all_measured_sp)
    titles(
        ax2,
        f"Zero-C primitives accelerate pipelines {min_sp:.0f}x to {max_sp:,.0f}x",
        f"strux-rs 0.5.1 suite across Laptop, Cloud, & Data Center; hollow = estimated baseline",
    )

    fig.tight_layout(w_pad=3.8)

    # Save to both target filenames
    out_all = bdir / "hardware_scaling_comparison.png"
    out_l4 = bdir / "hardware_l4_scaling_comparison.png"
    out_artifact = Path(r"C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f\hardware_scaling_comparison.png")
    out_artifact_l4 = Path(r"C:\Users\Gebruiker\.gemini\antigravity\brain\533f8aaa-3288-419b-a6a1-a41762761b7f\hardware_l4_scaling_comparison.png")

    fig.savefig(out_all, bbox_inches="tight", dpi=240)
    fig.savefig(out_l4, bbox_inches="tight", dpi=240)
    try:
        fig.savefig(out_artifact, bbox_inches="tight", dpi=240)
        fig.savefig(out_artifact_l4, bbox_inches="tight", dpi=240)
    except Exception:
        pass
    plt.close(fig)
    print("Generated hardware_scaling_comparison.png and updated hardware_l4_scaling_comparison.png.")


if __name__ == "__main__":
    main()
