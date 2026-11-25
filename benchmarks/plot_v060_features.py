"""Figure for the strux-rs v0.6.0 additions, in the shared benchmarks/ house style.

Panels A and B are correctness demonstrations computed here from a seeded synthetic
system, so they are deterministic. Panels C and D are timings read from
benchmarks/v060_results.json, which benchmark_v060.py writes; the script fails rather
than draw literals if that file is missing.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import strux_rs

from plot_style import (
    COBALT, DASH, DOT, INK, INK2, JADE, OCHRE, PLUM, RUST,
    plain_log_axis, plt, provenance_footnote, setup, titles,
)

BDIR = Path(__file__).resolve().parent
RESULTS = BDIR / "v060_results.json"
OUT = BDIR / "v060_features_verification.png"


def load_results() -> dict:
    if not RESULTS.exists():
        raise SystemExit(f"missing {RESULTS}: run benchmarks/benchmark_v060.py first")
    return json.loads(RESULTS.read_text())


def build_label(d: dict) -> str:
    ver, commit = d.get("strux_rs_version", "unknown"), d.get("git_commit")
    return f"{ver} ({commit})" if commit and commit != "unknown" else ver


def panel_a(ax) -> None:
    """Bonded chain crossing the periodic boundary, before and after unwrap_pbc."""
    box = (50.0, 50.0, 50.0)
    n = 30
    coords = np.zeros((n, 3), dtype=np.float32)
    coords[:, 0] = np.mod(np.arange(n) * 3.8, box[0])  # wraps twice across 50 A
    unwrapped = strux_rs.unwrap_pbc(coords, box)

    wrapped_d = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    unwrapped_d = np.linalg.norm(np.diff(unwrapped, axis=0), axis=1)
    x = np.arange(1, n)

    ax.axhline(3.8, color=INK2, lw=0.9, ls=DASH)
    ax.plot(x, wrapped_d, "-o", color=RUST, ms=3.4, mec="white", mew=0.6, lw=1.3)
    ax.plot(x, unwrapped_d, "-s", color=JADE, ms=3.4, mec="white", mew=0.6, lw=1.3)

    worst = float(wrapped_d.max())
    ax.set_ylim(-3, worst * 1.30)
    ax.text(0.5, worst * 1.12, f"raw PBC, {worst:.1f} A apparent jump", fontsize=6.4, color=RUST)
    ax.text(0.5, 7.4, "unwrap_pbc, bonded spacing preserved", fontsize=6.4, color=JADE)
    ax.text(1.0, -1.6, "ideal CA-CA 3.8 A", fontsize=6.2, color=INK2, va="center")
    ax.set_xlabel("residue pair index (i to i+1)")
    ax.set_ylabel("apparent CA-CA distance (A)")
    titles(ax, "A. Bonded chain unwrapping",
           f"{n}-residue chain in a {box[0]:.0f} A cubic box; max error removed: {worst - 3.8:.1f} A")


def panel_b(ax) -> None:
    """Centre of mass of a diffusing molecule, before and after nojump unwrapping."""
    box = (50.0, 50.0, 50.0)
    rng = np.random.default_rng(3)
    n_frames, n_atoms = 150, 40
    # A whole molecule drifting steadily in x, written back into the box each frame:
    # the molecule stays intact, so only its centre of mass jumps by one box length.
    true_com = 25.0 + 0.62 * np.arange(n_frames) + np.cumsum(rng.normal(0, 0.25, n_frames))
    offsets = rng.normal(0, 1.5, (n_atoms, 3))
    offsets -= offsets.mean(axis=0)
    traj = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for f in range(n_frames):
        traj[f] = offsets + np.array([np.mod(true_com[f], box[0]), 25.0, 25.0])
    raw = traj.mean(axis=1)[:, 0]
    fixed = strux_rs.unwrap_trajectory_nojump(traj, box).mean(axis=1)[:, 0]

    t = np.arange(n_frames)
    ax.plot(t, raw, color=RUST, lw=1.2)
    ax.plot(t, fixed, color=COBALT, lw=1.4)
    ax.set_xlabel("trajectory frame")
    ax.set_ylabel("COM x-coordinate (A)")
    ax.set_xlim(-2, n_frames + 34)
    n_jumps = int((np.abs(np.diff(raw)) > box[0] / 2).sum())
    ax.text(n_frames + 2, raw[-1], " raw, wrapped", fontsize=6.4, color=RUST, va="center")
    ax.text(n_frames + 2, fixed[-1], " nojump", fontsize=6.4, color=COBALT, va="center")
    ax.axhline(box[0], color=INK2, lw=0.8, ls=DASH)
    ax.text(0, box[0], " box edge 50 A", fontsize=6.2, color=INK2, va="bottom")
    titles(ax, "B. Trajectory COM nojump unwrapping",
           f"{n_frames} frames; {n_jumps} box-crossing discontinuities removed")


def panel_c(ax, recs: list[dict]) -> None:
    """unwrap_pbc throughput against the NumPy loop it replaces."""
    rows = sorted([r for r in recs if r["benchmark"] == "unwrap_pbc"], key=lambda r: r["n_atoms"])
    if not rows:
        raise SystemExit("v060_results.json has no unwrap_pbc rows")
    n = np.array([r["n_atoms"] for r in rows], float)
    strux = np.array([r["strux_ms"] for r in rows])
    base = np.array([r["baseline_ms"] for r in rows])

    ax.plot(n, base, "-o", color=RUST, ms=4.2, mec="white", mew=0.6, label="NumPy loop")
    ax.plot(n, strux, "-D", color=JADE, ms=4.2, mec="white", mew=0.6, label="strux-rs")
    ax.set_xscale("log")
    ax.set_yscale("log")
    plain_log_axis(ax, "x")
    plain_log_axis(ax, "y")
    ax.set_xlabel("chain length (atoms)")
    ax.set_ylabel("time per call (ms, log)")
    ax.set_xlim(n[0] * 0.55, n[-1] * 4.2)
    ax.text(n[-1] * 1.25, base[-1], " NumPy", fontsize=6.6, color=RUST, va="center")
    ax.text(n[-1] * 1.25, strux[-1], " strux-rs", fontsize=6.6, color=JADE, va="center")
    best = max(r["speedup"] for r in rows)
    titles(ax, f"C. unwrap_pbc is up to {best:.0f}x the NumPy loop",
           "same minimum-image result; NumPy timed on 2,000 atoms and scaled (the loop is O(N))")


def panel_d(ax, recs: list[dict]) -> None:
    """Measured read/write throughput per trajectory format."""
    rows = [r for r in recs if r["benchmark"] == "trajectory_io"]
    if not rows:
        raise SystemExit("v060_results.json has no trajectory_io rows")
    rows.sort(key=lambda r: r["frames_per_s"])
    colours = {"XTC": COBALT, "DCD": JADE, "GRO": OCHRE, "PDB (ASCII)": RUST}

    def colour(label: str) -> str:
        for k, c in colours.items():
            if label.startswith(k):
                return c
        return PLUM

    lo = min(r["frames_per_s"] for r in rows)
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows):
        fps = r["frames_per_s"]
        c = colour(r["format"])
        ax.hlines(y, lo * 0.45, fps, color=c, lw=2.2)
        ax.plot(fps, y, "o", color=c, ms=5.0, mec="white", mew=0.7, zorder=3)
        ax.text(fps * 1.04, y, f" {fps:,.0f} fps", fontsize=6.6, color=INK, va="center")

    ax.set_yticks(ys, [r["format"] for r in rows], fontsize=6.8)
    ax.set_xscale("log")
    plain_log_axis(ax, "x")
    ax.set_xlim(lo * 0.45, max(r["frames_per_s"] for r in rows) * 4.5)
    ax.set_ylim(-0.8, len(rows) - 0.3)
    ax.grid(False)
    ax.xaxis.grid(True, color="#e2dacb", lw=0.8, ls=DOT)
    ax.set_xlabel("frames / second (log scale)")
    n_f, n_a = rows[0]["n_frames"], rows[0]["n_atoms"]
    titles(ax, "D. Trajectory I/O throughput",
           f"{n_f} frames x {n_a} atoms, single thread, warm page cache")


def main() -> None:
    data = load_results()
    recs = data.get("benchmarks", [])
    setup()
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6))
    panel_a(axes[0][0])
    panel_b(axes[0][1])
    panel_c(axes[1][0], recs)
    panel_d(axes[1][1], recs)
    fig.tight_layout(h_pad=3.0, w_pad=3.4)
    hw = data.get("hardware", {})
    provenance_footnote(
        fig,
        f"strux-rs {build_label(data)} on {hw.get('cpu_model', 'unknown CPU')}; "
        f"panels C and D read from benchmarks/{RESULTS.name}, panels A and B computed from a seeded synthetic system.",
    )
    fig.savefig(OUT, bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
