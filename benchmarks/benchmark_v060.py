#!/usr/bin/env python3
"""Empirical microbenchmarks for the strux-rs v0.6.0 additions.

Measures:
  1. unwrap_pbc                  bonded-chain minimum-image unwrapping
  2. unwrap_trajectory_nojump    cumulative COM trajectory unwrapping
  3. trajectory I/O              .gro / .pdb parsing and PDB writing, plus
                                 .xtc / .dcd reads when mdtraj is available to
                                 write the fixtures
Every number is written to benchmarks/v060_results.json together with the
version, git commit and host, so benchmarks/plot_v060_features.py can draw the
figure from measurements instead of literals.
"""
from __future__ import annotations

import json
import platform
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import strux_rs

OUT_JSON = Path(__file__).resolve().parent / "v060_results.json"
records: list[dict] = []


def record(name: str, **fields) -> None:
    records.append({"benchmark": name, **fields})


def git_commit() -> str:
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


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


print("=" * 80)
print(f"strux-rs v{strux_rs.__version__} Empirical Upgrade Benchmark Suite")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. unwrap_pbc (minimum-image bonded unwrapping)
# -----------------------------------------------------------------------------
print("\n[1] BENCHMARK: unwrap_pbc (Bonded Chain PBC Unwrapping)")
print("-" * 75)
print(f"{'Chain Length (N)':<20} | {'strux-rs Time':<15} | {'Throughput':<20} | {'NumPy Equivalent':<18} | {'Speedup'}")
print("-" * 75)

box_dims = (50.0, 50.0, 50.0)


def numpy_unwrap_pbc(c, box):
    unwrapped = c.copy()
    for i in range(1, len(c)):
        dx = unwrapped[i] - unwrapped[i - 1]
        unwrapped[i] -= np.round(dx / box) * box
    return unwrapped


for n_atoms in [500, 2000, 10000, 50000, 200000]:
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    coords[:, 0] = np.mod(np.arange(n_atoms) * 3.8, box_dims[0])

    _ = strux_rs.unwrap_pbc(coords, box_dims)

    reps = max(5, 500000 // n_atoms)
    t0 = time.perf_counter()
    for _ in range(reps):
        strux_rs.unwrap_pbc(coords, box_dims)
    t_strux = (time.perf_counter() - t0) / reps

    # NumPy reference timed on a capped slice, then scaled: the loop is O(N).
    cap = min(n_atoms, 2000)
    reps_np = max(2, min(50, 5000 // n_atoms))
    t0 = time.perf_counter()
    for _ in range(reps_np):
        numpy_unwrap_pbc(coords[:cap], np.array(box_dims, dtype=np.float32))
    t_np = ((time.perf_counter() - t0) / reps_np) * (n_atoms / cap)

    speedup = t_np / t_strux
    thru = (n_atoms / t_strux) / 1e6
    ts = f"{t_strux * 1e3:.3f} ms" if t_strux >= 1e-3 else f"{t_strux * 1e6:.1f} us"
    tn = f"{t_np * 1e3:.3f} ms" if t_np >= 1e-3 else f"{t_np * 1e6:.1f} us"
    print(f"{n_atoms:<20} | {ts:<15} | {thru:>7.2f} M atoms/s   | {tn:<18} | {speedup:>6.1f}x")
    record("unwrap_pbc", n_atoms=n_atoms, strux_ms=t_strux * 1e3,
           baseline_ms=t_np * 1e3, baseline_name="NumPy_loop_scaled_from_2000",
           m_atoms_per_s=thru, speedup=speedup)

# -----------------------------------------------------------------------------
# 2. unwrap_trajectory_nojump (trajectory COM unwrapping)
# -----------------------------------------------------------------------------
print("\n[2] BENCHMARK: unwrap_trajectory_nojump (Trajectory COM Nojump)")
print("-" * 75)
print(f"{'Frames x Atoms':<20} | {'strux-rs Time':<15} | {'Throughput (fps)':<20} | {'Bandwidth (GB/s)':<18}")
print("-" * 75)

rng = np.random.default_rng(0)
for n_frames, n_atoms in [(100, 1000), (500, 1000), (2000, 1000), (5000, 2000)]:
    traj = rng.uniform(0, 50, (n_frames, n_atoms, 3)).astype(np.float32)
    _ = strux_rs.unwrap_trajectory_nojump(traj, box_dims)

    reps = max(3, 20000 // (n_frames * n_atoms // 1000))
    t0 = time.perf_counter()
    for _ in range(reps):
        strux_rs.unwrap_trajectory_nojump(traj, box_dims)
    t_strux = (time.perf_counter() - t0) / reps

    fps = n_frames / t_strux
    gb_s = (n_frames * n_atoms * 3 * 4 * 2 / t_strux) / 1e9  # read + write
    print(f"{f'{n_frames} x {n_atoms}':<20} | {t_strux*1e3:>8.2f} ms     | {fps:>10.1f} frames/s | {gb_s:>10.2f} GB/s")
    record("unwrap_trajectory_nojump", n_frames=n_frames, n_atoms=n_atoms,
           strux_ms=t_strux * 1e3, frames_per_s=fps, gb_per_s=gb_s)

# -----------------------------------------------------------------------------
# 3. Trajectory I/O throughput
# -----------------------------------------------------------------------------
print("\n[3] BENCHMARK: trajectory I/O throughput")
print("-" * 75)
print(f"{'Format & Workload':<26} | {'Frames x Atoms':<16} | {'Time':<12} | {'Throughput'}")
print("-" * 75)

N_FRAMES, N_ATOMS, REPS = 200, 500, 10

with tempfile.TemporaryDirectory() as tmpdir:
    tmp = Path(tmpdir)
    traj = rng.uniform(10, 40, (N_FRAMES, N_ATOMS, 3)).astype(np.float32)

    pdb_file = tmp / "test_traj.pdb"
    gro_file = tmp / "test_traj.gro"
    strux_rs.write_pdb_trajectory(str(pdb_file), traj)
    with open(gro_file, "w") as f:
        for fr in range(N_FRAMES):
            f.write(f"Generated frame {fr}\n{N_ATOMS:5d}\n")
            for a in range(N_ATOMS):
                p = traj[fr, a] / 10.0  # Angstrom -> nm
                f.write(f"    1ALA     CA{a+1:5d}{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
            f.write("   5.00000   5.00000   5.00000\n")

    def bench(fn, reps=REPS):
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    cases = [
        ("GRO (GROMACS) read", lambda: strux_rs.parse_gro(str(gro_file))),
        ("PDB (ASCII) read", lambda: strux_rs.parse_pdb(str(pdb_file))),
        ("PDB multi-model write", lambda: strux_rs.write_pdb_trajectory(str(tmp / "out.pdb"), traj)),
    ]

    # Binary formats need mdtraj to write the fixtures; skip rather than invent numbers.
    try:
        import mdtraj as md
        from mdtraj.formats import DCDTrajectoryFile, XTCTrajectoryFile

        xtc_file, dcd_file = tmp / "test.xtc", tmp / "test.dcd"
        with XTCTrajectoryFile(str(xtc_file), "w") as f:
            f.write(traj / 10.0)
        with DCDTrajectoryFile(str(dcd_file), "w") as f:
            f.write(traj)
        cases += [
            ("XTC (XDR) read", lambda: strux_rs.parse_xtc(str(xtc_file))),
            ("DCD (CHARMM) read", lambda: strux_rs.parse_dcd(str(dcd_file))),
        ]
    except ImportError:
        print("  (mdtraj not installed: skipping XTC and DCD, no fixtures to read)")

    for label, fn in cases:
        t = bench(fn)
        fps = N_FRAMES / t
        print(f"{label:<26} | {f'{N_FRAMES} x {N_ATOMS}':<16} | {t*1e3:>7.2f} ms  | {fps:>9.1f} frames/s")
        record("trajectory_io", format=label, n_frames=N_FRAMES, n_atoms=N_ATOMS,
               strux_ms=t * 1e3, frames_per_s=fps)

payload = {
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "strux_rs_version": getattr(strux_rs, "__version__", "unknown"),
    "strux_rs_file": getattr(strux_rs, "__file__", "unknown"),
    "git_commit": git_commit(),
    "hardware": {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu_model": cpu_model(),
    },
    "benchmarks": records,
}
OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

print("=" * 80)
print(f"Wrote {OUT_JSON.relative_to(Path.cwd()) if OUT_JSON.is_relative_to(Path.cwd()) else OUT_JSON}")
print("=" * 80)
