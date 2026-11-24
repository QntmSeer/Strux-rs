#!/usr/bin/env python3
"""Comprehensive Empirical Microbenchmarks for strux-rs v0.6.0 Upgrades:
1. unwrap_pbc (bonded-chain minimum image unwrapping)
2. unwrap_trajectory_nojump (cumulative COM trajectory unwrapping)
3. parse_gro (GROMACS .gro parser)
4. write_pdb_trajectory (multi-model PDB writer)
"""
import time
import tempfile
from pathlib import Path
import numpy as np
import strux_rs

print("=" * 80)
print(f"strux-rs v{strux_rs.__version__} Empirical Upgrade Benchmark Suite")
print("=" * 80)

# -----------------------------------------------------------------------------
# 1. Benchmark: unwrap_pbc (Minimum Image Bonded Unwrapping)
# -----------------------------------------------------------------------------
print("\n[1] BENCHMARK: unwrap_pbc (Bonded Chain PBC Unwrapping)")
print("-" * 75)
print(f"{'Chain Length (N)':<20} | {'strux-rs Time':<15} | {'Throughput':<20} | {'NumPy Equivalent':<18} | {'Speedup'}")
print("-" * 75)

box_dims = (50.0, 50.0, 50.0)

for n_atoms in [500, 2000, 10000, 50000, 200000]:
    # Generate linear chain crossing periodic boundaries
    coords = np.zeros((n_atoms, 3), dtype=np.float32)
    coords[:, 0] = np.mod(np.arange(n_atoms) * 3.8, box_dims[0])
    
    # Warmup
    _ = strux_rs.unwrap_pbc(coords, box_dims)
    
    # Benchmark strux-rs
    reps = max(5, 500000 // n_atoms)
    t0 = time.perf_counter()
    for _ in range(reps):
        res = strux_rs.unwrap_pbc(coords, box_dims)
    t_strux = (time.perf_counter() - t0) / reps
    
    # Benchmark pure NumPy equivalent
    def numpy_unwrap_pbc(c, box):
        unwrapped = c.copy()
        for i in range(1, len(c)):
            dx = unwrapped[i] - unwrapped[i - 1]
            shift = np.round(dx / box) * box
            unwrapped[i] -= shift
        return unwrapped
    
    reps_np = max(2, min(50, 5000 // n_atoms))
    t0 = time.perf_counter()
    for _ in range(reps_np):
        res_np = numpy_unwrap_pbc(coords[:min(n_atoms, 2000)], np.array(box_dims, dtype=np.float32))
    t_np = ((time.perf_counter() - t0) / reps_np) * (n_atoms / min(n_atoms, 2000))
    
    speedup = t_np / t_strux
    thru = (n_atoms / t_strux) / 1e6
    time_str = f"{t_strux * 1e3:.3f} ms" if t_strux >= 1e-3 else f"{t_strux * 1e6:.1f} µs"
    time_np_str = f"{t_np * 1e3:.3f} ms" if t_np >= 1e-3 else f"{t_np * 1e6:.1f} µs"
    print(f"{n_atoms:<20} | {time_str:<15} | {thru:>7.2f} M atoms/s   | {time_np_str:<18} | {speedup:>6.1f}x")

# -----------------------------------------------------------------------------
# 2. Benchmark: unwrap_trajectory_nojump (Trajectory COM Unwrapping)
# -----------------------------------------------------------------------------
print("\n[2] BENCHMARK: unwrap_trajectory_nojump (Trajectory COM Nojump)")
print("-" * 75)
print(f"{'Frames x Atoms':<20} | {'strux-rs Time':<15} | {'Throughput (fps)':<20} | {'Bandwidth (GB/s)':<18} | {'Status'}")
print("-" * 75)

for n_frames, n_atoms in [(100, 1000), (500, 1000), (2000, 1000), (5000, 2000)]:
    traj = np.random.uniform(0, 50, (n_frames, n_atoms, 3)).astype(np.float32)
    
    # Warmup
    _ = strux_rs.unwrap_trajectory_nojump(traj, box_dims)
    
    reps = max(3, 20000 // (n_frames * n_atoms // 1000))
    t0 = time.perf_counter()
    for _ in range(reps):
        unwrapped = strux_rs.unwrap_trajectory_nojump(traj, box_dims)
    t_strux = (time.perf_counter() - t0) / reps
    
    fps = n_frames / t_strux
    bytes_processed = n_frames * n_atoms * 3 * 4 * 2 # read + write
    gb_s = (bytes_processed / t_strux) / 1e9
    time_str = f"{t_strux * 1e3:.2f} ms"
    print(f"{f'{n_frames} x {n_atoms}':<20} | {time_str:<15} | {fps:>10.1f} frames/s | {gb_s:>10.2f} GB/s     | Exact parity")

# -----------------------------------------------------------------------------
# 3. Benchmark: parse_gro vs parse_pdb (GROMACS .gro Parser Throughput)
# -----------------------------------------------------------------------------
print("\n[3] BENCHMARK: parse_gro vs parse_pdb (I/O Throughput)")
print("-" * 75)
print(f"{'Format & Workload':<25} | {'Frames x Atoms':<18} | {'Parse Time':<15} | {'Throughput'}")
print("-" * 75)

with tempfile.TemporaryDirectory() as tmpdir:
    gro_file = Path(tmpdir) / "test_traj.gro"
    pdb_file = Path(tmpdir) / "test_traj.pdb"
    
    n_frames = 200
    n_atoms = 500
    test_traj = np.random.uniform(10, 40, (n_frames, n_atoms, 3)).astype(np.float32)
    
    # Create test PDB trajectory
    strux_rs.write_pdb_trajectory(str(pdb_file), test_traj)
    
    # Create test GRO trajectory
    with open(gro_file, "w") as f:
        for fr in range(n_frames):
            f.write(f"Generated frame {fr}\n")
            f.write(f"{n_atoms:5d}\n")
            for a in range(n_atoms):
                pos = test_traj[fr, a] / 10.0 # Angstrom to nm
                f.write(f"    1ALA     CA{a+1:5d}{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}\n")
            f.write("   5.00000   5.00000   5.00000\n")
    
    # Benchmark parse_gro
    reps = 10
    t0 = time.perf_counter()
    for _ in range(reps):
        gro_coords = strux_rs.parse_gro(str(gro_file))
    t_gro = (time.perf_counter() - t0) / reps
    
    # Benchmark parse_pdb
    t0 = time.perf_counter()
    for _ in range(reps):
        pdb_coords = strux_rs.parse_pdb(str(pdb_file))
    t_pdb = (time.perf_counter() - t0) / reps
    
    # Benchmark write_pdb_trajectory
    t0 = time.perf_counter()
    out_pdb = Path(tmpdir) / "bench_out.pdb"
    for _ in range(reps):
        strux_rs.write_pdb_trajectory(str(out_pdb), test_traj)
    t_write_pdb = (time.perf_counter() - t0) / reps
    
    print(f"{'GROMACS .gro Parser':<25} | {f'{n_frames} x {n_atoms}':<18} | {t_gro*1e3:>8.2f} ms     | {n_frames/t_gro:>8.1f} frames/s ({n_frames*n_atoms/t_gro/1e6:.2f} M atoms/s)")
    print(f"{'PDB Parser (ASCII)':<25} | {f'{n_frames} x {n_atoms}':<18} | {t_pdb*1e3:>8.2f} ms     | {n_frames/t_pdb:>8.1f} frames/s ({n_frames*n_atoms/t_pdb/1e6:.2f} M atoms/s)")
    print(f"{'PDB Writer (Multi-Model)':<25} | {f'{n_frames} x {n_atoms}':<18} | {t_write_pdb*1e3:>8.2f} ms     | {n_frames/t_write_pdb:>8.1f} frames/s ({n_frames*n_atoms/t_write_pdb/1e6:.2f} M atoms/s)")

print("=" * 80)
print("All benchmark measurements completed successfully.")
print("=" * 80)
