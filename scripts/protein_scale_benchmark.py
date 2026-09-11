# ponytail: multi-protein size scaling spectrum benchmark (N_atoms = 300 to 10,000+).
import os
import sys
import time
import urllib.request
import numpy as np
import strux_rs

def fetch_or_generate_protein(pdb_id, num_atoms_synth=None):
    if num_atoms_synth is not None:
        # Synthetic large macromolecular complex
        print(f"\nSynthesizing macromolecular complex with {num_atoms_synth:,d} atoms...")
        coords = np.random.randn(num_atoms_synth, 3).astype(np.float32) * 20.0
        return coords, f"Macromolecular Complex ({num_atoms_synth:,d} atoms)"

    path = f"{pdb_id}.pdb"
    if not os.path.exists(path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        print(f"Downloading {pdb_id} from RCSB ({url})...")
        urllib.request.urlretrieve(url, path)

    traj = strux_rs.parse_pdb(path)
    return traj[0], f"PDB {pdb_id}"

def main():
    print("=" * 80)
    print("  STRUX-RS MULTI-PROTEIN SIZE SCALING SPECTRUM BENCHMARK")
    print("  Comparing 16-Core Core i9 vs RTX A2000 across N_atoms = 300 to 10,000+")
    print("=" * 80)

    test_targets = [
        ("1L2Y", None, "Trp-cage Mini-Protein (Small)"),
        ("1GFL", None, "GFP Globular Domain (Medium)"),
        ("1HTQ", None, "Hemoglobin Tetramer (Large)"),
        (None, 10000, "Macromolecular Complex (Very Large)"),
    ]

    T_FRAMES = 1000
    TOTAL_PAIRS = T_FRAMES * T_FRAMES

    print(f"\nSweep Configuration: Fixed T = {T_FRAMES:,d} frames ({TOTAL_PAIRS:,d} pairwise alignments per target)\n")
    print(f"{'Target':<30} | {'Atoms (N)':>10} | {'16-Core CPU Time':>18} | {'RTX A2000 Time':>16} | {'GPU Throughput':>18} | {'Speedup':>9}")
    print("-" * 115)

    for pdb_id, synth_n, desc in test_targets:
        base_coords, label = fetch_or_generate_protein(pdb_id, synth_n)
        n_atoms = base_coords.shape[0]

        # Generate realistic 1,000-frame ensemble
        np.random.seed(42)
        noise = np.random.normal(0, 0.05, size=(T_FRAMES, n_atoms, 3)).astype(np.float32)
        ensemble = np.ascontiguousarray(base_coords[np.newaxis, :, :] + noise, dtype=np.float32)

        # 16-Core CPU Rayon
        t0 = time.perf_counter()
        _ = strux_rs.cpu_pairwise_rmsd(ensemble)
        t_cpu = time.perf_counter() - t0
        cpu_rate = TOTAL_PAIRS / t_cpu

        # GPU RTX A2000
        t0 = time.perf_counter()
        _ = strux_rs.cuda_pairwise_rmsd(ensemble)
        t_gpu = time.perf_counter() - t0
        gpu_rate = TOTAL_PAIRS / t_gpu
        speedup = t_cpu / t_gpu

        print(f"{desc:<30} | {n_atoms:>10,d} | {t_cpu:>17.3f}s | {t_gpu:>15.4f}s | {gpu_rate:>16,.1f}/s | {speedup:>8.1f}x")

    print("=" * 115)
    print("  MULTI-PROTEIN SIZE SCALING BENCHMARK COMPLETE")
    print("=" * 115)

if __name__ == "__main__":
    main()
