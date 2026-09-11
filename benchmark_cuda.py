# ponytail: automated validation and benchmark comparing CPU Kabsch vs CUDA QCP.
import os
import time
import numpy as np
import strux_rs

def main():
    traj_path = "trajectory.pdb"
    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found.")
        return

    print("=== STRUX-RS BIOPHYSICAL BENCHMARK ===")
    print(f"Loading {traj_path}...")
    t0 = time.perf_counter()
    traj = strux_rs.parse_pdb(traj_path)
    t_parse = time.perf_counter() - t0
    num_frames, num_atoms, _ = traj.shape
    print(f"Parsed {num_frames} frames ({num_atoms} atoms each) in {t_parse:.4f}s.")

    cuda_available = strux_rs.cuda_is_available()
    print(f"CUDA Hardware Acceleration Available: {cuda_available}")

    if not cuda_available:
        print("\nNote: Running on CPU-only host. GPU CUDA acceleration is compiled in with --features cuda.")
        print("Benchmarking CPU Kabsch baseline across all pairs...")
        
        # Benchmark CPU all-to-all on first 50 frames
        n_test = min(50, num_frames)
        t0 = time.perf_counter()
        cpu_mat = np.zeros((n_test, n_test), dtype=np.float32)
        for i in range(n_test):
            for j in range(i + 1, n_test):
                rmsd = strux_rs.calculate_rmsd_kabsch(traj[i], traj[j])
                cpu_mat[i, j] = rmsd
                cpu_mat[j, i] = rmsd
        t_cpu = time.perf_counter() - t0
        print(f"CPU All-to-all Kabsch ({n_test} frames, {n_test*(n_test-1)//2} alignments): {t_cpu:.4f}s")
        print(f"Projected CPU time for 50,000 frames: {(t_cpu / (n_test*(n_test-1)/2) * (50000*49999/2)) / 3600:.2f} hours.")
        print("\nSUCCESS: Baseline ready. Deploy to GPU workstation (agni) to execute CUDA QCP acceleration.")
        return

    # CUDA is available!
    print(f"\n1. RUNNING GPU QCP PAIRWISE RMSD ({num_frames}x{num_frames} = {num_frames*num_frames} pairs)...")
    t0 = time.perf_counter()
    gpu_dist = strux_rs.cuda_pairwise_rmsd(traj)
    t_gpu = time.perf_counter() - t0
    print(f"GPU QCP Execution Time: {t_gpu:.6f}s")

    # Verify numerical parity against CPU Kabsch
    print("\n2. VERIFYING NUMERICAL PARITY (GPU QCP vs CPU Kabsch)...")
    max_err = 0.0
    checked_pairs = 0
    for i in range(min(20, num_frames)):
        for j in range(i + 1, min(20, num_frames)):
            cpu_val = strux_rs.calculate_rmsd_kabsch(traj[i], traj[j])
            gpu_val = gpu_dist[i, j]
            err = abs(cpu_val - gpu_val)
            if err > max_err:
                max_err = err
            checked_pairs += 1

    print(f"Checked {checked_pairs} frame pairs.")
    print(f"Maximum absolute deviation: {max_err:.6e} Å")
    assert max_err < 1e-4, f"Numerical parity failed! Max error = {max_err}"
    print("PASS: GPU QCP matches CPU Kabsch within 10^-4 Å tolerance!")

    # 3. GPU Daura Clustering
    cutoff = 2.0
    print(f"\n3. RUNNING GPU GREEDY DAURA CLUSTERING (cutoff={cutoff} Å)...")
    t0 = time.perf_counter()
    labels, centroids = strux_rs.cuda_cluster_daura(traj, cutoff)
    t_cluster = time.perf_counter() - t0
    print(f"Clustered {num_frames} frames into {len(centroids)} clusters in {t_cluster:.6f}s.")
    print(f"Cluster centroids (frame indices): {centroids[:10]}...")

    # 4. GPU Residue Contact Bitmask
    print(f"\n4. RUNNING GPU PACKED RESIDUE CONTACT MAP (cutoff=4.5 Å)...")
    t0 = time.perf_counter()
    bitmasks = strux_rs.cuda_contact_map_bitmask(traj, 4.5)
    t_contact = time.perf_counter() - t0
    print(f"Computed bitmask shape {bitmasks.shape} in {t_contact:.6f}s.")

    print("\nALL GPU BENCHMARKS AND VALIDATIONS COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
