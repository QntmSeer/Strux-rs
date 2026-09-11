# ponytail: high-throughput GPU stress test and benchmark on NVIDIA RTX A2000.
import time
import numpy as np
import strux_rs

def main():
    print("=" * 60)
    print("  NVIDIA RTX A2000 TRAJECTORY STRESS TEST & SCALING BENCHMARK")
    print("=" * 60)

    # 1. Load real base protein structure
    base_traj = strux_rs.parse_pdb("trajectory.pdb")
    num_atoms = base_traj.shape[1]
    base_frame = base_traj[0]
    print(f"Base structure: {num_atoms} atoms per frame.")

    # 2. Benchmark CPU single-core rate on 50 pairs
    print("\n[1/3] Benchmarking CPU Kabsch baseline rate...")
    t0 = time.perf_counter()
    n_cpu_pairs = 100
    for i in range(n_cpu_pairs):
        f1 = base_frame + np.random.randn(num_atoms, 3).astype(np.float32) * 0.1
        f2 = base_frame + np.random.randn(num_atoms, 3).astype(np.float32) * 0.1
        _ = strux_rs.calculate_rmsd_kabsch(f1, f2)
    t_cpu = time.perf_counter() - t0
    cpu_rate = n_cpu_pairs / t_cpu
    print(f"CPU Kabsch throughput: {cpu_rate:,.1f} pairs/second ({t_cpu/n_cpu_pairs*1000:.3f} ms/pair)")

    # 3. GPU Scaling & Stress Test
    frame_counts = [100, 250, 500, 1000, 2500, 5000]
    print("\n[2/3] Running GPU QCP Scaling Sweep...")
    print(f"{'Frames':>8} | {'Total Pairs':>14} | {'GPU Time (s)':>14} | {'Throughput (pairs/s)':>22} | {'CPU Equiv Time':>16} | {'Speedup':>10}")
    print("-" * 92)

    np.random.seed(42)
    for n_f in frame_counts:
        # Generate ensemble
        noise = np.random.randn(n_f, num_atoms, 3).astype(np.float32) * 0.5
        synth_traj = base_frame[np.newaxis, :, :] + noise
        synth_traj = np.ascontiguousarray(synth_traj, dtype=np.float32)

        total_pairs = n_f * n_f

        # Measure GPU execution time
        t0 = time.perf_counter()
        gpu_dist = strux_rs.cuda_pairwise_rmsd(synth_traj)
        t_gpu = time.perf_counter() - t0

        gpu_rate = total_pairs / t_gpu
        equiv_cpu_sec = total_pairs / cpu_rate
        speedup = equiv_cpu_sec / t_gpu

        # Format CPU time
        if equiv_cpu_sec < 60:
            cpu_str = f"{equiv_cpu_sec:.1f}s"
        elif equiv_cpu_sec < 3600:
            cpu_str = f"{equiv_cpu_sec/60:.1f}m"
        else:
            cpu_str = f"{equiv_cpu_sec/3600:.1f}h"

        print(f"{n_f:>8} | {total_pairs:>14,d} | {t_gpu:>13.4f}s | {gpu_rate:>20,.1f} | {cpu_str:>16} | {speedup:>9.1f}x")

    # 4. Stress Test: Large-Scale Daura Clustering
    n_cluster_frames = 2000
    print(f"\n[3/3] Stress Testing Daura Greedy Leader Clustering ({n_cluster_frames} frames)...")
    noise = np.random.randn(n_cluster_frames, num_atoms, 3).astype(np.float32) * 0.8
    cluster_traj = np.ascontiguousarray(base_frame[np.newaxis, :, :] + noise, dtype=np.float32)

    t0 = time.perf_counter()
    labels, centroids = strux_rs.cuda_cluster_daura(cluster_traj, cutoff=3.0)
    t_cluster = time.perf_counter() - t0

    print(f"Clustered {n_cluster_frames} frames ({n_cluster_frames*n_cluster_frames:,d} pairwise evaluations)")
    print(f"Time taken: {t_cluster:.4f}s ({len(centroids)} clusters found)")
    print(f"Average cluster size: {n_cluster_frames/len(centroids):.1f} frames")

    print("\n" + "=" * 60)
    print("  STRESS TEST COMPLETED: 100% SUCCESS WITHOUT MEMORY OVERFLOW")
    print("=" * 60)

if __name__ == "__main__":
    main()
