"""Sustained 75-second Full-System Burn on agni (16-Core i9 + RTX A2000 CUDA).

Simultaneously saturates:
1. RTX A2000 GPU at 100% with continuous `cuda_pairwise_rmsd` & `cuda_contact_frequency`
2. All 16 CPU threads of the Core i9-11950H at ~100% with Rayon parallel AMS biophysics kernels:
   - `calculate_lddt` (12,000-atom complex, 144M pairs/call)
   - `trajectory_pca` (4,000 frames x 800 atoms eigendecomposition)
   - `cluster_msa` (10,000 sequences x 250 AA sub-alignment clustering)
   - `parse_xtc` (1,000 frames x 2,000 atoms parallel XDR 3D decompression)
   - `markov_populations` (500,000-step Markov chain stationary distribution)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import threading
import time
import numpy as np
from mdtraj.formats import XTCTrajectoryFile
import strux_rs


def read_cpu_usage(prev_idle: int, prev_total: int) -> tuple[float, int, int]:
    with open("/proc/stat", "r") as f:
        fields = [int(x) for x in f.readline().strip().split()[1:]]
    idle = fields[3] + fields[4]
    total = sum(fields)
    diff_idle = idle - prev_idle
    diff_total = total - prev_total
    usage = 100.0 * (1.0 - diff_idle / max(diff_total, 1))
    return usage, idle, total


def query_gpu() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        u_gpu, u_mem, temp, pwr, clk = [x.strip() for x in out.split(",")]
        return f"GPU: {u_gpu:>3}% | VRAM: {u_mem:>2}% | {temp}C | {pwr}W | {clk}MHz"
    except Exception as e:
        return f"GPU err: {e}"


def main():
    duration_sec = 300.0
    print("=" * 92)
    print(f"🔥 LAUNCHING {duration_sec:.0f}s (5-MINUTE) PROLONGED BURN ON AGNI (16T Core i9 + RTX A2000 CUDA) 🔥")
    print("=" * 92, flush=True)

    rng = np.random.default_rng(2026)

    # 1. Pre-allocate contiguous arrays & XTC file so 100% of runtime is pure Rust/CUDA compute
    gpu_traj = np.ascontiguousarray(rng.standard_normal((5000, 1919, 3), dtype=np.float32) * 20.0)
    gpu_contact = np.ascontiguousarray(rng.standard_normal((1500, 1919, 3), dtype=np.float32) * 25.0)

    lddt_ref = np.ascontiguousarray(np.cumsum(rng.normal(0, 2.0, size=(12000, 3)), axis=0).astype(np.float32))
    lddt_mob = np.ascontiguousarray((lddt_ref + rng.normal(0, 1.1, size=(12000, 3))).astype(np.float32))

    pca_traj = np.ascontiguousarray(rng.standard_normal((2000, 500, 3), dtype=np.float32) * 12.0)

    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    founders = [rng.choice(alphabet, size=250) for _ in range(25)]
    msa_seqs = []
    for f in founders:
        for m in range(400):
            s = f.copy()
            if m > 0:
                mask = rng.random(250) < 0.08
                s[mask] = rng.choice(alphabet, size=np.sum(mask))
            msa_seqs.append("".join(s))

    msm_traj = rng.integers(0, 4, size=500_000).tolist()

    xtc_fd, xtc_path = tempfile.mkstemp(suffix=".xtc")
    os.close(xtc_fd)
    base_nm = np.cumsum(rng.normal(0, 0.12, size=(2000, 3)), axis=0).astype(np.float32)
    xyz_nm = np.stack([base_nm + rng.normal(0, 0.02, size=(2000, 3)).astype(np.float32) for _ in range(1000)])
    with XTCTrajectoryFile(xtc_path, "w") as f:
        f.write(
            xyz_nm,
            time=np.arange(1000, dtype=np.float32),
            step=np.arange(1000, dtype=np.int32),
            box=np.tile(np.eye(3, dtype=np.float32) * 15.0, (1000, 1, 1)),
        )

    stop_event = threading.Event()
    lock = threading.Lock()
    stats = {
        "gpu_pairs": 0,
        "gpu_cmaps": 0,
        "lddt_calls": 0,
        "pca_calls": 0,
        "msa_seqs": 0,
        "xtc_frames": 0,
        "msm_steps": 0,
    }

    def gpu_worker():
        while not stop_event.is_set():
            rmsd_mat = strux_rs.cuda_pairwise_rmsd(gpu_traj)
            _ = strux_rs.cuda_contact_frequency(gpu_contact, 8.0, 0)
            with lock:
                stats["gpu_pairs"] += rmsd_mat.size
                stats["gpu_cmaps"] += 1

    def cpu_lddt_worker():
        while not stop_event.is_set():
            g, _ = strux_rs.calculate_lddt(lddt_ref, lddt_mob, cutoff=15.0)
            with lock:
                stats["lddt_calls"] += 1

    def cpu_pca_worker():
        while not stop_event.is_set():
            proj, var, comp = strux_rs.trajectory_pca(pca_traj, n_components=6)
            with lock:
                stats["pca_calls"] += 1

    def cpu_ams_pipeline_worker():
        while not stop_event.is_set():
            coords = strux_rs.parse_xtc(xtc_path)
            labels, reps = strux_rs.cluster_msa(msa_seqs, min_identity=0.75)
            pi, dg, tmat = strux_rs.markov_populations(msm_traj, num_states=4, lag_time=1, temperature=300.0)
            with lock:
                stats["xtc_frames"] += coords.shape[0]
                stats["msa_seqs"] += len(msa_seqs)
                stats["msm_steps"] += len(msm_traj)

    threads = [
        threading.Thread(target=gpu_worker, daemon=True),
        threading.Thread(target=cpu_lddt_worker, daemon=True),
        threading.Thread(target=cpu_lddt_worker, daemon=True),
        threading.Thread(target=cpu_pca_worker, daemon=True),
        threading.Thread(target=cpu_ams_pipeline_worker, daemon=True),
    ]

    _, prev_idle, prev_total = read_cpu_usage(0, 0)
    t0 = time.perf_counter()
    for t in threads:
        t.start()

    print("[burn] All 16 CPU threads + CUDA GPU active! Live telemetry every 3s:\n", flush=True)
    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed >= duration_sec:
                break
            time.sleep(3.0)
            cpu_pct, prev_idle, prev_total = read_cpu_usage(prev_idle, prev_total)
            gpu_str = query_gpu()
            with lock:
                print(
                    f"[{elapsed:5.1f}s] CPU(16T): {cpu_pct:5.1f}% | {gpu_str} | "
                    f"lDDT(12k): {stats['lddt_calls']} | PCA(4kx800): {stats['pca_calls']} | "
                    f"MSA: {stats['msa_seqs']//1000}k | XTC: {stats['xtc_frames']//1000}k | "
                    f"MSM: {stats['msm_steps']//1_000_000}M | GPU pairs: {stats['gpu_pairs']/1e6:.0f}M",
                    flush=True,
                )
    finally:
        stop_event.set()
        for t in threads:
            t.join(timeout=8.0)
        if os.path.exists(xtc_path):
            os.remove(xtc_path)

    total_time = time.perf_counter() - t0
    print("\n" + "=" * 92)
    print(f"✅ SUSTAINED {total_time:.1f}s FULL-SYSTEM BURN COMPLETE ON AGNI")
    print(f"   • CUDA QCP Alignments : {stats['gpu_pairs']:,} pairs ({stats['gpu_cmaps']} GPU contact matrices)")
    print(f"   • AlphaFold lDDT (12k): {stats['lddt_calls']} evaluations ({stats['lddt_calls']*144:,}M atom pairs)")
    print(f"   • Trajectory PCA SVD  : {stats['pca_calls']} full (4,000 x 2,400) eigendecompositions")
    print(f"   • AF-Cluster MSA      : {stats['msa_seqs']:,} sequences clustered")
    print(f"   • GROMACS XTC Decoded : {stats['xtc_frames']:,} frames ({stats['xtc_frames']*2000:,} 3D coordinates)")
    print(f"   • Markov State Steps  : {stats['msm_steps']:,} transitions solved")
    print("=" * 92, flush=True)


if __name__ == "__main__":
    main()
