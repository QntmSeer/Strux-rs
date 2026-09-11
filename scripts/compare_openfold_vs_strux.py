# ponytail: direct head-to-head performance comparison: OpenFold vs strux-rs.
import os
import sys
import time
import string
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer

# Add openfold and strux-rs to path if needed
sys.path.insert(0, os.path.abspath("."))
import strux_rs

# ------------------------------------------------------------------------------
# 1. OpenFold Reference Implementations
# ------------------------------------------------------------------------------
def openfold_superimpose_np(reference, coords):
    """Exact function from openfold/utils/superimposition.py line 19."""
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()

def openfold_parse_a3m(a3m_string: str):
    """Exact legacy parser from openfold/data/parsers.py line 132."""
    def parse_fasta(fasta_string: str):
        sequences = []
        descriptions = []
        index = -1
        for line in fasta_string.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                index += 1
                descriptions.append(line[1:])
                sequences.append("")
                continue
            elif not descriptions:
                raise ValueError("Expected description line")
            sequences[index] += line
        return sequences, descriptions

    sequences, descriptions = parse_fasta(a3m_string)
    deletion_matrix = []
    for msa_sequence in sequences:
        deletion_vec = []
        deletion_count = 0
        for j in msa_sequence:
            if j.islower():
                deletion_count += 1
            else:
                deletion_vec.append(deletion_count)
                deletion_count = 0
        deletion_matrix.append(deletion_vec)

    deletion_table = str.maketrans("", "", string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix, descriptions

# ------------------------------------------------------------------------------
# 2. Benchmark Suite
# ------------------------------------------------------------------------------
def benchmark_superimposition():
    print("\n" + "=" * 80)
    print("BENCHMARK 1: Structural Superimposition & Kabsch RMSD")
    print("Comparing OpenFold's Bio.SVDSuperimposer vs strux-rs")
    print("=" * 80)

    # Load 1,919-atom protein
    pdb_path = "trajectory.pdb"
    if not os.path.exists(pdb_path):
        pdb_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trajectory.pdb")
    traj = strux_rs.parse_pdb(pdb_path)
    base_frame = traj[0]
    num_atoms = base_frame.shape[0]

    N_EVALS = 1000
    print(f"Dataset: {num_atoms:,d} atoms per structure | Evaluating {N_EVALS:,d} superimpositions...")

    np.random.seed(42)
    noise = np.random.normal(0, 0.5, size=(N_EVALS, num_atoms, 3)).astype(np.float32)
    ensemble = np.ascontiguousarray(base_frame[np.newaxis, :, :] + noise, dtype=np.float32)

    # A) OpenFold (Bio.SVDSuperimposer)
    t0 = time.perf_counter()
    of_rmsds = []
    for i in range(N_EVALS):
        _, r = openfold_superimpose_np(base_frame, ensemble[i])
        of_rmsds.append(r)
    t_openfold = time.perf_counter() - t0
    of_rate = N_EVALS / t_openfold
    print(f"\n[1] OpenFold Reference (Bio.SVDSuperimposer):")
    print(f"    Time:       {t_openfold:.4f} s")
    print(f"    Throughput: {of_rate:,.1f} alignments / sec")

    # B) strux-rs Single-Thread CPU Kabsch
    t0 = time.perf_counter()
    strux_1_rmsds = [strux_rs.calculate_rmsd_kabsch(base_frame, ensemble[i]) for i in range(N_EVALS)]
    t_strux_1 = time.perf_counter() - t0
    strux_1_rate = N_EVALS / t_strux_1
    print(f"\n[2] strux-rs Single-Thread CPU Kabsch:")
    print(f"    Time:       {t_strux_1:.4f} s ({t_openfold / t_strux_1:.1f}x faster than OpenFold)")
    print(f"    Throughput: {strux_1_rate:,.1f} alignments / sec")

    # C) strux-rs 16-Core Rayon CPU Parallel
    t0 = time.perf_counter()
    strux_rayon_rmsds = strux_rs.cpu_trajectory_rmsd(ensemble, base_frame)
    t_strux_rayon = time.perf_counter() - t0
    strux_rayon_rate = N_EVALS / t_strux_rayon
    print(f"\n[3] strux-rs Multi-Core CPU (Rayon Parallel):")
    print(f"    Time:       {t_strux_rayon*1000:.2f} ms ({t_openfold / t_strux_rayon:.1f}x faster than OpenFold)")
    print(f"    Throughput: {strux_rayon_rate:,.1f} alignments / sec")

    # D) strux-rs CUDA GPU (if available)
    if strux_rs.cuda_is_available():
        t0 = time.perf_counter()
        _ = strux_rs.cuda_pairwise_rmsd(ensemble)
        t_strux_gpu = time.perf_counter() - t0
        gpu_rate = (N_EVALS * N_EVALS) / t_strux_gpu
        print(f"\n[4] strux-rs GPU CUDA Engine (NVIDIA RTX A2000):")
        print(f"    Pairwise Time: {t_strux_gpu:.4f} s across {N_EVALS*N_EVALS:,d} pairs")
        print(f"    Throughput:    {gpu_rate:,.1f} alignments / sec ({gpu_rate / of_rate:,.0f}x faster than OpenFold)")

    # Parity verification
    max_dev = np.max(np.abs(np.array(of_rmsds) - np.array(strux_1_rmsds)))
    print(f"\nNumerical Parity between OpenFold and strux-rs: Max Deviation = {max_dev:.6e} Å")

def benchmark_a3m_parsing():
    print("\n" + "=" * 80)
    print("BENCHMARK 2: A3M Alignment Parsing & Deletion Matrix Generation")
    print("Comparing OpenFold's Python parser vs strux-rs zero-copy parser")
    print("=" * 80)

    # Generate synthetic A3M dataset
    NUM_SEQS = 25000
    SEQ_LEN = 150
    print(f"Generating realistic A3M dataset: {NUM_SEQS:,d} sequences of length {SEQ_LEN}...")

    alphabet = list("ACDEFGHIKLMNPQRSTVWY")
    np.random.seed(42)
    lines = [">query\n" + "".join(np.random.choice(alphabet, SEQ_LEN))]
    for i in range(1, NUM_SEQS):
        seq = []
        for j in range(SEQ_LEN):
            if np.random.random() < 0.05:
                seq.append("a")  # insertion relative to query
            if np.random.random() < 0.10:
                seq.append("-")  # gap in query column
            else:
                seq.append(np.random.choice(alphabet))
        lines.append(f">seq_{i}\n" + "".join(seq))

    a3m_str = "\n".join(lines)
    data_mb = len(a3m_str.encode('utf-8')) / (1024 * 1024)
    print(f"Synthetic A3M Size: {data_mb:.2f} MB in memory\n")

    # A) OpenFold Python Parser
    t0 = time.perf_counter()
    of_seqs, of_dels, of_descs = openfold_parse_a3m(a3m_str)
    t_openfold = time.perf_counter() - t0
    of_rate = NUM_SEQS / t_openfold
    print(f"[1] OpenFold Reference (Pure Python parsers.py):")
    print(f"    Time:       {t_openfold:.4f} s")
    print(f"    Throughput: {of_rate:,.1f} seqs / sec ({data_mb / t_openfold:.1f} MB/s)")

    # B) strux-rs Rust Zero-Copy Parser
    t0 = time.perf_counter()
    rust_msa = strux_rs.parse_a3m(a3m_str)
    t_strux = time.perf_counter() - t0
    strux_rate = NUM_SEQS / t_strux
    print(f"\n[2] strux-rs Parser (Zero-Copy Rust FFI):")
    print(f"    Time:       {t_strux:.4f} s ({t_openfold / t_strux:.1f}x faster than OpenFold)")
    print(f"    Throughput: {strux_rate:,.1f} seqs / sec ({data_mb / t_strux:.1f} MB/s)")

    # Parity check
    print(f"\nParity Verification:")
    print(f"    Sequences Count Match: {len(of_seqs) == len(rust_msa.sequences)}")
    print(f"    Descriptions Match:    {of_descs[:5] == rust_msa.descriptions[:5]}")
    print(f"    Deletion Matrix Shape: {len(of_dels)}x{len(of_dels[0])} vs {len(rust_msa.deletion_matrix)}x{len(rust_msa.deletion_matrix[0])}")

def main():
    print("================================================================================")
    print("  HEAD-TO-HEAD BENCHMARK: OPENFOLD vs STRUX-RS")
    print("================================================================================")
    benchmark_superimposition()
    benchmark_a3m_parsing()
    print("\n" + "=" * 80)
    print("  BENCHMARK COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
