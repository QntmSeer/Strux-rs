import time
import string
import numpy as np
import strux_rs
from Bio.SVDSuperimposer import SVDSuperimposer

# OpenFold reference implementations
def openfold_superimpose_np(reference, coords):
    sup = SVDSuperimposer()
    sup.set(reference, coords)
    sup.run()
    return sup.get_transformed(), sup.get_rms()

def openfold_parse_a3m(a3m_string: str):
    def parse_fasta(fasta_string: str):
        sequences, descriptions = [], []
        index = -1
        for line in fasta_string.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                index += 1
                descriptions.append(line[1:])
                sequences.append('')
                continue
            elif not descriptions:
                raise ValueError('Expected description line')
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

    deletion_table = str.maketrans('', '', string.ascii_lowercase)
    aligned_sequences = [s.translate(deletion_table) for s in sequences]
    return aligned_sequences, deletion_matrix, descriptions

def generate_a3m(num_seqs, seq_len):
    alphabet = list('ACDEFGHIKLMNPQRSTVWY')
    np.random.seed(42)
    lines = ['>query\n' + ''.join(np.random.choice(alphabet, seq_len))]
    for i in range(1, num_seqs):
        seq = []
        for j in range(seq_len):
            if np.random.random() < 0.05:
                seq.append('a')
            if np.random.random() < 0.10:
                seq.append('-')
            else:
                seq.append(np.random.choice(alphabet))
        lines.append(f'>seq_{i}\n' + ''.join(seq))
    return '\n'.join(lines)

print('=== 1. A3M Parsing vs Sequence Depth (seq_len=150) ===')
depths = [1000, 2500, 5000, 10000, 25000, 50000]
depth_results = []
for n in depths:
    a3m_data = generate_a3m(n, 150)
    data_mb = len(a3m_data.encode('utf-8')) / (1024 * 1024)
    
    # Warmup
    _ = strux_rs.parse_a3m(a3m_data[:5000])
    
    # OpenFold
    t0 = time.perf_counter()
    _ = openfold_parse_a3m(a3m_data)
    t_of = time.perf_counter() - t0
    
    # strux-rs
    t0 = time.perf_counter()
    _ = strux_rs.parse_a3m(a3m_data)
    t_st = time.perf_counter() - t0
    
    sp = t_of / t_st
    rate_of = n / t_of
    rate_st = n / t_st
    print(f'N={n:5d} ({data_mb:.2f} MB) | OF: {t_of:.4f}s ({rate_of:,.0f} seq/s) | strux: {t_st:.4f}s ({rate_st:,.0f} seq/s) | {sp:.1f}x')
    depth_results.append((n, data_mb, t_of, t_st, rate_of, rate_st, sp))

print('\n=== 2. A3M Parsing vs Sequence Length (num_seqs=10,000) ===')
lengths = [50, 100, 250, 500, 1000]
length_results = []
for l in lengths:
    a3m_data = generate_a3m(10000, l)
    data_mb = len(a3m_data.encode('utf-8')) / (1024 * 1024)
    
    t0 = time.perf_counter()
    _ = openfold_parse_a3m(a3m_data)
    t_of = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    _ = strux_rs.parse_a3m(a3m_data)
    t_st = time.perf_counter() - t0
    
    sp = t_of / t_st
    mb_s_of = data_mb / t_of
    mb_s_st = data_mb / t_st
    print(f'L={l:4d} ({data_mb:.2f} MB) | OF: {t_of:.4f}s ({mb_s_of:.1f} MB/s) | strux: {t_st:.4f}s ({mb_s_st:.1f} MB/s) | {sp:.1f}x')
    length_results.append((l, data_mb, t_of, t_st, mb_s_of, mb_s_st, sp))

print('\n=== 3. Kabsch RMSD vs Atom Count (N_evals=250) ===', flush=True)
atom_counts = [100, 300, 1000, 1919, 3500]
atom_results = []
for n_atoms in atom_counts:
    np.random.seed(42)
    ref = np.random.randn(n_atoms, 3).astype(np.float32)
    frames = [(ref + np.random.randn(n_atoms, 3) * 0.5).astype(np.float32) for _ in range(250)]
    
    # OpenFold
    t0 = time.perf_counter()
    for f in frames:
        _ = openfold_superimpose_np(ref, f)
    t_of = time.perf_counter() - t0
    
    # strux-rs
    t0 = time.perf_counter()
    for f in frames:
        _ = strux_rs.calculate_rmsd_kabsch(ref, f)
    t_st = time.perf_counter() - t0
    
    sp = t_of / t_st
    rate_of = 250 / t_of
    rate_st = 250 / t_st
    print(f'Atoms={n_atoms:4d} | OF: {t_of:.4f}s ({rate_of:,.0f}/s) | strux: {t_st:.4f}s ({rate_st:,.0f}/s) | {sp:.1f}x', flush=True)
    atom_results.append((n_atoms, t_of, t_st, rate_of, rate_st, sp))

# Save gnuplot .dat files
import os
os.makedirs('benchmarks', exist_ok=True)
with open('benchmarks/a3m_depth_scaling.dat', 'w') as f:
    f.write('# Seqs DataMB OpenFoldTimeSec StruxTimeSec OpenFoldRate StruxRate Speedup\n')
    for r in depth_results:
        f.write(f'{r[0]} {r[1]:.3f} {r[2]:.5f} {r[3]:.5f} {r[4]:.1f} {r[5]:.1f} {r[6]:.2f}\n')

with open('benchmarks/a3m_length_scaling.dat', 'w') as f:
    f.write('# SeqLen DataMB OpenFoldTimeSec StruxTimeSec OpenFoldMBs StruxMBs Speedup\n')
    for r in length_results:
        f.write(f'{r[0]} {r[1]:.3f} {r[2]:.5f} {r[3]:.5f} {r[4]:.2f} {r[5]:.2f} {r[6]:.2f}\n')

with open('benchmarks/superimpose_atom_scaling.dat', 'w') as f:
    f.write('# Atoms OpenFoldTimeSec StruxTimeSec OpenFoldRate StruxRate Speedup\n')
    for r in atom_results:
        f.write(f'{r[0]} {r[1]:.5f} {r[2]:.5f} {r[3]:.1f} {r[4]:.1f} {r[5]:.2f}\n')

print('\nBenchmark data saved to benchmarks/*.dat successfully!')
