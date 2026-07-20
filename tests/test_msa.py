import time
import string
import numpy as np
import strux_rs

# ==========================================
# Reference Python Implementations (AlphaFold)
# ==========================================

def parse_fasta(fasta_string: str):
    sequences = []
    descriptions = []
    index = -1
    for line in fasta_string.splitlines():
        line = line.strip()
        if line.startswith('>'):
            index += 1
            descriptions.append(line[1:])
            sequences.append('')
            continue
        elif not line:
            continue
        sequences[index] += line
    return sequences, descriptions

def parse_a3m_py(a3m_string: str):
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

def parse_stockholm_py(stockholm_string: str):
    name_to_sequence = {}
    for line in stockholm_string.splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '//')):
            continue
        name, sequence = line.split()
        if name not in name_to_sequence:
            name_to_sequence[name] = ''
        name_to_sequence[name] += sequence

    msa = []
    deletion_matrix = []
    query = ''
    keep_columns = []
    for seq_index, sequence in enumerate(name_to_sequence.values()):
        if seq_index == 0:
            query = sequence
            keep_columns = [i for i, res in enumerate(query) if res != '-']

        aligned_sequence = ''.join([sequence[c] for c in keep_columns])
        msa.append(aligned_sequence)

        deletion_vec = []
        deletion_count = 0
        for seq_res, query_res in zip(sequence, query):
            if seq_res != '-' or query_res != '-':
                if query_res == '-':
                    deletion_count += 1
                else:
                    deletion_vec.append(deletion_count)
                    deletion_count = 0
        deletion_matrix.append(deletion_vec)

    return msa, deletion_matrix, list(name_to_sequence.keys())

# ==========================================
# Test Data Generation
# ==========================================

def generate_mock_a3m(num_seqs=1000, seq_len=300):
    lines = []
    # Query sequence (all uppercase)
    query_seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY"), seq_len))
    lines.append(f">query_sequence")
    lines.append(query_seq)
    
    # Aligned sequences (with insertions and gaps)
    for i in range(num_seqs - 1):
        lines.append(f">seq_{i}")
        seq_parts = []
        for char in query_seq:
            r = np.random.random()
            if r < 0.1:  # Gap in sequence
                seq_parts.append('-')
            elif r < 0.2:  # Insertion (lowercase characters)
                seq_parts.append(char)
                seq_parts.append("".join(np.random.choice(list("acdefghiklmnpqrstvwy"), np.random.randint(1, 4))))
            else:
                seq_parts.append(char)
        lines.append("".join(seq_parts))
        
    return "\n".join(lines)

def generate_mock_stockholm(num_seqs=1000, seq_len=300):
    # Stockholm requires matching sequence lengths for alignment
    query_seq = "".join(np.random.choice(list("ACDEFGHIKLMNPQRSTVWY-"), seq_len))
    names = [f"seq_{i}" for i in range(num_seqs)]
    
    lines = ["# STOCKHOLM 1.0"]
    for i, name in enumerate(names):
        if i == 0:
            seq = query_seq
        else:
            # Random mutations/gaps/insertions represented as lowercase
            seq_chars = []
            for char in query_seq:
                r = np.random.random()
                if r < 0.15:
                    seq_chars.append('-')
                elif r < 0.3:
                    seq_chars.append(char.lower())
                else:
                    seq_chars.append(char.upper())
            seq = "".join(seq_chars)
        lines.append(f"{name:<15} {seq}")
    lines.append("//")
    return "\n".join(lines)

# ==========================================
# Run Verification and Benchmarks
# ==========================================

def run_tests():
    print("Generating mock alignment data...")
    a3m_data = generate_mock_a3m(500, 200)
    sto_data = generate_mock_stockholm(500, 200)

    print("\n--- Testing A3M Parser Correctness ---")
    py_seqs, py_del, py_desc = parse_a3m_py(a3m_data)
    rust_msa = strux_rs.parse_a3m(a3m_data)

    assert len(rust_msa) == len(py_seqs), "Sequence count mismatch"
    assert rust_msa.sequences == py_seqs, "Sequences mismatch"
    assert rust_msa.descriptions == py_desc, "Descriptions mismatch"
    assert rust_msa.deletion_matrix == py_del, "Deletion matrix mismatch"
    
    # Test NumPy fast getter
    rust_del_np = rust_msa.deletion_matrix_np
    assert np.array_equal(rust_del_np, np.array(py_del, dtype=np.int32)), "NumPy deletion matrix mismatch"
    
    # Test truncate method
    truncated = rust_msa.truncate(10)
    assert len(truncated) == 10
    assert truncated.sequences == py_seqs[:10]
    assert truncated.descriptions == py_desc[:10]
    assert truncated.deletion_matrix == py_del[:10]
    
    print("A3M Parser matches reference implementation exactly!")

    print("\n--- Testing Stockholm Parser Correctness ---")
    py_seqs, py_del, py_desc = parse_stockholm_py(sto_data)
    rust_msa = strux_rs.parse_stockholm(sto_data)

    assert len(rust_msa) == len(py_seqs), "Sequence count mismatch"
    assert rust_msa.sequences == py_seqs, "Sequences mismatch"
    assert rust_msa.descriptions == py_desc, "Descriptions mismatch"
    assert rust_msa.deletion_matrix == py_del, "Deletion matrix mismatch"
    
    # Test NumPy fast getter
    rust_del_np = rust_msa.deletion_matrix_np
    assert np.array_equal(rust_del_np, np.array(py_del, dtype=np.int32)), "NumPy deletion matrix mismatch"
    
    print("Stockholm Parser matches reference implementation exactly!")

def run_benchmarks():
    print("\n==========================================")
    print("Running Performance Benchmarks...")
    print("==========================================")
    
    num_seqs = 20000
    seq_len = 500
    print(f"Generating large benchmark A3M alignment ({num_seqs} sequences, length {seq_len})...")
    large_a3m = generate_mock_a3m(num_seqs, seq_len)
    
    # Write to a file for file-parsing benchmarks
    file_path = "large_temp_msa.a3m"
    with open(file_path, "w") as f:
        f.write(large_a3m)
    
    print("Benchmarking Python A3M parsing...")
    t0 = time.perf_counter()
    py_seqs, py_del, py_desc = parse_a3m_py(large_a3m)
    t_py = time.perf_counter() - t0
    print(f"Python parser: {t_py:.4f} seconds")
    
    print("Benchmarking Rust A3M parsing (from string)...")
    t0 = time.perf_counter()
    rust_msa = strux_rs.parse_a3m(large_a3m)
    # Access properties to force conversion
    _ = rust_msa.sequences
    _ = rust_msa.deletion_matrix
    t_rust_str = time.perf_counter() - t0
    print(f"Rust parser (from string, list output): {t_rust_str:.4f} seconds (Speedup: {t_py/t_rust_str:.1f}x)")
    
    print("Benchmarking Rust A3M parsing (from file + NumPy matrix)...")
    t0 = time.perf_counter()
    rust_msa = strux_rs.parse_a3m_file(file_path)
    _ = rust_msa.sequences
    _ = rust_msa.deletion_matrix_np
    t_rust_file_np = time.perf_counter() - t0
    print(f"Rust parser (from file, NumPy output): {t_rust_file_np:.4f} seconds (Speedup: {t_py/t_rust_file_np:.1f}x)")

    # Clean up
    import os
    if os.path.exists(file_path):
        os.remove(file_path)

if __name__ == "__main__":
    run_tests()
    run_benchmarks()
