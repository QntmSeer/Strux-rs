# strux-rs

[![Language](https://img.shields.io/badge/Language-Rust-orange.svg)]()
[![Python Bindings](https://img.shields.io/badge/Python-PyO3_/_Maturin-blue.svg)]()
[![PyPI Package](https://img.shields.io/badge/PyPI-v0.2.3-blue.svg)](https://pypi.org/project/strux-rs/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **strux-rs** is a blazingly fast, performance-engineered structural biology library written in Rust, offering direct, zero-copy Python bindings via PyO3 and NumPy.

It acts as a drop-in accelerator for slow CPU-bound bottlenecks in protein folding pipelines (AlphaFold / OpenFold), molecular dynamics (MD) trajectory analysis, and generative structural biology workflows.

---

## Performance Speedups (Workstation Benchmarks)

*All benchmarks measured on Linux Workstation `agni` (Intel Core i9-11950H @ 2.60GHz, 16 Cores, 62 GB RAM).*

### 1. MSA Parsing & Preprocessing (OpenFold / AlphaFold Bottleneck)
Benchmark performed on a **250,000 sequence (144.6 MB `.a3m` file)** dataset:

| Task | Python Reference | Rust (`strux-rs`) | Speedup | Throughput | Peak RAM Delta |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A3M Preprocessing** | 10.485s | **0.801s** | **13.1x** | **312,000 seqs/s** | **966 MB** (Saved 383 MB) |

### 2. Massive MD Trajectory Analysis
Benchmark performed on a **500-frame PDB trajectory** containing **5,000 atoms per frame** (2,500,000 total coordinates, 188.4 MB `.pdb` file):

| Task | Python (Pure/NumPy) | Rust (`strux-rs`) | Speedup | Peak RAM Delta |
| :--- | :--- | :--- | :--- | :--- |
| **188.4 MB PDB Trajectory Parsing** | 5.0113s | **0.5166s** | **9.7x** | **0.00 MB** (Saved 558 MB) |
| **Radius of Gyration ($R_g$) (500 frames)** | 0.1343s | **0.0158s** | **8.5x** | Instantaneous |
| **RMSF Fluctuation (2.5M Coordinates)** | 0.0718s | **0.0216s** | **3.3x** | Instantaneous |
| **Kabsch SVD Alignment (100 pairs)** | 0.0531s | **0.0074s** | **7.2x** | SIMD Vectorized |
| **Interface Contacts (5.0Å Cutoff)** | 0.2362s | **0.0046s** | **51.5x** | Spatial Voxel Hash |

---

## Core Optimization Features

* **Zero-Copy Memory-Mapped MSA Parsing (`memmap2`)**: Memory-maps gigabyte-scale `.a3m` and `.sto` (Stockholm) files directly from kernel page cache, parsing records and counting deletions in a single pass without heap allocation overhead.
* **Direct NumPy Matrix Output**: Exposes `deletion_matrix_np` as a contiguous 2D NumPy array (`int32`), completely bypassing slow Python list-of-lists conversions.
* **SVD-based Kabsch Alignment (~7.2x faster)**: Mathematical alignment centered at centroids, computing covariance matrices, and executing reflection-corrected Singular Value Decomposition (SVD) using optimized SIMD-capable `nalgebra` structures.
* **Spatial Hashing Cell Lists (~51.5x faster)**: Indexes coordinate grids into cubical voxels and wraps boundary cells under Periodic Boundary Conditions (PBC).
* **Multi-Threaded Trajectory & MSA Scaling**: Bypasses the Python Global Interpreter Lock (GIL) entirely. Heavy computations scale automatically across all CPU cores using `rayon`.

---

## Installation

Install directly from PyPI:

```bash
pip install strux-rs
```

To build and compile `strux-rs` from source:

```bash
git clone https://github.com/QntmSeer/strux-rs.git
cd strux-rs
pip install maturin
maturin develop --release
```

---

## Quickstart (Python API)

### 1. High-Speed MSA Preprocessing (AlphaFold / OpenFold)

```python
import strux_rs

# Parse an A3M alignment file using zero-copy memory mapping
msa = strux_rs.parse_a3m_file("uniref90_hits.a3m")

print(f"Parsed {len(msa)} aligned sequences.")
print(f"Aligned sequence 0 length: {len(msa.sequences[0])}")

# Zero-copy 2D NumPy array for OpenFold feature compilation
deletion_matrix = msa.deletion_matrix_np
print(f"Deletion Matrix shape: {deletion_matrix.shape}")
```

### 2. Trajectory Analysis & Biophysical Metrics

```python
import strux_rs

# Load trajectory [Frames, Atoms, 3] from large multi-model PDB
traj = strux_rs.parse_pdb("trajectory.pdb")

# Radius of Gyration
rg = strux_rs.calculate_rg(traj[0])

# Kabsch Aligned RMSD
aligned_rmsd = strux_rs.calculate_rmsd_kabsch(traj[0], traj[1])

# Interface contact mapping (PBC-aware spatial hashing)
contacts = strux_rs.find_interface_contacts(
    target=traj[0][:2500],
    binder=traj[0][2500:],
    cutoff=5.0,
    box_dims=[100.0, 100.0, 100.0]
)
```

---

## Repository Layout

```text
strux-rs/
├── Cargo.toml          # Cargo configuration and dependencies (pyo3, numpy, memmap2, rayon)
├── pyproject.toml      # Maturin build configuration
├── src/
│   ├── lib.rs          # PyO3 bindings and module glue
│   ├── msa.rs          # Zero-copy Rayon memory-mapped A3M & Stockholm parsers
│   ├── pdb.rs          # Zero-allocation buffered PDB trajectory scanner
│   ├── spatial.rs      # PBC-aware Voxel spatial hashing / Cell Lists
│   └── analysis.rs     # Biophysical math: Rg, RMSD, SVD-Kabsch, and RMSF
└── tests/
    ├── test_msa.py     # Correctness and AlphaFold parser equivalence suite
    └── stress_test_msa.py # Multi-threaded workstation stress benchmark
```

---

## License
This project is licensed under the MIT License.
