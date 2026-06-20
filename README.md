# strux-rs

[![Language](https://img.shields.io/badge/Language-Rust-orange.svg)]()
[![Python Bindings](https://img.shields.io/badge/Python-PyO3_/_Maturin-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

> **strux-rs** is a blazingly fast, performance-engineered structural biology library written in Rust, offering direct, zero-copy Python bindings via PyO3 and NumPy.

It acts as a drop-in accelerator for slow CPU-bound bottlenecks in molecular dynamics (MD) analysis and generative structural biology pipelines.

---

## Performance Speedups

Benchmark performed on a **21-frame trajectory** of a designed E3 ubiquitin ligase binder containing **1,924 atoms per frame**:

| Task | Python (Pure/NumPy) | Rust (`strux-rs`) | Speedup |
| :--- | :--- | :--- | :--- |
| **PDB Trajectory Parsing** | 0.0781s | 0.0513s | **1.5x** |
| **Radius of Gyration ($R_g$)** | 0.000140s | 0.000009s | **16.0x** |
| **Raw RMSD** | 0.000037s | 0.000008s | **4.5x** |
| **Kabsch Aligned RMSD** | 0.004737s | 0.000019s | **255.6x** |
| **RMSF** | 0.000601s | 0.000131s | **4.6x** |
| **Interface Contacts (5.0Å)** | 0.026400s | 0.000500s | **55.5x** |

### Core Optimization Features
* **SVD-based Kabsch Alignment (~255x faster)**: Mathematical alignment centered at centroids, computing covariance matrices, and executing reflection-corrected Singular Value Decomposition (SVD) using optimized SIMD-capable `nalgebra` structures.
* **Spatial Hashing Cell Lists (~55x faster)**: Bypasses naive $O(N_A \times N_B)$ double-loop contact mapping. It indexes coordinate grids into cubical voxels and wraps boundary cells under Periodic Boundary Conditions (PBC) using minimum image conventions.
* **Multi-Threaded Trajectory Scaling**: Bypasses the Python Global Interpreter Lock (GIL) entirely. Heavy computations scale automatically across all CPU cores using `rayon`.

---

## Features

* **Trajectory Parsing**: High-speed multi-model PDB text scanner returning a 3D NumPy array `[Frames, Atoms, 3]`.
* **Kabsch Superposition**: RMSD alignment for coordinate superposition.
* **Biophysical Metrics**: Radius of Gyration ($R_g$), Root Mean Square Fluctuation (RMSF), and Raw RMSD.
* **Interface Contact Mapper**: Rapid neighbor lookup under cutoff distances supporting periodic boxes.

---

## Installation

To build and compile `strux-rs` from source, ensure you have the Rust toolchain installed.

```bash
# Clone the repository
git clone https://github.com/QntmSeer/strux-rs.git
cd strux-rs

# Compile and install inside your active Python environment
pip install .
```

*For active local development:*
```bash
pip install maturin
maturin develop --release
```

---

## Quickstart (Python API)

```python
import numpy as np
import strux_rs

# 1. Parse a multi-model PDB trajectory into a [Frames, Atoms, 3] numpy array
traj = strux_rs.parse_pdb("trajectory.pdb")
print(f"Loaded trajectory with shape: {traj.shape}")

# 2. Compute Radius of Gyration for the first frame
rg = strux_rs.calculate_rg(traj[0])
print(f"Radius of Gyration: {rg:.3f} Å")

# 3. Calculate aligned RMSD using the Kabsch algorithm
aligned_rmsd = strux_rs.calculate_rmsd_kabsch(traj[0], traj[1])
print(f"Aligned RMSD: {aligned_rmsd:.3f} Å")

# 4. Rapid interface contact mapping under Periodic Boundary Conditions (PBC)
target = traj[0][:1023]
binder = traj[0][1023:]
cutoff = 5.0  # Ångstroms
box_dims = [25.0, 25.0, 25.0]  # Periodic boundary box length

contacts = strux_rs.find_interface_contacts(target, binder, cutoff, box_dims)
print(f"Detected {len(contacts)} contact pairs at the interface.")
```

---

## Repository Layout

```text
strux-rs/
├── Cargo.toml          # Cargo configuration and dependencies (pyo3, numpy, nalgebra, rayon)
├── benchmark.py        # Correctness and performance benchmarking suite
├── src/
│   ├── lib.rs          # PyO3 bindings and array conversion layer
│   ├── pdb.rs          # Zero-allocation buffered PDB trajectory scanner
│   ├── spatial.rs      # PBC-aware Voxel spatial hashing / Cell Lists
│   └── analysis.rs     # Biophysical math: Rg, RMSD, SVD-Kabsch, and RMSF
└── tests/
    └── grueling_tests.rs # Edge-case, boundary wrapping, and degeneracy verification
```

---

## License
This project is licensed under the MIT License.
