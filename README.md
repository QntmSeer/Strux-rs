<p align="center">
  <img src="assets/strux_logo.png" alt="strux-rs demoscene insignia" width="360">
</p>

# strux-rs

[![Language](https://img.shields.io/badge/Language-Rust-orange.svg)]()
[![Python Bindings](https://img.shields.io/badge/Python-PyO3_/_Maturin-blue.svg)]()
[![PyPI Package](https://img.shields.io/badge/PyPI-v0.4.0-blue.svg)](https://pypi.org/project/strux-rs/)
[![CUDA](https://img.shields.io/badge/CUDA-12.x_QCP-76b900.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

`strux-rs` is a zero-C-dependency Rust + CUDA engine with Python bindings (`PyO3` / `NumPy`) that bridges **AI protein structure prediction** (AlphaFold 3, Boltz-1, Chai-1, OpenFold, AF-Cluster) and **molecular dynamics (MD) ensemble biophysics** (GROMACS `.xtc`, OpenMM/NAMD `.dcd`, PCA, and Markov State Models).

---

## Key Capabilities

* **Multi-Format Structure & Binary Trajectory Ingestion (`memmap2` + `rayon`)**:
  * **GROMACS `.xtc` (`parse_xtc`)**: Pure-Rust two-pass memory-mapped XDR 3D decompression across all CPU cores (**7.3×** faster than MDAnalysis, **2.8×** faster than MDTraj C `xdrfile`, `1.53e-5 Å` parity).
  * **CHARMM/NAMD/OpenMM `.dcd` (`parse_dcd`)**: Memory-mapped binary reader (**73.6×** faster than MDAnalysis, **1.4×** faster than MDTraj).
  * **AlphaFold 3 / Boltz-2 / Chai-1 / RCSB `mmCIF` (`parse_cif`, `parse_cif_with_plddt`, `parse_cif_full`) & `.pdb` (`parse_pdb`)**: Dynamic `_atom_site.` header loop parser handling arbitrary column order, `auth_asym_id` vs `label_asym_id` multimers, alternate conformers (`label_alt_id`), optional hydrogen stripping (`strip_hydrogens=True`), and single-pass `(coords, pLDDT, chain_ids, res_ids, res_names, atom_names, elements)` ingestion.
  * **In-Flight $C_\alpha$ & Solute Subsetting (`ca_only=True`, `atom_indices=[...]`)**: Filter explicit solvent (`SOL`/`HOH`) or sidechains *during* Rust decompression to cut RAM usage by 10×–100× with upfront bounds validation (`PyValueError`).
* **De Novo Protein & Therapeutic Design Biophysics Primitives**:
  * **Target-Anchored Pose & Self-Consistency RMSD (`calculate_pose_and_sc_rmsd`, `batch_pose_and_sc_rmsd`)**: Superimposes exclusively on target $C_\alpha$ atoms and applies the resulting Kabsch transform to binder $C_\alpha$ (`Pose_RMSD`) alongside standalone monomer `scRMSD` (**1,000 complexes in 0.76 ms**).
  * **All-Atom Interface Physics & Buried Unsatisfied Polars (`analyze_interface_physics`)**: Single-pass interfacial heavy-atom contacts ($\le 4.5\text{ \AA}$), polar N/O H-bonds ($\le 3.5\text{ \AA}$), cationic-anionic salt bridges ($\le 4.0\text{ \AA}$), and Buried Unsatisfied Polar Atoms (`buns`, zero-dependency replacement for PyRosetta `delta_unsatHbonds` & legacy `hbplus`).
  * **96-Point Golden-Spiral Shrake-Rupley SASA & BSA (`calculate_sasa`, `calculate_bsa`)**: 16-thread Rayon solvent-accessible and buried surface area (**800-atom complex BSA in 1.25 ms**, `4.22e-6 Å²` analytical sphere parity).
  * **Screened Debye-Hückel $\Delta\Delta G_{\text{pH}}$ (`calculate_ph_switch_ddg`)**: Henderson-Hasselbalch fractional protonation (`HIS`, `ASP`, `GLU`, `CYS`, `TYR`, `LYS`, `ARG`) + ionic-strength-screened interfacial electrostatic energy difference between acidic (e.g., tumor/endosomal pH 6.0) and physiological (pH 7.4) states.
  * **Anisotropic Network Model Normal Modes (`anm_modes`)**: $3N \times 3N$ Hessian assembly + Householder/QR symmetric eigendecomposition returning lowest non-trivial eigenvalues, eigenvectors, and theoretical B-factors (`5.50 ms` for 60 $C_\alpha$, `1.86e-9` max eigenvalue Δ vs LAPACK `eigh`).
  * **MPNN Sequence Developability & $k$-mer Novelty (`screen_sequences_biophysics`, `sliding_kmer_novelty`)**: Rayon + SIMD byte screening for bisection pI, GRAVY, net charge at pH 7.4, 7-residue CamSol solubility score, Aggregation-Prone Regions (`apr_count`), chemical liabilities (`DP`, `NG`/`NS`/`NT`, `N[^P][ST]` sequons), and sliding $k$-mer novelty (**5,000 sequences in 4.15 ms**).
* **AI-Accelerated Molecular Simulation (AMS) Biophysics**:
  * **Superposition-Free AlphaFold lDDT (`calculate_lddt`)**: Multi-threaded Local Distance Difference Test (`0.5, 1.0, 2.0, 4.0 Å` thresholds) running **201.2×** faster than NumPy (`0.00e+00` Δ).
  * **AF-Cluster Sub-Alignment Generation (`cluster_msa`, `Msa.cluster`, `Msa.get_cluster`)**: Greedy pairwise non-gap sequence identity clustering running **33.8×** faster than Python (`22.2 ms` for 10,000 sequences × 250 AA).
  * **Cartesian Trajectory PCA (`trajectory_pca`)**: Rayon-parallelized covariance deflation returning principal components, projections, and explained variance **5.1×** faster than NumPy LAPACK SVD (`0.00e+00` variance Δ).
  * **Markov State Populations & Free Energies (`markov_populations`)**: Maximum-likelihood transition matrix, ergodic stationary distribution $\pi_i$, and relative free energy $\Delta G_i = -k_B T \ln(\pi_i / \max \pi)$ at **103.9M frames/s** (`19.3 ms` for $2 \times 10^6$ frames).
* **CUDA GPU Trajectory Alignment & Contact Maps (`cudarc`)**:
  * All-to-all pairwise RMSD (`cuda_pairwise_rmsd`) reaching **2.05M alignments/s** (**4,348×** over Biopython, **74.5×** over MDTraj AVX QCP, **25.3×** over Google JAX JIT), Daura clustering (`cuda_cluster_daura`), and dense contact frequency matrices (`cuda_contact_frequency`, **19.5×** speedup).
* **Zero-Copy MSA Token Tensors**: Memory-mapped `.a3m` and `.sto` parsers producing contiguous 2D `uint8` residue token tensors (`tokens_np`, **664×** speedup, `12.6 µs`) for PyTorch DataLoaders.

---

## Benchmarks

Measured on an Intel Core i9-11950H (16 hardware threads, 62 GB RAM) and an NVIDIA RTX A2000 Laptop GPU (4 GB physical VRAM, 35W TGP).

### 1. Gold-Standard Ecosystem Comparison
Head-to-head comparison across standard computational biology packages on identical hardware:

| Task & Biological Target | Tool / Implementation | Runtime | Throughput | Speedup vs Ref |
| :--- | :--- | :---: | :---: | :---: |
| **Ubiquitin Solution Ensemble**<br>(`PDB 2K39`, 116 models, 13,456 pairs) | Biopython (SVD) | 30.13 s | 446.6 align/s | 1.0× (Baseline) |
| | SciPy (`align_vectors`) | 2.48 s | 5,436.6 align/s | 12.2× |
| | MDAnalysis (QCPROT) | 1.27 s | 10,567.4 align/s | 23.7× |
| | MDTraj (C / AVX QCP) | 0.517 s | 26,046.1 align/s | 58.3× |
| | Gemmi (C++ `superpose_positions`) | 0.339 s | 39,710.8 align/s | 88.9× |
| | **`strux-rs` (1-Thread CPU)** | **0.311 s** | **43,241.4 align/s** | **96.8×** |
| | Google JAX (`vmap JIT`) | 0.176 s | 76,612.5 align/s | 171.5× |
| | ProDy (C-Kabsch) | 0.146 s | 92,281.5 align/s | 206.6× |
| | **`strux-rs` (16-Thread Rayon CPU)** | **22.57 ms** | **596,176.5 align/s** | **1,335×** |
| | **`strux-rs` (CUDA QCP GPU)** | **6.93 ms** | **1,941,698 align/s** | **4,348×** |
| **GFP Trajectory Dynamics ($R_g$)**<br>(`trajectory.pdb`, 1,919 atoms) | MDAnalysis | 59.4 ms | 336.9 fps | 1.0× (Baseline) |
| | MDTraj | 2.24 ms | 8,912.2 fps | 26.5× |
| | **`strux-rs` (SIMD)** | **0.29 ms** | **68,770.8 fps** | **204×** |

![Gold-Standard Ecosystem Benchmark](benchmarks/gold_standard_comparison.png?v=0.4.0)

### 2. AMS Biophysics, Binary Trajectory Readers & Zero-Copy Accelerators

| Kernel / Task | Workload | Reference Baseline | `strux-rs` | Speedup | Numerical Parity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **AlphaFold lDDT** (`calculate_lddt`) | 4,000 atoms (16M pairs) | 943.42 ms (NumPy) | **4.69 ms** | **201.2×** | `0.00e+00` Δ (SO(3) invariant) |
| **AF-Cluster MSA** (`cluster_msa`) | 10,000 seqs × 250 AA | 750.4 ms (Python) | **22.21 ms** | **33.8×** | Exact bit-for-bit cluster IDs |
| **GROMACS `.xtc` Read** (`parse_xtc`) | 500 frames × 1,000 atoms | 39.04 ms (MDAnalysis) | **5.32 ms** | **7.3×** (2.8× vs MDTraj) | `1.53e-5 Å` max Δ vs C `xdrfile` |
| **CHARMM `.dcd` Read** (`parse_dcd`) | 21 frames × 1,919 atoms | 27.97 ms (MDAnalysis) | **0.380 ms** | **73.6×** (1.4× vs MDTraj) | `0.00e+00 Å` exact match |
| **Trajectory PCA** (`trajectory_pca`) | 2,500 frames × 500 atoms | 978.12 ms (LAPACK SVD) | **190.20 ms** | **5.1×** | `0.00e+00` variance Δ, $\|\cos\theta\| \ge 0.9999999$ |
| **Markov State $\Delta G$** (`markov_populations`) | 2,000,000 state steps | — | **19.25 ms** | **103.9M frames/s** | `1.36e-8` Δ vs `np.linalg.eig` (0.7% rare state) |
| **MSA `uint8` Tensor** (`tokens_np`) | 1,024 seqs × 73 residues | 8.40 ms (Python) | **12.6 µs** | **664×** | Exact byte-for-byte match |
| **GPU Contact Map** (`cuda_contact_frequency`) | 1,919 × 1,919 grid | 3,677.7 ms (NumPy) | **188.8 ms** | **19.5×** | `0.00e+00` exact parity |

![New Features Verification](benchmarks/new_features_verification.png?v=0.4.0)

### 3. Pairwise Trajectory RMSD ($O(T^2)$) & Macromolecular Size Scaling

| Trajectory Frames ($T$) | Pairwise Alignments | 16-Thread Core i9 (Rayon) | RTX A2000 (CUDA QCP) | GPU Throughput | GPU vs 16T CPU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 250 | 62,500 | 0.118s | 0.201s | 311,700 pairs/s | 0.6× |
| 1,000 | 1,000,000 | 1.88s | 0.643s | 1,554,800 pairs/s | 2.9× |
| 2,500 | 6,250,000 | 11.8s | 3.211s | 1,946,400 pairs/s | 3.7× |
| 5,000 | 25,000,000 | 47.0s | 12.169s | **2,054,488 pairs/s** | **3.9×** |

![Benchmark Scaling](benchmarks/benchmark_scaling.png?v=0.4.0)

### 4. Hardware Scaling & VRAM Capacity: NVIDIA L4 (24 GB) vs. Laptop RTX A2000 (4 GB)

To evaluate GPU scaling beyond the laptop thermal and memory envelope, the exact same trajectory QCP kernel and therapeutic triage benchmark battery were deployed to a cloud **NVIDIA L4 GPU** (Ada Lovelace, 24 GB VRAM, 72W TDP):

| Workload & System Scale | Laptop RTX A2000 (4 GB, 35W) | Cloud NVIDIA L4 (24 GB, 72W) | L4 Throughput / Speedup |
| :--- | :---: | :---: | :---: |
| **1,000 × 1,000 frames** (1.0M pairs) | 211.4 ms | **159.6 ms** | 1.57 B atom-pairs/s (4.27× vs 8T CPU) |
| **5,000 × 5,000 frames** (25.0M pairs) | 1,843.1 ms | **551.7 ms** | 11.33 B atom-pairs/s (30.89× vs CPU) |
| **10,000 × 10,000 frames** (100.0M pairs) | 6,969.6 ms | **1,767.5 ms** | 14.14 B atom-pairs/s (38.57× vs CPU) |
| **20,000 × 20,000 frames** (400.0M pairs) | *OOM (>4 GB VRAM barrier)* | **6,606.1 ms (6.60s)** | **15.14 B atom-pairs/s (41.28× vs CPU)** |
| **5,000-Frame AMS Feedback Loop** (lDDT + PCA + Daura + MSM) | 11,164.7 ms | **7,280.9 ms** | Complete ensemble analysis in 7.28s |
| **Target-Anchored Pose + scRMSD** (10,000 designs) | 4.03 ms | **8.45 ms** | **326.3× vs NumPy SVD** ($\Delta < 1.55 \times 10^{-6}\text{ \AA}$) |
| **Shrake-Rupley BSA** (2,048 atoms, 96 points) | 1.96 ms | **8.30 ms** | **90.6× vs pure Python** |
| **Sliding 15-mer Novelty DB** (5k queries × 500 refs) | 9,477.5 ms | **7,441.5 ms** | **1,793.8× vs Python** (7.4s vs ~3.7 hours) |

![Hardware L4 Scaling Comparison](benchmarks/hardware_l4_scaling_comparison.png?v=0.4.0)

### 5. Prolonged 5-Minute Full-System Hardware Burn (`309.3s` Concurrent 16T CPU + CUDA GPU)
Under 5+ continuous minutes of simultaneous **16-thread Rayon CPU torture (`97.5%` mean utilization)** and **100% CUDA GPU saturation** inside a shared laptop thermal envelope:
* **Thermal Equilibrium**: GPU plateaued at **70–71 °C** (`16 °C` below the `87 °C` thermal throttle limit) drawing **27.8 W** at a locked **1,710 MHz** boost clock with zero thermal throttling.
* **Cumulative Work Completed in 309.3 Seconds**:
  * **CUDA QCP Alignments**: `600,000,000` pairwise 1,919-atom alignments + `24` GPU contact matrices
  * **AlphaFold `calculate_lddt` (12,000 atoms)**: `5,658` macromolecular evaluations (`814.75 billion` atom pairs scored)
  * **GROMACS `parse_xtc`**: `1,463,000` compressed XDR frames decoded (`2.926 billion` 3D coordinates)
  * **AF-Cluster `cluster_msa`**: `14,630,000` sequences clustered (`3.66 billion` amino acid residues)
  * **Markov State Steps**: `731,500,000` state transitions solved

![Hardware Soak Telemetry](benchmarks/hardware_soak_telemetry.png?v=0.4.0)

---

## Installation

### From PyPI (Pre-Built Binary Wheels for Linux, macOS Apple Silicon/Intel & Windows)
```bash
pip install strux-rs
```

### From Source (With CUDA Support)
Requires CUDA Toolkit ($\ge 12.0$) and Rust ($\ge 1.75$):
```bash
git clone https://github.com/QntmSeer/strux-rs.git
cd strux-rs
pip install maturin
maturin develop --release --features cuda
```

---

## Python API Usage

### 1. Structure & Trajectory Ingestion (`.cif`, `.pdb`, `.xtc`, `.dcd` + In-Flight Subsetting)
```python
import strux_rs

# AlphaFold 3 / Boltz-1 / Chai-1 / RCSB mmCIF with per-atom pLDDT confidence scores
coords_cif, plddt = strux_rs.parse_cif_with_plddt("fold_model_0.cif", ca_only=True)

# Pure-Rust 16T Rayon GROMACS .xtc & mmap .dcd readers with optional in-flight solute/CA selection
traj_xtc = strux_rs.parse_xtc("md_100ns.xtc", atom_indices=range(0, 3000, 15))
traj_dcd = strux_rs.parse_dcd("trajectory.dcd")
traj_pdb = strux_rs.parse_pdb("ensemble.pdb", ca_only=True)
```

### 2. AI-Accelerated Molecular Simulation (AMS) Biophysics Workflow
```python
# Step 1: AF-Cluster sub-alignment generation from deep MSA
msa = strux_rs.parse_a3m_file("target.a3m")
labels, rep_indices = msa.cluster(min_identity=0.75)
sub_msa = msa.get_cluster(cluster_id=0, labels=labels.tolist())
tokens_u8 = sub_msa.tokens_np()  # Zero-copy [N_seq, L] uint8 tensor (12.6 µs)

# Step 2: Superposition-free AlphaFold lDDT evaluation (201.2x faster than NumPy)
global_lddt, per_res_lddt = strux_rs.calculate_lddt(traj_xtc[0], traj_xtc[-1], cutoff=15.0)

# Step 3: Cartesian Trajectory PCA (Rayon-parallel covariance deflation)
projections, explained_var_ratio, components = strux_rs.trajectory_pca(traj_xtc, n_components=4)

# Step 4: Markov State Model stationary populations (pi) & free energies (dG in kcal/mol)
populations, delta_g_kcal, transition_mat = strux_rs.markov_populations(
    state_sequence=state_assignments, num_states=4, lag_time=10, temperature=300.0
)
```

### 3. Therapeutic Binder & De Novo Design Biophysics (RFdiffusion / BindCraft / Boltz-2 / MPNN)
```python
# Parse co-folded complex with full metadata:
# coords (3D), pLDDT (2D), chain_ids (list), res_ids (1D), res_names (list), atom_names (list)
coords, plddt, chain_ids, res_ids, res_names, atom_names = strux_rs.parse_cif_full(
    "boltz2_binder_complex.cif", ca_only=False, strip_hydrogens=True
)

# 1. Target-anchored Pose_RMSD (catches binders drifting onto wrong epitope) + monomer scRMSD
# Across 10,000 designs in 8.45 ms (326x faster than SciPy SVD)
sc_rmsd, pose_rmsd = strux_rs.batch_pose_and_sc_rmsd(
    ref_target_ca, ref_binder_ca, mob_target_ca, mob_binder_ca
)

# 2. All-atom interface physics: contacts, H-bonds, salt bridges & Buried Unsatisfied Polars (BUNS)
# Zero-dependency replacement for PyRosetta delta_unsatHbonds & legacy 1994 hbplus
n_contacts, n_hbonds, n_salt_bridges, n_buns = strux_rs.analyze_interface_physics(
    coords[0], atom_names, res_names, chain_ids, res_ids, target_chain="A", binder_chain="B"
)
print(f"Salt bridges: {n_salt_bridges}, H-bonds: {n_hbonds}, BUNS: {n_buns}")

# 3. 96-point Shrake-Rupley Buried Surface Area (BSA) & pH-switch electrostatic ddG (pH 6.0 vs 7.4)
bsa_total, sasa_tgt, sasa_bnd, sasa_cpx = strux_rs.calculate_bsa(
    target_coords, binder_coords, probe_radius=1.4, n_points=96
)
ddg_ph, dg_low, dg_high = strux_rs.calculate_ph_switch_ddg(
    coords[0], atom_names, res_names, chain_ids, target_chain="A", binder_chain="B",
    ph_low=6.0, ph_high=7.4, ionic_strength=0.15, dielectric=8.0
)

# 4. MPNN sequence developability (pI, GRAVY, CamSol, APRs, chemical liabilities) & 15-mer novelty
# 100,000 sequences screened in 270 ms; 5,000 queries vs 500 reference binders in 7.4 s
dev_metrics = strux_rs.screen_sequences_biophysics(mpnn_sequences)
novelty_scores = strux_rs.sliding_kmer_novelty(mpnn_sequences, reference_binders, k=15)
```

### 4. GPU Trajectory Alignment, Clustering & Contact Maps
```python
if strux_rs.cuda_is_available():
    rmsd_matrix = strux_rs.cuda_pairwise_rmsd(traj_xtc)
    cluster_labels, centroids = strux_rs.cuda_cluster_daura(traj_xtc, cutoff=2.0)
    contact_freq = strux_rs.cuda_contact_frequency(traj_xtc, cutoff=4.5)
```

---

## License
MIT License.
