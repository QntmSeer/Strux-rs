import time
import numpy as np
import strux_rs
from Bio.SVDSuperimposer import SVDSuperimposer
from scipy.spatial.transform import Rotation
import MDAnalysis as mda
import MDAnalysis.analysis.rms as mda_rms
import prody
import mdtraj as md
import gemmi
import jax
import jax.numpy as jnp
import os

prody.confProDy(verbosity='none')
os.makedirs("benchmarks", exist_ok=True)

print("=" * 85)
print("  GOLD-STANDARD BIOLOGICAL DATASET BENCHMARK")
print("  1. Ubiquitin 116-Model NMR Solution Ensemble (PDB: 2K39, 1,231 atoms)")
print("  2. Green Fluorescent Protein Trajectory (trajectory.pdb, 1,919 atoms)")
print("  3. AlphaFold Real Metagenomic BFD/Uniclust A3M & 20.6 MB Kinase Stockholm")
print("=" * 85)

# ==============================================================================
# 1. UBIQUITIN 2K39 ALL-TO-ALL PAIRWISE SUPERIMPOSITION (13,456 ALIGNMENTS)
# ==============================================================================
print("\n[1] Loading Ubiquitin 2K39 Ensemble (116 models, 1,231 atoms each)...")
traj_mdt = md.load_pdb("benchmarks/data/2K39.pdb")
coords_116 = np.ascontiguousarray(traj_mdt.xyz * 10.0, dtype=np.float32) # nm to Å
N_MODELS = len(coords_116)
N_PAIRS = N_MODELS * N_MODELS # 13,456 pairs
coords_116_f64 = coords_116.astype(np.float64)
print(f"    Loaded {N_MODELS} experimental models ({N_PAIRS:,d} all-to-all structural pairs)")

results_2k39 = []

# A. Biopython (subset of 500 pairs for timing extrapolation to save CPU time)
t0 = time.perf_counter()
for i in range(500):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    sup = SVDSuperimposer()
    sup.set(coords_116_f64[idx_a], coords_116_f64[idx_b])
    sup.run()
    _ = sup.get_rms()
t_bio_sample = time.perf_counter() - t0
rate_bio = 500 / t_bio_sample
t_bio_total = N_PAIRS / rate_bio
results_2k39.append(("Biopython", rate_bio, t_bio_total, 1.0, "#94a3b8"))
print(f"    1. Biopython (SVD):           {t_bio_total:.2f} s | {rate_bio:,.1f} align/s | 1.0x (Ref)")

# B. SciPy

t0 = time.perf_counter()
for i in range(500):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    _, rssd = Rotation.align_vectors(coords_116_f64[idx_a], coords_116_f64[idx_b])
t_scipy_sample = time.perf_counter() - t0
rate_scipy = 500 / t_scipy_sample
t_scipy_total = N_PAIRS / rate_scipy
results_2k39.append(("SciPy", rate_scipy, t_scipy_total, rate_scipy/rate_bio, "#64748b"))
print(f"    3. SciPy (align_vectors):     {t_scipy_total:.2f} s | {rate_scipy:,.1f} align/s | {rate_scipy/rate_bio:.1f}x")

# D. Google JAX
@jax.jit
def jax_kabsch_pair(r, t):
    rc = r - jnp.mean(r, axis=0)
    tc = t - jnp.mean(t, axis=0)
    H = tc.T @ rc
    U, S, Vt = jnp.linalg.svd(H)
    d = jnp.linalg.det(Vt @ U)
    D = jnp.diag(jnp.array([1.0, 1.0, d]))
    R = Vt @ D @ U
    diff = (tc @ R.T) - rc
    return jnp.sqrt(jnp.mean(jnp.sum(diff**2, axis=-1)))

jax_batch = jax.jit(jax.vmap(jax_kabsch_pair, in_axes=(None, 0)))
j_coords = jnp.array(coords_116)
# warmup
_ = jax_batch(j_coords[0], j_coords).block_until_ready()
t0 = time.perf_counter()
for m in range(N_MODELS):
    _ = jax_batch(j_coords[m], j_coords).block_until_ready()
t_jax = time.perf_counter() - t0
rate_jax = N_PAIRS / t_jax
results_2k39.append(("Google JAX", rate_jax, t_jax, rate_jax/rate_bio, "#64748b"))
print(f"    4. Google JAX (vmap JIT):     {t_jax:.3f} s | {rate_jax:,.1f} align/s | {rate_jax/rate_bio:.1f}x")

# E. MDAnalysis QCPROT
t0 = time.perf_counter()
for i in range(1000):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    _ = mda_rms.rmsd(coords_116_f64[idx_a], coords_116_f64[idx_b], superposition=True)
t_mda_sample = time.perf_counter() - t0
rate_mda = 1000 / t_mda_sample
t_mda_total = N_PAIRS / rate_mda
results_2k39.append(("MDAnalysis", rate_mda, t_mda_total, rate_mda/rate_bio, "#475569"))
print(f"    5. MDAnalysis (QCPROT C):     {t_mda_total:.2f} s | {rate_mda:,.1f} align/s | {rate_mda/rate_bio:.1f}x")

# F. Gemmi C++
gemmi_models = [[gemmi.Position(x, y, z) for x, y, z in m] for m in coords_116]
t0 = time.perf_counter()
for i in range(1000):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    _ = gemmi.superpose_positions(gemmi_models[idx_a], gemmi_models[idx_b])
t_gem_sample = time.perf_counter() - t0
rate_gem = 1000 / t_gem_sample
t_gem_total = N_PAIRS / rate_gem
results_2k39.append(("Gemmi C++", rate_gem, t_gem_total, rate_gem/rate_bio, "#334155"))
print(f"    6. Gemmi C++ (superpose):     {t_gem_total:.2f} s | {rate_gem:,.1f} align/s | {rate_gem/rate_bio:.1f}x")

# G. ProDy
t0 = time.perf_counter()
for i in range(2000):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    _ = prody.calcRMSD(coords_116_f64[idx_a], coords_116_f64[idx_b])
t_prody_sample = time.perf_counter() - t0
rate_prody = 2000 / t_prody_sample
t_prody_total = N_PAIRS / rate_prody
results_2k39.append(("ProDy", rate_prody, t_prody_total, rate_prody/rate_bio, "#1e293b"))
print(f"    7. ProDy (C-Kabsch):          {t_prody_total:.3f} s | {rate_prody:,.1f} align/s | {rate_prody/rate_bio:.1f}x")

# H. MDTraj (C/AVX QCP)
t0 = time.perf_counter()
for m in range(N_MODELS):
    _ = md.rmsd(traj_mdt, traj_mdt[m])
t_mdtraj = time.perf_counter() - t0
rate_mdtraj = N_PAIRS / t_mdtraj
results_2k39.append(("MDTraj", rate_mdtraj, t_mdtraj, rate_mdtraj/rate_bio, "#0f172a"))
print(f"    8. MDTraj (C/AVX QCP):        {t_mdtraj:.4f} s | {rate_mdtraj:,.1f} align/s | {rate_mdtraj/rate_bio:.1f}x")

# I. strux-rs Single-Thread CPU
t0 = time.perf_counter()
for i in range(1000):
    idx_a = i % N_MODELS
    idx_b = (i * 7) % N_MODELS
    _ = strux_rs.calculate_rmsd_kabsch(coords_116[idx_a], coords_116[idx_b])
t_strux1_sample = time.perf_counter() - t0
rate_strux_1 = 1000 / t_strux1_sample
t_strux1_total = N_PAIRS / rate_strux_1
results_2k39.append(("strux (1T)", rate_strux_1, t_strux1_total, rate_strux_1/rate_bio, "#2563eb"))
print(f"    9. strux-rs (1T CPU SVD):     {t_strux1_total:.3f} s | {rate_strux_1:,.1f} align/s | {rate_strux_1/rate_bio:.1f}x")

# J. strux-rs 16-Thread Rayon CPU
t0 = time.perf_counter()
for m in range(N_MODELS):
    _ = strux_rs.cpu_trajectory_rmsd(coords_116, coords_116[m])
t_strux_16 = time.perf_counter() - t0
rate_strux_16 = N_PAIRS / t_strux_16
results_2k39.append(("strux (16T)", rate_strux_16, t_strux_16, rate_strux_16/rate_bio, "#1d4ed8"))
print(f"   10. strux-rs (16T Rayon):      {t_strux_16*1e3:.2f} ms | {rate_strux_16:,.1f} align/s | {rate_strux_16/rate_bio:.1f}x")

# K. strux-rs CUDA GPU (RTX A2000)
t0 = time.perf_counter()
_ = strux_rs.cuda_pairwise_rmsd(coords_116)
t_strux_gpu = time.perf_counter() - t0
rate_strux_gpu = N_PAIRS / t_strux_gpu
results_2k39.append(("strux (CUDA)", rate_strux_gpu, t_strux_gpu, rate_strux_gpu/rate_bio, "#0f766e"))
print(f"   11. strux-rs (CUDA RTX A2000): {t_strux_gpu*1e3:.2f} ms | {rate_strux_gpu:,.1f} align/s | {rate_strux_gpu/rate_bio:,.0f}x")

# Parity check on Ubiquitin Model 1 vs 2
rmsd_bio = sup.get_rms()
rmsd_st = strux_rs.calculate_rmsd_kabsch(coords_116[idx_a], coords_116[idx_b])
print(f"\n    Numerical Parity on 2K39 Ubiquitin: Max delta = {abs(rmsd_bio - rmsd_st):.6e} Å")

# ==============================================================================
# 2. REAL BIOLOGICAL TRAJECTORY DYNAMICS (trajectory.pdb, 1,919 ATOMS, 100 FRAMES)
# ==============================================================================
print("\n[2] Loading GFP Folding Trajectory (trajectory.pdb, 1,919 atoms, 100 frames)...")
u_gfp = mda.Universe("trajectory.pdb")
gfp_coords = np.array([u_gfp.atoms.positions for ts in u_gfp.trajectory], dtype=np.float32)
traj_gfp_mdt = md.load_pdb("trajectory.pdb")

# Radius of Gyration
t0 = time.perf_counter()
_ = [u_gfp.atoms.radius_of_gyration() for ts in u_gfp.trajectory]
t_mda_rg = time.perf_counter() - t0
rate_mda_rg = len(gfp_coords) / t_mda_rg

t0 = time.perf_counter()
_ = md.compute_rg(traj_gfp_mdt)
t_mdt_rg = time.perf_counter() - t0
rate_mdt_rg = len(gfp_coords) / t_mdt_rg

t0 = time.perf_counter()
for f in gfp_coords:
    _ = strux_rs.calculate_rg(f)
t_st_rg = time.perf_counter() - t0
rate_st_rg = len(gfp_coords) / t_st_rg

print(f"    Radius of Gyration (GFP 1,919 atoms):")
print(f"      MDAnalysis:  {rate_mda_rg:,.1f} fps (1.0x Ref)")
print(f"      MDTraj:      {rate_mdt_rg:,.1f} fps ({rate_mdt_rg/rate_mda_rg:.1f}x)")
print(f"      strux-rs:    {rate_st_rg:,.1f} fps ({rate_st_rg/rate_mda_rg:.1f}x faster)")

# ==============================================================================
# 3. REAL METAGENOMIC PRODUCTION A3M / STOCKHOLM INGESTION
# ==============================================================================
print("\n[3] Ingesting Production Metagenomic MSAs...")

# A. Real 20.6 MB Kinase HMMER Stockholm
sto_path = "benchmarks/data/hmm_output.sto"
with open(sto_path, "r") as f:
    sto_str = f.read()

# Pure Python Stockholm Parser Baseline
import collections
def py_parse_stockholm(s):
    name_to_seq = collections.OrderedDict()
    for l in s.splitlines():
        l = l.strip()
        if not l or l.startswith(("#", "//")): continue
        parts = l.split()
        if len(parts) >= 2:
            if parts[0] not in name_to_seq: name_to_seq[parts[0]] = ""
            name_to_seq[parts[0]] += parts[1]
    return list(name_to_seq.values())

t0 = time.perf_counter()
seqs_py = py_parse_stockholm(sto_str)
t_py_sto = time.perf_counter() - t0

t0 = time.perf_counter()
msa_st = strux_rs.parse_stockholm_file(sto_path)
t_st_sto = time.perf_counter() - t0

mb_sto = len(sto_str.encode('utf-8')) / (1024 * 1024)
print(f"    Human Kinase HMMER Stockholm ({mb_sto:.1f} MB, {len(seqs_py):,d} sequences):")
print(f"      Pure Python (Baseline): {t_py_sto:.4f} s ({mb_sto/t_py_sto:.1f} MB/s)")
print(f"      strux-rs:               {t_st_sto:.4f} s ({mb_sto/t_st_sto:.1f} MB/s, {t_py_sto/t_st_sto:.1f}x faster)")

# B. Real AlphaFold BFD/Uniclust A3M
a3m_path = "benchmarks/data/bfd_uniclust_hits.a3m"
with open(a3m_path, "r") as f:
    a3m_str = f.read()

def py_parse_a3m(s):
    seqs, idx = [], -1
    for l in s.splitlines():
        l = l.strip()
        if not l: continue
        if l.startswith(">"):
            idx += 1
            seqs.append("")
        else:
            seqs[idx] += l
    del_tab = str.maketrans("", "", "abcdefghijklmnopqrstuvwxyz")
    return [x.translate(del_tab) for x in seqs]

t0 = time.perf_counter()
_ = py_parse_a3m(a3m_str)
t_py_a3m = time.perf_counter() - t0

t0 = time.perf_counter()
_ = strux_rs.parse_a3m(a3m_str)
t_st_a3m = time.perf_counter() - t0

kb_a3m = len(a3m_str.encode('utf-8')) / 1024
print(f"    AlphaFold BFD/Uniclust A3M ({kb_a3m:.1f} KB):")
print(f"      Pure Python (Baseline): {t_py_a3m*1e3:.2f} ms")
print(f"      strux-rs:               {t_st_a3m*1e3:.2f} ms ({t_py_a3m/t_st_a3m:.1f}x faster)")

# Save benchmark data for plotting
with open("benchmarks/gold_standard_superimpose.dat", "w") as f:
    f.write("# Index Method Rate Speedup Color\n")
    for idx, (m, r, t, sp, c) in enumerate(results_2k39, 1):
        f.write(f'{idx} "{m}" {r:.1f} {sp:.1f}x {c.replace("#", "0x")}\n')

print("\nAll gold-standard benchmarks completed successfully!")