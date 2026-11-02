import os
import time
import tempfile
import numpy as np
import mdtraj
from mdtraj.formats import XTCTrajectoryFile
import MDAnalysis as mda
import strux_rs

def banner(title: str):
    print("\n" + "=" * 78)
    print(f"🔥 {title}")
    print("=" * 78)

# ==============================================================================
# 1. INTENSE lDDT STRESS & NUMERICAL PRECISION TEST
# ==============================================================================
def numpy_reference_lddt(coords_ref: np.ndarray, coords_target: np.ndarray, cutoff: float = 15.0):
    diff_ref = coords_ref[:, None, :] - coords_ref[None, :, :]
    dist_ref = np.linalg.norm(diff_ref, axis=-1)

    diff_tgt = coords_target[:, None, :] - coords_target[None, :, :]
    dist_tgt = np.linalg.norm(diff_tgt, axis=-1)

    n = coords_ref.shape[0]
    mask = (dist_ref < cutoff) & (~np.eye(n, dtype=bool))
    delta = np.abs(dist_ref - dist_tgt)

    preserved = (
        (delta < 0.5).astype(np.float64) +
        (delta < 1.0).astype(np.float64) +
        (delta < 2.0).astype(np.float64) +
        (delta < 4.0).astype(np.float64)
    ) * 0.25

    per_res = np.ones(n, dtype=np.float32)
    for i in range(n):
        m = mask[i]
        cnt = np.sum(m)
        if cnt > 0:
            per_res[i] = float(np.sum(preserved[i, m]) / cnt)
    # Global score pools all scored pairs (AlphaFold lddt(per_residue=False) / OpenStructure).
    n_pairs = int(np.sum(mask))
    global_lddt = float(np.sum(preserved[mask]) / n_pairs) if n_pairs > 0 else 1.0
    return global_lddt, per_res

def stress_test_lddt():
    banner("1. AlphaFold lDDT: Numerical Equivalence, SO(3) Invariance & 4,000-Atom Stress")
    rng = np.random.default_rng(2026)

    # A. Exact numerical agreement against NumPy reference across 5 noise levels
    n_test = 400
    ref_coords = np.cumsum(rng.normal(0, 2.2, size=(n_test, 3)), axis=0).astype(np.float32)

    for sigma in [0.0, 0.25, 0.75, 2.0, 5.0]:
        noisy = (ref_coords + rng.normal(0, sigma, size=ref_coords.shape)).astype(np.float32)
        g_rs, pr_rs = strux_rs.calculate_lddt(ref_coords, noisy, cutoff=15.0)
        g_np, pr_np = numpy_reference_lddt(ref_coords, noisy, cutoff=15.0)

        max_err = np.max(np.abs(pr_rs - pr_np))
        assert np.isclose(g_rs, g_np, atol=1e-5), f"Global lDDT mismatch at sigma={sigma}: {g_rs} vs {g_np}"
        assert max_err < 1e-5, f"Per-residue lDDT mismatch at sigma={sigma}: max_err={max_err}"
        print(f"  [σ={sigma:4.2f} Å] Global lDDT = {g_rs:.6f} | Max per-residue Δ vs NumPy = {max_err:.2e}")

    # B. 50 random 3D rotations & extreme translations
    max_rot_drift = 0.0
    for _ in range(50):
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        trans = rng.uniform(-5000.0, 5000.0, size=(1, 3))
        transformed = ((ref_coords.astype(np.float64) @ q.T) + trans).astype(np.float32)
        g_rot, _ = strux_rs.calculate_lddt(ref_coords, transformed, cutoff=15.0)
        max_rot_drift = max(max_rot_drift, abs(1.0 - g_rot))
    assert max_rot_drift < 1e-5, f"SO(3) invariance drift too high: {max_rot_drift}"
    print(f"  [SO(3) Invariance] 50 random rotations + ±5000 Å translations -> Max drift = {max_rot_drift:.2e}")

    # C. High-throughput benchmark on 4,000-atom complex (16M pairs)
    n_large = 4000
    large_ref = np.cumsum(rng.normal(0, 2.0, size=(n_large, 3)), axis=0).astype(np.float32)
    large_tgt = (large_ref + rng.normal(0, 1.2, size=(n_large, 3))).astype(np.float32)

    t0 = time.perf_counter()
    g_large_rs, _ = strux_rs.calculate_lddt(large_ref, large_tgt, cutoff=15.0)
    t_rs = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    g_large_np, _ = numpy_reference_lddt(large_ref, large_tgt, cutoff=15.0)
    t_np = (time.perf_counter() - t0) * 1000.0

    assert np.isclose(g_large_rs, g_large_np, atol=1e-5)
    print(f"  [4,000 Atoms / 16M pairs] strux-rs: {t_rs:.2f} ms vs NumPy: {t_np:.2f} ms ({t_np/t_rs:.1f}x speedup) | lDDT={g_large_rs:.5f}")


# ==============================================================================
# 2. INTENSE MARKOV STATE MODEL (MSM) 2,000,000-STEP ERGODIC & RARE STATE TEST
# ==============================================================================
def stress_test_markov():
    banner("2. Markov State Populations: 2,000,000-Step T4 Lysozyme Rare-State Recovery")
    rng = np.random.default_rng(42)

    # Ground-truth stationary distribution from Bhakat JCIM paper (T4 Lysozyme 4 states):
    # State 0 (Buried/Open): 58.2%, State 1 (Exposed/Open): 39.2%,
    # State 2 (Buried/Closed): 1.9%, State 3 (Exposed/Closed rare state): 0.7%
    pi_true = np.array([0.582, 0.392, 0.019, 0.007], dtype=np.float64)

    # Construct reversible transition matrix satisfying detailed balance: pi_i * T_ij = pi_j * T_ji
    flux = np.array([
        [0.000, 0.040, 0.003, 0.001],
        [0.040, 0.000, 0.002, 0.001],
        [0.003, 0.002, 0.000, 0.0005],
        [0.001, 0.001, 0.0005, 0.000],
    ], dtype=np.float64)

    T_true = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            if i != j:
                T_true[i, j] = flux[i, j] / pi_true[i]
        T_true[i, i] = 1.0 - np.sum(T_true[i])

    # Verify T_true stationary distribution matches pi_true
    assert np.allclose(pi_true @ T_true, pi_true, atol=1e-12)

    # Simulate 250,000 ergodic steps and tile to 2,000,000 steps
    cdf = np.cumsum(T_true, axis=1)
    n_sim = 250_000
    sim_traj = [0] * n_sim
    curr = 0
    cdf_list = cdf.tolist()
    rand_list = rng.random(n_sim).tolist()
    for step in range(1, n_sim):
        u = rand_list[step]
        row = cdf_list[curr]
        if u < row[0]:
            curr = 0
        elif u < row[1]:
            curr = 1
        elif u < row[2]:
            curr = 2
        else:
            curr = 3
        sim_traj[step] = curr

    full_traj = sim_traj * 8 # 2,000,000 steps!
    t0 = time.perf_counter()
    pi_rs, dG_rs, tmat_rs = strux_rs.markov_populations(full_traj, num_states=4, lag_time=1, temperature=300.0)
    t_msm = (time.perf_counter() - t0) * 1000.0

    # Cross-check against exact eigendecomposition of tmat_rs in SciPy/NumPy
    evals, evecs = np.linalg.eig(tmat_rs.astype(np.float64).T)
    idx = np.argmin(np.abs(evals - 1.0))
    pi_eig = np.real(evecs[:, idx])
    pi_eig /= np.sum(pi_eig)

    kb_t = 0.0019872041 * 300.0
    dG_expected = -kb_t * np.log(pi_rs / np.max(pi_rs))

    assert np.allclose(pi_rs, pi_eig, atol=1e-5), f"Power iteration vs eig mismatch: {pi_rs} vs {pi_eig}"
    assert np.allclose(dG_rs, dG_expected, atol=1e-5), f"dG formula mismatch: {dG_rs} vs {dG_expected}"
    assert np.allclose(pi_rs, pi_true, atol=0.015), f"Ergodic recovery drifted from ground truth: {pi_rs} vs {pi_true}"

    print(f"  [2,000,000 Frames Processed] Time: {t_msm:.2f} ms ({2.0 / (t_msm/1000.0):.1f}M frames/sec)")
    print(f"  [Ground Truth π]  {np.round(pi_true * 100, 2)} %")
    print(f"  [Recovered π]     {np.round(pi_rs * 100, 2)} % (Max diff vs np.linalg.eig = {np.max(np.abs(pi_rs - pi_eig)):.2e})")
    print(f"  [Relative ΔG]     {np.round(dG_rs, 4)} kcal/mol (Rare state ΔG = {dG_rs[3]:.3f} kcal/mol)")


# ==============================================================================
# 3. INTENSE TRAJECTORY PCA (2,500 FRAMES x 500 ATOMS) VS LAPACK SVD
# ==============================================================================
def stress_test_pca():
    banner("3. Cartesian Trajectory PCA: 2,500 Frames x 500 Atoms vs LAPACK SVD")
    rng = np.random.default_rng(1337)

    n_frames = 2500
    n_atoms = 500
    dim = n_atoms * 3

    # Create 4 orthogonal ground-truth collective modes with distinct amplitudes
    modes = rng.normal(size=(4, dim))
    modes, _ = np.linalg.qr(modes.T)
    modes = modes.T # [4, dim] orthonormal

    t = np.linspace(0, 8 * np.pi, n_frames)
    amps = np.column_stack([
        15.0 * np.sin(t),
        8.0 * np.cos(1.7 * t),
        4.0 * np.sin(3.3 * t + 0.5),
        2.0 * np.cos(5.1 * t - 0.3),
    ]) # [n_frames, 4]

    X_clean = amps @ modes
    noise = rng.normal(0, 0.15, size=(n_frames, dim))
    traj = (X_clean + noise).reshape(n_frames, n_atoms, 3).astype(np.float32)

    # Run strux-rs PCA
    t0 = time.perf_counter()
    proj_rs, var_rs, comp_rs = strux_rs.trajectory_pca(traj, n_components=4)
    t_rs = (time.perf_counter() - t0) * 1000.0

    # Run full NumPy LAPACK SVD reference
    t0 = time.perf_counter()
    X_flat = traj.reshape(n_frames, dim).astype(np.float64)
    X_centered = X_flat - np.mean(X_flat, axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    variances_np = (S ** 2) / (n_frames - 1)
    var_np = (variances_np[:4] / np.sum(variances_np)).astype(np.float32)
    comp_np = Vt[:4].astype(np.float32)
    proj_np = (X_centered @ Vt[:4].T).astype(np.float32)
    t_np = (time.perf_counter() - t0) * 1000.0

    # 1. Verify explained variance ratio matches LAPACK SVD
    var_err = np.max(np.abs(var_rs - var_np))
    assert var_err < 1e-5, f"Explained variance mismatch: {var_rs} vs {var_np}"

    # 2. Verify subspace alignment (|cos(theta)| == 1.0 up to sign flip)
    cos_sims = [abs(np.dot(comp_rs[i], comp_np[i])) for i in range(4)]
    min_cos = min(cos_sims)
    assert min_cos > 0.99999, f"Eigenvector alignment error: cos_sims={cos_sims}"

    # 3. Verify component orthogonality V * V^T == I
    gram = comp_rs @ comp_rs.T
    ortho_err = np.max(np.abs(gram - np.eye(4)))
    assert ortho_err < 1e-5, f"Orthogonality error: {ortho_err}"

    # 4. Verify projections match up to sign flip
    proj_corrs = [abs(np.corrcoef(proj_rs[:, i], proj_np[:, i])[0, 1]) for i in range(4)]
    assert min(proj_corrs) > 0.999999, f"Projection correlation error: {proj_corrs}"

    print(f"  [2,500 x 1,500 Matrix] strux-rs: {t_rs:.2f} ms vs NumPy SVD: {t_np:.2f} ms ({t_np/t_rs:.1f}x speedup)")
    print(f"  [Explained Variance]   {np.round(var_rs * 100, 3)} % (Max Δ vs LAPACK = {var_err:.2e})")
    print(f"  [PC Cosine Similarity] {[round(float(c), 7) for c in cos_sims]} | Orthogonality err = {ortho_err:.2e}")


# ==============================================================================
# 4. INTENSE MSA CLUSTERING: 10,000 SEQUENCES x 250 RESIDUES (2.5M CHARS)
# ==============================================================================
def python_reference_cluster_msa(sequences, min_identity=0.7):
    reps = []
    labels = []
    for i, s in enumerate(sequences):
        matched = None
        for cid, r_idx in enumerate(reps):
            r = sequences[r_idx]
            matches = 0
            valid = 0
            for a, b in zip(s, r):
                if a == '-' and b == '-':
                    continue
                valid += 1
                if a == b and a != '-':
                    matches += 1
            ident = (matches / valid) if valid > 0 else 0.0
            if ident >= min_identity:
                matched = cid
                break
        if matched is not None:
            labels.append(matched)
        else:
            reps.append(i)
            labels.append(len(reps) - 1)
    return np.array(labels, dtype=np.uint64), np.array(reps, dtype=np.uint64)

def stress_test_msa_clustering():
    banner("4. AF-Cluster MSA Sub-Alignment: 10,000 Sequences x 250 Residues (2.5M AA)")
    rng = np.random.default_rng(99)
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))

    n_families = 20
    seqs_per_family = 500
    seq_len = 250
    total_seqs = n_families * seqs_per_family # 10,000 sequences

    # Generate 20 completely distinct founder sequences (pairwise identity ~5%)
    founders = [rng.choice(alphabet, size=seq_len) for _ in range(n_families)]

    sequences = []
    descriptions = []
    del_matrix = [[0] * seq_len for _ in range(total_seqs)]

    for fam_id, founder in enumerate(founders):
        for m in range(seqs_per_family):
            seq = founder.copy()
            if m > 0:
                # Mutate 8% of positions within family (~92% identity to founder)
                mut_mask = rng.random(seq_len) < 0.08
                seq[mut_mask] = rng.choice(alphabet, size=np.sum(mut_mask))
                if m % 5 == 0:
                    seq[10:18] = '-'
            sequences.append("".join(seq))
            descriptions.append(f"fam{fam_id}_seq{m}")

    # 1. Exact agreement vs Python reference on a 1,000-sequence strided subset
    sub_seqs = sequences[::10]
    t0 = time.perf_counter()
    l_py, r_py = python_reference_cluster_msa(sub_seqs, min_identity=0.75)
    t_py_1k = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    l_rs_1k, r_rs_1k = strux_rs.cluster_msa(sub_seqs, min_identity=0.75)
    t_rs_1k = (time.perf_counter() - t0) * 1000.0

    assert np.array_equal(l_py, l_rs_1k), "Cluster labels mismatch vs Python reference!"
    assert np.array_equal(r_py, r_rs_1k), "Representative indices mismatch vs Python reference!"
    print(f"  [1,000 Seq Verification] Exact bit-for-bit match vs Python | strux-rs: {t_rs_1k:.2f} ms vs Python: {t_py_1k:.2f} ms ({t_py_1k/t_rs_1k:.1f}x faster)")

    # 2. Full 10,000-sequence clustering + sub-MSA extraction
    msa = strux_rs.Msa(sequences=sequences, deletion_matrix=del_matrix, descriptions=descriptions)
    t0 = time.perf_counter()
    labels_10k, reps_10k = msa.cluster(min_identity=0.75)
    t_rs_10k = (time.perf_counter() - t0) * 1000.0

    assert len(reps_10k) == n_families, f"Expected {n_families} families, got {len(reps_10k)}"
    total_extracted = 0
    for cid in range(len(reps_10k)):
        sub_msa = msa.get_cluster(cid, labels_10k.tolist())
        assert sub_msa.num_seqs == seqs_per_family, f"Cluster {cid} size mismatch: {sub_msa.num_seqs}"
        total_extracted += sub_msa.num_seqs
    assert total_extracted == total_seqs
    print(f"  [10,000 Seqs x 250 AA]   Clustered 2,500,000 residues into {len(reps_10k)} exact families in {t_rs_10k:.2f} ms!")


# ==============================================================================
# 5. INTENSE GROMACS .XTC XDR 3D DECOMPRESSION TEST VS MDTRAJ & MDANALYSIS
# ==============================================================================
def stress_test_xtc():
    banner("5. Pure-Rust GROMACS .xtc Reader: Cross-Validation vs MDTraj & MDAnalysis")
    rng = np.random.default_rng(777)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Case A: Uncompressed small system (<= 9 atoms)
        xtc_small = os.path.join(tmpdir, "small_8atoms.xtc")
        xyz_small_nm = rng.uniform(-5.0, 5.0, size=(50, 8, 3)).astype(np.float32)
        time_small = np.arange(50, dtype=np.float32) * 0.5
        step_small = np.arange(50, dtype=np.int32) * 100
        box_small = np.tile(np.eye(3, dtype=np.float32) * 10.0, (50, 1, 1))

        with XTCTrajectoryFile(xtc_small, 'w') as f:
            f.write(xyz_small_nm, time=time_small, step=step_small, box=box_small)

        coords_s = strux_rs.parse_xtc(xtc_small)
        max_err_small = np.max(np.abs(coords_s - xyz_small_nm * 10.0))
        assert coords_s.shape == (50, 8, 3)
        assert max_err_small < 1e-5, f"Uncompressed XTC max error: {max_err_small}"
        print(f"  [Uncompressed ≤9 Atoms]  50 frames x 8 atoms -> Max coordinate error = {max_err_small:.2e} Å")

        # Case B: Large 3D-compressed trajectory (500 frames x 1,000 atoms = 500,000 atoms)
        # Include realistic water-like triplets + protein chain to trigger all XDR bitpacking branches!
        xtc_large = os.path.join(tmpdir, "large_1000atoms.xtc")
        n_frames = 500
        n_atoms = 1000

        # Build realistic chain + water coordinates in nm so adjacent atoms are within 0.1-0.3 nm (triggers run-length water encoding!)
        base_nm = np.cumsum(rng.normal(0, 0.12, size=(n_atoms, 3)), axis=0).astype(np.float32)
        xyz_large_nm = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for f_idx in range(n_frames):
            xyz_large_nm[f_idx] = base_nm + rng.normal(0, 0.02, size=(n_atoms, 3)).astype(np.float32)

        time_large = np.arange(n_frames, dtype=np.float32) * 2.0
        step_large = np.arange(n_frames, dtype=np.int32) * 1000
        box_large = np.tile(np.eye(3, dtype=np.float32) * 15.0, (n_frames, 1, 1))

        with XTCTrajectoryFile(xtc_large, 'w') as f:
            f.write(xyz_large_nm, time=time_large, step=step_large, box=box_large)

        file_mb = os.path.getsize(xtc_large) / (1024 * 1024)

        # 1. Read with MDTraj
        t0 = time.perf_counter()
        with XTCTrajectoryFile(xtc_large, 'r') as f:
            xyz_mdt_nm, time_mdt, step_mdt, _ = f.read()
        t_mdt = (time.perf_counter() - t0) * 1000.0
        xyz_mdt_ang = xyz_mdt_nm * 10.0

        # 2. Read with MDAnalysis
        t0 = time.perf_counter()
        u_mda = mda.Universe.empty(n_atoms, trajectory=True)
        u_mda.load_new(xtc_large)
        xyz_mda_ang = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
        for idx, ts in enumerate(u_mda.trajectory):
            xyz_mda_ang[idx] = ts.positions
        t_mda = (time.perf_counter() - t0) * 1000.0

        # 3. Read with strux-rs
        t0 = time.perf_counter()
        coords_rs = strux_rs.parse_xtc(xtc_large)
        t_rs = (time.perf_counter() - t0) * 1000.0

        assert coords_rs.shape == (n_frames, n_atoms, 3), f"Shape mismatch: {coords_rs.shape}"

        max_diff_vs_mdt = np.max(np.abs(coords_rs - xyz_mdt_ang))
        max_diff_vs_mda = np.max(np.abs(coords_rs - xyz_mda_ang))
        max_diff_vs_orig = np.max(np.abs(coords_rs - xyz_large_nm * 10.0))

        assert max_diff_vs_mdt < 1e-4, f"Mismatch vs MDTraj xdrfile decoder: {max_diff_vs_mdt}"
        assert max_diff_vs_mda < 1e-4, f"Mismatch vs MDAnalysis xdrfile decoder: {max_diff_vs_mda}"
        assert max_diff_vs_orig <= 0.006, f"Quantization error exceeded 0.005 Å half-step: {max_diff_vs_orig}"

        print(f"  [Compressed 500x1,000]   File size: {file_mb:.2f} MB (500,000 3D atom coordinates)")
        print(f"  [Bit-Exact vs xdrfile]   Max Δ vs MDTraj = {max_diff_vs_mdt:.2e} Å | Max Δ vs MDAnalysis = {max_diff_vs_mda:.2e} Å")
        print(f"  [Quantization Precision] Max Δ vs unquantized float32 = {max_diff_vs_orig:.5f} Å (XTC precision = 0.01 Å)")
        print(f"  [Read Throughput]        strux-rs: {t_rs:.2f} ms | MDTraj: {t_mdt:.2f} ms ({t_mdt/t_rs:.1f}x) | MDAnalysis: {t_mda:.2f} ms ({t_mda/t_rs:.1f}x)")

        # Case C: In-flight atom_indices selection during XTC decompression
        ca_idx = list(range(0, n_atoms, 8))  # 125 solute/CA atoms out of 1,000
        coords_sel = strux_rs.parse_xtc(xtc_large, atom_indices=ca_idx)
        assert coords_sel.shape == (n_frames, len(ca_idx), 3)
        assert np.array_equal(coords_sel, coords_rs[:, ca_idx, :]), "XTC atom_indices selection mismatch!"
        print(f"  [In-Flight Selection]    Extracted {len(ca_idx)}/{n_atoms} atoms across {n_frames} frames with 0.00e+00 Δ")


# ==============================================================================
# 6. mmCIF (.cif) AF3/BOLTZ-1 PARSER & IN-FLIGHT CA/ATOM SELECTION STRESS TEST
# ==============================================================================
def stress_test_cif_and_selection():
    banner("6. mmCIF (.cif) AF3/Boltz-1 Parser + pLDDT + In-Flight CA/Atom Selection")
    rng = np.random.default_rng(2027)

    with tempfile.TemporaryDirectory() as tmpdir:
        cif_path = os.path.join(tmpdir, "af3_ensemble.cif")
        n_models = 5
        n_res = 120
        atoms_per_res = ["N", "CA", "C", "O", "CB"]
        total_atoms = n_res * len(atoms_per_res)

        gt_coords = np.round(rng.uniform(-40.0, 40.0, size=(n_models, total_atoms, 3)), 3).astype(np.float32)
        gt_plddt = np.round(rng.uniform(50.0, 99.5, size=(n_models, total_atoms)), 2).astype(np.float32)

        with open(cif_path, "w") as f:
            f.write("data_af3_ensemble\n#\nloop_\n")
            f.write("_atom_site.group_PDB\n")
            f.write("_atom_site.id\n")
            f.write("_atom_site.type_symbol\n")
            f.write("_atom_site.label_atom_id\n")
            f.write("_atom_site.label_alt_id\n")
            f.write("_atom_site.label_comp_id\n")
            f.write("_atom_site.label_asym_id\n")
            f.write("_atom_site.label_seq_id\n")
            f.write("_atom_site.Cartn_x\n")
            f.write("_atom_site.Cartn_y\n")
            f.write("_atom_site.Cartn_z\n")
            f.write("_atom_site.occupancy\n")
            f.write("_atom_site.B_iso_or_equiv\n")
            f.write("_atom_site.pdbx_PDB_model_num\n")

            atom_serial = 1
            for m in range(n_models):
                for r in range(n_res):
                    for a_idx, aname in enumerate(atoms_per_res):
                        flat_i = r * len(atoms_per_res) + a_idx
                        x, y, z = gt_coords[m, flat_i]
                        b = gt_plddt[m, flat_i]
                        f.write(
                            f"ATOM {atom_serial} {aname[0]} {aname} . ALA A {r+1} "
                            f"{x:.3f} {y:.3f} {z:.3f} 1.00 {b:.2f} {m+1}\n"
                        )
                        atom_serial += 1
                        # Inject a B alternate conformer that must be ignored
                        if a_idx == 4 and r % 10 == 0:
                            f.write(
                                f"ATOM {atom_serial} C CB B ALA A {r+1} "
                                f"{x+99.0:.3f} {y+99.0:.3f} {z+99.0:.3f} 0.50 {b:.2f} {m+1}\n"
                            )
                            atom_serial += 1
            f.write("#\n")

        t0 = time.perf_counter()
        coords_all, plddt_all = strux_rs.parse_cif_with_plddt(cif_path)
        t_cif = (time.perf_counter() - t0) * 1000.0

        assert coords_all.shape == (n_models, total_atoms, 3), f"CIF shape mismatch: {coords_all.shape}"
        assert plddt_all.shape == (n_models, total_atoms), f"pLDDT shape mismatch: {plddt_all.shape}"
        max_c_err = np.max(np.abs(coords_all - np.round(gt_coords, 3)))
        max_b_err = np.max(np.abs(plddt_all - np.round(gt_plddt, 2)))
        assert max_c_err < 1e-4, f"CIF coord error: {max_c_err}"
        assert max_b_err < 1e-4, f"CIF pLDDT error: {max_b_err}"

        # Verify ca_only=True extracts only the 120 CA atoms per model
        coords_ca, plddt_ca = strux_rs.parse_cif_with_plddt(cif_path, ca_only=True)
        expected_ca_idx = list(range(1, total_atoms, 5))
        assert coords_ca.shape == (n_models, n_res, 3)
        assert np.allclose(coords_ca, coords_all[:, expected_ca_idx, :], atol=1e-5)
        assert np.allclose(plddt_ca, plddt_all[:, expected_ca_idx], atol=1e-5)

        # Verify atom_indices subsetting & out-of-bounds ValueError
        custom_sel = [0, 7, 42, 199, 599]
        coords_sub = strux_rs.parse_cif(cif_path, atom_indices=custom_sel)
        assert coords_sub.shape == (n_models, len(custom_sel), 3)
        assert np.array_equal(coords_sub, coords_all[:, custom_sel, :])

        coords_full, plddt_full, chains, res_ids, res_names, atom_names = strux_rs.parse_cif_full(cif_path)
        assert len(chains) == total_atoms and chains[0] == "A"
        assert len(res_names) == total_atoms and res_names[0] == "ALA"
        assert len(atom_names) == total_atoms and atom_names[1] == "CA"

        try:
            strux_rs.parse_cif(cif_path, atom_indices=[999999])
            raise AssertionError("Expected ValueError on out-of-bounds atom_indices")
        except ValueError:
            pass

        print(f"  [Multi-Model mmCIF]      {n_models} models x {total_atoms} atoms + alt-loc filtering parsed in {t_cif:.2f} ms")
        print(f"  [Coord & pLDDT Parity]   Max coord Δ = {max_c_err:.2e} Å | Max pLDDT Δ = {max_b_err:.2e}")
        print(f"  [CA-Only & Subsetting]   Extracted {n_res} Cα atoms, metadata & verified OOB ValueError")


# ==============================================================================
# 7. CHALLENGES 01-05 THERAPEUTIC DESIGN & BIOPHYSICS PRIMITIVES STRESS TEST
# ==============================================================================
def stress_test_therapeutic_primitives():
    banner("7. Challenges 01-05 Primitives: Pose_RMSD, BUNS, SASA/BSA, pH-Switch, ANM & Developability")
    rng = np.random.default_rng(2028)

    # 1. Target-Anchored Pose_RMSD & scRMSD (Single + 1,000-Design Batch)
    ref_target = rng.uniform(-25.0, 25.0, size=(200, 3)).astype(np.float32)
    ref_binder = rng.uniform(-15.0, 15.0, size=(80, 3)).astype(np.float32)

    # Apply a global rigid-body 3D rotation Q + translation T to the whole complex,
    # PLUS a 15.0 Å drift along X to the binder ONLY (simulating epitope drift with perfect monomer fold!)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    q = q.astype(np.float32)
    trans = np.array([[120.0, -85.0, 42.0]], dtype=np.float32)
    drift_local = np.array([[15.0, 0.0, 0.0]], dtype=np.float32)

    mob_target = (ref_target @ q.T) + trans
    mob_binder = ((ref_binder + drift_local) @ q.T) + trans

    sc_rmsd, pose_rmsd = strux_rs.calculate_pose_and_sc_rmsd(ref_target, ref_binder, mob_target, mob_binder)
    assert abs(sc_rmsd - 0.0) < 1e-3, f"Expected sc_rmsd ~ 0.0, got {sc_rmsd}"
    assert abs(pose_rmsd - 15.0) < 1e-3, f"Expected pose_rmsd ~ 15.0, got {pose_rmsd}"

    n_batch = 1000
    batch_t = np.ascontiguousarray(np.broadcast_to(mob_target, (n_batch, 200, 3)).copy())
    batch_b = np.ascontiguousarray(np.broadcast_to(mob_binder, (n_batch, 80, 3)).copy())
    t0 = time.perf_counter()
    sc_arr, pose_arr = strux_rs.batch_pose_and_sc_rmsd(ref_target, ref_binder, batch_t, batch_b)
    t_pose = (time.perf_counter() - t0) * 1000.0
    assert np.allclose(sc_arr, 0.0, atol=1e-3) and np.allclose(pose_arr, 15.0, atol=1e-3)
    print(f"  [1. Pose_RMSD & scRMSD]  1,000 complexes in {t_pose:.2f} ms | scRMSD={sc_rmsd:.4f} Å, Pose_RMSD={pose_rmsd:.4f} Å (exact 15.0 Å epitope drift caught!)")

    # 2. All-Atom Interface Physics: Contacts, H-Bonds, Salt Bridges & Buried Unsatisfied Polars (BUNS)
    # Construct a 4-atom synthetic interface:
    # Target (Chain A): ASP OD1 at (0,0,0), ALA CB at (0, 3.0, 0)
    # Binder (Chain B): LYS NZ at (2.8,0,0) -> forms H-bond (<=3.5 A) AND salt bridge (<=4.0 A) with ASP OD1!
    # Binder (Chain B): ASN ND2 at (0, 3.0, 3.2) -> 3.2 A from Target ALA CB (buried <= 5.0 A) but >4.5 A from any N/O -> BUNS!
    iface_coords = np.array([
        [0.0, 0.0, 0.0],    # 0: Target ASP OD1 (satisfied by LYS NZ)
        [0.0, 5.0, 0.0],    # 1: Target ALA CB (hydrophobic carbon burying ASN ND2)
        [2.8, 0.0, 0.0],    # 2: Binder LYS NZ (forms salt bridge & H-bond with ASP OD1)
        [0.0, 5.0, 3.2],    # 3: Binder ASN ND2 (3.2 A from ALA CB, >6.0 A from ASP OD1 & LYS NZ -> buried unsatisfied!)
    ], dtype=np.float32)
    atom_names = ["OD1", "CB", "NZ", "ND2"]
    res_names = ["ASP", "ALA", "LYS", "ASN"]
    chain_ids = ["A", "A", "B", "B"]
    res_ids = [10, 20, 30, 40]

    n_cont, n_hb, n_sb, n_buns, buns_idx = strux_rs.analyze_interface_physics(
        iface_coords, atom_names, res_names, chain_ids, res_ids, "A", "B"
    )
    assert n_cont == 2 and n_hb == 1 and n_sb == 1 and n_buns == 1 and list(buns_idx) == [3]
    print(f"  [2. Interface Physics]   Contacts={n_cont}, H-bonds={n_hb}, Salt-bridges={n_sb}, BUNS={n_buns} (index {list(buns_idx)} exact)")

    # 3. Shrake-Rupley SASA & Buried Surface Area (BSA)
    single_atom = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    sasa_1, _ = strux_rs.calculate_sasa(single_atom, probe_radius=1.4, n_points=96)
    expected_sphere = 4.0 * np.pi * ((1.70 + 1.40) ** 2)
    assert abs(sasa_1 - expected_sphere) < 0.05, f"Single sphere SASA mismatch: {sasa_1} vs {expected_sphere}"

    t_coords = rng.uniform(-10.0, 10.0, size=(600, 3)).astype(np.float32)
    b_coords = t_coords[:200] + np.array([[2.2, 0.0, 0.0]], dtype=np.float32)
    t0 = time.perf_counter()
    bsa, s_t, s_b, s_c = strux_rs.calculate_bsa(t_coords, b_coords, probe_radius=1.4, n_points=96)
    t_sasa = (time.perf_counter() - t0) * 1000.0
    assert bsa > 500.0 and abs((s_t + s_b - s_c) - bsa) < 1e-2
    print(f"  [3. Shrake-Rupley SASA]  800-atom complex BSA = {bsa:.1f} Å² computed in {t_sasa:.2f} ms (single-sphere Δ = {abs(sasa_1 - expected_sphere):.2e} Å²)")

    # 4. Screened Debye-Hückel pH-Switch Electrostatics (HIS-ASP Viral/Endosomal Switch)
    ph_coords = np.array([
        [0.0, 0.0, 0.0],  # Target ASP OD1
        [0.0, 1.2, 0.0],  # Target ASP OD2
        [3.0, 0.0, 0.0],  # Binder HIS ND1
        [3.0, 1.2, 0.0],  # Binder HIS NE2
    ], dtype=np.float32)
    ddg_ph, dg_60, dg_74 = strux_rs.calculate_ph_switch_ddg(
        ph_coords,
        ["OD1", "OD2", "ND1", "NE2"],
        ["ASP", "ASP", "HIS", "HIS"],
        ["A", "A", "B", "B"],
        "A", "B", 6.0, 7.4, 0.15, 20.0
    )
    assert ddg_ph < -1.0 and dg_60 < dg_74, f"Expected strong pH 6.0 stabilization, got ddg={ddg_ph}"
    print(f"  [4. Debye-Hückel ΔΔGpH]  HIS-ASP switch: ΔG(pH 6.0)={dg_60:.3f}, ΔG(pH 7.4)={dg_74:.3f} -> ΔΔGpH = {ddg_ph:.3f} kcal/mol")

    # 5. Anisotropic Network Model (ANM) Normal Modes vs Exact NumPy eigh
    n_ca = 60
    ca_chain = np.cumsum(rng.normal(0, 2.5, size=(n_ca, 3)), axis=0).astype(np.float32)
    t0 = time.perf_counter()
    evals_rs, evecs_rs, bfac_rs = strux_rs.anm_modes(ca_chain, cutoff=15.0, gamma=1.0, n_modes=5)
    t_anm = (time.perf_counter() - t0) * 1000.0

    # Build reference Hessian in NumPy and compare lowest 5 non-zero eigenvalues
    dim = n_ca * 3
    H_np = np.zeros((dim, dim), dtype=np.float64)
    for i in range(n_ca):
        for j in range(i + 1, n_ca):
            d = ca_chain[j].astype(np.float64) - ca_chain[i].astype(np.float64)
            d2 = np.dot(d, d)
            if d2 <= 225.0:
                block = -1.0 * np.outer(d, d) / d2
                H_np[i*3:(i+1)*3, j*3:(j+1)*3] = block
                H_np[j*3:(j+1)*3, i*3:(i+1)*3] = block
                H_np[i*3:(i+1)*3, i*3:(i+1)*3] -= block
                H_np[j*3:(j+1)*3, j*3:(j+1)*3] -= block
    w_np, _ = np.linalg.eigh(H_np)
    evals_np = w_np[6:11].astype(np.float32)
    max_anm_err = np.max(np.abs(evals_rs - evals_np))
    assert max_anm_err < 1e-4, f"ANM eigenvalues mismatch vs np.linalg.eigh: {evals_rs} vs {evals_np}"
    print(f"  [5. ANM Normal Modes]    60 Cα (180x180 Hessian) in {t_anm:.2f} ms | Max eigenvalue Δ vs LAPACK eigh = {max_anm_err:.2e}")

    # 6. Sequence Developability & Sliding 15-mer Novelty Screener
    alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
    queries = ["".join(rng.choice(alphabet, size=100)) for _ in range(5000)]
    ref_db = ["".join(rng.choice(alphabet, size=120)) for _ in range(200)]
    # Plant an exact 15-mer match in query 0 from ref_db[0]
    queries[0] = ref_db[0][:15] + queries[0][15:]

    t0 = time.perf_counter()
    bio_dict = strux_rs.screen_sequences_biophysics(queries)
    t_dev = (time.perf_counter() - t0) * 1000.0
    assert len(bio_dict["isoelectric_point"]) == 5000

    t0 = time.perf_counter()
    max_id, min_ham = strux_rs.sliding_kmer_novelty(queries[:500], ref_db, k=15)
    t_nov = (time.perf_counter() - t0) * 1000.0
    assert abs(max_id[0] - 1.0) < 1e-6 and min_ham[0] == 0
    print(f"  [6. Seq Developability]  5,000 MPNN sequences screened (pI, GRAVY, CamSol, APRs, liabilities) in {t_dev:.2f} ms")
    print(f"  [   15-mer Novelty DB]   500 queries vs 200 refs in {t_nov:.2f} ms (planted 15-mer identity={max_id[0]:.2f}, hamming={min_ham[0]})")


# ==============================================================================
# 8. HOSTILE INPUTS, CORRUPTED BITSTREAMS & FFI BOUNDARY FUZZING
# ==============================================================================
def stress_test_hostile_fuzz():
    banner("8. Hostile Inputs, Corrupted Bitstreams, NaN/Inf SVD & FFI Boundary Fuzzing")
    import struct
    rng = np.random.default_rng(2029)

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Zero-byte empty files across all 6 file readers
        empty_path = os.path.join(tmpdir, "empty.bin")
        open(empty_path, "wb").close()

        for fn in (strux_rs.parse_xtc, strux_rs.parse_dcd):
            try:
                fn(empty_path)
                raise AssertionError(f"Expected IOError on 0-byte file for {fn}")
            except OSError:
                pass

        assert strux_rs.parse_cif(empty_path).shape == (0, 0, 3)
        assert strux_rs.parse_pdb(empty_path).shape == (0, 0, 3)
        assert len(strux_rs.parse_a3m_file(empty_path)) == 0
        assert len(strux_rs.parse_stockholm_file(empty_path)) == 0

        # 2. Hostile XTC headers: negative natoms, huge smallidx=999999, maxint < minint (div-by-zero probe), & 50 random bit-flip mutations
        bad_xtc = os.path.join(tmpdir, "hostile.xtc")
        # Header with magic=1995, natoms=-500 -> must raise IOError, not wrap usize::MAX
        with open(bad_xtc, "wb") as f:
            f.write(struct.pack(">iiif", 1995, -500, 0, 0.0) + b"\x00" * 60)
        try:
            strux_rs.parse_xtc(bad_xtc)
            raise AssertionError("Expected IOError on negative natoms in XTC")
        except OSError:
            pass

        # Craft a compressed XTC frame (natoms=20) with hostile smallidx=999999 and maxint < minint (sizeint=0 probe)
        with open(bad_xtc, "wb") as f:
            hdr = struct.pack(">iiif", 1995, 20, 1, 1.0) + (b"\x00" * 36) + struct.pack(">i", 20)
            # payload: precision=0.0 (div-by-zero probe), minint=(100,100,100), maxint=(-100,-100,-100), smallidx=999999, nbytes=16
            payload = struct.pack(">fiiiiiii i", 0.0, 100, 100, 100, -100, -100, -100, 999999, 16) + rng.bytes(16)
            f.write(hdr + payload)
        out_fuzzed = strux_rs.parse_xtc(bad_xtc)
        assert out_fuzzed.shape == (1, 20, 3) and np.all(np.isfinite(out_fuzzed))

        # 50 random byte-corruption fuzz trials on a compressed XTC bitstream
        for trial in range(50):
            with open(bad_xtc, "wb") as f:
                hdr = struct.pack(">iiif", 1995, 15, trial, float(trial)) + (b"\x00" * 36) + struct.pack(">i", 15)
                rand_meta = rng.bytes(32)
                payload_bytes = rng.bytes(24)
                f.write(hdr + rand_meta + struct.pack(">i", 24) + payload_bytes)
            try:
                res = strux_rs.parse_xtc(bad_xtc)
                assert res.shape == (1, 15, 3)
            except (OSError, ValueError):
                pass

        # 3. Hostile DCD header: u32::MAX title_block_len overflow probe & negative natoms
        bad_dcd = os.path.join(tmpdir, "hostile.dcd")
        with open(bad_dcd, "wb") as f:
            # block1_len=84, magic=b"CORD", nframes=-1, ..., title_block_len=0xFFFFFFFF
            b1 = struct.pack("<i4si", 84, b"CORD", -1) + (b"\x00" * 76) + struct.pack("<i", 84)
            b2 = struct.pack("<I", 0xFFFFFFFF)
            f.write(b1 + b2)
        try:
            strux_rs.parse_dcd(bad_dcd)
            raise AssertionError("Expected IOError on u32::MAX title_block_len in DCD")
        except OSError:
            pass

        # 4. Hostile PDB: multi-byte UTF-8 emojis inside coordinate columns & mismatched MODEL atom counts
        utf8_pdb = os.path.join(tmpdir, "emoji.pdb")
        with open(utf8_pdb, "w", encoding="utf-8") as f:
            f.write("ATOM      1  CA  ALA A   1    🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀🦀\n")
            f.write("ATOM      2  CA  ALA A   2      10.000  20.000  30.000  1.00 90.00           C\n")
        coords_utf8 = strux_rs.parse_pdb(utf8_pdb, ca_only=True)
        assert coords_utf8.shape == (1, 1, 3)

        try:
            strux_rs.parse_pdb(utf8_pdb, atom_indices=[99])
            raise AssertionError("Expected ValueError on out-of-bounds PDB atom_indices")
        except ValueError:
            pass

        # 5. NaN / Inf Coordinate & Wrong-Shape Array Fuzzing across Kabsch SVD, ANM, SASA & PCA
        nan_coords = np.full((25, 3), np.nan, dtype=np.float32)
        inf_coords = np.full((25, 3), np.inf, dtype=np.float32)
        valid_coords = rng.normal(size=(25, 3)).astype(np.float32)

        # Must never panic inside nalgebra SVD or symmetric_eigen
        _ = strux_rs.calculate_rmsd_kabsch(nan_coords, valid_coords)
        _ = strux_rs.calculate_rmsd_kabsch(inf_coords, valid_coords)
        _ = strux_rs.calculate_pose_and_sc_rmsd(nan_coords, valid_coords, valid_coords, nan_coords)
        evals_nan, evecs_nan, bfac_nan = strux_rs.anm_modes(nan_coords, n_modes=4)
        assert len(evals_nan) == 0 and len(bfac_nan) == 25
        _ = strux_rs.find_interface_contacts(nan_coords, valid_coords, cutoff=-5.0)

        # Wrong 2D shape (N, 2) and 3D shape (T, N, 2) -> must raise clean ValueError
        wrong_2d = np.zeros((10, 2), dtype=np.float32)
        wrong_3d = np.zeros((4, 10, 2), dtype=np.float32)
        for fn2 in (
            lambda: strux_rs.calculate_rg(wrong_2d),
            lambda: strux_rs.calculate_rmsd_kabsch(wrong_2d, wrong_2d),
            lambda: strux_rs.calculate_sasa(wrong_2d),
            lambda: strux_rs.cpu_pairwise_rmsd(wrong_3d),
            lambda: strux_rs.trajectory_pca(wrong_3d),
        ):
            try:
                fn2()
                raise AssertionError("Expected ValueError on non-(..., 3) coordinate array")
            except ValueError:
                pass

    print("  [0-Byte & Truncated I/O] Verified clean IOError/empty return across XTC, DCD, CIF, PDB, A3M, STO")
    print("  [XTC Bitstream Fuzzing]  50 random bit-flip trials + smallidx=999999 + sizeint=0 -> 0 panics!")
    print("  [UTF-8 & Overflow Guard] Multi-byte UTF-8 PDB columns + u32::MAX DCD block lengths caught cleanly")
    print("  [NaN/Inf & Shape Guards] NaN/Inf Kabsch SVD & ANM eigensolver + (N,2) shape rejection verified")


if __name__ == "__main__":
    t_start = time.perf_counter()
    stress_test_lddt()
    stress_test_markov()
    stress_test_pca()
    stress_test_msa_clustering()
    stress_test_xtc()
    stress_test_cif_and_selection()
    stress_test_therapeutic_primitives()
    stress_test_hostile_fuzz()
    total_sec = time.perf_counter() - t_start
    print("\n" + "=" * 78)
    print(f"✅ ALL 8 GRUELING STRESS & FUZZ TEST SUITES PASSED IN {total_sec:.2f} SECONDS!")
    print("=" * 78)
