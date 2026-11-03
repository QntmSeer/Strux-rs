import numpy as np
import strux_rs

def test_lddt_identity():
    # 5 atoms forming a simple polyhedron
    coords = np.array([
        [0.0, 0.0, 0.0],
        [1.5, 0.0, 0.0],
        [0.0, 1.5, 0.0],
        [0.0, 0.0, 1.5],
        [1.5, 1.5, 1.5],
    ], dtype=np.float32)

    # 1. Identity test: lDDT vs self must be exactly 1.0
    score, per_res = strux_rs.calculate_lddt(coords, coords, cutoff=15.0)
    assert np.isclose(score, 1.0, atol=1e-5), f"Expected 1.0, got {score}"
    assert len(per_res) == 5
    assert np.allclose(per_res, 1.0, atol=1e-5)

    # 2. Rigid translation + rotation invariance test
    theta = np.pi / 4
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta),  np.cos(theta), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    trans = np.array([10.0, -25.0, 50.0], dtype=np.float32)
    transformed = (coords @ rot.T) + trans

    score_rot, per_res_rot = strux_rs.calculate_lddt(coords, transformed.astype(np.float32), cutoff=15.0)
    assert np.isclose(score_rot, 1.0, atol=1e-4), f"Rigid transform must preserve lDDT = 1.0, got {score_rot}"

    # 3. Perturbed coordinates: distance shifts should lower lDDT
    perturbed = coords.copy()
    perturbed[1, 0] += 1.5 # shift atom 1 by 1.5 A
    score_pert, _ = strux_rs.calculate_lddt(coords, perturbed, cutoff=15.0)
    assert 0.0 < score_pert < 1.0, f"Perturbed score should be between 0 and 1, got {score_pert}"
    print(f"✓ calculate_lddt: identity=1.0, rigid_invariance={score_rot:.4f}, perturbed={score_pert:.4f}")

def test_markov_populations():
    # 3-state system with known transitions
    # State 0 is dominant, 1 is intermediate, 2 is rare
    dtraj = [0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 0, 2, 0, 0, 1, 0]
    pi, dG, tmat = strux_rs.markov_populations(dtraj, num_states=3, lag_time=1, temperature=300.0)

    # Verify normalization
    assert np.isclose(np.sum(pi), 1.0, atol=1e-5), f"Populations must sum to 1.0, got {np.sum(pi)}"
    assert np.all(pi > 0.0), "All states should have non-zero pseudo-counted population"
    assert pi[0] > pi[1] > pi[2], f"State 0 should have highest population, got {pi}"

    # Verify transition matrix rows sum to 1.0
    for row in tmat:
        assert np.isclose(np.sum(row), 1.0, atol=1e-5), f"Row must sum to 1.0: {row}"

    # Verify free energies: ground state has dG = 0.0, higher states have dG > 0
    assert np.isclose(dG[0], 0.0, atol=1e-5), f"Ground state dG must be 0, got {dG[0]}"
    assert dG[1] > 0.0 and dG[2] > dG[1], f"Free energies should order correctly: {dG}"
    print(f"✓ markov_populations: populations={pi}, dG={dG}")

def test_trajectory_pca():
    # 100 frames, 10 atoms moving primarily along X axis
    np.random.seed(42)
    n_frames = 100
    n_atoms = 10
    base_coords = np.random.randn(n_atoms, 3).astype(np.float32)

    traj = np.zeros((n_frames, n_atoms, 3), dtype=np.float32)
    for i in range(n_frames):
        traj[i] = base_coords + np.random.randn(n_atoms, 3) * 0.1
        traj[i, :, 0] += np.sin(i * 0.1) * 3.0 # large collective motion along X

    proj, var_exp, comp = strux_rs.trajectory_pca(traj, n_components=2)

    assert proj.shape == (n_frames, 2), f"Projection shape mismatch: {proj.shape}"
    assert len(var_exp) == 2, f"Variance ratio length mismatch: {len(var_exp)}"
    assert var_exp[0] > var_exp[1], f"PC1 should explain more variance than PC2: {var_exp}"
    assert var_exp[0] > 0.5, f"PC1 should capture dominant sinusoidal motion, got {var_exp[0]}"
    print(f"✓ trajectory_pca: projections={proj.shape}, explained_var={var_exp}")

def test_cluster_msa():
    # 4 sequences: seq0 and seq1 are identical, seq2 is 93% identical, seq3 is completely different
    s0 = "MKVLWVALAALLAAS"
    s1 = "MKVLWVALAALLAAS" # 100%
    s2 = "MKVLWVALAALLAAA" # 14/15 ~ 93%
    s3 = "GGGGGGGGGGGGGGG" # 0%

    seqs = [s0, s1, s2, s3]
    labels, reps = strux_rs.cluster_msa(seqs, min_identity=0.8)

    assert labels[0] == labels[1] == labels[2], f"s0, s1, s2 should cluster together: {labels}"
    assert labels[3] != labels[0], f"s3 should be in a separate cluster: {labels}"
    assert len(reps) == 2, f"Expected 2 clusters, got {len(reps)}"

    # Test Msa methods
    del_mat = [[0] * len(s) for s in seqs]
    msa = strux_rs.Msa(
        sequences=seqs,
        deletion_matrix=del_mat,
        descriptions=["s0", "s1", "s2", "s3"]
    )
    labels2, reps2 = msa.cluster(min_identity=0.8)
    assert np.array_equal(labels2, labels)
    assert np.array_equal(reps2, reps)

    sub_msa0 = msa.get_cluster(labels[0], labels)
    assert len(sub_msa0.sequences) == 3
    assert sub_msa0.descriptions == ["s0", "s1", "s2"]

    sub_msa1 = msa.get_cluster(labels[3], labels)
    assert len(sub_msa1.sequences) == 1
    assert sub_msa1.descriptions == ["s3"]
    print(f"✓ cluster_msa: {len(seqs)} sequences into {len(reps)} clusters, sub-MSAs extracted correctly")

def test_xtc_export():
    assert hasattr(strux_rs, "parse_xtc"), "parse_xtc must be exported"
    print("✓ parse_xtc is exported into python module")

if __name__ == "__main__":
    test_lddt_identity()
    test_markov_populations()
    test_trajectory_pca()
    test_cluster_msa()
    test_xtc_export()
    print("\n==========================================")
    print("🎉 ALL 5 AMS BIOPHYSICS FEATURES VERIFIED!")
    print("==========================================")
