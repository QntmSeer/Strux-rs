// ponytail: grueling integration tests testing edge cases, degeneracies, and PBC boundary conditions.
use strux_rs::analysis::{calculate_rmsd_kabsch, calculate_rmsd_raw, find_interface_contacts};

fn pseudo_random_coords(size: usize, seed: u32) -> Vec<[f32; 3]> {
    let mut coords = Vec::with_capacity(size);
    let mut state = seed;
    for _ in 0..size {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let x = (state as f32) / (u32::MAX as f32) * 50.0;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let y = (state as f32) / (u32::MAX as f32) * 50.0;
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let z = (state as f32) / (u32::MAX as f32) * 50.0;
        coords.push([x, y, z]);
    }
    coords
}

#[test]
fn test_kabsch_identical() {
    let coords = pseudo_random_coords(100, 42);
    let rmsd = calculate_rmsd_kabsch(&coords, &coords);
    assert!(rmsd < 1e-5, "RMSD of identical coordinates should be 0, got {}", rmsd);
}

#[test]
fn test_kabsch_pure_translation_rotation() {
    let coords1 = pseudo_random_coords(100, 42);
    
    // Apply a translation [10.0, -5.0, 3.0] and a 90-degree rotation around Z-axis
    let mut coords2 = Vec::new();
    for p in &coords1 {
        let x = -p[1] + 10.0;
        let y = p[0] - 5.0;
        let z = p[2] + 3.0;
        coords2.push([x, y, z]);
    }

    let rmsd_raw = calculate_rmsd_raw(&coords1, &coords2);
    assert!(rmsd_raw > 1.0, "Raw RMSD should be large before superposition");

    let rmsd_kabsch = calculate_rmsd_kabsch(&coords1, &coords2);
    assert!(rmsd_kabsch < 1e-4, "Kabsch RMSD should be 0 after alignment, got {}", rmsd_kabsch);
}

#[test]
fn test_kabsch_reflection() {
    let coords1 = pseudo_random_coords(10, 101);
    
    // Create coords2 as a reflection of coords1 along X-axis
    // (a pure reflection cannot be achieved via rotation)
    let mut coords2 = Vec::new();
    for p in &coords1 {
        coords2.push([-p[0], p[1], p[2]]);
    }

    let rmsd = calculate_rmsd_kabsch(&coords1, &coords2);
    // Since Kabsch should prevent reflection (det(R) must be +1), the RMSD should remain
    // positive and reflect the best rotation (not the reflection itself).
    assert!(rmsd > 0.1, "Kabsch must not allow reflections, RMSD should be positive, got {}", rmsd);
}

#[test]
fn test_kabsch_degeneracy() {
    // All points collinear along the X-axis
    let mut coords1 = Vec::new();
    let mut coords2 = Vec::new();
    for i in 0..10 {
        let val = i as f32;
        coords1.push([val, 0.0, 0.0]);
        // coords2 is rotated 90 degrees around Z
        coords2.push([0.0, val, 0.0]);
    }

    let rmsd = calculate_rmsd_kabsch(&coords1, &coords2);
    assert!(rmsd < 1e-4, "Kabsch SVD should converge on degenerate collinear inputs, got {}", rmsd);
}

#[test]
fn test_spatial_hashing_correctness_under_pbc() {
    let target = pseudo_random_coords(500, 123);
    let binder = pseudo_random_coords(100, 456);
    let cutoff = 6.0;
    let box_dims = [25.0, 25.0, 25.0];

    // Naive pairwise calculation with PBC
    let mut expected_contacts = Vec::new();
    for (b_idx, &b) in binder.iter().enumerate() {
        for (t_idx, &t) in target.iter().enumerate() {
            let mut dist_sq = 0.0;
            for i in 0..3 {
                let mut diff = b[i] - t[i];
                let l = box_dims[i];
                diff -= l * (diff / l).round();
                dist_sq += diff * diff;
            }
            if dist_sq < cutoff * cutoff {
                expected_contacts.push((t_idx, b_idx));
            }
        }
    }
    expected_contacts.sort_unstable();

    // Custom cell list interface contacts calculation
    let mut actual_contacts = find_interface_contacts(&target, &binder, cutoff, Some(box_dims));
    actual_contacts.sort_unstable();

    assert_eq!(expected_contacts.len(), actual_contacts.len(), "Contact counts mismatch!");
    assert_eq!(expected_contacts, actual_contacts, "Contact indices mismatch!");
}

#[test]
fn test_spatial_hashing_exact_boundary() {
    // Coordinate exactly on the box boundary
    let target = vec![[0.0, 0.0, 0.0]];
    let binder = vec![[20.0, 0.0, 0.0]]; // 20.0 under PBC box 20.0 is equivalent to 0.0
    let box_dims = [20.0, 20.0, 20.0];
    let cutoff = 1.0;

    let contacts = find_interface_contacts(&target, &binder, cutoff, Some(box_dims));
    assert_eq!(contacts.len(), 1, "Should detect contact at wrapped boundary");
    assert_eq!(contacts[0], (0, 0));
}

#[test]
fn test_spatial_hashing_negative_coords() {
    // Negative coordinate wrapping
    let target = vec![[-1.0, -1.0, -1.0]]; // wraps to [19.0, 19.0, 19.0]
    let binder = vec![[19.0, 19.0, 19.0]];
    let box_dims = [20.0, 20.0, 20.0];
    let cutoff = 1.0;

    let contacts = find_interface_contacts(&target, &binder, cutoff, Some(box_dims));
    assert_eq!(contacts.len(), 1, "Should detect contact for negative coordinates");
    assert_eq!(contacts[0], (0, 0));
}

#[test]
fn test_spatial_hashing_small_box() {
    // Box dimension is smaller than the cutoff (e.g. box 4.0, cutoff 5.0)
    let target = vec![[1.0, 1.0, 1.0]];
    let binder = vec![[2.0, 2.0, 2.0]];
    let box_dims = [4.0, 4.0, 4.0];
    let cutoff = 5.0;

    let contacts = find_interface_contacts(&target, &binder, cutoff, Some(box_dims));
    assert_eq!(contacts.len(), 1, "Should execute safely without duplicates on small boxes");
    assert_eq!(contacts[0], (0, 0));
}

#[test]
fn test_unwrap_pbc_broken_chain() {
    use strux_rs::biophysics::unwrap_pbc;
    // A linear polymer chain with bonded distance 1.5 A along X-axis
    // Atom 2 crosses the periodic boundary of a 30 A box
    let box_dims = [30.0, 30.0, 30.0];
    let original = vec![
        [28.0, 5.0, 5.0],
        [29.5, 5.0, 5.0],
        [1.0, 5.0, 5.0],  // wrapped: should unwrap to 31.0
        [2.5, 5.0, 5.0],  // wrapped: should unwrap to 32.5
    ];

    let unwrapped = unwrap_pbc(&original, box_dims);
    assert_eq!(unwrapped.len(), 4);
    assert!((unwrapped[0][0] - 28.0).abs() < 1e-4);
    assert!((unwrapped[1][0] - 29.5).abs() < 1e-4);
    assert!((unwrapped[2][0] - 31.0).abs() < 1e-4, "Atom 2 should unwrap to 31.0, got {}", unwrapped[2][0]);
    assert!((unwrapped[3][0] - 32.5).abs() < 1e-4, "Atom 3 should unwrap to 32.5, got {}", unwrapped[3][0]);

    // Check consecutive bond distances are ~1.5 A
    for i in 1..unwrapped.len() {
        let dx = unwrapped[i][0] - unwrapped[i - 1][0];
        assert!((dx - 1.5).abs() < 1e-4, "Bond distance should be 1.5 A");
    }
}

#[test]
fn test_unwrap_trajectory_nojump() {
    use strux_rs::biophysics::unwrap_trajectory_nojump;
    let box_dims = [50.0, 50.0, 50.0];
    // 3 frames, 2 atoms each
    // Frame 0: COM at [5.0, 0.0, 0.0]
    // Frame 1: Molecule jumped across boundary to [55.0, 0.0, 0.0] (wrapped to 5.0 + 50.0)
    // Frame 2: Molecule jumped again by 50.0 to [105.0, 0.0, 0.0]
    let frames = vec![
        // Frame 0
        4.5, 0.0, 0.0,
        5.5, 0.0, 0.0,
        // Frame 1 (jumped by +50)
        54.5, 0.0, 0.0,
        55.5, 0.0, 0.0,
        // Frame 2 (jumped by +100 total)
        104.5, 0.0, 0.0,
        105.5, 0.0, 0.0,
    ];

    let unwrapped = unwrap_trajectory_nojump(&frames, 3, 2, box_dims);
    // After nojump unwrapping, COM should stay near 5.0 for all frames!
    for f in 0..3 {
        let base = f * 6;
        let com_x = (unwrapped[base] + unwrapped[base + 3]) * 0.5;
        assert!((com_x - 5.0).abs() < 1e-4, "Frame {} COM should be continuous at 5.0, got {}", f, com_x);
    }
}

#[test]
fn test_pdb_trajectory_writer_roundtrip() {
    use strux_rs::pdb::{parse_pdb_trajectory, write_pdb_trajectory};
    let tmp_path = std::env::temp_dir().join("strux_test_traj.pdb");

    // 2 frames, 3 atoms each
    let coords = vec![
        // Frame 0
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        // Frame 1
        10.0, 11.0, 12.0,
        13.0, 14.0, 15.0,
        16.0, 17.0, 18.0,
    ];

    write_pdb_trajectory(&tmp_path, &coords, 2, 3).expect("Failed to write PDB trajectory");
    let parsed = parse_pdb_trajectory(&tmp_path).expect("Failed to parse back PDB trajectory");
    let _ = std::fs::remove_file(&tmp_path);

    assert_eq!(parsed.len(), 2, "Must have 2 frames");
    assert_eq!(parsed[0].len(), 3, "Must have 3 atoms per frame");
    assert_eq!(parsed[0][0], [1.0, 2.0, 3.0]);
    assert_eq!(parsed[0][1], [4.0, 5.0, 6.0]);
    assert_eq!(parsed[0][2], [7.0, 8.0, 9.0]);
    assert_eq!(parsed[1][0], [10.0, 11.0, 12.0]);
    assert_eq!(parsed[1][1], [13.0, 14.0, 15.0]);
    assert_eq!(parsed[1][2], [16.0, 17.0, 18.0]);
}

#[test]
fn test_gro_parser() {
    use std::io::Write;
    use strux_rs::pdb::parse_gro_trajectory;
    let tmp_path = std::env::temp_dir().join("strux_test.gro");

    // Sample GRO with 2 atoms in nm (1.0 nm = 10.0 A) formatted to fixed columns
    let gro_content = "\
Protein test
    2
    1ALA     CA    1   1.234   2.345   3.456
    1ALA      C    2   1.350   2.450   3.550
   5.00000   5.00000   5.00000
";
    {
        let mut f = std::fs::File::create(&tmp_path).expect("Failed to create test gro");
        f.write_all(gro_content.as_bytes()).expect("Write error");
    }

    let parsed = parse_gro_trajectory(&tmp_path).expect("Failed to parse gro");
    let _ = std::fs::remove_file(&tmp_path);

    assert_eq!(parsed.len(), 1, "Must have 1 frame");
    assert_eq!(parsed[0].len(), 2, "Must have 2 atoms");
    // Check nm -> Angstroms conversion (10x)
    assert!((parsed[0][0][0] - 12.34).abs() < 1e-3);
    assert!((parsed[0][0][1] - 23.45).abs() < 1e-3);
    assert!((parsed[0][0][2] - 34.56).abs() < 1e-3);
    assert!((parsed[0][1][0] - 13.50).abs() < 1e-3);
    assert!((parsed[0][1][1] - 24.50).abs() < 1e-3);
    assert!((parsed[0][1][2] - 35.50).abs() < 1e-3);
}

#[test]
fn test_pdb_writer_atom_limit() {
    use strux_rs::pdb::write_pdb_trajectory;
    let tmp_path = std::env::temp_dir().join("strux_test_overflow.pdb");
    let dummy_coords = vec![0.0f32; 100_000 * 3];
    let result = write_pdb_trajectory(&tmp_path, &dummy_coords, 1, 100_000);
    assert!(result.is_err(), "Must reject atom counts > 99,999 to comply with PDB v3.3 specification");
}

fn write_test_dcd(
    path: &std::path::Path,
    frames: &[Vec<[f32; 3]>],
    unitcells: Option<&[[f64; 6]]>,
    charmm_version: i32,
    truncate_bytes: usize,
) {
    use std::io::Write;
    let mut buf = Vec::new();
    let nframes = frames.len() as i32;
    let natoms = frames[0].len() as i32;

    // Header block 1: 4 + 84 + 4 = 92 bytes
    buf.extend_from_slice(&84u32.to_le_bytes());
    buf.extend_from_slice(b"CORD");
    let mut header_ints = [0i32; 20];
    header_ints[0] = nframes;
    header_ints[19] = charmm_version;
    if unitcells.is_some() {
        header_ints[10] = 1; // QCRYS at file offset 48..52
    } else if charmm_version == 0 {
        // X-PLOR DELTA test: bits equal 1 at offsets 44..52 must NOT trigger has_unitcell
        header_ints[9] = 1;
        header_ints[10] = 1;
    }
    for v in &header_ints {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(&84u32.to_le_bytes());

    // Block 2: Title (1 line of 80 bytes)
    let title_len = 4u32 + 80u32;
    buf.extend_from_slice(&title_len.to_le_bytes());
    buf.extend_from_slice(&1i32.to_le_bytes());
    buf.extend_from_slice(&[b' '; 80]);
    buf.extend_from_slice(&title_len.to_le_bytes());

    // Block 3: natoms
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&natoms.to_le_bytes());
    buf.extend_from_slice(&4u32.to_le_bytes());

    for (fi, frame) in frames.iter().enumerate() {
        if let Some(ucs) = unitcells {
            buf.extend_from_slice(&48u32.to_le_bytes());
            for &val in &ucs[fi] {
                buf.extend_from_slice(&val.to_le_bytes());
            }
            buf.extend_from_slice(&48u32.to_le_bytes());
        }
        let coord_bytes = (frame.len() * 4) as u32;
        for d in 0..3 {
            buf.extend_from_slice(&coord_bytes.to_le_bytes());
            for atom in frame {
                buf.extend_from_slice(&atom[d].to_le_bytes());
            }
            buf.extend_from_slice(&coord_bytes.to_le_bytes());
        }
    }

    if truncate_bytes > 0 && truncate_bytes < buf.len() {
        let new_len = buf.len() - truncate_bytes;
        buf.truncate(new_len);
    }
    let mut f = std::fs::File::create(path).expect("create dcd fixture");
    f.write_all(&buf).expect("write dcd fixture");
}

fn write_test_xtc_uncompressed(
    path: &std::path::Path,
    magic: i32,
    frames: &[Vec<[f32; 3]>], // in nm
    boxes_nm: &[[f32; 9]],
    truncate_bytes: usize,
) {
    use std::io::Write;
    let mut buf = Vec::new();
    for (fi, frame) in frames.iter().enumerate() {
        let natoms = frame.len() as i32;
        assert!(natoms <= 9, "uncompressed XTC requires natoms <= 9");
        buf.extend_from_slice(&magic.to_be_bytes());
        buf.extend_from_slice(&natoms.to_be_bytes());
        buf.extend_from_slice(&(fi as i32).to_be_bytes());
        buf.extend_from_slice(&(fi as f32 * 10.0).to_be_bytes());
        for &b in &boxes_nm[fi] {
            buf.extend_from_slice(&b.to_be_bytes());
        }
        buf.extend_from_slice(&natoms.to_be_bytes()); // lsize
        for atom in frame {
            for d in 0..3 {
                buf.extend_from_slice(&atom[d].to_be_bytes());
            }
        }
    }
    if truncate_bytes > 0 && truncate_bytes < buf.len() {
        let new_len = buf.len() - truncate_bytes;
        buf.truncate(new_len);
    }
    let mut f = std::fs::File::create(path).expect("create xtc fixture");
    f.write_all(&buf).expect("write xtc fixture");
}

#[test]
fn test_dcd_variants_and_box_vectors() {
    use strux_rs::dcd::{parse_dcd_trajectory, parse_dcd_trajectory_selected};
    let frames = vec![
        vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        vec![[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]],
        vec![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
    ];

    // 1. OpenMM style: flag at offset 48, angles in degrees, triclinic NPT
    // [A, gamma, B, beta, alpha, C]
    let ucs_deg = [
        [40.0, 90.0, 40.0, 60.0, 60.0, 40.0],
        [41.0, 90.0, 41.0, 60.0, 60.0, 41.0],
        [42.0, 90.0, 42.0, 60.0, 60.0, 42.0],
    ];
    let p_openmm = std::env::temp_dir().join("strux_test_openmm.dcd");
    write_test_dcd(&p_openmm, &frames, Some(&ucs_deg), 24, 0);
    let traj = parse_dcd_trajectory(&p_openmm).expect("parse openmm dcd");
    let _ = std::fs::remove_file(&p_openmm);
    assert_eq!(traj.num_frames, 3);
    assert_eq!(traj.num_atoms, 3);
    let bv = traj.box_vectors.expect("must have box vectors");
    assert_eq!(bv.len(), 27);
    // Frame 0: a=(40,0,0), b=(0,40,0), c=(20, 20, 40/sqrt(2)=28.28427)
    assert!((bv[0] - 40.0).abs() < 1e-4);
    assert!((bv[4] - 40.0).abs() < 1e-4);
    assert!((bv[6] - 20.0).abs() < 1e-4);
    assert!((bv[7] - 20.0).abs() < 1e-4);
    assert!((bv[8] - (40.0 / 2.0f32.sqrt())).abs() < 1e-3);

    // 2. CHARMM/OpenMM cosine style: icntrl[19]=24, icntrl[10]=1, angles as cos(theta) in [-1, 1]
    let ucs_cos = [
        [40.0, 0.0, 40.0, 0.5, 0.5, 40.0],
        [41.0, 0.0, 41.0, 0.5, 0.5, 41.0],
        [42.0, 0.0, 42.0, 0.5, 0.5, 42.0],
    ];
    let p_charmm = std::env::temp_dir().join("strux_test_charmm.dcd");
    write_test_dcd(&p_charmm, &frames, Some(&ucs_cos), 24, 0);
    let traj_c = parse_dcd_trajectory(&p_charmm).expect("parse charmm dcd");
    let _ = std::fs::remove_file(&p_charmm);
    let bvc = traj_c.box_vectors.expect("charmm box vectors");
    for i in 0..27 {
        assert!((bv[i] - bvc[i]).abs() < 1e-4, "CHARMM cos vs NAMD deg mismatch at {}", i);
    }

    // 3. X-PLOR no unit cell (icntrl[19]=0, even with bits=1 at offsets 44..52) + OOB + truncated
    let p_nouc = std::env::temp_dir().join("strux_test_nouc.dcd");
    write_test_dcd(&p_nouc, &frames, None, 0, 0);
    let traj_no = parse_dcd_trajectory(&p_nouc).expect("parse xplor no-uc dcd");
    assert!(traj_no.box_vectors.is_none());
    assert!(parse_dcd_trajectory_selected(&p_nouc, Some(&[0, 99])).is_err());
    let _ = std::fs::remove_file(&p_nouc);

    let p_trunc = std::env::temp_dir().join("strux_test_trunc.dcd");
    write_test_dcd(&p_trunc, &frames, Some(&ucs_deg), 24, 12);
    let traj_t = parse_dcd_trajectory(&p_trunc).expect("parse truncated dcd");
    let _ = std::fs::remove_file(&p_trunc);
    assert_eq!(traj_t.num_frames, 2);
    assert_eq!(traj_t.box_vectors.unwrap().len(), 18);
}

#[test]
fn test_xtc_box_vectors_and_magic_variants() {
    use strux_rs::xtc::{parse_xtc_trajectory, parse_xtc_trajectory_selected};
    let frames_nm = vec![
        vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        vec![[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
        vec![[0.3, 0.4, 0.5], [0.6, 0.7, 0.8]],
    ];
    let boxes_nm = [
        [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 1.5, 1.5, 2.1213203],
        [3.1, 0.0, 0.0, 0.0, 3.1, 0.0, 1.55, 1.55, 2.192031],
        [3.2, 0.0, 0.0, 0.0, 3.2, 0.0, 1.6, 1.6, 2.2627418],
    ];

    for &magic in &[1995i32, 2023i32] {
        let p = std::env::temp_dir().join(format!("strux_test_xtc_{}.xtc", magic));
        write_test_xtc_uncompressed(&p, magic, &frames_nm, &boxes_nm, 0);
        let traj = parse_xtc_trajectory(&p).expect("parse xtc");
        assert_eq!(traj.num_frames, 3);
        assert_eq!(traj.num_atoms, 2);
        assert_eq!(traj.box_vectors.len(), 27);
        for f in 0..3 {
            for k in 0..9 {
                assert!(
                    (traj.box_vectors[f * 9 + k] - boxes_nm[f][k] * 10.0).abs() < 1e-4,
                    "XTC box mismatch at frame {} idx {}", f, k
                );
            }
        }
        assert!((traj.coords[0] - 1.0).abs() < 1e-4);
        assert!(parse_xtc_trajectory_selected(&p, Some(&[5])).is_err());
        let _ = std::fs::remove_file(&p);
    }

    // Truncated final frame
    let p_trunc = std::env::temp_dir().join("strux_test_xtc_trunc.xtc");
    write_test_xtc_uncompressed(&p_trunc, 1995, &frames_nm, &boxes_nm, 8);
    let traj_t = parse_xtc_trajectory(&p_trunc).expect("parse truncated xtc");
    let _ = std::fs::remove_file(&p_trunc);
    assert_eq!(traj_t.num_frames, 2);
    assert_eq!(traj_t.box_vectors.len(), 18);
}

#[test]
fn test_image_molecules_triclinic_and_npt() {
    use strux_rs::biophysics::{image_molecules, unwrap_trajectory_nojump_box};

    // Rhombic dodecahedron box in Angstroms: a=(40,0,0), b=(0,40,0), c=(20,20,28.284271)
    let a = [40.0f32, 0.0, 0.0];
    let b = [0.0f32, 40.0, 0.0];
    let c = [20.0f32, 20.0, 40.0 / 2.0f32.sqrt()];
    let box_f0 = [a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2]];
    let box_f1 = [42.0, 0.0, 0.0, 0.0, 42.0, 0.0, 21.0, 21.0, 42.0 / 2.0f32.sqrt()];

    // True intact complex: Mol 0 (atoms 0..3), Mol 1 (atoms 3..6), 4.0 A interface separation
    let true_f0 = [
        [10.0, 10.0, 10.0],
        [11.5, 10.0, 10.0],
        [13.0, 10.0, 10.0],
        [13.0, 14.0, 10.0],
        [14.5, 14.0, 10.0],
        [16.0, 14.0, 10.0],
    ];

    // Wrap atom 2 by +c, and shift entire Mol 1 by +a - c (plus wrap atom 5 by +b)
    let mut wrapped_f0 = true_f0;
    for d in 0..3 { wrapped_f0[2][d] += c[d]; }
    for ai in 3..6 {
        for d in 0..3 { wrapped_f0[ai][d] += a[d] - c[d]; }
    }
    for d in 0..3 { wrapped_f0[5][d] += b[d]; }

    // Frame 1: entire complex shifted by +c_f1 (for nojump test too)
    let mut wrapped_f1 = true_f0;
    for ai in 0..6 {
        for d in 0..3 { wrapped_f1[ai][d] += box_f1[6 + d]; }
    }
    // Split Mol 1 across -a_f1 in frame 1
    for ai in 3..6 { wrapped_f1[ai][0] -= box_f1[0]; }

    let mut coords = Vec::new();
    for p in wrapped_f0.iter().chain(wrapped_f1.iter()) {
        coords.extend_from_slice(p);
    }
    let mut boxes = Vec::new();
    boxes.extend_from_slice(&box_f0);
    boxes.extend_from_slice(&box_f1);

    let imaged = image_molecules(&coords, 2, 6, &boxes, &[(0, 3), (3, 6)], 0);

    // Check exact bond lengths (1.5 A) and exact interface distance (4.0 A between atom 2 and atom 3)
    for f in 0..2 {
        let base = f * 18;
        let get = |i: usize| [imaged[base + i * 3], imaged[base + i * 3 + 1], imaged[base + i * 3 + 2]];
        let dist = |p: [f32; 3], q: [f32; 3]| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        };
        assert!((dist(get(0), get(1)) - 1.5).abs() < 1e-4);
        assert!((dist(get(1), get(2)) - 1.5).abs() < 1e-4);
        assert!((dist(get(3), get(4)) - 1.5).abs() < 1e-4);
        assert!((dist(get(4), get(5)) - 1.5).abs() < 1e-4);
        assert!((dist(get(2), get(3)) - 4.0).abs() < 1e-4);
    }

    // Verify unwrap_trajectory_nojump_box removes the +c_f1 jump between Frame 0 and Frame 1
    let nojump = unwrap_trajectory_nojump_box(&imaged, 2, 6, &boxes);
    let com0_z: f32 = (0..6).map(|i| nojump[i * 3 + 2]).sum::<f32>() / 6.0;
    let com1_z: f32 = (0..6).map(|i| nojump[18 + i * 3 + 2]).sum::<f32>() / 6.0;
    assert!((com1_z - com0_z).abs() < 1e-4, "Triclinic nojump COM z mismatch: {} vs {}", com0_z, com1_z);

    // Explicit rhombic dodecahedron Voronoi-corner counterexamples (d = 100 A):
    // 1) Bound complex at dr = (0, 0, 36.0) must NOT be torn apart to 78.8 A
    // 2) Wrapped binder at s = (0.45, 0.45, 0.45) -> (67.5, 67.5, 31.8198) (100.62 A) must snap to 46.10 A
    let d100 = [100.0f32, 0.0, 0.0, 0.0, 100.0, 0.0, 50.0, 50.0, 100.0 / 2.0f32.sqrt()];
    let c_bound = vec![0.0, 0.0, 0.0, 0.0, 0.0, 36.0];
    let img_bound = image_molecules(&c_bound, 1, 2, &d100, &[(0, 1), (1, 2)], 0);
    let dist_bound = ((img_bound[3] - img_bound[0]).powi(2)
        + (img_bound[4] - img_bound[1]).powi(2)
        + (img_bound[5] - img_bound[2]).powi(2))
        .sqrt();
    assert!((dist_bound - 36.0).abs() < 1e-4, "Bound complex torn apart: got {}", dist_bound);

    let c_corner = vec![0.0, 0.0, 0.0, 67.5, 67.5, 45.0 / 2.0f32.sqrt()];
    let img_corner = image_molecules(&c_corner, 1, 2, &d100, &[(0, 1), (1, 2)], 0);
    let dist_corner = ((img_corner[3] - img_corner[0]).powi(2)
        + (img_corner[4] - img_corner[1]).powi(2)
        + (img_corner[5] - img_corner[2]).powi(2))
        .sqrt();
    assert!((dist_corner - 46.1008).abs() < 1e-2, "Voronoi corner failed to snap: got {}", dist_corner);
}
