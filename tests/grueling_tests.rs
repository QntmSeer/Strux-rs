// ponytail: grueling integration tests testing edge cases, degeneracies, and PBC boundary conditions.
use strux_rs::analysis::{calculate_rg, calculate_rmsd_kabsch, calculate_rmsd_raw, find_interface_contacts};

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
