// ponytail: standard biophysical calculations.
use crate::spatial::CellList;
use nalgebra::{Matrix3, SVD, Vector3};

pub fn calculate_rg(coords: &[[f32; 3]]) -> f32 {
    if coords.is_empty() {
        return 0.0;
    }
    let mut mean = [0.0, 0.0, 0.0];
    for c in coords {
        mean[0] += c[0];
        mean[1] += c[1];
        mean[2] += c[2];
    }
    let n = coords.len() as f32;
    mean[0] /= n;
    mean[1] /= n;
    mean[2] /= n;

    let mut sum_sq_dist = 0.0;
    for c in coords {
        let dx = c[0] - mean[0];
        let dy = c[1] - mean[1];
        let dz = c[2] - mean[2];
        sum_sq_dist += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq_dist / n).sqrt()
}

pub fn calculate_rmsd_raw(coords1: &[[f32; 3]], coords2: &[[f32; 3]]) -> f32 {
    let n = coords1.len();
    if n == 0 || n != coords2.len() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for i in 0..n {
        let dx = coords1[i][0] - coords2[i][0];
        let dy = coords1[i][1] - coords2[i][1];
        let dz = coords1[i][2] - coords2[i][2];
        sum_sq += dx * dx + dy * dy + dz * dz;
    }
    (sum_sq / n as f32).sqrt()
}

pub fn calculate_rmsd_kabsch(coords1: &[[f32; 3]], coords2: &[[f32; 3]]) -> f32 {
    let n = coords1.len();
    if n == 0 || n != coords2.len() {
        return 0.0;
    }

    // 1. Calculate centroids
    let mut c1 = Vector3::zeros();
    let mut c2 = Vector3::zeros();
    for i in 0..n {
        c1 += Vector3::new(coords1[i][0], coords1[i][1], coords1[i][2]);
        c2 += Vector3::new(coords2[i][0], coords2[i][1], coords2[i][2]);
    }
    let nf = n as f32;
    c1 /= nf;
    c2 /= nf;

    // 2. Center coordinates & Compute covariance matrix H
    let mut h = Matrix3::zeros();
    for i in 0..n {
        let x = Vector3::new(coords1[i][0], coords1[i][1], coords1[i][2]) - c1;
        let y = Vector3::new(coords2[i][0], coords2[i][1], coords2[i][2]) - c2;
        h += x * y.transpose();
    }

    // 3. Compute SVD of H
    let svd = SVD::new(h, true, true);
    let u = svd.u.expect("SVD U failed");
    let v_t = svd.v_t.expect("SVD V_t failed");
    let v = v_t.transpose();

    // 4. Calculate rotation matrix R with reflection correction
    let mut d = (v * u.transpose()).determinant();
    if d < 0.0 {
        d = -1.0;
    } else {
        d = 1.0;
    }
    let mut w = Matrix3::identity();
    w[(2, 2)] = d;
    let r = v * w * u.transpose();

    // 5. Apply rotation and calculate RMSD
    let mut sum_sq = 0.0;
    for i in 0..n {
        let x = Vector3::new(coords1[i][0], coords1[i][1], coords1[i][2]) - c1;
        let y = Vector3::new(coords2[i][0], coords2[i][1], coords2[i][2]) - c2;
        let x_rotated = r * x;
        let diff = x_rotated - y;
        sum_sq += diff.norm_squared();
    }
    (sum_sq / nf).sqrt()
}

pub fn calculate_rmsf(trajectory: &[Vec<[f32; 3] >]) -> Vec<f32> {
    if trajectory.is_empty() {
        return Vec::new();
    }
    let num_frames = trajectory.len();
    let num_atoms = trajectory[0].len();

    let mut means = vec![[0.0f32; 3]; num_atoms];
    for frame in trajectory {
        for i in 0..num_atoms {
            means[i][0] += frame[i][0];
            means[i][1] += frame[i][1];
            means[i][2] += frame[i][2];
        }
    }

    let f = num_frames as f32;
    for i in 0..num_atoms {
        means[i][0] /= f;
        means[i][1] /= f;
        means[i][2] /= f;
    }

    let mut rmsf = vec![0.0f32; num_atoms];
    for frame in trajectory {
        for i in 0..num_atoms {
            let dx = frame[i][0] - means[i][0];
            let dy = frame[i][1] - means[i][1];
            let dz = frame[i][2] - means[i][2];
            rmsf[i] += dx * dx + dy * dy + dz * dz;
        }
    }

    for i in 0..num_atoms {
        rmsf[i] = (rmsf[i] / f).sqrt();
    }

    rmsf
}

pub fn find_interface_contacts(
    target: &[[f32; 3]],
    binder: &[[f32; 3]],
    cutoff: f32,
    box_dims: Option<[f32; 3]>,
) -> Vec<(usize, usize)> {
    let cell_list = CellList::build(target, cutoff, box_dims);
    let mut contacts = Vec::new();

    for (b_idx, &b_pos) in binder.iter().enumerate() {
        let neighbors = cell_list.query_neighbors(b_pos, target);
        for t_idx in neighbors {
            contacts.push((t_idx, b_idx));
        }
    }

    contacts
}
