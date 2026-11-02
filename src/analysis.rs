// ponytail: standard biophysical calculations.
use crate::spatial::CellList;
use nalgebra::{Matrix3, SVD, Vector3};
use rayon::prelude::*;

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

/// Computes optimal Kabsch rotation matrix R and centroids (c_ref, c_mob)
/// such that x_aligned = R * (x_mob - c_mob) + c_ref minimizes RMSD(coords_ref, coords_mob).
/// Gracefully falls back to identity rotation if coordinates contain NaN/Inf.
#[inline]
pub fn kabsch_transform(
    coords_ref: &[[f32; 3]],
    coords_mob: &[[f32; 3]],
) -> (Matrix3<f32>, Vector3<f32>, Vector3<f32>) {
    let n = coords_ref.len();
    if n == 0 || n != coords_mob.len() {
        return (Matrix3::identity(), Vector3::zeros(), Vector3::zeros());
    }

    let mut c_ref = Vector3::<f32>::zeros();
    let mut c_mob = Vector3::<f32>::zeros();
    for i in 0..n {
        c_ref += Vector3::from(coords_ref[i]);
        c_mob += Vector3::from(coords_mob[i]);
    }
    let inv_n = 1.0 / (n as f32);
    c_ref *= inv_n;
    c_mob *= inv_n;

    let mut cov = Matrix3::<f32>::zeros();
    for i in 0..n {
        let p = Vector3::from(coords_mob[i]) - c_mob;
        let q = Vector3::from(coords_ref[i]) - c_ref;
        cov += p * q.transpose();
    }

    if !cov.iter().all(|x| x.is_finite()) {
        return (Matrix3::identity(), c_ref, c_mob);
    }

    let svd = SVD::new(cov, true, true);
    let (u, v_t) = match (svd.u, svd.v_t) {
        (Some(u), Some(v_t)) => (u, v_t),
        _ => return (Matrix3::identity(), c_ref, c_mob),
    };

    let d = (v_t.transpose() * u.transpose()).determinant();
    let mut corr = Matrix3::<f32>::identity();
    if d < 0.0 {
        corr[(2, 2)] = -1.0;
    }

    let r = v_t.transpose() * corr * u.transpose();
    (r, c_ref, c_mob)
}

pub fn calculate_rmsd_kabsch(coords1: &[[f32; 3]], coords2: &[[f32; 3]]) -> f32 {
    let n = coords1.len();
    if n == 0 || n != coords2.len() {
        return 0.0;
    }

    let (r, c2, c1) = kabsch_transform(coords2, coords1);
    let mut sum_sq = 0.0f32;
    for i in 0..n {
        let x = Vector3::from(coords1[i]) - c1;
        let y = Vector3::from(coords2[i]) - c2;
        let diff = (r * x) - y;
        sum_sq += diff.norm_squared();
    }
    (sum_sq / (n as f32)).sqrt()
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

/// AlphaFold Local Distance Difference Test (lDDT).
/// Alignment-free metric comparing all pairs with reference distance < cutoff (default 15.0 Å).
/// Returns (global_lddt [0.0..1.0], per_residue_lddt).
pub fn calculate_lddt(
    coords_ref: &[[f32; 3]],
    coords_target: &[[f32; 3]],
    cutoff: f32,
) -> (f32, Vec<f32>) {
    let n = coords_ref.len();
    if n == 0 || n != coords_target.len() {
        return (0.0, Vec::new());
    }

    let cutoff_sq = cutoff * cutoff;
    let thresholds = [0.5f32, 1.0, 2.0, 4.0];

    let results: Vec<(f32, bool)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut preserved_count = 0usize;
            let mut total_pairs = 0usize;
            let ref_i = coords_ref[i];
            let tgt_i = coords_target[i];

            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx_ref = ref_i[0] - coords_ref[j][0];
                let dy_ref = ref_i[1] - coords_ref[j][1];
                let dz_ref = ref_i[2] - coords_ref[j][2];
                let dist_ref_sq = dx_ref * dx_ref + dy_ref * dy_ref + dz_ref * dz_ref;

                if dist_ref_sq < cutoff_sq {
                    let dist_ref = dist_ref_sq.sqrt();
                    let dx_tgt = tgt_i[0] - coords_target[j][0];
                    let dy_tgt = tgt_i[1] - coords_target[j][1];
                    let dz_tgt = tgt_i[2] - coords_target[j][2];
                    let dist_tgt = (dx_tgt * dx_tgt + dy_tgt * dy_tgt + dz_tgt * dz_tgt).sqrt();
                    let delta = (dist_tgt - dist_ref).abs();

                    total_pairs += 4;
                    for &t in &thresholds {
                        if delta < t {
                            preserved_count += 1;
                        }
                    }
                }
            }

            if total_pairs > 0 {
                (preserved_count as f32 / total_pairs as f32, true)
            } else {
                (1.0, false)
            }
        })
        .collect();

    let mut per_residue = Vec::with_capacity(n);
    let mut total_score = 0.0f64;
    let mut evaluated_residues = 0usize;

    for (score, evaluated) in results {
        per_residue.push(score);
        if evaluated {
            total_score += score as f64;
            evaluated_residues += 1;
        }
    }

    let global_lddt = if evaluated_residues > 0 {
        (total_score / evaluated_residues as f64) as f32
    } else {
        1.0
    };

    (global_lddt, per_residue)
}

pub struct MarkovPopulationsResult {
    pub populations: Vec<f32>,
    pub delta_g: Vec<f32>,
    pub transition_matrix: Vec<f32>,
    pub num_states: usize,
}

/// Computes transition probability matrix, stationary populations, and relative free energies (kcal/mol)
/// across a trajectory of discrete conformational cluster state labels.
pub fn calculate_markov_populations(
    state_sequence: &[usize],
    num_states: usize,
    lag_time: usize,
    temperature_k: f32,
) -> MarkovPopulationsResult {
    if state_sequence.is_empty() || num_states == 0 {
        return MarkovPopulationsResult {
            populations: Vec::new(),
            delta_g: Vec::new(),
            transition_matrix: Vec::new(),
            num_states: 0,
        };
    }

    // 1. Build transition count matrix C_ij with lag time
    let mut counts = vec![0.0f64; num_states * num_states];
    let lag = lag_time.max(1);

    if state_sequence.len() > lag {
        for t in 0..(state_sequence.len() - lag) {
            let from_state = state_sequence[t];
            let to_state = state_sequence[t + lag];
            if from_state < num_states && to_state < num_states {
                counts[from_state * num_states + to_state] += 1.0;
            }
        }
    }

    // 2. Row-normalize to build transition probability matrix T_ij
    let mut transition_matrix = vec![0.0f32; num_states * num_states];
    for i in 0..num_states {
        let mut row_sum = 0.0f64;
        for j in 0..num_states {
            row_sum += counts[i * num_states + j];
        }
        if row_sum > 0.0 {
            for j in 0..num_states {
                transition_matrix[i * num_states + j] = (counts[i * num_states + j] / row_sum) as f32;
            }
        } else {
            transition_matrix[i * num_states + i] = 1.0;
        }
    }

    // 3. Reversible count symmetrization for robust stationary populations
    let mut state_counts = vec![0.0f64; num_states];
    let mut total_symmetrized = 0.0f64;
    for i in 0..num_states {
        for j in 0..num_states {
            let sym = counts[i * num_states + j] + counts[j * num_states + i];
            state_counts[i] += sym;
            total_symmetrized += sym;
        }
    }

    let mut populations = vec![0.0f32; num_states];
    if total_symmetrized > 0.0 {
        for i in 0..num_states {
            populations[i] = (state_counts[i] / total_symmetrized) as f32;
        }
    } else {
        let uniform = 1.0 / num_states as f32;
        populations.fill(uniform);
    }

    // 4. Compute relative free energies: ΔG_i = -k_B * T * ln(P_i / P_max)
    // k_B in kcal / (mol * K) = 0.001987204
    let kb = 0.001_987_204_2_f32;
    let kbt = kb * temperature_k;
    let max_p = populations.iter().cloned().fold(0.0f32, f32::max).max(1e-12);

    let mut delta_g = vec![0.0f32; num_states];
    for i in 0..num_states {
        if populations[i] > 1e-12 {
            delta_g[i] = -kbt * (populations[i] / max_p).ln();
        } else {
            delta_g[i] = 20.0;
        }
    }

    MarkovPopulationsResult {
        populations,
        delta_g,
        transition_matrix,
        num_states,
    }
}

pub struct PcaResult {
    pub projections: Vec<f32>,              // [num_frames * n_components]
    pub explained_variance_ratio: Vec<f32>, // [n_components]
    pub components: Vec<f32>,               // [n_components * dim]
}

/// Cartesian Principal Component Analysis (PCA) on trajectory frames.
/// Mean-centers atomic coordinates and computes top-k principal component projections.
pub fn calculate_trajectory_pca(
    trajectory_flat: &[f32],
    num_frames: usize,
    num_atoms: usize,
    n_components: usize,
) -> PcaResult {
    let dim = num_atoms * 3;
    let k = n_components.min(num_frames).min(dim);
    if num_frames == 0 || dim == 0 || k == 0 {
        return PcaResult {
            projections: Vec::new(),
            explained_variance_ratio: Vec::new(),
            components: Vec::new(),
        };
    }

    // 1. Calculate coordinate mean across all frames [dim]
    let mut mean_coords = vec![0.0f64; dim];
    for f in 0..num_frames {
        let frame_slice = &trajectory_flat[f * dim..(f + 1) * dim];
        for d in 0..dim {
            mean_coords[d] += frame_slice[d] as f64;
        }
    }
    let nf = num_frames as f64;
    for d in 0..dim {
        mean_coords[d] /= nf;
    }

    // 2. Build centered matrix X [num_frames, dim] and compute exact total variance
    let mut centered_data = Vec::with_capacity(num_frames * dim);
    let mut total_sq_sum = 0.0f64;
    for f in 0..num_frames {
        let frame_slice = &trajectory_flat[f * dim..(f + 1) * dim];
        for d in 0..dim {
            let val = frame_slice[d] as f64 - mean_coords[d];
            total_sq_sum += val * val;
            centered_data.push(val);
        }
    }

    let denom = (num_frames.saturating_sub(1).max(1)) as f64;
    let total_variance = total_sq_sum / denom;

    // 3. Deflated power iteration for top-k principal components
    let mut deflated_x = centered_data.clone();
    let mut components_f64: Vec<Vec<f64>> = Vec::with_capacity(k);
    let mut explained_variance_ratio = Vec::with_capacity(k);
    let mut projections = vec![0.0f32; num_frames * k];

    for comp in 0..k {
        // Initialize v from the frame with maximum remaining norm
        let mut best_f = 0;
        let mut best_norm = -1.0f64;
        for f in 0..num_frames {
            let row = &deflated_x[f * dim..(f + 1) * dim];
            let nsq: f64 = row.iter().map(|&x| x * x).sum();
            if nsq > best_norm {
                best_norm = nsq;
                best_f = f;
            }
        }

        let mut v: Vec<f64> = if best_norm > 1e-20 {
            deflated_x[best_f * dim..(best_f + 1) * dim].to_vec()
        } else {
            (0..dim).map(|d| ((d * 7 + comp * 13 + 1) as f64).sin()).collect()
        };

        // Normalize initial v
        let mut v_norm = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if v_norm > 1e-20 {
            for x in v.iter_mut() {
                *x /= v_norm;
            }
        }

        let mut u = vec![0.0f64; num_frames];
        for _iter in 0..150 {
            // u = X * v
            u.par_iter_mut()
                .zip(deflated_x.par_chunks(dim))
                .for_each(|(uf, row)| {
                    let mut dot = 0.0f64;
                    for d in 0..dim {
                        dot += row[d] * v[d];
                    }
                    *uf = dot;
                });

            // v_new = X^T * u
            let mut v_new = vec![0.0f64; dim];
            for f in 0..num_frames {
                let uf = u[f];
                let row = &deflated_x[f * dim..(f + 1) * dim];
                for d in 0..dim {
                    v_new[d] += uf * row[d];
                }
            }

            // Double Gram-Schmidt orthogonalization against previous components
            for _pass in 0..2 {
                for prev_v in &components_f64 {
                    let dot: f64 = v_new.iter().zip(prev_v.iter()).map(|(&a, &b)| a * b).sum();
                    for d in 0..dim {
                        v_new[d] -= dot * prev_v[d];
                    }
                }
            }

            v_norm = v_new.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if v_norm <= 1e-20 {
                break;
            }
            for x in v_new.iter_mut() {
                *x /= v_norm;
            }

            let cos_sim: f64 = v.iter().zip(v_new.iter()).map(|(&a, &b)| a * b).sum();
            v = v_new;
            if (1.0 - cos_sim.abs()) < 1e-11 {
                break;
            }
        }

        // Compute projections p = X * v and deflate X
        let mut comp_sq_sum = 0.0f64;
        for f in 0..num_frames {
            let row = &mut deflated_x[f * dim..(f + 1) * dim];
            let mut dot = 0.0f64;
            for d in 0..dim {
                dot += row[d] * v[d];
            }
            comp_sq_sum += dot * dot;
            projections[f * k + comp] = dot as f32;
            for d in 0..dim {
                row[d] -= dot * v[d];
            }
        }

        let comp_var = comp_sq_sum / denom;
        let ratio = if total_variance > 0.0 {
            (comp_var / total_variance) as f32
        } else {
            0.0
        };
        explained_variance_ratio.push(ratio);
        components_f64.push(v);
    }

    let mut components = vec![0.0f32; k * dim];
    for (comp, v) in components_f64.iter().enumerate() {
        for d in 0..dim {
            components[comp * dim + d] = v[d] as f32;
        }
    }

    PcaResult {
        projections,
        explained_variance_ratio,
        components,
    }
}
