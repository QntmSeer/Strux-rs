// ponytail: high-throughput therapeutic binder & ensemble biophysics kernels (Challenges 01-05).
use nalgebra::{DMatrix, Vector3};
use rayon::prelude::*;
use std::f32::consts::PI;

use crate::analysis::{calculate_rmsd_kabsch, kabsch_transform};

// ============================================================================
// 1. TARGET-ANCHORED POSE_RMSD & MONOMER SC_RMSD
// ============================================================================

/// Computes both monomer self-consistency RMSD (`sc_rmsd`) and target-anchored `pose_rmsd`.
pub fn calculate_pose_and_sc_rmsd(
    ref_target_ca: &[[f32; 3]],
    ref_binder_ca: &[[f32; 3]],
    mob_target_ca: &[[f32; 3]],
    mob_binder_ca: &[[f32; 3]],
) -> (f32, f32) {
    if ref_target_ca.is_empty()
        || ref_binder_ca.is_empty()
        || ref_target_ca.len() != mob_target_ca.len()
        || ref_binder_ca.len() != mob_binder_ca.len()
    {
        return (0.0, 0.0);
    }

    let sc_rmsd = calculate_rmsd_kabsch(ref_binder_ca, mob_binder_ca);
    let (r_target, c_ref_t, c_mob_t) = kabsch_transform(ref_target_ca, mob_target_ca);

    let mut sum_sq = 0.0f32;
    let n_b = ref_binder_ca.len();
    for i in 0..n_b {
        let p_mob = Vector3::from(mob_binder_ca[i]) - c_mob_t;
        let p_aligned = (r_target * p_mob) + c_ref_t;
        let p_ref = Vector3::from(ref_binder_ca[i]);
        let diff = p_aligned - p_ref;
        sum_sq += diff.norm_squared();
    }
    let pose_rmsd = (sum_sq / (n_b as f32)).sqrt();
    (sc_rmsd, pose_rmsd)
}

/// Batch version across M co-folded designs parallelized with Rayon.
pub fn batch_pose_and_sc_rmsd(
    ref_target_ca: &[[f32; 3]],
    ref_binder_ca: &[[f32; 3]],
    mob_targets_flat: &[f32],
    mob_binders_flat: &[f32],
    num_models: usize,
) -> (Vec<f32>, Vec<f32>) {
    let n_t = ref_target_ca.len();
    let n_b = ref_binder_ca.len();
    let stride_t = n_t * 3;
    let stride_b = n_b * 3;

    let results: Vec<(f32, f32)> = (0..num_models)
        .into_par_iter()
        .map(|m| {
            let t_slice = &mob_targets_flat[m * stride_t..(m + 1) * stride_t];
            let b_slice = &mob_binders_flat[m * stride_b..(m + 1) * stride_b];
            let mob_t: &[[f32; 3]] =
                unsafe { std::slice::from_raw_parts(t_slice.as_ptr() as *const [f32; 3], n_t) };
            let mob_b: &[[f32; 3]] =
                unsafe { std::slice::from_raw_parts(b_slice.as_ptr() as *const [f32; 3], n_b) };
            calculate_pose_and_sc_rmsd(ref_target_ca, ref_binder_ca, mob_t, mob_b)
        })
        .collect();

    let mut sc = Vec::with_capacity(num_models);
    let mut pose = Vec::with_capacity(num_models);
    for (s, p) in results {
        sc.push(s);
        pose.push(p);
    }
    (sc, pose)
}

// ============================================================================
// 2. ALL-ATOM INTERFACE PHYSICS (CONTACTS, H-BONDS, SALT BRIDGES, BUNS)
// ============================================================================

pub struct InterfacePhysicsResult {
    pub interface_contacts: usize,
    pub h_bonds: usize,
    pub salt_bridges: usize,
    pub buns_count: usize,
    pub buns_indices: Vec<usize>,
}

#[inline]
fn is_polar_atom(atom_name: &str) -> bool {
    let bytes = atom_name.trim().as_bytes();
    if bytes.is_empty() {
        return false;
    }
    bytes[0] == b'N' || bytes[0] == b'O'
}

#[inline]
fn is_cationic_atom(res_name: &str, atom_name: &str) -> bool {
    let r = res_name.trim();
    let a = atom_name.trim();
    (r == "LYS" && a == "NZ")
        || (r == "ARG" && (a == "NH1" || a == "NH2" || a == "NE"))
        || (r == "HIS" && (a == "ND1" || a == "NE2"))
}

#[inline]
fn is_anionic_atom(res_name: &str, atom_name: &str) -> bool {
    let r = res_name.trim();
    let a = atom_name.trim();
    (r == "ASP" && (a == "OD1" || a == "OD2")) || (r == "GLU" && (a == "OE1" || a == "OE2"))
}

pub fn analyze_interface_physics(
    coords: &[[f32; 3]],
    atom_names: &[String],
    res_names: &[String],
    chain_ids: &[String],
    res_ids: &[i32],
    target_chain: &str,
    binder_chain: &str,
) -> InterfacePhysicsResult {
    let n = coords.len();
    let mut target_idx = Vec::new();
    let mut binder_idx = Vec::new();

    for i in 0..n {
        if chain_ids[i] == target_chain {
            target_idx.push(i);
        } else if chain_ids[i] == binder_chain {
            binder_idx.push(i);
        }
    }

    let cutoff_contact_sq = 4.5f32 * 4.5f32;
    let cutoff_hbond_sq = 3.5f32 * 3.5f32;
    let cutoff_salt_sq = 4.0f32 * 4.0f32;
    let cutoff_buried_sq = 5.0f32 * 5.0f32;
    let cutoff_sat_sq = 3.9f32 * 3.9f32;

    let mut interface_contacts = 0usize;
    let mut h_bonds = 0usize;
    let mut salt_bridges = 0usize;

    let mut is_buried_interface = vec![false; n];

    for &ti in &target_idx {
        let ct = coords[ti];
        let pt = is_polar_atom(&atom_names[ti]);
        let cat_t = is_cationic_atom(&res_names[ti], &atom_names[ti]);
        let ani_t = is_anionic_atom(&res_names[ti], &atom_names[ti]);

        for &bi in &binder_idx {
            let cb = coords[bi];
            let dx = ct[0] - cb[0];
            let dy = ct[1] - cb[1];
            let dz = ct[2] - cb[2];
            let d2 = dx * dx + dy * dy + dz * dz;

            if d2 <= cutoff_buried_sq {
                is_buried_interface[ti] = true;
                is_buried_interface[bi] = true;
            }
            if d2 <= cutoff_contact_sq {
                interface_contacts += 1;
            }
            if pt && is_polar_atom(&atom_names[bi]) {
                if d2 <= cutoff_hbond_sq {
                    h_bonds += 1;
                }
                if d2 <= cutoff_salt_sq {
                    let cat_b = is_cationic_atom(&res_names[bi], &atom_names[bi]);
                    let ani_b = is_anionic_atom(&res_names[bi], &atom_names[bi]);
                    if (cat_t && ani_b) || (ani_t && cat_b) {
                        salt_bridges += 1;
                    }
                }
            }
        }
    }

    // Collect all polar atoms across both chains to test satisfaction of buried interface polars
    let mut polar_indices = Vec::new();
    for i in 0..n {
        if (chain_ids[i] == target_chain || chain_ids[i] == binder_chain)
            && is_polar_atom(&atom_names[i])
        {
            polar_indices.push(i);
        }
    }

    let mut buns_indices = Vec::new();
    for &pi in &polar_indices {
        if !is_buried_interface[pi] {
            continue;
        }
        let cp = coords[pi];
        let mut satisfied = false;
        for &pj in &polar_indices {
            if pi == pj {
                continue;
            }
            // Skip intra-residue or adjacent peptide bond N/O on the same chain
            if chain_ids[pi] == chain_ids[pj] && (res_ids[pi] - res_ids[pj]).abs() <= 1 {
                let ai = atom_names[pi].trim();
                let aj = atom_names[pj].trim();
                if res_ids[pi] == res_ids[pj] || ((ai == "N" || ai == "O") && (aj == "N" || aj == "O"))
                {
                    continue;
                }
            }
            let cq = coords[pj];
            let dx = cp[0] - cq[0];
            let dy = cp[1] - cq[1];
            let dz = cp[2] - cq[2];
            if dx * dx + dy * dy + dz * dz <= cutoff_sat_sq {
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            buns_indices.push(pi);
        }
    }

    InterfacePhysicsResult {
        interface_contacts,
        h_bonds,
        salt_bridges,
        buns_count: buns_indices.len(),
        buns_indices,
    }
}

// ============================================================================
// 3. SHRAKE-RUPLEY SASA & BURIED SURFACE AREA (BSA)
// ============================================================================

/// Generates `n_points` uniformly distributed on the unit sphere via a Golden Spiral.
fn golden_spiral_sphere(n_points: usize) -> Vec<[f32; 3]> {
    let mut pts = Vec::with_capacity(n_points);
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());
    for i in 0..n_points {
        let y = 1.0 - (2.0 * (i as f32) + 1.0) / (n_points as f32);
        let r = (1.0 - y * y).max(0.0).sqrt();
        let theta = golden_angle * (i as f32);
        pts.push([r * theta.cos(), y, r * theta.sin()]);
    }
    pts
}

pub fn calculate_sasa(
    coords: &[[f32; 3]],
    radii: Option<&[f32]>,
    probe_radius: f32,
    n_points: usize,
) -> (f32, Vec<f32>) {
    let n = coords.len();
    if n == 0 {
        return (0.0, Vec::new());
    }
    let n_pts = n_points.max(16);
    let sphere_pts = golden_spiral_sphere(n_pts);

    let eff_radii: Vec<f32> = (0..n)
        .map(|i| radii.and_then(|r| r.get(i).copied()).unwrap_or(1.70) + probe_radius)
        .collect();

    let weight_per_point = 4.0 * PI / (n_pts as f32);

    let per_atom_sasa: Vec<f32> = (0..n)
        .into_par_iter()
        .map(|i| {
            let ci = coords[i];
            let ri = eff_radii[i];
            // Pre-filter candidate neighbors within ri + rj
            let mut neighbors = Vec::with_capacity(32);
            for j in 0..n {
                if i == j {
                    continue;
                }
                let cj = coords[j];
                let rj = eff_radii[j];
                let max_d = ri + rj;
                let dx = ci[0] - cj[0];
                let dy = ci[1] - cj[1];
                let dz = ci[2] - cj[2];
                let d2 = dx * dx + dy * dy + dz * dz;
                if d2 < max_d * max_d {
                    neighbors.push((cj, rj * rj));
                }
            }

            let mut accessible = 0usize;
            'pt_loop: for sp in &sphere_pts {
                let px = ci[0] + ri * sp[0];
                let py = ci[1] + ri * sp[1];
                let pz = ci[2] + ri * sp[2];
                for &(cj, rj_sq) in &neighbors {
                    let dx = px - cj[0];
                    let dy = py - cj[1];
                    let dz = pz - cj[2];
                    if dx * dx + dy * dy + dz * dz < rj_sq {
                        continue 'pt_loop;
                    }
                }
                accessible += 1;
            }
            (accessible as f32) * weight_per_point * ri * ri
        })
        .collect();

    let total_sasa: f32 = per_atom_sasa.iter().sum();
    (total_sasa, per_atom_sasa)
}

pub fn calculate_bsa(
    target_coords: &[[f32; 3]],
    binder_coords: &[[f32; 3]],
    probe_radius: f32,
    n_points: usize,
) -> (f32, f32, f32, f32) {
    let (sasa_target, _) = calculate_sasa(target_coords, None, probe_radius, n_points);
    let (sasa_binder, _) = calculate_sasa(binder_coords, None, probe_radius, n_points);

    let mut complex_coords = Vec::with_capacity(target_coords.len() + binder_coords.len());
    complex_coords.extend_from_slice(target_coords);
    complex_coords.extend_from_slice(binder_coords);

    let (sasa_complex, _) = calculate_sasa(&complex_coords, None, probe_radius, n_points);
    let bsa = (sasa_target + sasa_binder - sasa_complex).max(0.0);
    (bsa, sasa_target, sasa_binder, sasa_complex)
}

// ============================================================================
// 4. SCREENED DEBYE-HUCKEL pH-SWITCH ELECTROSTATIC ddG
// ============================================================================

#[inline]
fn titratable_charge(res_name: &str, atom_name: &str, ph: f32) -> f32 {
    let r = res_name.trim();
    let a = atom_name.trim();
    // Fractional Henderson-Hasselbalch charge split across sidechain terminal charge centers
    if r == "ASP" && (a == "OD1" || a == "OD2") {
        let frac_deprot = 1.0 / (1.0 + 10.0f32.powf(3.9 - ph));
        return -0.5 * frac_deprot;
    }
    if r == "GLU" && (a == "OE1" || a == "OE2") {
        let frac_deprot = 1.0 / (1.0 + 10.0f32.powf(4.3 - ph));
        return -0.5 * frac_deprot;
    }
    if r == "HIS" && (a == "ND1" || a == "NE2") {
        let frac_prot = 1.0 / (1.0 + 10.0f32.powf(ph - 6.0));
        return 0.5 * frac_prot;
    }
    if r == "LYS" && a == "NZ" {
        return 1.0 / (1.0 + 10.0f32.powf(ph - 10.5));
    }
    if r == "ARG" && (a == "NH1" || a == "NH2") {
        let frac_prot = 1.0 / (1.0 + 10.0f32.powf(ph - 12.5));
        return 0.5 * frac_prot;
    }
    if r == "CYS" && a == "SG" {
        return -1.0 / (1.0 + 10.0f32.powf(8.3 - ph));
    }
    if r == "TYR" && a == "OH" {
        return -1.0 / (1.0 + 10.0f32.powf(10.1 - ph));
    }
    0.0
}

pub fn calculate_ph_switch_ddg(
    coords: &[[f32; 3]],
    atom_names: &[String],
    res_names: &[String],
    chain_ids: &[String],
    target_chain: &str,
    binder_chain: &str,
    ph_low: f32,
    ph_high: f32,
    ionic_strength: f32,
    dielectric: f32,
) -> (f32, f32, f32) {
    let n = coords.len();
    let mut target_sites = Vec::new();
    let mut binder_sites = Vec::new();

    for i in 0..n {
        let q_low = titratable_charge(&res_names[i], &atom_names[i], ph_low);
        let q_high = titratable_charge(&res_names[i], &atom_names[i], ph_high);
        if q_low.abs() > 1e-5 || q_high.abs() > 1e-5 {
            if chain_ids[i] == target_chain {
                target_sites.push((coords[i], q_low, q_high));
            } else if chain_ids[i] == binder_chain {
                binder_sites.push((coords[i], q_low, q_high));
            }
        }
    }

    // Debye-Huckel screening parameter kappa (in A^-1) at 298 K: kappa = 0.329 * sqrt(I)
    let kappa = 0.329 * ionic_strength.max(0.001).sqrt();
    let coulomb_const = 332.0637 / dielectric.max(1.0);

    let mut dg_low = 0.0f32;
    let mut dg_high = 0.0f32;

    for &(ct, qt_low, qt_high) in &target_sites {
        for &(cb, qb_low, qb_high) in &binder_sites {
            let dx = ct[0] - cb[0];
            let dy = ct[1] - cb[1];
            let dz = ct[2] - cb[2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1.5);
            if r <= 20.0 {
                let factor = coulomb_const * (-kappa * r).exp() / r;
                dg_low += qt_low * qb_low * factor;
                dg_high += qt_high * qb_high * factor;
            }
        }
    }

    let ddg_ph = dg_low - dg_high;
    (ddg_ph, dg_low, dg_high)
}

// ============================================================================
// 5. ANISOTROPIC NETWORK MODEL (ANM) NORMAL MODES & THEORETICAL B-FACTORS
// ============================================================================

pub struct AnmResult {
    pub eigenvalues: Vec<f32>,
    pub eigenvectors: Vec<f32>, // [n_modes * 3 * n_atoms]
    pub bfactors: Vec<f32>,     // [n_atoms]
}

pub fn calculate_anm_modes(
    ca_coords: &[[f32; 3]],
    cutoff: f32,
    gamma: f32,
    n_modes: usize,
) -> AnmResult {
    let n = ca_coords.len();
    let dim = n * 3;
    if n < 3 || n_modes == 0 || !ca_coords.iter().all(|c| c[0].is_finite() && c[1].is_finite() && c[2].is_finite()) {
        return AnmResult {
            eigenvalues: Vec::new(),
            eigenvectors: Vec::new(),
            bfactors: vec![0.0; n],
        };
    }

    // 1. Build 3N x 3N Hessian matrix H in f64
    let mut hessian = vec![0.0f64; dim * dim];
    let cutoff_sq = (cutoff as f64) * (cutoff as f64);
    let g = gamma as f64;

    for i in 0..n {
        for j in (i + 1)..n {
            let dx = (ca_coords[j][0] - ca_coords[i][0]) as f64;
            let dy = (ca_coords[j][1] - ca_coords[i][1]) as f64;
            let dz = (ca_coords[j][2] - ca_coords[i][2]) as f64;
            let d2 = dx * dx + dy * dy + dz * dz;
            if d2 <= cutoff_sq && d2 > 1e-8 {
                let d_vec = [dx, dy, dz];
                let scale = -g / d2;
                for a in 0..3 {
                    for b in 0..3 {
                        let hij = scale * d_vec[a] * d_vec[b];
                        let row_i = i * 3 + a;
                        let col_j = j * 3 + b;
                        hessian[row_i * dim + col_j] = hij;
                        hessian[col_j * dim + row_i] = hij;

                        let row_ii = i * 3 + a;
                        let col_ib = i * 3 + b;
                        hessian[row_ii * dim + col_ib] -= hij;

                        let row_jj = j * 3 + a;
                        let col_jb = j * 3 + b;
                        hessian[row_jj * dim + col_jb] -= hij;
                    }
                }
            }
        }
    }

    // 2. Exact Householder tridiagonalization + QR symmetric eigendecomposition via nalgebra
    let mat = DMatrix::<f64>::from_row_slice(dim, dim, &hessian);
    let eig = mat.symmetric_eigen();

    let mut indices: Vec<usize> = (0..dim).collect();
    indices.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Skip the 6 trivial rigid-body modes (3 translations + 3 rotations)
    let start_idx = 6.min(dim);
    let k_modes = n_modes.min(dim.saturating_sub(start_idx));

    let mut eigenvalues = Vec::with_capacity(k_modes);
    let mut eigenvectors = Vec::with_capacity(k_modes * dim);
    let mut bfactors = vec![0.0f32; n];

    for &col_idx in indices.iter().skip(start_idx).take(k_modes) {
        let eval = eig.eigenvalues[col_idx].max(1e-6) as f32;
        eigenvalues.push(eval);
        let inv_lambda = 1.0 / eval;
        let col = eig.eigenvectors.column(col_idx);
        for i in 0..n {
            let vx = col[i * 3] as f32;
            let vy = col[i * 3 + 1] as f32;
            let vz = col[i * 3 + 2] as f32;
            eigenvectors.push(vx);
            eigenvectors.push(vy);
            eigenvectors.push(vz);
            bfactors[i] += (vx * vx + vy * vy + vz * vz) * inv_lambda;
        }
    }

    AnmResult {
        eigenvalues,
        eigenvectors,
        bfactors,
    }
}

// ============================================================================
// 6. SEQUENCE DEVELOPABILITY BIOPHYSICS & SLIDING K-MER NOVELTY SCREENER
// ============================================================================

pub struct SequenceBiophysicsBatch {
    pub isoelectric_point: Vec<f32>,
    pub gravy: Vec<f32>,
    pub net_charge_ph74: Vec<f32>,
    pub camsol_score: Vec<f32>,
    pub apr_count: Vec<usize>,
    pub liability_count: Vec<usize>,
    pub hqnst_fraction: Vec<f32>,
}

#[inline]
fn kyte_doolittle(aa: u8) -> f32 {
    match aa {
        b'I' => 4.5,
        b'V' => 4.2,
        b'L' => 3.8,
        b'F' => 2.8,
        b'C' => 2.5,
        b'M' => 1.9,
        b'A' => 1.8,
        b'G' => -0.4,
        b'T' => -0.7,
        b'S' => -0.8,
        b'W' => -0.9,
        b'Y' => -1.3,
        b'P' => -1.6,
        b'H' => -3.2,
        b'E' => -3.5,
        b'Q' => -3.5,
        b'D' => -3.5,
        b'N' => -3.5,
        b'K' => -3.9,
        b'R' => -4.5,
        _ => 0.0,
    }
}

#[inline]
fn camsol_intrinsic(aa: u8) -> f32 {
    // Intrinsic solubility propensity scale (Sormanni et al., JMB 2015)
    match aa {
        b'D' => 3.09,
        b'E' => 2.92,
        b'K' => 2.89,
        b'R' => 2.50,
        b'N' => 0.85,
        b'Q' => 0.78,
        b'P' => 0.65,
        b'S' => 0.42,
        b'T' => 0.25,
        b'H' => 0.18,
        b'G' => 0.08,
        b'A' => -0.35,
        b'M' => -0.82,
        b'C' => -1.25,
        b'Y' => -1.55,
        b'V' => -1.80,
        b'L' => -2.05,
        b'I' => -2.35,
        b'F' => -2.65,
        b'W' => -2.85,
        _ => 0.0,
    }
}

#[inline]
fn net_charge_at_ph(counts: &[usize; 26], ph: f32) -> f32 {
    let n_term = 1.0 / (1.0 + 10.0f32.powf(ph - 8.0));
    let c_term = -1.0 / (1.0 + 10.0f32.powf(3.1 - ph));
    let k = (counts[(b'K' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(ph - 10.5));
    let r = (counts[(b'R' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(ph - 12.5));
    let h = (counts[(b'H' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(ph - 6.0));
    let d = -(counts[(b'D' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(3.9 - ph));
    let e = -(counts[(b'E' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(4.3 - ph));
    let c = -(counts[(b'C' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(8.3 - ph));
    let y = -(counts[(b'Y' - b'A') as usize] as f32) / (1.0 + 10.0f32.powf(10.1 - ph));
    n_term + c_term + k + r + h + d + e + c + y
}

pub fn screen_sequences_biophysics(sequences: &[String]) -> SequenceBiophysicsBatch {
    let n = sequences.len();
    let results: Vec<(f32, f32, f32, f32, usize, usize, f32)> = sequences
        .par_iter()
        .map(|seq| {
            let bytes: Vec<u8> = seq
                .bytes()
                .filter(|&b| b.is_ascii_alphabetic())
                .map(|b| b.to_ascii_uppercase())
                .collect();
            let len = bytes.len();
            if len == 0 {
                return (7.0, 0.0, 0.0, 0.0, 0, 0, 0.0);
            }

            let mut counts = [0usize; 26];
            let mut kd_sum = 0.0f32;
            let mut hqnst = 0usize;
            let mut apr_count = 0usize;
            let mut cur_hydro_run = 0usize;

            for &b in &bytes {
                if (b'A'..=b'Z').contains(&b) {
                    counts[(b - b'A') as usize] += 1;
                }
                let kd = kyte_doolittle(b);
                kd_sum += kd;
                if matches!(b, b'H' | b'Q' | b'N' | b'S' | b'T') {
                    hqnst += 1;
                }
                if matches!(b, b'I' | b'V' | b'L' | b'F' | b'Y' | b'W' | b'M') {
                    cur_hydro_run += 1;
                    if cur_hydro_run == 5 {
                        apr_count += 1;
                    }
                } else {
                    cur_hydro_run = 0;
                }
            }

            // Bisection for isoelectric point (pI)
            let mut lo = 0.0f32;
            let mut hi = 14.0f32;
            for _ in 0..24 {
                let mid = 0.5 * (lo + hi);
                if net_charge_at_ph(&counts, mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let pi = 0.5 * (lo + hi);
            let q74 = net_charge_at_ph(&counts, 7.4);
            let gravy = kd_sum / (len as f32);

            // 7-residue window CamSol profile + gatekeeper charge bonus
            let mut camsol_sum = 0.0f32;
            for i in 0..len {
                let w_start = i.saturating_sub(3);
                let w_end = (i + 4).min(len);
                let mut w_val = 0.0f32;
                for j in w_start..w_end {
                    w_val += camsol_intrinsic(bytes[j]);
                }
                camsol_sum += w_val / ((w_end - w_start) as f32);
            }
            let camsol = camsol_sum / (len as f32);

            // Chemical liability motifs: DP (acid cleavage), NG/NS/NT (deamidation), N[^P][ST] (N-glycosylation)
            let mut liabilities = 0usize;
            for i in 0..len.saturating_sub(1) {
                let a = bytes[i];
                let b = bytes[i + 1];
                if (a == b'D' && b == b'P')
                    || (a == b'N' && (b == b'G' || b == b'S' || b == b'T'))
                {
                    liabilities += 1;
                }
                if i + 2 < len && a == b'N' && bytes[i + 1] != b'P' && (bytes[i + 2] == b'S' || bytes[i + 2] == b'T') {
                    liabilities += 1;
                }
            }

            let hqnst_frac = (hqnst as f32) / (len as f32);
            (pi, gravy, q74, camsol, apr_count, liabilities, hqnst_frac)
        })
        .collect();

    let mut isoelectric_point = Vec::with_capacity(n);
    let mut gravy = Vec::with_capacity(n);
    let mut net_charge_ph74 = Vec::with_capacity(n);
    let mut camsol_score = Vec::with_capacity(n);
    let mut apr_count = Vec::with_capacity(n);
    let mut liability_count = Vec::with_capacity(n);
    let mut hqnst_fraction = Vec::with_capacity(n);

    for (pi, gr, q, cs, apr, liab, hq) in results {
        isoelectric_point.push(pi);
        gravy.push(gr);
        net_charge_ph74.push(q);
        camsol_score.push(cs);
        apr_count.push(apr);
        liability_count.push(liab);
        hqnst_fraction.push(hq);
    }

    SequenceBiophysicsBatch {
        isoelectric_point,
        gravy,
        net_charge_ph74,
        camsol_score,
        apr_count,
        liability_count,
        hqnst_fraction,
    }
}

/// Computes maximum sliding k-mer identity (0.0 to 1.0) and minimum Hamming distance
/// of each query sequence against a reference database across all CPU threads.
pub fn sliding_kmer_novelty(
    queries: &[String],
    reference_db: &[String],
    k: usize,
) -> (Vec<f32>, Vec<usize>) {
    let k_len = k.max(1);
    if k_len <= 16 {
        // ponytail: pack windows into 16-byte aligned blocks so LLVM lowers comparison to SIMD pcmpeqb
        let mut ref_blocks: Vec<[u8; 16]> = Vec::new();
        for s in reference_db {
            let b = s.as_bytes();
            if b.len() >= k_len {
                for w in b.windows(k_len) {
                    let mut blk = [0u8; 16];
                    blk[..k_len].copy_from_slice(w);
                    ref_blocks.push(blk);
                }
            }
        }
        ref_blocks.sort_unstable();
        ref_blocks.dedup();

        let results: Vec<(f32, usize)> = queries
            .par_iter()
            .map(|q| {
                let qb = q.as_bytes();
                if qb.len() < k_len || ref_blocks.is_empty() {
                    return (0.0, k_len);
                }
                let mut max_matches = 0usize;
                for w in qb.windows(k_len) {
                    let mut q_blk = [0u8; 16];
                    q_blk[..k_len].copy_from_slice(w);
                    if ref_blocks.binary_search(&q_blk).is_ok() {
                        return (1.0, 0);
                    }
                    for r_blk in &ref_blocks {
                        let mut eq = 0usize;
                        for idx in 0..16 {
                            eq += (q_blk[idx] == r_blk[idx]) as usize;
                        }
                        let m = eq - (16 - k_len);
                        if m > max_matches {
                            max_matches = m;
                        }
                    }
                }
                let ident = (max_matches as f32) / (k_len as f32);
                (ident, k_len - max_matches)
            })
            .collect();

        let mut idents = Vec::with_capacity(queries.len());
        let mut hammings = Vec::with_capacity(queries.len());
        for (id, hm) in results {
            idents.push(id);
            hammings.push(hm);
        }
        return (idents, hammings);
    }

    let mut ref_windows: Vec<&[u8]> = Vec::new();
    for s in reference_db {
        let b = s.as_bytes();
        if b.len() >= k_len {
            ref_windows.extend(b.windows(k_len));
        }
    }

    let results: Vec<(f32, usize)> = queries
        .par_iter()
        .map(|q| {
            let qb = q.as_bytes();
            if qb.len() < k_len || ref_windows.is_empty() {
                return (0.0, k_len);
            }
            let mut max_matches = 0usize;
            for q_win in qb.windows(k_len) {
                for &r_win in &ref_windows {
                    let mut m = 0usize;
                    for idx in 0..k_len {
                        m += (q_win[idx] == r_win[idx]) as usize;
                    }
                    if m > max_matches {
                        max_matches = m;
                        if max_matches == k_len {
                            return (1.0, 0);
                        }
                    }
                }
            }
            let ident = (max_matches as f32) / (k_len as f32);
            let min_hamming = k_len - max_matches;
            (ident, min_hamming)
        })
        .collect();

    let mut idents = Vec::with_capacity(queries.len());
    let mut hammings = Vec::with_capacity(queries.len());
    for (id, hm) in results {
        idents.push(id);
        hammings.push(hm);
    }
    (idents, hammings)
}
