// ponytail: zero-allocation memory-mapped DCD binary trajectory reader.
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use memmap2::MmapOptions;

pub struct DcdTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    /// Flat array of coordinates [num_frames * num_atoms * 3] in Angstroms
    pub coords: Vec<f32>,
    /// Optional per-frame 3x3 unit-cell matrices [num_frames * 9] in Angstroms (row-major: a, b, c)
    pub box_vectors: Option<Vec<f32>>,
}

/// Convert CHARMM/NAMD/OpenMM 6-double unit cell [A, gamma, B, beta, alpha, C] to a 3x3
/// row-major Cartesian cell matrix in Angstroms (rows are a, b, c).
/// Disambiguates angles stored as cos(theta) in [-1, 1] (CHARMM/OpenMM) vs degrees (NAMD).
#[inline]
fn dcd_unitcell_to_matrix(raw: [f64; 6]) -> [f32; 9] {
    if !raw.iter().all(|v| v.is_finite()) {
        return [0.0f32; 9];
    }
    let a = raw[0];
    let gamma_raw = raw[1];
    let b = raw[2];
    let beta_raw = raw[3];
    let alpha_raw = raw[4];
    let c = raw[5];

    if a <= 0.0 || b <= 0.0 || c <= 0.0 {
        return [0.0f32; 9];
    }

    let is_cosines = gamma_raw.abs() <= 1.0 && beta_raw.abs() <= 1.0 && alpha_raw.abs() <= 1.0;
    let to_rad = |v: f64| -> f64 {
        if is_cosines {
            v.clamp(-1.0, 1.0).acos()
        } else {
            v.to_radians()
        }
    };

    let alpha = to_rad(alpha_raw);
    let beta = to_rad(beta_raw);
    let gamma = to_rad(gamma_raw);

    let cos_a = alpha.cos();
    let cos_b = beta.cos();
    let cos_g = gamma.cos();
    let sin_g = gamma.sin();

    let bx = b * cos_g;
    let by = b * sin_g;
    let cx = c * cos_b;
    let cy = if sin_g.abs() > 1e-12 {
        c * (cos_a - cos_b * cos_g) / sin_g
    } else {
        0.0
    };
    let cz = (c * c - cx * cx - cy * cy).max(0.0).sqrt();

    [
        a as f32, 0.0, 0.0,
        bx as f32, by as f32, 0.0,
        cx as f32, cy as f32, cz as f32,
    ]
}

pub fn parse_dcd_trajectory<P: AsRef<Path>>(path: P) -> Result<DcdTrajectory> {
    parse_dcd_trajectory_selected(path, None)
}

pub fn parse_dcd_trajectory_selected<P: AsRef<Path>>(
    path: P,
    atom_indices: Option<&[usize]>,
) -> Result<DcdTrajectory> {
    let file = File::open(path)?;
    if file.metadata()?.len() < 92 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "DCD file too short for header"));
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let data = &mmap[..];

    // Check header block 1
    // Fortran block 1: 4 bytes length (84), "CORD", 20 ints (80 bytes), 4 bytes length (84)
    let block1_len = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    if block1_len != 84 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid DCD magic block length"));
    }

    let magic = &data[4..8];
    if magic != b"CORD" && magic != b"VELD" {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid DCD signature; expected CORD or VELD"));
    }

    let namnf = i32::from_le_bytes(data[40..44].try_into().unwrap());
    if namnf != 0 {
        return Err(Error::new(ErrorKind::InvalidData, "DCD fixed atoms (NAMNF != 0) are not supported"));
    }
    let is_charmm = i32::from_le_bytes(data[84..88].try_into().unwrap()) != 0;
    // In CHARMM/NAMD/OpenMM DCD format, QCRYS (has_unitcell) is icntrl[10] at file offset 48..52
    // and is only valid when icntrl[19] (CHARMM version, offset 84..88) is non-zero.
    let has_unitcell = is_charmm && i32::from_le_bytes(data[48..52].try_into().unwrap()) == 1;
    let has_4dims = is_charmm && i32::from_le_bytes(data[52..56].try_into().unwrap()) == 1;
    if has_4dims {
        return Err(Error::new(ErrorKind::InvalidData, "4-dimensional DCD files are not supported"));
    }

    // Advance past block 1: 4 + 84 + 4 = 92 bytes
    let mut offset = 92usize;

    // Block 2: Title / remarks block
    if offset.saturating_add(4) > data.len() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF reading title block"));
    }
    let title_block_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset = offset.saturating_add(4).saturating_add(title_block_len).saturating_add(4);

    // Block 3: Number of atoms
    if offset.saturating_add(12) > data.len() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF reading atom count block"));
    }
    let natoms_block_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    if natoms_block_len != 4 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid natoms block size in DCD"));
    }
    let natoms_raw = i32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
    if natoms_raw <= 0 || natoms_raw > 50_000_000 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid atom count in DCD"));
    }
    let num_atoms = natoms_raw as usize;
    offset += 12;

    if let Some(sel) = atom_indices {
        for &idx in sel {
            if idx >= num_atoms {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Atom index {} out of bounds for DCD with {} atoms", idx, num_atoms),
                ));
            }
        }
    }

    let out_atoms = atom_indices.map(|s| s.len()).unwrap_or(num_atoms);
    let coord_block_bytes = 4usize.saturating_add(num_atoms.saturating_mul(4)).saturating_add(4);
    let mut frame_bytes = coord_block_bytes.saturating_mul(3);
    if has_unitcell {
        frame_bytes = frame_bytes.saturating_add(56);
    }

    let remaining = data.len().saturating_sub(offset);
    // Trust file-size-derived frame count (handles appended/restarted DCDs with stale header counts)
    let num_frames = if frame_bytes > 0 { remaining / frame_bytes } else { 0 };
    let expected_coord_len = (num_atoms * 4) as u32;

    let mut coords = Vec::with_capacity(num_frames * out_atoms * 3);
    let mut box_vectors = if has_unitcell {
        Some(Vec::with_capacity(num_frames * 9))
    } else {
        None
    };

    let mut actual_frames = 0usize;

    for _ in 0..num_frames {
        let mut frame_box = [0.0f32; 9];
        if has_unitcell {
            if offset.saturating_add(4) > data.len() { break; }
            let uc_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            if uc_len != 48 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    format!("Invalid DCD unit-cell record length: expected 48, got {}", uc_len),
                ));
            }
            if offset.saturating_add(56) > data.len() { break; }
            let uc_start = offset + 4;
            let mut raw = [0.0f64; 6];
            for k in 0..6 {
                let s = uc_start + k * 8;
                raw[k] = f64::from_le_bytes(data[s..s + 8].try_into().unwrap());
            }
            frame_box = dcd_unitcell_to_matrix(raw);
            offset += 56;
        }

        // Read X
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        if u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) != expected_coord_len {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid DCD X coordinate block length"));
        }
        let x_start = offset + 4;
        let x_slice = &data[x_start..x_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Y
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        if u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) != expected_coord_len {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid DCD Y coordinate block length"));
        }
        let y_start = offset + 4;
        let y_slice = &data[y_start..y_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Z
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        if u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) != expected_coord_len {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid DCD Z coordinate block length"));
        }
        let z_start = offset + 4;
        let z_slice = &data[z_start..z_start + num_atoms * 4];
        offset += coord_block_bytes;

        if let Some(ref mut bv) = box_vectors {
            bv.extend_from_slice(&frame_box);
        }
        actual_frames += 1;

        if let Some(sel) = atom_indices {
            for &a in sel {
                let byte_idx = a * 4;
                let x = f32::from_le_bytes(x_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                let y = f32::from_le_bytes(y_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                let z = f32::from_le_bytes(z_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        } else {
            for a in 0..num_atoms {
                let byte_idx = a * 4;
                let x = f32::from_le_bytes(x_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                let y = f32::from_le_bytes(y_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                let z = f32::from_le_bytes(z_slice[byte_idx..byte_idx + 4].try_into().unwrap());
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }

    Ok(DcdTrajectory {
        num_frames: actual_frames,
        num_atoms: out_atoms,
        coords,
        box_vectors,
    })
}
