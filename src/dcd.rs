// ponytail: zero-allocation memory-mapped DCD binary trajectory reader.
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use memmap2::MmapOptions;

pub struct DcdTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    /// Flat array of coordinates [num_frames * num_atoms * 3]
    pub coords: Vec<f32>,
}

pub fn parse_dcd_trajectory<P: AsRef<Path>>(path: P) -> Result<DcdTrajectory> {
    parse_dcd_trajectory_selected(path, None)
}

pub fn parse_dcd_trajectory_selected<P: AsRef<Path>>(
    path: P,
    atom_indices: Option<&[usize]>,
) -> Result<DcdTrajectory> {
    let file = File::open(path)?;
    if file.metadata()?.len() < 84 {
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

    let nframes_header = i32::from_le_bytes(data[8..12].try_into().unwrap()).max(0) as usize;
    let has_unitcell = i32::from_le_bytes(data[44..48].try_into().unwrap()) == 1;

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
    let estimated_frames = if frame_bytes > 0 { remaining / frame_bytes } else { 0 };
    let num_frames = if nframes_header > 0 && nframes_header <= estimated_frames {
        nframes_header
    } else {
        estimated_frames
    };

    let mut coords = Vec::with_capacity(num_frames * out_atoms * 3);

    for _ in 0..num_frames {
        if has_unitcell {
            if offset.saturating_add(4) > data.len() { break; }
            let uc_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset = offset.saturating_add(4).saturating_add(uc_len).saturating_add(4);
        }

        // Read X
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        let x_start = offset + 4;
        let x_slice = &data[x_start..x_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Y
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        let y_start = offset + 4;
        let y_slice = &data[y_start..y_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Z
        if offset.saturating_add(coord_block_bytes) > data.len() { break; }
        let z_start = offset + 4;
        let z_slice = &data[z_start..z_start + num_atoms * 4];
        offset += coord_block_bytes;

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

    let actual_frames = if out_atoms > 0 { coords.len() / (out_atoms * 3) } else { 0 };

    Ok(DcdTrajectory {
        num_frames: actual_frames,
        num_atoms: out_atoms,
        coords,
    })
}
