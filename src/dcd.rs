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
    let file = File::open(path)?;
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let data = &mmap[..];

    if data.len() < 84 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "DCD file too short for header"));
    }

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

    let nframes_header = i32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let has_unitcell = i32::from_le_bytes(data[44..48].try_into().unwrap()) == 1;

    // Advance past block 1: 4 + 84 + 4 = 92 bytes
    let mut offset = 92;

    // Block 2: Title / remarks block
    if offset + 4 > data.len() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF reading title block"));
    }
    let title_block_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4 + title_block_len + 4;

    // Block 3: Number of atoms
    if offset + 12 > data.len() {
        return Err(Error::new(ErrorKind::UnexpectedEof, "Unexpected EOF reading atom count block"));
    }
    let natoms_block_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    if natoms_block_len != 4 {
        return Err(Error::new(ErrorKind::InvalidData, "Invalid natoms block size in DCD"));
    }
    let num_atoms = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
    offset += 12;

    // Coordinate frames follow
    // Each frame:
    // If has_unitcell: 4 bytes len (48 or 56), unit cell doubles, 4 bytes len
    // X block: 4 bytes len (num_atoms * 4), float32 * num_atoms, 4 bytes len
    // Y block: 4 bytes len (num_atoms * 4), float32 * num_atoms, 4 bytes len
    // Z block: 4 bytes len (num_atoms * 4), float32 * num_atoms, 4 bytes len

    let coord_block_bytes = 4 + (num_atoms * 4) + 4;
    let mut frame_bytes = 3 * coord_block_bytes;
    if has_unitcell {
        // Typically 4 + 48 + 4 = 56 bytes for unit cell (6 doubles)
        frame_bytes += 56;
    }

    let remaining = data.len().saturating_sub(offset);
    let estimated_frames = if frame_bytes > 0 { remaining / frame_bytes } else { 0 };
    let num_frames = if nframes_header > 0 && nframes_header <= estimated_frames {
        nframes_header
    } else {
        estimated_frames
    };

    let mut coords = Vec::with_capacity(num_frames * num_atoms * 3);

    for _ in 0..num_frames {
        if has_unitcell {
            if offset + 4 > data.len() { break; }
            let uc_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            offset += 4 + uc_len + 4;
        }

        // Read X
        if offset + coord_block_bytes > data.len() { break; }
        let x_start = offset + 4;
        let x_slice = &data[x_start..x_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Y
        if offset + coord_block_bytes > data.len() { break; }
        let y_start = offset + 4;
        let y_slice = &data[y_start..y_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Read Z
        if offset + coord_block_bytes > data.len() { break; }
        let z_start = offset + 4;
        let z_slice = &data[z_start..z_start + num_atoms * 4];
        offset += coord_block_bytes;

        // Interleave X, Y, Z directly into coords [num_atoms, 3]
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

    let actual_frames = coords.len() / (num_atoms * 3);

    Ok(DcdTrajectory {
        num_frames: actual_frames,
        num_atoms,
        coords,
    })
}
