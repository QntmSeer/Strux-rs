// ponytail: simple line-by-line PDB/GRO parser and writer, minimal allocations.
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Error, ErrorKind, Write};
use std::path::Path;

pub fn parse_pdb_trajectory<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Vec<[f32; 3]>>, Error> {
    parse_pdb_trajectory_selected(path, false, None)
}

pub fn parse_pdb_trajectory_selected<P: AsRef<Path>>(
    path: P,
    ca_only: bool,
    atom_indices: Option<&[usize]>,
) -> Result<Vec<Vec<[f32; 3]>>, Error> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut frames = Vec::new();
    let mut current_frame = Vec::new();

    for line_result in reader.lines() {
        let line = line_result?;
        if line.starts_with("MODEL") {
            if !current_frame.is_empty() {
                frames.push(current_frame);
                current_frame = Vec::new();
            }
        } else if line.starts_with("ENDMDL") {
            if !current_frame.is_empty() {
                frames.push(current_frame);
                current_frame = Vec::new();
            }
        } else if line.starts_with("ATOM  ") || line.starts_with("HETATM") {
            let b = line.as_bytes();
            if b.len() >= 54 {
                if ca_only {
                    let atom_field = std::str::from_utf8(&b[12..16]).unwrap_or("");
                    if atom_field.trim() != "CA" {
                        continue;
                    }
                }
                let x_str = std::str::from_utf8(&b[30..38]).unwrap_or("").trim();
                let y_str = std::str::from_utf8(&b[38..46]).unwrap_or("").trim();
                let z_str = std::str::from_utf8(&b[46..54]).unwrap_or("").trim();
                if let (Ok(x), Ok(y), Ok(z)) = (
                    x_str.parse::<f32>(),
                    y_str.parse::<f32>(),
                    z_str.parse::<f32>(),
                ) {
                    current_frame.push([x, y, z]);
                }
            }
        }
    }

    if !current_frame.is_empty() {
        frames.push(current_frame);
    }

    if frames.is_empty() {
        return Ok(frames);
    }

    let raw_atoms = frames[0].len();
    if frames.iter().any(|f| f.len() != raw_atoms) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Inconsistent atom count across PDB models",
        ));
    }

    if let Some(sel) = atom_indices {
        for &idx in sel {
            if idx >= raw_atoms {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Atom index {} out of bounds for PDB with {} atoms", idx, raw_atoms),
                ));
            }
        }
        let mut filtered_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let mut sub = Vec::with_capacity(sel.len());
            for &idx in sel {
                sub.push(frame[idx]);
            }
            filtered_frames.push(sub);
        }
        return Ok(filtered_frames);
    }

    Ok(frames)
}

// ============================================================================
// GROMACS .gro Trajectory & Structure Parser
// ============================================================================

pub fn parse_gro_trajectory<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Vec<[f32; 3]>>, Error> {
    parse_gro_trajectory_selected(path, false, None)
}

pub fn parse_gro_trajectory_selected<P: AsRef<Path>>(
    path: P,
    ca_only: bool,
    atom_indices: Option<&[usize]>,
) -> Result<Vec<Vec<[f32; 3]>>, Error> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut frames = Vec::new();
    let mut line_buf = String::new();

    loop {
        line_buf.clear();
        // 1. Title line
        if reader.read_line(&mut line_buf)? == 0 {
            break;
        }
        let title = line_buf.trim();
        if title.is_empty() {
            continue;
        }

        // 2. Number of atoms line
        line_buf.clear();
        if reader.read_line(&mut line_buf)? == 0 {
            break;
        }
        let num_atoms: usize = line_buf
            .trim()
            .parse()
            .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Failed to parse GRO atom count: {}", e)))?;

        // 3. Atom lines
        let mut frame_coords = Vec::with_capacity(num_atoms);
        for atom_idx in 0..num_atoms {
            line_buf.clear();
            if reader.read_line(&mut line_buf)? == 0 {
                return Err(Error::new(
                    ErrorKind::UnexpectedEof,
                    format!("Unexpected EOF reading GRO atom {} of {}", atom_idx + 1, num_atoms),
                ));
            }
            let b = line_buf.as_bytes();
            if ca_only && b.len() >= 15 {
                let atom_name = std::str::from_utf8(&b[10..15]).unwrap_or("").trim();
                if atom_name != "CA" {
                    continue;
                }
            }

            let (x, y, z) = if b.len() >= 44 {
                let x_str = std::str::from_utf8(&b[20..28]).unwrap_or("").trim();
                let y_str = std::str::from_utf8(&b[28..36]).unwrap_or("").trim();
                let z_str = std::str::from_utf8(&b[36..44]).unwrap_or("").trim();
                if let (Ok(x), Ok(y), Ok(z)) = (x_str.parse::<f32>(), y_str.parse::<f32>(), z_str.parse::<f32>()) {
                    (x, y, z)
                } else {
                    parse_gro_coords_fallback(&line_buf)?
                }
            } else {
                parse_gro_coords_fallback(&line_buf)?
            };

            // Convert nm to Angstroms (1 nm = 10 A)
            frame_coords.push([x * 10.0, y * 10.0, z * 10.0]);
        }

        // 4. Box vectors line
        line_buf.clear();
        if reader.read_line(&mut line_buf)? == 0 {
            return Err(Error::new(ErrorKind::UnexpectedEof, "Missing GRO box vector line"));
        }

        frames.push(frame_coords);
    }

    if frames.is_empty() {
        return Ok(frames);
    }

    let raw_atoms = frames[0].len();
    if frames.iter().any(|f| f.len() != raw_atoms) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Inconsistent atom count across GRO frames",
        ));
    }

    if let Some(sel) = atom_indices {
        for &idx in sel {
            if idx >= raw_atoms {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Atom index {} out of bounds for GRO with {} atoms", idx, raw_atoms),
                ));
            }
        }
        let mut filtered_frames = Vec::with_capacity(frames.len());
        for frame in frames {
            let mut sub = Vec::with_capacity(sel.len());
            for &idx in sel {
                sub.push(frame[idx]);
            }
            filtered_frames.push(sub);
        }
        return Ok(filtered_frames);
    }

    Ok(frames)
}

fn parse_gro_coords_fallback(line: &str) -> Result<(f32, f32, f32), Error> {
    let parts: Vec<&str> = line.split_whitespace().collect();
    if parts.len() < 6 {
        return Err(Error::new(ErrorKind::InvalidData, format!("Malformed GRO atom line: {}", line)));
    }
    // GRO lines without velocities end with (x, y, z) [6 or 7 tokens].
    // GRO lines with velocities end with (x, y, z, vx, vy, vz) [9 or 10 tokens].
    let (x_str, y_str, z_str) = if parts.len() >= 9 {
        (parts[parts.len() - 6], parts[parts.len() - 5], parts[parts.len() - 4])
    } else {
        (parts[parts.len() - 3], parts[parts.len() - 2], parts[parts.len() - 1])
    };
    let x = x_str.parse::<f32>().map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    let y = y_str.parse::<f32>().map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    let z = z_str.parse::<f32>().map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
    Ok((x, y, z))
}

// ============================================================================
// Multi-model PDB Trajectory & Centroid Writer
// ============================================================================

pub fn write_pdb_trajectory<P: AsRef<Path>>(
    path: P,
    coords_flat: &[f32],
    num_frames: usize,
    num_atoms: usize,
) -> Result<(), Error> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    if num_atoms > 99999 {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("PDB format strictly limits atom serial numbers to 99,999 (got {}). Use mmCIF for larger structures.", num_atoms),
        ));
    }

    let frame_stride = num_atoms * 3;
    if coords_flat.len() < num_frames * frame_stride {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "Coordinates buffer smaller than num_frames * num_atoms * 3",
        ));
    }

    for f in 0..num_frames {
        if num_frames > 1 {
            writeln!(writer, "MODEL     {:4}", (f + 1).min(9999))?;
        }
        let f_offset = f * frame_stride;
        for a in 0..num_atoms {
            let base = f_offset + a * 3;
            let x = coords_flat[base];
            let y = coords_flat[base + 1];
            let z = coords_flat[base + 2];
            let atom_num = a + 1;
            let res_num = (a % 9999) + 1;
            writeln!(
                writer,
                "ATOM  {:5}  CA  GLY A{:4}    {:8.3}{:8.3}{:8.3}  1.00  0.00           C",
                atom_num, res_num, x, y, z
            )?;
        }
        if num_frames > 1 {
            writeln!(writer, "ENDMDL")?;
        }
    }
    writeln!(writer, "END")?;
    writer.flush()?;
    Ok(())
}
