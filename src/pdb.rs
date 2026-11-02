// ponytail: simple line-by-line PDB parser, minimal allocations.
use std::fs::File;
use std::io::{BufRead, BufReader, Error, ErrorKind};
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
