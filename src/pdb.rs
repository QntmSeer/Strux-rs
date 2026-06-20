// ponytail: simple line-by-line PDB parser, minimal allocations.
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub fn parse_pdb_trajectory<P: AsRef<Path>>(
    path: P,
) -> Result<Vec<Vec<[f32; 3]>>, std::io::Error> {
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
            if line.len() >= 54 {
                let x_str = line[30..38].trim();
                let y_str = line[38..46].trim();
                let z_str = line[46..54].trim();
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

    Ok(frames)
}
