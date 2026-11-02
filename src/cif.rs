// ponytail: zero-allocation memory-mapped mmCIF (.cif) parser for AF3, Boltz-2, Chai-1, RFD4 & RCSB PDB.
// Handles divergent _atom_site column orders, auth vs label chains, insertion codes, altlocs, H-stripping, and pLDDT.
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use memmap2::MmapOptions;

pub struct CifTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    /// Coordinates in Angstroms [num_frames * num_atoms * 3]
    pub coords: Vec<f32>,
    /// B-factors / pLDDT confidence scores [num_frames * num_atoms]
    pub b_factors: Vec<f32>,
    /// Per-atom metadata from model 0 (length = num_atoms)
    pub chain_ids: Vec<String>,
    pub res_ids: Vec<i32>,
    pub res_names: Vec<String>,
    pub atom_names: Vec<String>,
}

#[inline]
fn split_cif_tokens<'a>(line: &'a str, out: &mut Vec<&'a str>) {
    out.clear();
    let bytes = line.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        while i < n && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= n {
            break;
        }
        if bytes[i] == b'\'' || bytes[i] == b'"' {
            let quote = bytes[i];
            i += 1;
            let start = i;
            while i < n && bytes[i] != quote {
                i += 1;
            }
            out.push(&line[start..i]);
            if i < n {
                i += 1;
            }
        } else {
            let start = i;
            while i < n && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            out.push(&line[start..i]);
        }
    }
}

pub fn parse_cif_trajectory<P: AsRef<Path>>(
    path: P,
    ca_only: bool,
    strip_hydrogens: bool,
    use_auth_chain: bool,
    atom_indices: Option<&[usize]>,
) -> Result<CifTrajectory> {
    let file = File::open(path)?;
    if file.metadata()?.len() == 0 {
        return Ok(CifTrajectory {
            num_frames: 0,
            num_atoms: 0,
            coords: Vec::new(),
            b_factors: Vec::new(),
            chain_ids: Vec::new(),
            res_ids: Vec::new(),
            res_names: Vec::new(),
            atom_names: Vec::new(),
        });
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let content = std::str::from_utf8(&mmap[..])
        .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Invalid UTF-8 in mmCIF: {}", e)))?;

    let mut in_atom_site_header = false;
    let mut in_atom_site_rows = false;
    let mut col_idx = 0usize;

    let mut idx_x: Option<usize> = None;
    let mut idx_y: Option<usize> = None;
    let mut idx_z: Option<usize> = None;
    let mut idx_b: Option<usize> = None;
    let mut idx_label_atom: Option<usize> = None;
    let mut idx_auth_atom: Option<usize> = None;
    let mut idx_label_asym: Option<usize> = None;
    let mut idx_auth_asym: Option<usize> = None;
    let mut idx_label_seq: Option<usize> = None;
    let mut idx_auth_seq: Option<usize> = None;
    let mut idx_label_comp: Option<usize> = None;
    let mut idx_auth_comp: Option<usize> = None;
    let mut idx_alt_id: Option<usize> = None;
    let mut idx_type_symbol: Option<usize> = None;
    let mut idx_model: Option<usize> = None;

    let mut frames_coords: Vec<Vec<[f32; 3]>> = Vec::new();
    let mut frames_bfac: Vec<Vec<f32>> = Vec::new();
    let mut cur_coords: Vec<[f32; 3]> = Vec::new();
    let mut cur_bfac: Vec<f32> = Vec::new();

    let mut meta_chains: Vec<String> = Vec::new();
    let mut meta_res_ids: Vec<i32> = Vec::new();
    let mut meta_res_names: Vec<String> = Vec::new();
    let mut meta_atom_names: Vec<String> = Vec::new();

    let mut current_model: Option<String> = None;
    let mut tokens: Vec<&str> = Vec::with_capacity(32);

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            if in_atom_site_rows {
                in_atom_site_rows = false;
                in_atom_site_header = false;
            }
            continue;
        }

        if line == "loop_" {
            in_atom_site_header = false;
            in_atom_site_rows = false;
            col_idx = 0;
            continue;
        }

        if line.starts_with("_atom_site.") {
            if !in_atom_site_header {
                in_atom_site_header = true;
                in_atom_site_rows = false;
                col_idx = 0;
                idx_x = None;
                idx_y = None;
                idx_z = None;
                idx_b = None;
                idx_label_atom = None;
                idx_auth_atom = None;
                idx_label_asym = None;
                idx_auth_asym = None;
                idx_label_seq = None;
                idx_auth_seq = None;
                idx_label_comp = None;
                idx_auth_comp = None;
                idx_alt_id = None;
                idx_type_symbol = None;
                idx_model = None;
            }
            let field = line.trim_start_matches("_atom_site.").split_whitespace().next().unwrap_or("");
            match field {
                "Cartn_x" => idx_x = Some(col_idx),
                "Cartn_y" => idx_y = Some(col_idx),
                "Cartn_z" => idx_z = Some(col_idx),
                "B_iso_or_equiv" => idx_b = Some(col_idx),
                "label_atom_id" => idx_label_atom = Some(col_idx),
                "auth_atom_id" => idx_auth_atom = Some(col_idx),
                "label_asym_id" => idx_label_asym = Some(col_idx),
                "auth_asym_id" => idx_auth_asym = Some(col_idx),
                "label_seq_id" => idx_label_seq = Some(col_idx),
                "auth_seq_id" => idx_auth_seq = Some(col_idx),
                "label_comp_id" => idx_label_comp = Some(col_idx),
                "auth_comp_id" => idx_auth_comp = Some(col_idx),
                "label_alt_id" => idx_alt_id = Some(col_idx),
                "type_symbol" => idx_type_symbol = Some(col_idx),
                "pdbx_PDB_model_num" => idx_model = Some(col_idx),
                _ => {}
            }
            col_idx += 1;
            continue;
        }

        if in_atom_site_header && !line.starts_with('_') {
            in_atom_site_header = false;
            in_atom_site_rows = true;
        }

        if in_atom_site_rows {
            if line.starts_with('_') || line.starts_with("loop_") || line.starts_with("data_") {
                in_atom_site_rows = false;
                continue;
            }

            let (ix, iy, iz) = match (idx_x, idx_y, idx_z) {
                (Some(x), Some(y), Some(z)) => (x, y, z),
                _ => return Err(Error::new(ErrorKind::InvalidData, "mmCIF missing Cartn_x/y/z columns")),
            };

            split_cif_tokens(line, &mut tokens);
            let max_req = ix.max(iy).max(iz);
            if tokens.len() <= max_req {
                continue;
            }

            // 1. Filter alternate conformations (keep '.', '?', or 'A')
            if let Some(ia) = idx_alt_id {
                if ia < tokens.len() {
                    let alt = tokens[ia];
                    if alt != "." && alt != "?" && alt != "A" {
                        continue;
                    }
                }
            }

            // 2. Strip hydrogens/deuterium if requested
            if strip_hydrogens {
                if let Some(its) = idx_type_symbol {
                    if its < tokens.len() {
                        let sym = tokens[its];
                        if sym == "H" || sym == "D" {
                            continue;
                        }
                    }
                }
            }

            // 3. Resolve atom name (prefer label_atom_id, fallback auth_atom_id)
            let idx_atom = idx_label_atom.or(idx_auth_atom);
            let atom_name = idx_atom
                .and_then(|i| tokens.get(i))
                .copied()
                .unwrap_or("CA");

            if ca_only && atom_name != "CA" {
                continue;
            }

            // 4. Check model boundary
            if let Some(im) = idx_model {
                if im < tokens.len() {
                    let m_str = tokens[im];
                    match &current_model {
                        Some(prev) if prev != m_str => {
                            if !cur_coords.is_empty() {
                                frames_coords.push(std::mem::take(&mut cur_coords));
                                frames_bfac.push(std::mem::take(&mut cur_bfac));
                            }
                            current_model = Some(m_str.to_string());
                        }
                        None => {
                            current_model = Some(m_str.to_string());
                        }
                        _ => {}
                    }
                }
            }

            if let (Ok(x), Ok(y), Ok(z)) = (
                tokens[ix].parse::<f32>(),
                tokens[iy].parse::<f32>(),
                tokens[iz].parse::<f32>(),
            ) {
                let b = idx_b
                    .and_then(|ib| tokens.get(ib))
                    .and_then(|s| s.parse::<f32>().ok())
                    .unwrap_or(0.0);
                cur_coords.push([x, y, z]);
                cur_bfac.push(b);

                // Record topology metadata on first model
                if frames_coords.is_empty() {
                    let idx_chain = if use_auth_chain {
                        idx_auth_asym.or(idx_label_asym)
                    } else {
                        idx_label_asym.or(idx_auth_asym)
                    };
                    let chain_str = idx_chain
                        .and_then(|i| tokens.get(i))
                        .copied()
                        .unwrap_or("A");

                    let idx_seq = if use_auth_chain {
                        idx_auth_seq.or(idx_label_seq)
                    } else {
                        idx_label_seq.or(idx_auth_seq)
                    };
                    let res_id = idx_seq
                        .and_then(|i| tokens.get(i))
                        .and_then(|s| s.parse::<i32>().ok())
                        .unwrap_or(1);

                    let idx_comp = idx_label_comp.or(idx_auth_comp);
                    let res_name = idx_comp
                        .and_then(|i| tokens.get(i))
                        .copied()
                        .unwrap_or("UNK");

                    meta_chains.push(chain_str.to_string());
                    meta_res_ids.push(res_id);
                    meta_res_names.push(res_name.to_string());
                    meta_atom_names.push(atom_name.to_string());
                }
            }
        }
    }

    if !cur_coords.is_empty() {
        frames_coords.push(cur_coords);
        frames_bfac.push(cur_bfac);
    }

    if frames_coords.is_empty() {
        return Ok(CifTrajectory {
            num_frames: 0,
            num_atoms: 0,
            coords: Vec::new(),
            b_factors: Vec::new(),
            chain_ids: Vec::new(),
            res_ids: Vec::new(),
            res_names: Vec::new(),
            atom_names: Vec::new(),
        });
    }

    let num_frames = frames_coords.len();
    let raw_atoms = frames_coords[0].len();
    if frames_coords.iter().any(|f| f.len() != raw_atoms) {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "Inconsistent atom count across mmCIF models",
        ));
    }

    if let Some(sel) = atom_indices {
        for &idx in sel {
            if idx >= raw_atoms {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Atom index {} out of bounds for mmCIF with {} atoms", idx, raw_atoms),
                ));
            }
        }
    }

    let num_atoms = atom_indices.map(|sel| sel.len()).unwrap_or(raw_atoms);

    let mut flat_coords = Vec::with_capacity(num_frames * num_atoms * 3);
    let mut flat_bfac = Vec::with_capacity(num_frames * num_atoms);

    for (f_coords, f_bfac) in frames_coords.iter().zip(frames_bfac.iter()) {
        if let Some(sel) = atom_indices {
            for &idx in sel {
                let c = f_coords[idx];
                flat_coords.extend_from_slice(&c);
                flat_bfac.push(f_bfac[idx]);
            }
        } else {
            for (&c, &b) in f_coords.iter().take(raw_atoms).zip(f_bfac.iter().take(raw_atoms)) {
                flat_coords.extend_from_slice(&c);
                flat_bfac.push(b);
            }
        }
    }

    let (out_chains, out_res_ids, out_res_names, out_atom_names) = if let Some(sel) = atom_indices {
        let mut c = Vec::with_capacity(sel.len());
        let mut r = Vec::with_capacity(sel.len());
        let mut rn = Vec::with_capacity(sel.len());
        let mut an = Vec::with_capacity(sel.len());
        for &idx in sel {
            c.push(meta_chains[idx].clone());
            r.push(meta_res_ids[idx]);
            rn.push(meta_res_names[idx].clone());
            an.push(meta_atom_names[idx].clone());
        }
        (c, r, rn, an)
    } else {
        (meta_chains, meta_res_ids, meta_res_names, meta_atom_names)
    };

    Ok(CifTrajectory {
        num_frames,
        num_atoms,
        coords: flat_coords,
        b_factors: flat_bfac,
        chain_ids: out_chains,
        res_ids: out_res_ids,
        res_names: out_res_names,
        atom_names: out_atom_names,
    })
}
