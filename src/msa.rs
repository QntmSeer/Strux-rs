// ponytail: fast A3M/Stockholm parsing with 100% multi-threaded Rayon record splitting & memmap2.
use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use rayon::prelude::*;
use std::fs::File;
use std::path::Path;
use memmap2::MmapOptions;

#[pyclass]
#[derive(Clone)]
pub struct Msa {
    #[pyo3(get)]
    pub sequences: Vec<String>,
    #[pyo3(get)]
    pub descriptions: Vec<String>,
    pub deletion_matrix_flat: Vec<i32>,
    #[pyo3(get)]
    pub num_seqs: usize,
    #[pyo3(get)]
    pub num_res: usize,
}


#[pymethods]
impl Msa {
    #[new]
    fn new(
        sequences: Vec<String>,
        deletion_matrix: Vec<Vec<i32>>,
        descriptions: Vec<String>,
    ) -> PyResult<Self> {
        let num_seqs = sequences.len();
        if num_seqs != deletion_matrix.len() || num_seqs != descriptions.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "All fields for an MSA must have the same length.",
            ));
        }
        let num_res = if num_seqs > 0 { deletion_matrix[0].len() } else { 0 };
        let mut deletion_matrix_flat = Vec::with_capacity(num_seqs * num_res);
        for row in deletion_matrix {
            if row.len() != num_res {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "All rows in the deletion matrix must have the same length.",
                ));
            }
            deletion_matrix_flat.extend(row);
        }
        Ok(Msa {
            sequences,
            descriptions,
            deletion_matrix_flat,
            num_seqs,
            num_res,
        })
    }

    #[getter]
    fn deletion_matrix(&self) -> Vec<Vec<i32>> {
        let mut matrix = Vec::with_capacity(self.num_seqs);
        for i in 0..self.num_seqs {
            let start = i * self.num_res;
            let end = start + self.num_res;
            matrix.push(self.deletion_matrix_flat[start..end].to_vec());
        }
        matrix
    }

    #[getter]
    fn deletion_matrix_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i32>>> {
        let array = PyArray2::<i32>::zeros_bound(py, [self.num_seqs, self.num_res], false);
        {
            let mut writer = array.readwrite();
            let slice = writer.as_slice_mut().unwrap();
            slice.copy_from_slice(&self.deletion_matrix_flat);
        }
        Ok(array)
    }

    /// Converts all sequences directly into an [N_seq, N_res] uint8 ASCII token matrix
    #[getter]
    fn tokens_np<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<u8>>> {
        let array = PyArray2::<u8>::zeros_bound(py, [self.num_seqs, self.num_res], false);
        {
            let mut writer = array.readwrite();
            let slice = writer.as_slice_mut().unwrap();
            for (i, seq) in self.sequences.iter().enumerate() {
                let row_start = i * self.num_res;
                let seq_bytes = seq.as_bytes();
                let copy_len = std::cmp::min(seq_bytes.len(), self.num_res);
                slice[row_start..row_start + copy_len].copy_from_slice(&seq_bytes[..copy_len]);
            }
        }
        Ok(array)
    }


    fn truncate(&self, max_seqs: usize) -> Self {
        let limit = std::cmp::min(max_seqs, self.num_seqs);
        let sequences = self.sequences[..limit].to_vec();
        let descriptions = self.descriptions[..limit].to_vec();
        let deletion_matrix_flat = self.deletion_matrix_flat[..limit * self.num_res].to_vec();
        Msa {
            sequences,
            descriptions,
            deletion_matrix_flat,
            num_seqs: limit,
            num_res: self.num_res,
        }
    }

    fn __len__(&self) -> usize {
        self.num_seqs
    }

    /// Clusters sequences by pairwise sequence identity (e.g. min_identity = 0.70).
    /// Used by AF-Cluster to generate diverse conformational sub-alignments.
    /// Returns (labels, cluster_representatives).
    #[pyo3(signature = (min_identity=0.7))]
    fn cluster<'py>(
        &self,
        py: Python<'py>,
        min_identity: f32,
    ) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
        let (labels, reps) = cluster_sequences(&self.sequences, min_identity);
        let py_labels = PyArray1::from_vec_bound(py, labels);
        let py_reps = PyArray1::from_vec_bound(py, reps);
        Ok((py_labels, py_reps))
    }

    /// Extracts an AF-Cluster sub-MSA containing only sequences assigned to cluster_id.
    fn get_cluster(&self, cluster_id: usize, labels: Vec<usize>) -> PyResult<Self> {
        let mut new_seqs = Vec::new();
        let mut new_descs = Vec::new();
        let mut new_del = Vec::new();

        for (i, &lbl) in labels.iter().enumerate() {
            if lbl == cluster_id && i < self.num_seqs {
                new_seqs.push(self.sequences[i].clone());
                new_descs.push(self.descriptions[i].clone());
                let start = i * self.num_res;
                let end = start + self.num_res;
                new_del.extend_from_slice(&self.deletion_matrix_flat[start..end]);
            }
        }

        let num_seqs = new_seqs.len();
        Ok(Msa {
            sequences: new_seqs,
            descriptions: new_descs,
            deletion_matrix_flat: new_del,
            num_seqs,
            num_res: self.num_res,
        })
    }
}

/// Clusters MSA sequences by pairwise identity using greedy representative clustering.
/// Returns (cluster_labels: Vec<usize>, cluster_representatives: Vec<usize>).
pub fn cluster_sequences(
    sequences: &[String],
    min_identity: f32,
) -> (Vec<usize>, Vec<usize>) {
    let num_seqs = sequences.len();
    if num_seqs == 0 {
        return (Vec::new(), Vec::new());
    }

    let seq_bytes: Vec<&[u8]> = sequences.iter().map(|s| s.as_bytes()).collect();
    let mut labels = vec![usize::MAX; num_seqs];
    let mut representatives = Vec::new();

    for i in 0..num_seqs {
        let seq_i = seq_bytes[i];
        let len_i = seq_i.len();
        if len_i == 0 {
            continue;
        }

        let mut matched_cluster = None;
        for (cluster_id, &rep_idx) in representatives.iter().enumerate() {
            let rep_seq: &[u8] = seq_bytes[rep_idx];
            let compare_len = std::cmp::min(len_i, rep_seq.len());
            if compare_len == 0 {
                continue;
            }

            let mut matches = 0usize;
            let mut valid_positions = 0usize;

            for p in 0..compare_len {
                let a = seq_i[p];
                let b = rep_seq[p];
                if a == b'-' && b == b'-' {
                    continue;
                }
                valid_positions += 1;
                if a == b && a != b'-' {
                    matches += 1;
                }
            }

            let identity = if valid_positions > 0 {
                matches as f32 / valid_positions as f32
            } else {
                0.0
            };

            if identity >= min_identity {
                matched_cluster = Some(cluster_id);
                break;
            }
        }

        match matched_cluster {
            Some(cid) => {
                labels[i] = cid;
            }
            None => {
                let new_cid = representatives.len();
                representatives.push(i);
                labels[i] = new_cid;
            }
        }
    }

    (labels, representatives)
}

#[inline]
fn trim_bytes(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && (bytes[start] == b' ' || bytes[start] == b'\t' || bytes[start] == b'\r') {
        start += 1;
    }
    while end > start && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t' || bytes[end - 1] == b'\r') {
        end -= 1;
    }
    &bytes[start..end]
}

pub fn parse_a3m_impl(a3m_string: &str) -> Msa {
    parse_a3m_bytes(a3m_string.as_bytes())
}

pub fn parse_a3m_bytes(bytes: &[u8]) -> Msa {
    // 1. Split raw bytes by FASTA record delimiter '>'
    let records: Vec<&[u8]> = bytes
        .split(|&b| b == b'>')
        .filter(|r| !trim_bytes(r).is_empty())
        .collect();

    if records.is_empty() {
        return Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        };
    }

    // 2. Parse descriptions, sequence cleaning, and deletion counting 100% in parallel
    let results: Vec<(String, String, Vec<i32>)> = records
        .par_iter()
        .map(|record| {
            let record_trimmed = trim_bytes(record);
            // Split into header line and sequence body lines
            let mut line_iter = record_trimmed.split(|&b| b == b'\n');
            let header_raw = line_iter.next().unwrap_or(&[]);
            let header = String::from_utf8_lossy(trim_bytes(header_raw)).to_string();

            let mut aligned_seq = String::with_capacity(record_trimmed.len());
            let mut deletion_vec = Vec::with_capacity(record_trimmed.len());
            let mut deletion_count = 0;

            for seq_line in line_iter {
                let trimmed_line = trim_bytes(seq_line);
                if trimmed_line.is_empty() || trimmed_line[0] == b'#' {
                    continue;
                }
                for &b in trimmed_line {
                    if b.is_ascii_lowercase() {
                        deletion_count += 1;
                    } else {
                        aligned_seq.push(b as char);
                        deletion_vec.push(deletion_count);
                        deletion_count = 0;
                    }
                }
            }
            (header, aligned_seq, deletion_vec)
        })
        .collect();

    let num_seqs = results.len();
    if num_seqs == 0 {
        return Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        };
    }
    let num_res = results[0].2.len();

    let mut descriptions = Vec::with_capacity(num_seqs);
    let mut aligned_sequences = Vec::with_capacity(num_seqs);
    let mut deletion_matrix_flat = Vec::with_capacity(num_seqs * num_res);

    for (header, aligned_seq, mut deletion_vec) in results {
        descriptions.push(header);
        aligned_sequences.push(aligned_seq);
        if deletion_vec.len() < num_res {
            deletion_vec.resize(num_res, 0);
        } else if deletion_vec.len() > num_res {
            deletion_vec.truncate(num_res);
        }
        deletion_matrix_flat.extend(deletion_vec);
    }

    Msa {
        sequences: aligned_sequences,
        descriptions,
        deletion_matrix_flat,
        num_seqs,
        num_res,
    }
}

pub fn parse_a3m_file_impl<P: AsRef<Path>>(path: P) -> Result<Msa, std::io::Error> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        return Ok(Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        });
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(parse_a3m_bytes(&mmap[..]))
}

pub fn parse_stockholm_impl(stockholm_string: &str) -> Msa {
    parse_stockholm_bytes(stockholm_string.as_bytes())
}

pub fn parse_stockholm_bytes(bytes: &[u8]) -> Msa {
    let mut names = Vec::new();
    let mut name_to_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut raw_sequences: Vec<String> = Vec::new();

    for raw_line in bytes.split(|&b| b == b'\n') {
        let line = trim_bytes(raw_line);
        if line.is_empty() || line[0] == b'#' || line.starts_with(b"//") {
            continue;
        }
        let line_str = String::from_utf8_lossy(line);
        let mut parts = line_str.split_whitespace();
        if let (Some(name), Some(sequence)) = (parts.next(), parts.next()) {
            if let Some(&idx) = name_to_index.get(name) {
                raw_sequences[idx].push_str(sequence);
            } else {
                name_to_index.insert(name.to_string(), raw_sequences.len());
                names.push(name.to_string());
                raw_sequences.push(sequence.to_string());
            }
        }
    }

    if raw_sequences.is_empty() {
        return Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        };
    }

    let query = &raw_sequences[0];
    let query_bytes = query.as_bytes();

    let keep_columns: Vec<usize> = query_bytes
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b != b'-')
        .map(|(i, _)| i)
        .collect();

    let num_res = keep_columns.len();

    let results: Vec<(String, Vec<i32>)> = raw_sequences
        .par_iter()
        .map(|seq| {
            let seq_bytes = seq.as_bytes();
            let mut aligned_seq = String::with_capacity(num_res);
            let mut deletion_vec = Vec::with_capacity(num_res);
            let mut deletion_count = 0;

            for &idx in &keep_columns {
                if idx < seq_bytes.len() {
                    aligned_seq.push(seq_bytes[idx] as char);
                } else {
                    aligned_seq.push('-');
                }
            }

            let len = std::cmp::min(seq_bytes.len(), query_bytes.len());
            for idx in 0..len {
                let seq_res = seq_bytes[idx];
                let query_res = query_bytes[idx];
                if seq_res != b'-' || query_res != b'-' {
                    if query_res == b'-' {
                        deletion_count += 1;
                    } else {
                        deletion_vec.push(deletion_count);
                        deletion_count = 0;
                    }
                }
            }

            while deletion_vec.len() < num_res {
                deletion_vec.push(0);
            }

            (aligned_seq, deletion_vec)
        })
        .collect();

    let num_seqs = results.len();
    let mut aligned_sequences = Vec::with_capacity(num_seqs);
    let mut deletion_matrix_flat = Vec::with_capacity(num_seqs * num_res);

    for (aligned_seq, deletion_vec) in results {
        aligned_sequences.push(aligned_seq);
        deletion_matrix_flat.extend(deletion_vec);
    }

    Msa {
        sequences: aligned_sequences,
        descriptions: names,
        deletion_matrix_flat,
        num_seqs,
        num_res,
    }
}

pub fn parse_stockholm_file_impl<P: AsRef<Path>>(path: P) -> Result<Msa, std::io::Error> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;
    if metadata.len() == 0 {
        return Ok(Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        });
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    Ok(parse_stockholm_bytes(&mmap[..]))
}
