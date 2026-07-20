// ponytail: fast A3M/Stockholm parsing with single-pass and Rayon.
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
use rayon::prelude::*;

#[pyclass]
#[derive(Clone)]
pub struct Msa {
    #[pyo3(get)]
    pub sequences: Vec<String>,
    #[pyo3(get)]
    pub descriptions: Vec<String>,
    pub deletion_matrix_flat: Vec<i32>,
    pub num_seqs: usize,
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
}

pub fn parse_fasta(fasta_string: &str) -> (Vec<String>, Vec<String>) {
    let mut sequences = Vec::new();
    let mut descriptions = Vec::new();
    let mut current_seq = String::new();

    for line in fasta_string.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('>') {
            if !descriptions.is_empty() {
                sequences.push(current_seq);
                current_seq = String::new();
            }
            descriptions.push(line[1..].to_string());
        } else {
            current_seq.push_str(line);
        }
    }
    if !descriptions.is_empty() {
        sequences.push(current_seq);
    }
    (sequences, descriptions)
}

pub fn parse_a3m_impl(a3m_string: &str) -> Msa {
    let (sequences, descriptions) = parse_fasta(a3m_string);
    if sequences.is_empty() {
        return Msa {
            sequences: Vec::new(),
            descriptions: Vec::new(),
            deletion_matrix_flat: Vec::new(),
            num_seqs: 0,
            num_res: 0,
        };
    }

    // Process all sequences in parallel using Rayon.
    let results: Vec<(String, Vec<i32>)> = sequences
        .par_iter()
        .map(|seq| {
            let bytes = seq.as_bytes();
            let mut aligned_seq = String::with_capacity(bytes.len());
            let mut deletion_vec = Vec::with_capacity(bytes.len());
            let mut deletion_count = 0;

            for &b in bytes {
                if b.is_ascii_lowercase() {
                    deletion_count += 1;
                } else {
                    aligned_seq.push(b as char);
                    deletion_vec.push(deletion_count);
                    deletion_count = 0;
                }
            }
            (aligned_seq, deletion_vec)
        })
        .collect();

    let num_seqs = results.len();
    let num_res = results[0].1.len();

    let mut aligned_sequences = Vec::with_capacity(num_seqs);
    let mut deletion_matrix_flat = Vec::with_capacity(num_seqs * num_res);

    for (aligned_seq, deletion_vec) in results {
        aligned_sequences.push(aligned_seq);
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

pub fn parse_stockholm_impl(stockholm_string: &str) -> Msa {
    let mut names = Vec::new();
    let mut name_to_index: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    let mut raw_sequences: Vec<String> = Vec::new();

    for line in stockholm_string.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
            continue;
        }
        let mut parts = line.split_whitespace();
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

    // Process all sequences in parallel using Rayon.
    let results: Vec<(String, Vec<i32>)> = raw_sequences
        .par_iter()
        .map(|seq| {
            let seq_bytes = seq.as_bytes();
            let mut aligned_seq = String::with_capacity(num_res);
            let mut deletion_vec = Vec::with_capacity(num_res);
            let mut deletion_count = 0;

            // 1. Build aligned sequence.
            for &idx in &keep_columns {
                if idx < seq_bytes.len() {
                    aligned_seq.push(seq_bytes[idx] as char);
                } else {
                    aligned_seq.push('-');
                }
            }

            // 2. Count deletions.
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

            // Pad deletion_vec if sequence was shorter than query.
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
