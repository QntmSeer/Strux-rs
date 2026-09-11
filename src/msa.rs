// ponytail: fast A3M/Stockholm parsing with 100% multi-threaded Rayon record splitting & memmap2.
use pyo3::prelude::*;
use numpy::{PyArray2, PyArrayMethods};
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
