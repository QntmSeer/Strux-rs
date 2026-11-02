// ponytail: pure-Rust memory-mapped GROMACS .xtc trajectory reader.
// Two-pass mmap header scan + Rayon multi-threaded XDR 3D frame decompression.
use std::fs::File;
use std::io::{Error, ErrorKind, Result};
use std::path::Path;
use memmap2::MmapOptions;
use rayon::prelude::*;

pub struct XtcTrajectory {
    pub num_frames: usize,
    pub num_atoms: usize,
    /// Coordinates in Angstroms [num_frames * num_atoms * 3]
    pub coords: Vec<f32>,
    pub times: Vec<f32>,
    pub steps: Vec<i32>,
}

const XTC_MAGIC: i32 = 1995;
const XTC_NEW_MAGIC: i32 = 2023;
const FIRSTIDX: usize = 9;

const MAGICINTS: [u32; 73] = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216,
];

#[inline]
fn sizeofint(size: u32) -> u32 {
    let mut num = 1u32;
    let mut num_of_bits = 0u32;
    while size >= num && num_of_bits < 32 {
        num_of_bits += 1;
        num <<= 1;
    }
    num_of_bits
}

#[inline]
fn sizeofints(num_of_ints: usize, sizes: &[u32]) -> u32 {
    let mut num_of_bytes = 1usize;
    let mut bytes = [0u32; 32];
    bytes[0] = 1;

    for i in 0..num_of_ints {
        let mut tmp = 0u64;
        for j in 0..num_of_bytes {
            tmp += (bytes[j] as u64) * (sizes[i] as u64);
            bytes[j] = (tmp & 0xff) as u32;
            tmp >>= 8;
        }
        while tmp != 0 {
            bytes[num_of_bytes] = (tmp & 0xff) as u32;
            num_of_bytes += 1;
            tmp >>= 8;
        }
    }

    let mut num = 1u32;
    let mut num_of_bits = 0u32;
    num_of_bytes -= 1;
    while bytes[num_of_bytes] >= num {
        num_of_bits += 1;
        num <<= 1;
    }
    num_of_bits + (num_of_bytes as u32) * 8
}

struct BitReader<'a> {
    cbuf: &'a [u8],
    cnt: usize,
    lastbits: u32,
    lastbyte: u32,
}

impl<'a> BitReader<'a> {
    #[inline]
    fn new(cbuf: &'a [u8]) -> Self {
        Self {
            cbuf,
            cnt: 0,
            lastbits: 0,
            lastbyte: 0,
        }
    }

    #[inline]
    fn receive_bits(&mut self, mut num_of_bits: u32) -> i32 {
        let mask = if num_of_bits >= 32 {
            u32::MAX
        } else {
            (1u32 << num_of_bits) - 1
        };
        let mut num = 0u32;
        while num_of_bits >= 8 {
            let b = if self.cnt < self.cbuf.len() {
                self.cbuf[self.cnt] as u32
            } else {
                0
            };
            self.cnt += 1;
            self.lastbyte = (self.lastbyte << 8) | b;
            num |= (self.lastbyte >> self.lastbits) << (num_of_bits - 8);
            num_of_bits -= 8;
        }
        if num_of_bits > 0 {
            if self.lastbits < num_of_bits {
                self.lastbits += 8;
                let b = if self.cnt < self.cbuf.len() {
                    self.cbuf[self.cnt] as u32
                } else {
                    0
                };
                self.cnt += 1;
                self.lastbyte = (self.lastbyte << 8) | b;
            }
            self.lastbits -= num_of_bits;
            num |= (self.lastbyte >> self.lastbits) & ((1u32 << num_of_bits) - 1);
        }
        (num & mask) as i32
    }

    #[inline]
    fn receive_ints(
        &mut self,
        num_of_ints: usize,
        mut num_of_bits: u32,
        sizes: &[u32; 3],
        nums: &mut [i32; 3],
    ) {
        let mut bytes = [0u32; 32];
        let mut num_of_bytes = 0usize;
        while num_of_bits > 8 && num_of_bytes < 31 {
            bytes[num_of_bytes] = self.receive_bits(8) as u32;
            num_of_bytes += 1;
            num_of_bits -= 8;
        }
        if num_of_bits > 0 && num_of_bytes < 32 {
            bytes[num_of_bytes] = self.receive_bits(num_of_bits) as u32;
            num_of_bytes += 1;
        }
        for i in (1..num_of_ints).rev() {
            let mut num = 0u64;
            let sz = (sizes[i] as u64).max(1);
            for j in (0..num_of_bytes).rev() {
                num = (num << 8) | (bytes[j] as u64);
                let p = num / sz;
                bytes[j] = p as u32;
                num -= p * sz;
            }
            nums[i] = num as i32;
        }
        nums[0] = (bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24)) as i32;
    }
}

struct FrameSlice {
    payload_offset: usize,
    comp_offset: usize,
    comp_len: usize,
    compressed: bool,
}

#[inline]
fn decode_xtc_frame(data: &[u8], frame: &FrameSlice, natoms: usize, out: &mut [f32]) {
    if !frame.compressed {
        let off = frame.payload_offset;
        for i in 0..(natoms * 3) {
            let val = f32::from_be_bytes(data[off + i * 4..off + (i + 1) * 4].try_into().unwrap());
            out[i] = val * 10.0;
        }
        return;
    }

    let off = frame.payload_offset;
    let precision = f32::from_be_bytes(data[off..off + 4].try_into().unwrap());
    let inv_precision = if precision.abs() > 1e-12 && precision.is_finite() {
        10.0 / precision
    } else {
        0.0
    };

    let minint = [
        i32::from_be_bytes(data[off + 4..off + 8].try_into().unwrap()),
        i32::from_be_bytes(data[off + 8..off + 12].try_into().unwrap()),
        i32::from_be_bytes(data[off + 12..off + 16].try_into().unwrap()),
    ];
    let maxint = [
        i32::from_be_bytes(data[off + 16..off + 20].try_into().unwrap()),
        i32::from_be_bytes(data[off + 20..off + 24].try_into().unwrap()),
        i32::from_be_bytes(data[off + 24..off + 28].try_into().unwrap()),
    ];
    let smallidx_raw = i32::from_be_bytes(data[off + 28..off + 32].try_into().unwrap());
    let comp_bytes = &data[frame.comp_offset..frame.comp_offset + frame.comp_len];

    let mut sizeint = [0u32; 3];
    let mut bitsizeint = [0u32; 3];
    for i in 0..3 {
        let diff = (maxint[i] as i64) - (minint[i] as i64) + 1;
        sizeint[i] = diff.clamp(1, u32::MAX as i64) as u32;
    }
    let bitsize = if (sizeint[0] | sizeint[1] | sizeint[2]) > 0xffffff {
        bitsizeint[0] = sizeofint(sizeint[0]);
        bitsizeint[1] = sizeofint(sizeint[1]);
        bitsizeint[2] = sizeofint(sizeint[2]);
        0
    } else {
        sizeofints(3, &sizeint)
    };

    let mut smallidx = smallidx_raw.clamp(0, (MAGICINTS.len() - 1) as i32) as usize;
    let mut smaller = MAGICINTS[FIRSTIDX.max(smallidx.saturating_sub(1))] / 2;
    let mut smallnum = MAGICINTS[smallidx] / 2;
    let mut sizesmall = [MAGICINTS[smallidx]; 3];

    let mut reader = BitReader::new(comp_bytes);
    let mut i = 0usize;
    let mut out_idx = 0usize;
    let mut thiscoord = [0i32; 3];
    let mut run = 0i32;

    while i < natoms {
        if bitsize == 0 {
            thiscoord[0] = reader.receive_bits(bitsizeint[0]);
            thiscoord[1] = reader.receive_bits(bitsizeint[1]);
            thiscoord[2] = reader.receive_bits(bitsizeint[2]);
        } else {
            reader.receive_ints(3, bitsize, &sizeint, &mut thiscoord);
        }
        i += 1;
        thiscoord[0] += minint[0];
        thiscoord[1] += minint[1];
        thiscoord[2] += minint[2];

        let mut prevcoord = thiscoord;

        let flag = reader.receive_bits(1);
        let mut is_smaller = 0i32;
        if flag == 1 {
            run = reader.receive_bits(5);
            is_smaller = run % 3;
            run -= is_smaller;
            is_smaller -= 1;
        }

        if run > 0 {
            let bits_small = smallidx as u32;
            let mut k = 0;
            while k < run && i < natoms {
                reader.receive_ints(3, bits_small, &sizesmall, &mut thiscoord);
                i += 1;
                thiscoord[0] += prevcoord[0] - (smallnum as i32);
                thiscoord[1] += prevcoord[1] - (smallnum as i32);
                thiscoord[2] += prevcoord[2] - (smallnum as i32);
                if k == 0 {
                    std::mem::swap(&mut thiscoord, &mut prevcoord);
                    out[out_idx] = (prevcoord[0] as f32) * inv_precision;
                    out[out_idx + 1] = (prevcoord[1] as f32) * inv_precision;
                    out[out_idx + 2] = (prevcoord[2] as f32) * inv_precision;
                    out_idx += 3;
                } else {
                    prevcoord = thiscoord;
                }
                out[out_idx] = (thiscoord[0] as f32) * inv_precision;
                out[out_idx + 1] = (thiscoord[1] as f32) * inv_precision;
                out[out_idx + 2] = (thiscoord[2] as f32) * inv_precision;
                out_idx += 3;
                k += 3;
            }
        } else {
            out[out_idx] = (thiscoord[0] as f32) * inv_precision;
            out[out_idx + 1] = (thiscoord[1] as f32) * inv_precision;
            out[out_idx + 2] = (thiscoord[2] as f32) * inv_precision;
            out_idx += 3;
        }

        smallidx = ((smallidx as i32) + is_smaller).clamp(0, (MAGICINTS.len() - 1) as i32) as usize;
        if is_smaller < 0 {
            smallnum = smaller;
            if smallidx > FIRSTIDX {
                smaller = MAGICINTS[smallidx - 1] / 2;
            } else {
                smaller = 0;
            }
        } else if is_smaller > 0 {
            smaller = smallnum;
            smallnum = MAGICINTS[smallidx] / 2;
        }
        sizesmall = [MAGICINTS[smallidx]; 3];
    }
}

pub fn parse_xtc_trajectory<P: AsRef<Path>>(path: P) -> Result<XtcTrajectory> {
    parse_xtc_trajectory_selected(path, None)
}

pub fn parse_xtc_trajectory_selected<P: AsRef<Path>>(
    path: P,
    atom_indices: Option<&[usize]>,
) -> Result<XtcTrajectory> {
    let file = File::open(path)?;
    if file.metadata()?.len() < 56 {
        return Err(Error::new(ErrorKind::UnexpectedEof, "XTC file too small for header"));
    }
    let mmap = unsafe { MmapOptions::new().map(&file)? };
    let data = &mmap[..];

    let mut offset = 0usize;
    let mut frames = Vec::new();
    let mut times_all = Vec::new();
    let mut steps_all = Vec::new();
    let mut num_atoms_global = 0usize;

    // Pass 1: Ultrafast sequential scan of XTC frame offsets
    while offset + 56 <= data.len() {
        let magic = i32::from_be_bytes(data[offset..offset + 4].try_into().unwrap());
        if magic != XTC_MAGIC && magic != XTC_NEW_MAGIC {
            if offset == 0 {
                return Err(Error::new(ErrorKind::InvalidData, "Invalid XTC magic header"));
            }
            break;
        }

        let natoms_raw = i32::from_be_bytes(data[offset + 4..offset + 8].try_into().unwrap());
        if natoms_raw <= 0 || natoms_raw > 50_000_000 {
            return Err(Error::new(ErrorKind::InvalidData, "Invalid atom count in XTC frame"));
        }
        let natoms = natoms_raw as usize;
        let step = i32::from_be_bytes(data[offset + 8..offset + 12].try_into().unwrap());
        let time = f32::from_be_bytes(data[offset + 12..offset + 16].try_into().unwrap());
        offset += 56; // 16 header + 36 box + 4 lsize

        if num_atoms_global == 0 {
            num_atoms_global = natoms;
        } else if natoms != num_atoms_global {
            return Err(Error::new(ErrorKind::InvalidData, "Inconsistent atom count in XTC frame"));
        }

        if natoms <= 9 {
            let nbytes = natoms * 3 * 4;
            if offset.saturating_add(nbytes) > data.len() {
                break;
            }
            frames.push(FrameSlice {
                payload_offset: offset,
                comp_offset: offset,
                comp_len: nbytes,
                compressed: false,
            });
            offset += nbytes;
        } else {
            let header_len = if magic == XTC_NEW_MAGIC { 40 } else { 36 };
            if offset.saturating_add(header_len) > data.len() {
                break;
            }
            let nbytes = if magic == XTC_NEW_MAGIC {
                let nb = u64::from_be_bytes(data[offset + 32..offset + 40].try_into().unwrap());
                usize::try_from(nb).unwrap_or(usize::MAX)
            } else {
                let nb = i32::from_be_bytes(data[offset + 32..offset + 36].try_into().unwrap());
                if nb < 0 {
                    return Err(Error::new(ErrorKind::InvalidData, "Negative compressed payload size in XTC"));
                }
                nb as usize
            };
            let comp_offset = offset + header_len;
            let Some(end_offset) = comp_offset.checked_add(nbytes) else {
                break;
            };
            if end_offset > data.len() {
                break;
            }
            frames.push(FrameSlice {
                payload_offset: offset,
                comp_offset,
                comp_len: nbytes,
                compressed: true,
            });
            let padded_nbytes = nbytes.saturating_add(3) & !3;
            offset = comp_offset.saturating_add(padded_nbytes);
        }

        times_all.push(time);
        steps_all.push(step);
    }

    if let Some(sel) = atom_indices {
        for &idx in sel {
            if idx >= num_atoms_global {
                return Err(Error::new(
                    ErrorKind::InvalidInput,
                    format!("Atom index {} out of bounds for XTC with {} atoms", idx, num_atoms_global),
                ));
            }
        }
    }

    let num_frames = frames.len();
    let out_atoms = atom_indices.map(|s| s.len()).unwrap_or(num_atoms_global);
    let out_stride = out_atoms * 3;
    let mut coords_all = vec![0.0f32; num_frames * out_stride];

    // Pass 2: Multi-threaded Rayon decompression across all frames simultaneously
    if out_stride > 0 {
        if let Some(sel) = atom_indices {
            coords_all
                .par_chunks_mut(out_stride)
                .zip(frames.par_iter())
                .for_each_init(
                    || vec![0.0f32; num_atoms_global * 3],
                    |scratch, (out_slice, frame_meta)| {
                        decode_xtc_frame(data, frame_meta, num_atoms_global, scratch);
                        for (k, &atom_idx) in sel.iter().enumerate() {
                            let src = atom_idx * 3;
                            let dst = k * 3;
                            out_slice[dst] = scratch[src];
                            out_slice[dst + 1] = scratch[src + 1];
                            out_slice[dst + 2] = scratch[src + 2];
                        }
                    },
                );
        } else {
            coords_all
                .par_chunks_mut(out_stride)
                .zip(frames.par_iter())
                .for_each(|(out_slice, frame_meta)| {
                    decode_xtc_frame(data, frame_meta, num_atoms_global, out_slice);
                });
        }
    }

    Ok(XtcTrajectory {
        num_frames,
        num_atoms: out_atoms,
        coords: coords_all,
        times: times_all,
        steps: steps_all,
    })
}
