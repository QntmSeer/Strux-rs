// ponytail: simple spatial hashing with PBC support.
use std::collections::HashMap;

pub struct CellList {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cutoff: f32,
    box_dims: Option<[f32; 3]>,
}

impl CellList {
    pub fn build(coords: &[[f32; 3]], cutoff: f32, box_dims: Option<[f32; 3]>) -> Self {
        let safe_cutoff = if cutoff.is_finite() && cutoff > 1e-3 { cutoff } else { 1e-3 };
        let mut cells = HashMap::new();
        for (idx, &pos) in coords.iter().enumerate() {
            let cell_coord = Self::get_cell_coord(pos, safe_cutoff, box_dims);
            cells.entry(cell_coord).or_insert_with(Vec::new).push(idx);
        }
        CellList {
            cells,
            cutoff: safe_cutoff,
            box_dims,
        }
    }

    fn get_cell_coord(pos: [f32; 3], cutoff: f32, box_dims: Option<[f32; 3]>) -> (i32, i32, i32) {
        if !pos[0].is_finite() || !pos[1].is_finite() || !pos[2].is_finite() {
            return (0, 0, 0);
        }
        let mut p = pos;
        if let Some(box_l) = box_dims {
            let mut c = [0i32; 3];
            for i in 0..3 {
                let l = box_l[i];
                if l > 0.0 && l.is_finite() {
                    p[i] %= l;
                    if p[i] < 0.0 {
                        p[i] += l;
                    }
                    let n = (l / cutoff).floor().clamp(1.0, (i32::MAX / 2) as f32) as i32;
                    let cell_w = l / (n as f32);
                    let idx = (p[i] / cell_w).floor() as i32;
                    c[i] = idx.rem_euclid(n);
                } else {
                    c[i] = (p[i] / cutoff).floor().clamp((i32::MIN / 2) as f32, (i32::MAX / 2) as f32) as i32;
                }
            }
            return (c[0], c[1], c[2]);
        }
        let cx = (p[0] / cutoff).floor().clamp((i32::MIN / 2) as f32, (i32::MAX / 2) as f32) as i32;
        let cy = (p[1] / cutoff).floor().clamp((i32::MIN / 2) as f32, (i32::MAX / 2) as f32) as i32;
        let cz = (p[2] / cutoff).floor().clamp((i32::MIN / 2) as f32, (i32::MAX / 2) as f32) as i32;
        (cx, cy, cz)
    }

    pub fn query_neighbors(&self, pos: [f32; 3], coords: &[[f32; 3]]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let center_cell = Self::get_cell_coord(pos, self.cutoff, self.box_dims);
        let cutoff_sq = self.cutoff * self.cutoff;

        let num_cells = if let Some(box_l) = self.box_dims {
            [0, 1, 2].map(|i| {
                let l = box_l[i];
                if l > 0.0 && l.is_finite() {
                    (l / self.cutoff).floor().clamp(1.0, (i32::MAX / 2) as f32) as i32
                } else {
                    0
                }
            })
        } else {
            [0, 0, 0]
        };

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut cell_key = (
                        center_cell.0.saturating_add(dx),
                        center_cell.1.saturating_add(dy),
                        center_cell.2.saturating_add(dz),
                    );

                    if num_cells[0] > 0 {
                        cell_key.0 = cell_key.0.rem_euclid(num_cells[0]);
                    }
                    if num_cells[1] > 0 {
                        cell_key.1 = cell_key.1.rem_euclid(num_cells[1]);
                    }
                    if num_cells[2] > 0 {
                        cell_key.2 = cell_key.2.rem_euclid(num_cells[2]);
                    }

                    if let Some(indices) = self.cells.get(&cell_key) {
                        for &idx in indices {
                            let target_pos = coords[idx];
                            let mut dist_sq = 0.0;
                            for i in 0..3 {
                                let mut diff = pos[i] - target_pos[i];
                                if let Some(box_l) = self.box_dims {
                                    let l = box_l[i];
                                    if l > 0.0 && l.is_finite() {
                                        diff -= l * (diff / l).round();
                                    }
                                }
                                dist_sq += diff * diff;
                            }
                            if dist_sq < cutoff_sq {
                                neighbors.push(idx);
                            }
                        }
                    }
                }
            }
        }
        
        neighbors.sort_unstable();
        neighbors.dedup();
        neighbors
    }
}

#[cfg(test)]
mod tests {
    use super::CellList;

    #[test]
    fn test_non_divisible_box_cutoff_wraparound() {
        // Non-divisible box_l / cutoff ratios (e.g. 10.5/3.0=3.5, 14.9/3.0=4.967, 7.3/2.5=2.92)
        for &(box_l, cutoff) in &[
            ([10.5f32, 14.9, 7.3], 3.0f32),
            ([11.9, 13.7, 9.1], 2.5),
            ([5.0, 5.0, 5.0], 6.0), // box smaller than cutoff
            ([10.5, 10.5, 0.0], 3.0), // 2D slab (non-periodic Z)
        ] {
            let mut coords = Vec::new();
            let mut state = 0x1234_5678u32;
            for _ in 0..120 {
                let mut p = [0.0f32; 3];
                for d in 0..3 {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let span = if box_l[d] > 0.0 { box_l[d] } else { 10.0 };
                    p[d] = (state as f32 / u32::MAX as f32) * span;
                }
                coords.push(p);
            }
            // Add explicit boundary-hugging pairs just inside 0 and box_l
            coords.push([0.05, 0.05, 0.05]);
            coords.push([
                (box_l[0] - 0.05).max(0.0),
                (box_l[1] - 0.05).max(0.0),
                if box_l[2] > 0.0 { box_l[2] - 0.05 } else { 0.1 },
            ]);
            coords.push([(box_l[0] - 2.4).max(0.0), 0.1, 0.1]);

            let cl = CellList::build(&coords, cutoff, Some(box_l));
            for (i, &pi) in coords.iter().enumerate() {
                let mut expected = Vec::new();
                for (j, &pj) in coords.iter().enumerate() {
                    let mut d2 = 0.0f32;
                    for d in 0..3 {
                        let mut diff = pi[d] - pj[d];
                        if box_l[d] > 0.0 {
                            diff -= box_l[d] * (diff / box_l[d]).round();
                        }
                        d2 += diff * diff;
                    }
                    if d2 < cutoff * cutoff {
                        expected.push(j);
                    }
                }
                expected.sort_unstable();
                let got = cl.query_neighbors(pi, &coords);
                assert_eq!(got, expected, "Mismatch for atom {} at box {:?}, cutoff {}", i, box_l, cutoff);
            }
        }
    }
}
