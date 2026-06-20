// ponytail: simple spatial hashing with PBC support.
use std::collections::HashMap;

pub struct CellList {
    cells: HashMap<(i32, i32, i32), Vec<usize>>,
    cutoff: f32,
    box_dims: Option<[f32; 3]>,
}

impl CellList {
    pub fn build(coords: &[[f32; 3]], cutoff: f32, box_dims: Option<[f32; 3]>) -> Self {
        let mut cells = HashMap::new();
        for (idx, &pos) in coords.iter().enumerate() {
            let cell_coord = Self::get_cell_coord(pos, cutoff, box_dims);
            cells.entry(cell_coord).or_insert_with(Vec::new).push(idx);
        }
        CellList {
            cells,
            cutoff,
            box_dims,
        }
    }

    fn get_cell_coord(pos: [f32; 3], cutoff: f32, box_dims: Option<[f32; 3]>) -> (i32, i32, i32) {
        let mut p = pos;
        if let Some(box_l) = box_dims {
            for i in 0..3 {
                let l = box_l[i];
                p[i] = p[i] % l;
                if p[i] < 0.0 {
                    p[i] += l;
                }
            }
        }
        let mut cx = (p[0] / cutoff).floor() as i32;
        let mut cy = (p[1] / cutoff).floor() as i32;
        let mut cz = (p[2] / cutoff).floor() as i32;

        if let Some(box_l) = box_dims {
            let n_x = (box_l[0] / cutoff).floor() as i32;
            let n_y = (box_l[1] / cutoff).floor() as i32;
            let n_z = (box_l[2] / cutoff).floor() as i32;
            if n_x > 0 { cx = (cx % n_x + n_x) % n_x; }
            if n_y > 0 { cy = (cy % n_y + n_y) % n_y; }
            if n_z > 0 { cz = (cz % n_z + n_z) % n_z; }
        }
        (cx, cy, cz)
    }

    pub fn query_neighbors(&self, pos: [f32; 3], coords: &[[f32; 3]]) -> Vec<usize> {
        let mut neighbors = Vec::new();
        let center_cell = Self::get_cell_coord(pos, self.cutoff, self.box_dims);
        let cutoff_sq = self.cutoff * self.cutoff;

        let (num_cells_x, num_cells_y, num_cells_z) = if let Some(box_l) = self.box_dims {
            (
                (box_l[0] / self.cutoff).floor() as i32,
                (box_l[1] / self.cutoff).floor() as i32,
                (box_l[2] / self.cutoff).floor() as i32,
            )
        } else {
            (0, 0, 0)
        };

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let mut cell_key = (center_cell.0 + dx, center_cell.1 + dy, center_cell.2 + dz);
                    
                    if let Some(_box_l) = self.box_dims {
                        if num_cells_x > 0 {
                            cell_key.0 = (cell_key.0 % num_cells_x + num_cells_x) % num_cells_x;
                        }
                        if num_cells_y > 0 {
                            cell_key.1 = (cell_key.1 % num_cells_y + num_cells_y) % num_cells_y;
                        }
                        if num_cells_z > 0 {
                            cell_key.2 = (cell_key.2 % num_cells_z + num_cells_z) % num_cells_z;
                        }
                    }

                    if let Some(indices) = self.cells.get(&cell_key) {
                        for &idx in indices {
                            let target_pos = coords[idx];
                            let mut dist_sq = 0.0;
                            for i in 0..3 {
                                let mut diff = pos[i] - target_pos[i];
                                if let Some(box_l) = self.box_dims {
                                    let l = box_l[i];
                                    diff -= l * (diff / l).round();
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
