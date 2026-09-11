// ponytail: Rust CUDA driver for QCP trajectory calculations and clustering via cudarc.
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use std::sync::Arc;

pub const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/qcp.ptx"));

pub struct GpuTrajectoryEngine {
    dev: Arc<CudaDevice>,
}

impl GpuTrajectoryEngine {
    pub fn new(device_id: usize) -> Result<Self, DriverError> {
        let dev = CudaDevice::new(device_id)?;
        dev.load_ptx(
            PTX_SRC.into(),
            "qcp_module",
            &[
                "center_and_norms_kernel",
                "pairwise_qcp_tiled_kernel",
                "daura_count_neighbors_kernel",
                "contact_map_bitmask_kernel",
            ],
        )?;
        Ok(Self { dev })
    }

    /// Computes pairwise RMSD matrix [num_frames, num_frames] on GPU using Theobald QCP.
    pub fn pairwise_rmsd(
        &self,
        flat_coords: &[f32], // [num_frames, num_atoms, 3]
        num_frames: usize,
        num_atoms: usize,
    ) -> Result<Vec<f32>, DriverError> {
        if num_frames == 0 || num_atoms == 0 {
            return Ok(Vec::new());
        }

        // 1. Upload raw coordinates to device
        let d_coords = self.dev.htod_sync_copy(flat_coords)?;

        // 2. Allocate centered coords buffer and norms buffer
        let mut d_centered = self.dev.alloc_zeros::<f32>(num_frames * num_atoms * 3)?;
        let mut d_norms = self.dev.alloc_zeros::<f32>(num_frames)?;

        // 3. Launch Pass 1: Center coordinates & precompute norms
        let center_fn = self.dev.get_func("qcp_module", "center_and_norms_kernel").unwrap();
        let threads_x = 128;
        let blocks_x = (num_frames + threads_x - 1) / threads_x;
        let cfg_center = LaunchConfig {
            grid_dim: (blocks_x as u32, 1, 1),
            block_dim: (threads_x as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            center_fn.launch(
                cfg_center,
                (&d_coords, &mut d_centered, &mut d_norms, num_frames as i32, num_atoms as i32),
            )?;
        }

        // 4. Allocate distance matrix in device memory
        let mut d_dist = self.dev.alloc_zeros::<f32>(num_frames * num_frames)?;

        // 5. Launch Pass 2: Tiled QCP RMSD Kernel
        let qcp_fn = self.dev.get_func("qcp_module", "pairwise_qcp_tiled_kernel").unwrap();
        let tile_dim = 16;
        let grid_x = (num_frames + tile_dim - 1) / tile_dim;
        let grid_y = (num_frames + tile_dim - 1) / tile_dim;
        let cfg_qcp = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (tile_dim as u32, tile_dim as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            qcp_fn.launch(
                cfg_qcp,
                (
                    &d_centered,
                    &d_norms,
                    &mut d_dist,
                    num_frames as i32,
                    num_atoms as i32,
                    0i32, // row_offset
                    0i32, // col_offset
                    num_frames as i32,
                    num_frames as i32,
                ),
            )?;
        }

        // 6. Download distance matrix to host
        let host_dist = self.dev.dtoh_sync_copy(&d_dist)?;
        Ok(host_dist)
    }

    /// Performs Daura / GROMOS greedy leader clustering on the GPU.
    /// Returns (cluster_labels: [num_frames], centroid_indices: [num_clusters]).
    pub fn cluster_daura(
        &self,
        flat_coords: &[f32],
        num_frames: usize,
        num_atoms: usize,
        cutoff: f32,
    ) -> Result<(Vec<i32>, Vec<i32>), DriverError> {
        let dist_matrix = self.pairwise_rmsd(flat_coords, num_frames, num_atoms)?;

        // Upload distance matrix and run clustering logic
        let d_dist = self.dev.htod_sync_copy(&dist_matrix)?;
        let mut d_assigned = self.dev.alloc_zeros::<i32>(num_frames)?;
        let mut d_counts = self.dev.alloc_zeros::<i32>(num_frames)?;

        let count_fn = self.dev.get_func("qcp_module", "daura_count_neighbors_kernel").unwrap();
        let threads = 256;
        let blocks = (num_frames + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks as u32, 1, 1),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut labels = vec![-1i32; num_frames];
        let mut centroids = Vec::new();
        let mut current_cluster = 0i32;
        let mut assigned_mask = vec![0i32; num_frames];

        loop {
            // Count unassigned neighbors for all frames
            unsafe {
                count_fn.launch(
                    cfg,
                    (&d_dist, &d_assigned, &mut d_counts, cutoff, num_frames as i32),
                )?;
            }
            let host_counts = self.dev.dtoh_sync_copy(&d_counts)?;

            // Find unassigned frame with maximum neighbors
            let mut best_frame = -1i32;
            let mut max_neighbors = 0;
            for (idx, &cnt) in host_counts.iter().enumerate() {
                if cnt > max_neighbors {
                    max_neighbors = cnt;
                    best_frame = idx as i32;
                }
            }

            if best_frame == -1 || max_neighbors == 0 {
                // Assign remaining unassigned frames as singletons
                for i in 0..num_frames {
                    if assigned_mask[i] == 0 {
                        labels[i] = current_cluster;
                        centroids.push(i as i32);
                        current_cluster += 1;
                    }
                }
                break;
            }

            let best_idx = best_frame as usize;
            centroids.push(best_frame);

            // Assign neighbors of best_frame to current_cluster
            let row_offset = best_idx * num_frames;
            for j in 0..num_frames {
                if assigned_mask[j] == 0 && dist_matrix[row_offset + j] <= cutoff {
                    labels[j] = current_cluster;
                    assigned_mask[j] = 1;
                }
            }
            current_cluster += 1;

            // Update GPU assigned mask
            d_assigned = self.dev.htod_sync_copy(&assigned_mask)?;
        }

        Ok((labels, centroids))
    }

    /// Computes packed 64-bit contact frequency bitmasks across frames.
    pub fn contact_map_bitmask(
        &self,
        flat_coords: &[f32],
        num_frames: usize,
        num_res: usize,
        cutoff: f32,
    ) -> Result<Vec<u64>, DriverError> {
        let num_u64_per_frame = (num_res * num_res + 63) / 64;
        let d_coords = self.dev.htod_sync_copy(flat_coords)?;
        let mut d_bitmask = self.dev.alloc_zeros::<u64>(num_frames * num_u64_per_frame)?;

        let contact_fn = self.dev.get_func("qcp_module", "contact_map_bitmask_kernel").unwrap();
        let threads = 128;
        let blocks_x = (num_u64_per_frame + threads - 1) / threads;
        let cfg = LaunchConfig {
            grid_dim: (blocks_x as u32, 1, num_frames as u32),
            block_dim: (threads as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let cutoff_sq = cutoff * cutoff;
        unsafe {
            contact_fn.launch(
                cfg,
                (
                    &d_coords,
                    &mut d_bitmask,
                    cutoff_sq,
                    num_frames as i32,
                    num_res as i32,
                    num_u64_per_frame as i32,
                ),
            )?;
        }

        let host_bitmask = self.dev.dtoh_sync_copy(&d_bitmask)?;
        Ok(host_bitmask)
    }
}
