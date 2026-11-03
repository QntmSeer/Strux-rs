// ponytail: Rust CUDA driver for QCP trajectory calculations and clustering via cudarc.
pub mod nvtx;

use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

pub const PTX_SRC: &str = include_str!(concat!(env!("OUT_DIR"), "/qcp.ptx"));

// ponytail: Process-wide singleton cache per device ID.
// Eliminates the ~200 ms driver context recreation and PTX JIT compilation on repeated calls.
static ENGINES: OnceLock<Mutex<HashMap<usize, Arc<GpuTrajectoryEngine>>>> = OnceLock::new();

pub struct GpuTrajectoryEngine {
    dev: Arc<CudaDevice>,
}

impl GpuTrajectoryEngine {
    /// Retrieve an existing cached engine for `device_id` or initialize and cache a new one.
    pub fn get_or_create(device_id: usize) -> Result<Arc<Self>, DriverError> {
        let engines = ENGINES.get_or_init(|| Mutex::new(HashMap::new()));
        let mut map = engines.lock().unwrap();
        if let Some(engine) = map.get(&device_id) {
            return Ok(Arc::clone(engine));
        }
        let engine = Arc::new(Self::new(device_id)?);
        map.insert(device_id, Arc::clone(&engine));
        Ok(engine)
    }

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
                "contact_frequency_kernel",
            ],
        )?;

        Ok(Self { dev })
    }

    /// Computes pairwise RMSD matrix [num_frames, num_frames] on GPU using Theobald QCP.
    /// Uses 2D block tiling for large trajectories (>16,384 frames) to prevent VRAM OOM on 4GB GPUs.
    pub fn pairwise_rmsd(
        &self,
        flat_coords: &[f32], // [num_frames, num_atoms, 3]
        num_frames: usize,
        num_atoms: usize,
    ) -> Result<Vec<f32>, DriverError> {
        if num_frames == 0 || num_atoms == 0 {
            return Ok(Vec::new());
        }

        let _nvtx_all = nvtx::NvtxRange::new("strux_pairwise_rmsd");

        // 1. Upload raw coordinates to device
        let d_coords = {
            let _nvtx = nvtx::NvtxRange::new("upload_coords");
            self.dev.htod_sync_copy(flat_coords)?
        };

        // 2. Allocate centered coords buffer and norms buffer
        let mut d_centered = self.dev.alloc_zeros::<f32>(num_frames * num_atoms * 3)?;
        let mut d_norms = self.dev.alloc_zeros::<f32>(num_frames)?;

        // 3. Launch Pass 1: Center coordinates & precompute norms
        {
            let _nvtx = nvtx::NvtxRange::new("center_and_norms");
            let center_fn = self.dev.get_func("qcp_module", "center_and_norms_kernel").unwrap();
            let threads_x = 128;
            let blocks_x = num_frames.div_ceil(threads_x);
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
        }

        let qcp_fn = self.dev.get_func("qcp_module", "pairwise_qcp_tiled_kernel").unwrap();
        let tile_dim = 16;
        let max_oneshot_frames = 20480;

        // ponytail: If full distance matrix fits comfortably in GPU memory (<= 1.6 GB, <= 20,480 frames),
        // compute in one shot. Otherwise, tile across 2D blocks to prevent OOM on 4 GB laptop GPUs (A2000).
        if num_frames <= max_oneshot_frames {
            let mut d_dist = self.dev.alloc_zeros::<f32>(num_frames * num_frames)?;
            let grid_x = num_frames.div_ceil(tile_dim);
            let grid_y = num_frames.div_ceil(tile_dim);
            let cfg_qcp = LaunchConfig {
                grid_dim: (grid_x as u32, grid_y as u32, 1),
                block_dim: (tile_dim as u32, tile_dim as u32, 1),
                shared_mem_bytes: 0,
            };

            {
                let _nvtx = nvtx::NvtxRange::new("qcp_tiled_solve_oneshot");
                unsafe {
                    qcp_fn.clone().launch(
                        cfg_qcp,
                        (
                            &d_centered,
                            &d_norms,
                            &mut d_dist,
                            num_frames as i32,
                            num_atoms as i32,
                            0i32,
                            0i32,
                            num_frames as i32,
                            num_frames as i32,
                        ),
                    )?;
                }
            }

            let _nvtx = nvtx::NvtxRange::new("download_matrix");
            return self.dev.dtoh_sync_copy(&d_dist);
        }

        // Tiled block computation for massive trajectories
        let chunk_size = 8192;
        let mut host_dist = vec![0.0f32; num_frames * num_frames];
        let mut d_tile = self.dev.alloc_zeros::<f32>(chunk_size * chunk_size)?;

        let _nvtx_tiles = nvtx::NvtxRange::new("qcp_tiled_solve_blocks");
        for r_start in (0..num_frames).step_by(chunk_size) {
            let r_len = (num_frames - r_start).min(chunk_size);
            for c_start in (0..num_frames).step_by(chunk_size) {
                let c_len = (num_frames - c_start).min(chunk_size);

                let grid_x = c_len.div_ceil(tile_dim);
                let grid_y = r_len.div_ceil(tile_dim);
                let cfg_qcp = LaunchConfig {
                    grid_dim: (grid_x as u32, grid_y as u32, 1),
                    block_dim: (tile_dim as u32, tile_dim as u32, 1),
                    shared_mem_bytes: 0,
                };

                unsafe {
                    qcp_fn.clone().launch(
                        cfg_qcp,
                        (
                            &d_centered,
                            &d_norms,
                            &mut d_tile,
                            num_frames as i32,
                            num_atoms as i32,
                            r_start as i32,
                            c_start as i32,
                            r_len as i32,
                            c_len as i32,
                        ),
                    )?;
                }

                // Copy tile from device back to host
                let tile_data = self.dev.dtoh_sync_copy(&d_tile)?;
                for r in 0..r_len {
                    let src_offset = r * c_len;
                    let dst_offset = (r_start + r) * num_frames + c_start;
                    host_dist[dst_offset..dst_offset + c_len]
                        .copy_from_slice(&tile_data[src_offset..src_offset + c_len]);
                }
            }
        }

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
        let _nvtx = nvtx::NvtxRange::new("strux_cluster_daura");
        let dist_matrix = self.pairwise_rmsd(flat_coords, num_frames, num_atoms)?;

        // Upload distance matrix and run clustering logic
        let d_dist = self.dev.htod_sync_copy(&dist_matrix)?;
        let mut d_assigned = self.dev.alloc_zeros::<i32>(num_frames)?;
        let mut d_counts = self.dev.alloc_zeros::<i32>(num_frames)?;

        let count_fn = self.dev.get_func("qcp_module", "daura_count_neighbors_kernel").unwrap();
        let threads = 256;
        let blocks = num_frames.div_ceil(threads);
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
                count_fn.clone().launch(
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
        let _nvtx = nvtx::NvtxRange::new("strux_contact_map_bitmask");
        let num_u64_per_frame = (num_res * num_res).div_ceil(64);
        let mut host_bitmask = Vec::with_capacity(num_frames * num_u64_per_frame);

        let contact_fn = self.dev.get_func("qcp_module", "contact_map_bitmask_kernel").unwrap();
        let threads = 128;
        let blocks_x = num_u64_per_frame.div_ceil(threads);
        let cutoff_sq = cutoff * cutoff;

        // ponytail: stream contact bitmask computation in 500 MB chunks to bound VRAM usage on 4GB laptop GPUs
        let max_u64_buf = (500 * 1024 * 1024) / 8;
        let chunk_frames = (max_u64_buf / num_u64_per_frame).max(1).min(num_frames);

        for start_f in (0..num_frames).step_by(chunk_frames) {
            let cur_frames = (num_frames - start_f).min(chunk_frames);
            let coord_start = start_f * num_res * 3;
            let coord_end = coord_start + cur_frames * num_res * 3;

            let d_coords = self.dev.htod_sync_copy(&flat_coords[coord_start..coord_end])?;
            let mut d_bitmask = self.dev.alloc_zeros::<u64>(cur_frames * num_u64_per_frame)?;

            let cfg = LaunchConfig {
                grid_dim: (blocks_x as u32, 1, cur_frames as u32),
                block_dim: (threads as u32, 1, 1),
                shared_mem_bytes: 0,
            };

            unsafe {
                contact_fn.clone().launch(
                    cfg,
                    (
                        &d_coords,
                        &mut d_bitmask,
                        cutoff_sq,
                        cur_frames as i32,
                        num_res as i32,
                        num_u64_per_frame as i32,
                    ),
                )?;
            }

            let chunk_res = self.dev.dtoh_sync_copy(&d_bitmask)?;
            host_bitmask.extend_from_slice(&chunk_res);
        }

        Ok(host_bitmask)
    }

    /// Computes dense pairwise contact frequency matrix [num_res, num_res] directly on GPU.
    pub fn contact_frequency(
        &self,
        flat_coords: &[f32],
        num_frames: usize,
        num_res: usize,
        cutoff: f32,
    ) -> Result<Vec<f32>, DriverError> {
        if num_frames == 0 || num_res == 0 {
            return Ok(Vec::new());
        }

        let _nvtx = nvtx::NvtxRange::new("strux_contact_frequency");
        let d_coords = self.dev.htod_sync_copy(flat_coords)?;
        let mut d_freq = self.dev.alloc_zeros::<f32>(num_res * num_res)?;

        let freq_fn = self.dev.get_func("qcp_module", "contact_frequency_kernel").unwrap();
        let tile_dim = 16;
        let grid_x = num_res.div_ceil(tile_dim);
        let grid_y = num_res.div_ceil(tile_dim);

        let cfg = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (tile_dim as u32, tile_dim as u32, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            freq_fn.launch(
                cfg,
                (
                    &d_coords,
                    &mut d_freq,
                    cutoff * cutoff,
                    num_frames as i32,
                    num_res as i32,
                ),
            )?;
        }

        let host_freq = self.dev.dtoh_sync_copy(&d_freq)?;
        Ok(host_freq)
    }
}
