// ponytail: CUDA QCP Trajectory Engine (Theobald Quaternion Characteristic Polynomial)
// Pure registers, zero global memory allocations during solve, FP32 with double-precision accumulation.

typedef unsigned long long uint64_t;
typedef unsigned int size_t;

extern "C" {

// ============================================================================
// 1. Pass 1: Center coordinates to centroid and precompute G = sum(||x||^2)
// ============================================================================
__global__ void center_and_norms_kernel(
    const float* __restrict__ coords,        // [num_frames, num_atoms, 3]
    float* __restrict__ centered_coords,     // [num_frames, num_atoms, 3]
    float* __restrict__ norms,               // [num_frames]
    int num_frames,
    int num_atoms
) {
    int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (frame_idx >= num_frames) return;

    const float* in_frame = coords + (size_t)frame_idx * num_atoms * 3;
    float* out_frame = centered_coords + (size_t)frame_idx * num_atoms * 3;

    // 1. Compute centroid
    double cx = 0.0, cy = 0.0, cz = 0.0;
    for (int a = 0; a < num_atoms; ++a) {
        cx += in_frame[a * 3 + 0];
        cy += in_frame[a * 3 + 1];
        cz += in_frame[a * 3 + 2];
    }
    float inv_n = 1.0f / (float)num_atoms;
    float mean_x = (float)(cx * inv_n);
    float mean_y = (float)(cy * inv_n);
    float mean_z = (float)(cz * inv_n);

    // 2. Subtract centroid & compute sum of squared coordinates
    double sum_sq = 0.0;
    for (int a = 0; a < num_atoms; ++a) {
        float x = in_frame[a * 3 + 0] - mean_x;
        float y = in_frame[a * 3 + 1] - mean_y;
        float z = in_frame[a * 3 + 2] - mean_z;

        out_frame[a * 3 + 0] = x;
        out_frame[a * 3 + 1] = y;
        out_frame[a * 3 + 2] = z;

        sum_sq += (double)x * x + (double)y * y + (double)z * z;
    }

    norms[frame_idx] = (float)sum_sq;
}

// ============================================================================
// Helper: 4x4 symmetric matrix determinant in registers
// ============================================================================
__device__ __forceinline__ float det4x4_sym(
    float k00, float k01, float k02, float k03,
    float k11, float k12, float k13,
    float k22, float k23,
    float k33
) {
    // Sub-determinants for row 0
    float m00 = k11 * (k22 * k33 - k23 * k23) - k12 * (k12 * k33 - k23 * k13) + k13 * (k12 * k23 - k22 * k13);
    float m01 = k01 * (k22 * k33 - k23 * k23) - k12 * (k02 * k33 - k23 * k03) + k13 * (k02 * k23 - k22 * k03);
    float m02 = k01 * (k12 * k33 - k13 * k23) - k11 * (k02 * k33 - k23 * k03) + k13 * (k02 * k13 - k12 * k03);
    float m03 = k01 * (k12 * k23 - k13 * k22) - k11 * (k02 * k23 - k22 * k03) + k12 * (k02 * k13 - k12 * k03);

    return k00 * m00 - k01 * m01 + k02 * m02 - k03 * m03;
}

// ============================================================================
// 2. Pass 2: Tiled Pairwise QCP RMSD Kernel
// Computes RMSD between Frame i and Frame j
// ============================================================================
__global__ void pairwise_qcp_tiled_kernel(
    const float* __restrict__ centered_coords, // [num_frames, num_atoms, 3]
    const float* __restrict__ norms,           // [num_frames]
    float* __restrict__ dist_matrix,           // [num_frames, num_frames] or tile buffer
    int num_frames,
    int num_atoms,
    int row_offset,
    int col_offset,
    int tile_rows,
    int tile_cols
) {
    int local_r = blockIdx.y * blockDim.y + threadIdx.y;
    int local_c = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_r >= tile_rows || local_c >= tile_cols) return;

    int i = row_offset + local_r;
    int j = col_offset + local_c;

    if (i >= num_frames || j >= num_frames) return;

    // Diagonal is 0.0
    if (i == j) {
        dist_matrix[(size_t)local_r * tile_cols + local_c] = 0.0f;
        return;
    }

    const float* P = centered_coords + (size_t)i * num_atoms * 3;
    const float* Q = centered_coords + (size_t)j * num_atoms * 3;

    // 1. Inner product matrix S = P^T * Q (3x3)
    double Sxx = 0.0, Sxy = 0.0, Sxz = 0.0;
    double Syx = 0.0, Syy = 0.0, Syz = 0.0;
    double Szx = 0.0, Szy = 0.0, Szz = 0.0;

    for (int a = 0; a < num_atoms; ++a) {
        float px = P[a * 3 + 0];
        float py = P[a * 3 + 1];
        float pz = P[a * 3 + 2];

        float qx = Q[a * 3 + 0];
        float qy = Q[a * 3 + 1];
        float qz = Q[a * 3 + 2];

        Sxx += (double)px * qx;
        Sxy += (double)px * qy;
        Sxz += (double)px * qz;

        Syx += (double)py * qx;
        Syy += (double)py * qy;
        Syz += (double)py * qz;

        Szx += (double)pz * qx;
        Szy += (double)pz * qy;
        Szz += (double)pz * qz;
    }

    // 2. Construct 4x4 Key Matrix K
    float k00 = (float)(Sxx + Syy + Szz);
    float k11 = (float)(Sxx - Syy - Szz);
    float k22 = (float)(-Sxx + Syy - Szz);
    float k33 = (float)(-Sxx - Syy + Szz);

    float k01 = (float)(Syz - Szy);
    float k02 = (float)(Szx - Sxz);
    float k03 = (float)(Sxy - Syx);

    float k12 = (float)(Sxy + Syx);
    float k13 = (float)(Szx + Sxz);
    float k23 = (float)(Syz + Szy);

    // 3. Characteristic polynomial coefficients:
    // P(lambda) = lambda^4 - 2*c2*lambda^2 - 8*c1*lambda + c0
    float c2 = (float)(Sxx*Sxx + Sxy*Sxy + Sxz*Sxz +
                       Syx*Syx + Syy*Syy + Syz*Syz +
                       Szx*Szx + Szy*Szy + Szz*Szz);

    // Determinant of S (3x3)
    float c1 = (float)(Sxx * (Syy * Szz - Syz * Szy) -
                       Sxy * (Syx * Szz - Syz * Szx) +
                       Sxz * (Syx * Szy - Syy * Szx));

    float c0 = det4x4_sym(k00, k01, k02, k03, k11, k12, k13, k22, k23, k33);

    // 4. Initial guess: (G_P + G_Q) / 2
    float Gp = norms[i];
    float Gq = norms[j];
    float lambda = 0.5f * (Gp + Gq);

    // 5. Newton-Raphson iterations (monotonically converges to lambda_max)
    #pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        float l2 = lambda * lambda;
        float f = l2 * l2 - 2.0f * c2 * l2 - 8.0f * c1 * lambda + c0;
        float f_prime = 4.0f * (l2 * lambda - c2 * lambda - 2.0f * c1);
        if (fabsf(f_prime) > 1e-7f) {
            lambda -= f / f_prime;
        }
    }

    // 6. RMSD with floating-point underflow clamp
    float diff = (Gp + Gq - 2.0f * lambda) / (float)num_atoms;
    float rmsd = sqrtf(fmaxf(0.0f, diff));

    dist_matrix[(size_t)local_r * tile_cols + local_c] = rmsd;
}

// ============================================================================
// 3. Greedy Leader Clustering (Daura Algorithm Step)
// Counts unassigned neighbors within cutoff for all remaining frames
// ============================================================================
__global__ void daura_count_neighbors_kernel(
    const float* __restrict__ dist_matrix, // [num_frames, num_frames]
    const int* __restrict__ assigned_mask, // [num_frames] (1 if already in a cluster, 0 otherwise)
    int* __restrict__ neighbor_counts,     // [num_frames]
    float cutoff,
    int num_frames
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_frames) return;

    if (assigned_mask[i] != 0) {
        neighbor_counts[i] = -1; // Ignore already clustered frames
        return;
    }

    int count = 0;
    const float* row = dist_matrix + (size_t)i * num_frames;
    for (int j = 0; j < num_frames; ++j) {
        if (assigned_mask[j] == 0 && row[j] <= cutoff) {
            count++;
        }
    }
    neighbor_counts[i] = count;
}

// ============================================================================
// 4. Contact Map Kernel (Packed 64-bit Bitmask)
// Computes residue-residue distance < cutoff, packed 64 contacts per uint64_t
// ============================================================================
__global__ void contact_map_bitmask_kernel(
    const float* __restrict__ coords,       // [num_frames, num_res, 3]
    uint64_t* __restrict__ bitmask_out,     // [num_frames, (num_res * num_res + 63) / 64]
    float cutoff_sq,
    int num_frames,
    int num_res,
    int num_u64_per_frame
) {
    int frame_idx = blockIdx.z;
    if (frame_idx >= num_frames) return;

    int u64_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (u64_idx >= num_u64_per_frame) return;

    const float* frame_coords = coords + (size_t)frame_idx * num_res * 3;
    uint64_t mask = 0;

    #pragma unroll
    for (int bit = 0; bit < 64; ++bit) {
        int pair_idx = u64_idx * 64 + bit;
        int r1 = pair_idx / num_res;
        int r2 = pair_idx % num_res;

        if (r1 < num_res && r2 < num_res) {
            float dx = frame_coords[r1 * 3 + 0] - frame_coords[r2 * 3 + 0];
            float dy = frame_coords[r1 * 3 + 1] - frame_coords[r2 * 3 + 1];
            float dz = frame_coords[r1 * 3 + 2] - frame_coords[r2 * 3 + 2];
            float d_sq = dx * dx + dy * dy + dz * dz;

            if (d_sq <= cutoff_sq) {
                mask |= ((uint64_t)1 << bit);
            }
        }
    }

    bitmask_out[(size_t)frame_idx * num_u64_per_frame + u64_idx] = mask;
}

} // extern "C"
