// ponytail: pyo3 module glue, minimal code, fast array conversions.
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;

pub mod analysis;
pub mod biophysics;
pub mod cif;
pub mod dcd;
pub mod msa;
pub mod pdb;
pub mod spatial;
pub mod xtc;

#[cfg(feature = "cuda")]
pub mod cuda;

pub use msa::Msa;

#[inline]
fn map_io_err(e: std::io::Error) -> PyErr {
    if e.kind() == std::io::ErrorKind::InvalidInput {
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    } else {
        pyo3::exceptions::PyIOError::new_err(e.to_string())
    }
}

#[pyfunction]
#[pyo3(signature = (path, ca_only=false, atom_indices=None))]
fn parse_pdb(
    py: Python<'_>,
    path: &str,
    ca_only: bool,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let path_str = path.to_string();
    let frames = py.allow_threads(|| {
        pdb::parse_pdb_trajectory_selected(path_str, ca_only, atom_indices.as_deref())
    }).map_err(map_io_err)?;
    
    if frames.is_empty() {
        let array = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        return Ok(array.into());
    }
    
    let num_frames = frames.len();
    let num_atoms = frames[0].len();
    
    let array = PyArray3::<f32>::zeros_bound(py, [num_frames, num_atoms, 3], false);
    {
        let mut writer = array.readwrite();
        let slice = writer.as_slice_mut().unwrap();
        let mut idx = 0;
        for frame in frames {
            for atom in frame {
                slice[idx] = atom[0];
                slice[idx + 1] = atom[1];
                slice[idx + 2] = atom[2];
                idx += 3;
            }
        }
    }
    Ok(array.into())
}

#[pyfunction]
#[pyo3(signature = (path, ca_only=false, strip_hydrogens=true, use_auth_chain=true, atom_indices=None))]
fn parse_cif(
    py: Python<'_>,
    path: &str,
    ca_only: bool,
    strip_hydrogens: bool,
    use_auth_chain: bool,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let path_str = path.to_string();
    let cif_traj = py.allow_threads(|| {
        cif::parse_cif_trajectory(path_str, ca_only, strip_hydrogens, use_auth_chain, atom_indices.as_deref())
    }).map_err(map_io_err)?;

    if cif_traj.num_frames == 0 || cif_traj.num_atoms == 0 {
        let array = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        return Ok(array.into());
    }

    let array = PyArray3::<f32>::zeros_bound(py, [cif_traj.num_frames, cif_traj.num_atoms, 3], false);
    {
        let mut writer = array.readwrite();
        let slice = writer.as_slice_mut().unwrap();
        slice.copy_from_slice(&cif_traj.coords);
    }
    Ok(array.into())
}

#[pyfunction]
#[pyo3(signature = (path, ca_only=false, strip_hydrogens=true, use_auth_chain=true, atom_indices=None))]
fn parse_cif_with_plddt(
    py: Python<'_>,
    path: &str,
    ca_only: bool,
    strip_hydrogens: bool,
    use_auth_chain: bool,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<(Py<PyArray3<f32>>, Py<PyArray2<f32>>)> {
    let path_str = path.to_string();
    let cif_traj = py.allow_threads(|| {
        cif::parse_cif_trajectory(path_str, ca_only, strip_hydrogens, use_auth_chain, atom_indices.as_deref())
    }).map_err(map_io_err)?;

    if cif_traj.num_frames == 0 || cif_traj.num_atoms == 0 {
        let coords = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        let plddt = PyArray2::<f32>::zeros_bound(py, [0, 0], false);
        return Ok((coords.into(), plddt.into()));
    }

    let coords_arr = PyArray3::<f32>::zeros_bound(py, [cif_traj.num_frames, cif_traj.num_atoms, 3], false);
    {
        let mut writer = coords_arr.readwrite();
        writer.as_slice_mut().unwrap().copy_from_slice(&cif_traj.coords);
    }
    let plddt_arr = PyArray2::<f32>::zeros_bound(py, [cif_traj.num_frames, cif_traj.num_atoms], false);
    {
        let mut writer = plddt_arr.readwrite();
        writer.as_slice_mut().unwrap().copy_from_slice(&cif_traj.b_factors);
    }
    Ok((coords_arr.into(), plddt_arr.into()))
}

#[pyfunction]
#[pyo3(signature = (path, ca_only=false, strip_hydrogens=true, use_auth_chain=true, atom_indices=None))]
fn parse_cif_full(
    py: Python<'_>,
    path: &str,
    ca_only: bool,
    strip_hydrogens: bool,
    use_auth_chain: bool,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<(
    Py<PyArray3<f32>>,
    Py<PyArray2<f32>>,
    Vec<String>,
    Py<PyArray1<i32>>,
    Vec<String>,
    Vec<String>,
)> {
    let path_str = path.to_string();
    let cif_traj = py.allow_threads(|| {
        cif::parse_cif_trajectory(path_str, ca_only, strip_hydrogens, use_auth_chain, atom_indices.as_deref())
    }).map_err(map_io_err)?;

    if cif_traj.num_frames == 0 || cif_traj.num_atoms == 0 {
        let coords = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        let plddt = PyArray2::<f32>::zeros_bound(py, [0, 0], false);
        let res_ids = PyArray1::<i32>::from_vec_bound(py, Vec::new());
        return Ok((coords.into(), plddt.into(), Vec::new(), res_ids.into(), Vec::new(), Vec::new()));
    }

    let coords_arr = PyArray3::<f32>::zeros_bound(py, [cif_traj.num_frames, cif_traj.num_atoms, 3], false);
    {
        let mut writer = coords_arr.readwrite();
        writer.as_slice_mut().unwrap().copy_from_slice(&cif_traj.coords);
    }
    let plddt_arr = PyArray2::<f32>::zeros_bound(py, [cif_traj.num_frames, cif_traj.num_atoms], false);
    {
        let mut writer = plddt_arr.readwrite();
        writer.as_slice_mut().unwrap().copy_from_slice(&cif_traj.b_factors);
    }
    let res_ids_arr = PyArray1::<i32>::from_vec_bound(py, cif_traj.res_ids);

    Ok((
        coords_arr.into(),
        plddt_arr.into(),
        cif_traj.chain_ids,
        res_ids_arr.into(),
        cif_traj.res_names,
        cif_traj.atom_names,
    ))
}

#[pyfunction]
#[pyo3(signature = (path, atom_indices=None))]
fn parse_dcd(
    py: Python<'_>,
    path: &str,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let path_str = path.to_string();
    let dcd_traj = py.allow_threads(|| {
        dcd::parse_dcd_trajectory_selected(path_str, atom_indices.as_deref())
    }).map_err(map_io_err)?;

    if dcd_traj.num_frames == 0 || dcd_traj.num_atoms == 0 {
        let array = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        return Ok(array.into());
    }

    let array = PyArray3::<f32>::zeros_bound(py, [dcd_traj.num_frames, dcd_traj.num_atoms, 3], false);
    {
        let mut writer = array.readwrite();
        let slice = writer.as_slice_mut().unwrap();
        slice.copy_from_slice(&dcd_traj.coords);
    }
    Ok(array.into())
}

#[pyfunction]
#[pyo3(signature = (path, atom_indices=None))]
fn parse_xtc(
    py: Python<'_>,
    path: &str,
    atom_indices: Option<Vec<usize>>,
) -> PyResult<Py<PyArray3<f32>>> {
    let path_str = path.to_string();
    let xtc_traj = py.allow_threads(|| {
        xtc::parse_xtc_trajectory_selected(path_str, atom_indices.as_deref())
    }).map_err(map_io_err)?;

    if xtc_traj.num_frames == 0 || xtc_traj.num_atoms == 0 {
        let array = PyArray3::<f32>::zeros_bound(py, [0, 0, 3], false);
        return Ok(array.into());
    }

    let array = PyArray3::<f32>::zeros_bound(py, [xtc_traj.num_frames, xtc_traj.num_atoms, 3], false);
    {
        let mut writer = array.readwrite();
        let slice = writer.as_slice_mut().unwrap();
        slice.copy_from_slice(&xtc_traj.coords);
    }
    Ok(array.into())
}


#[inline]
fn readonly2_to_vec3(arr: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<[f32; 3]>> {
    let v = arr.as_array();
    let shape = v.shape();
    let n = shape[0];
    if n > 0 && shape[1] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "2D coordinate array must have shape (N, 3), got ({}, {})",
            n, shape[1]
        )));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push([v[[i, 0]], v[[i, 1]], v[[i, 2]]]);
    }
    Ok(out)
}

#[inline]
fn validate_traj3_shape(shape: &[usize]) -> PyResult<(usize, usize)> {
    let num_frames = shape[0];
    let num_atoms = shape[1];
    if num_frames > 0 && num_atoms > 0 && shape[2] != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "3D trajectory array must have shape (T, N, 3), got ({}, {}, {})",
            num_frames, num_atoms, shape[2]
        )));
    }
    Ok((num_frames, num_atoms))
}

#[pyfunction]
fn calculate_rg(py: Python<'_>, coords: PyReadonlyArray2<'_, f32>) -> PyResult<f32> {
    let v = readonly2_to_vec3(&coords)?;
    Ok(py.allow_threads(|| analysis::calculate_rg(&v)))
}

#[pyfunction]
fn calculate_rmsd_raw(
    py: Python<'_>,
    coords1: PyReadonlyArray2<'_, f32>,
    coords2: PyReadonlyArray2<'_, f32>,
) -> PyResult<f32> {
    let v1 = readonly2_to_vec3(&coords1)?;
    let v2 = readonly2_to_vec3(&coords2)?;
    Ok(py.allow_threads(|| analysis::calculate_rmsd_raw(&v1, &v2)))
}

#[pyfunction]
fn calculate_rmsd_kabsch(
    py: Python<'_>,
    coords1: PyReadonlyArray2<'_, f32>,
    coords2: PyReadonlyArray2<'_, f32>,
) -> PyResult<f32> {
    let v1 = readonly2_to_vec3(&coords1)?;
    let v2 = readonly2_to_vec3(&coords2)?;
    Ok(py.allow_threads(|| analysis::calculate_rmsd_kabsch(&v1, &v2)))
}

// ponytail: native Rayon multi-threaded 1-vs-T trajectory RMSD, pegging all CPU cores.
#[pyfunction]
fn cpu_trajectory_rmsd(
    py: Python<'_>,
    trajectory: PyReadonlyArray3<'_, f32>,
    reference: PyReadonlyArray2<'_, f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    let view = trajectory.as_array();
    let ref_view = reference.as_array();
    let (num_frames, num_atoms) = validate_traj3_shape(view.shape())?;
    if ref_view.shape()[0] != num_atoms || (num_atoms > 0 && ref_view.shape()[1] != 3) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Reference coordinate shape must match trajectory (N, 3)",
        ));
    }

    let slice = view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Trajectory array must be contiguous C-order")
    })?;
    let ref_slice = ref_view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Reference array must be contiguous C-order")
    })?;

    let stride = num_atoms * 3;
    let ref_coords: &[[f32; 3]] = unsafe {
        std::slice::from_raw_parts(ref_slice.as_ptr() as *const [f32; 3], num_atoms)
    };

    let rmsds: Vec<f32> = py.allow_threads(|| {
        (0..num_frames)
            .into_par_iter()
            .map(|i| {
                let frame: &[[f32; 3]] = unsafe {
                    std::slice::from_raw_parts(
                        slice[i * stride..].as_ptr() as *const [f32; 3],
                        num_atoms,
                    )
                };
                analysis::calculate_rmsd_kabsch(frame, ref_coords)
            })
            .collect()
    });

    let out = PyArray1::<f32>::from_vec_bound(py, rmsds);
    Ok(out.into())
}

// ponytail: native Rayon multi-threaded all-to-all pairwise RMSD across all available CPU threads.
#[pyfunction]
fn cpu_pairwise_rmsd(
    py: Python<'_>,
    trajectory: PyReadonlyArray3<'_, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let view = trajectory.as_array();
    let (num_frames, num_atoms) = validate_traj3_shape(view.shape())?;

    let slice = view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Trajectory array must be contiguous C-order")
    })?;

    let stride = num_atoms * 3;
    let matrix: Vec<f32> = py.allow_threads(|| {
        (0..num_frames * num_frames)
            .into_par_iter()
            .map(|idx| {
                let i = idx / num_frames;
                let j = idx % num_frames;
                if i == j || num_atoms == 0 {
                    return 0.0;
                }
                let f1: &[[f32; 3]] = unsafe {
                    std::slice::from_raw_parts(
                        slice[i * stride..].as_ptr() as *const [f32; 3],
                        num_atoms,
                    )
                };
                let f2: &[[f32; 3]] = unsafe {
                    std::slice::from_raw_parts(
                        slice[j * stride..].as_ptr() as *const [f32; 3],
                        num_atoms,
                    )
                };
                analysis::calculate_rmsd_kabsch(f1, f2)
            })
            .collect()
    });

    let out_array = PyArray2::<f32>::zeros_bound(py, [num_frames, num_frames], false);
    unsafe {
        let target_slice = out_array.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Buffer write error: {:?}", e))
        })?;
        target_slice.copy_from_slice(&matrix);
    }
    Ok(out_array.into())
}

#[pyfunction]
fn calculate_rmsf(py: Python<'_>, trajectory: PyReadonlyArray3<'_, f32>) -> PyResult<Vec<f32>> {
    let view = trajectory.as_array();
    let (num_frames, num_atoms) = validate_traj3_shape(view.shape())?;
    Ok(py.allow_threads(|| {
        let mut traj_vec = Vec::with_capacity(num_frames);
        for frame_idx in 0..num_frames {
            let mut frame_vec = Vec::with_capacity(num_atoms);
            for atom_idx in 0..num_atoms {
                frame_vec.push([
                    view[[frame_idx, atom_idx, 0]],
                    view[[frame_idx, atom_idx, 1]],
                    view[[frame_idx, atom_idx, 2]],
                ]);
            }
            traj_vec.push(frame_vec);
        }
        analysis::calculate_rmsf(&traj_vec)
    }))
}

#[pyfunction]
fn find_interface_contacts(
    py: Python<'_>,
    target: PyReadonlyArray2<'_, f32>,
    binder: PyReadonlyArray2<'_, f32>,
    cutoff: f32,
    box_dims: Option<[f32; 3]>,
) -> PyResult<Vec<(usize, usize)>> {
    let t = readonly2_to_vec3(&target)?;
    let b = readonly2_to_vec3(&binder)?;
    Ok(py.allow_threads(|| analysis::find_interface_contacts(&t, &b, cutoff, box_dims)))
}

#[pyfunction]
fn parse_a3m(py: Python<'_>, a3m_string: &str) -> PyResult<Msa> {
    py.allow_threads(|| {
        Ok(msa::parse_a3m_impl(a3m_string))
    })
}

#[pyfunction]
fn parse_a3m_file(py: Python<'_>, path: &str) -> PyResult<Msa> {
    let p = path.to_string();
    py.allow_threads(|| {
        msa::parse_a3m_file_impl(&p)
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[pyfunction]
fn parse_stockholm(py: Python<'_>, stockholm_string: &str) -> PyResult<Msa> {
    py.allow_threads(|| {
        Ok(msa::parse_stockholm_impl(stockholm_string))
    })
}

#[pyfunction]
fn parse_stockholm_file(py: Python<'_>, path: &str) -> PyResult<Msa> {
    let p = path.to_string();
    py.allow_threads(|| {
        msa::parse_stockholm_file_impl(&p)
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
}

#[cfg(feature = "cuda")]
const CUDA_REQUIREMENTS: &str =
    "strux-rs GPU functions need an NVIDIA GPU (compute capability 7.5 or newer) and a driver supporting CUDA 12";

/// True if the NVIDIA driver library can be opened under the names cudarc searches for.
#[cfg(feature = "cuda")]
fn cuda_driver_loadable() -> bool {
    ["cuda", "nvcuda"]
        .iter()
        .any(|name| unsafe { libloading::Library::new(libloading::library_filename(name)) }.is_ok())
}

/// Create the GPU engine, turning a missing or unusable driver into a Python RuntimeError.
/// cudarc panics when the driver library is absent or lacks CUDA 12 symbols, so the driver is
/// probed first (the common no-GPU case, silent) and any remaining panic is caught.
#[cfg(feature = "cuda")]
fn new_gpu_engine(device_id: usize) -> PyResult<std::sync::Arc<cuda::GpuTrajectoryEngine>> {
    if !cuda_driver_loadable() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "NVIDIA driver library (libcuda.so / nvcuda.dll) not found. {CUDA_REQUIREMENTS}."
        )));
    }
    match std::panic::catch_unwind(|| cuda::GpuTrajectoryEngine::get_or_create(device_id)) {
        Ok(Ok(engine)) => Ok(engine),
        Ok(Err(e)) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "CUDA init error: {e:?}. {CUDA_REQUIREMENTS}."
        ))),
        Err(_) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "The NVIDIA driver could not be loaded (it may be older than CUDA 12). {CUDA_REQUIREMENTS}."
        ))),
    }
}

#[pyfunction]
fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        new_gpu_engine(0).is_ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_pairwise_rmsd(
    py: Python<'_>,
    trajectory: PyReadonlyArray3<'_, f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    let view = trajectory.as_array();
    let shape = view.shape();
    let num_frames = shape[0];
    let num_atoms = shape[1];

    let slice = view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Trajectory array must be contiguous C-order")
    })?;

    let engine = new_gpu_engine(0)?;

    let dist = py.allow_threads(|| {
        engine.pairwise_rmsd(slice, num_frames, num_atoms)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA execution error: {:?}", e)))?;

    let out_array = PyArray2::<f32>::zeros_bound(py, [num_frames, num_frames], false);
    {
        let mut writer = out_array.readwrite();
        let target_slice = writer.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Buffer write error: {:?}", e))
        })?;
        target_slice.copy_from_slice(&dist);
    }
    Ok(out_array.into())
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_cluster_daura(
    py: Python<'_>,
    trajectory: PyReadonlyArray3<'_, f32>,
    cutoff: f32,
) -> PyResult<(Py<PyArray1<i32>>, Py<PyArray1<i32>>)> {
    let view = trajectory.as_array();
    let shape = view.shape();
    let num_frames = shape[0];
    let num_atoms = shape[1];

    let slice = view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Trajectory array must be contiguous C-order")
    })?;

    let engine = new_gpu_engine(0)?;

    let (labels, centroids) = py.allow_threads(|| {
        engine.cluster_daura(slice, num_frames, num_atoms, cutoff)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA clustering error: {:?}", e)))?;

    let out_labels = PyArray1::<i32>::from_vec_bound(py, labels);
    let out_centroids = PyArray1::<i32>::from_vec_bound(py, centroids);

    Ok((out_labels.into(), out_centroids.into()))
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_contact_map_bitmask(
    py: Python<'_>,
    trajectory: PyReadonlyArray3<'_, f32>,
    cutoff: f32,
) -> PyResult<Py<PyArray2<u64>>> {
    let view = trajectory.as_array();
    let shape = view.shape();
    let num_frames = shape[0];
    let num_res = shape[1];

    let slice = view.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Trajectory array must be contiguous C-order")
    })?;

    let engine = new_gpu_engine(0)?;

    let bitmasks = py.allow_threads(|| {
        engine.contact_map_bitmask(slice, num_frames, num_res, cutoff)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA contact error: {:?}", e)))?;

    let num_u64_per_frame = (num_res * num_res).div_ceil(64);
    let out_array = PyArray2::<u64>::zeros_bound(py, [num_frames, num_u64_per_frame], false);
    {
        let mut writer = out_array.readwrite();
        let target_slice = writer.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Buffer write error: {:?}", e))
        })?;
        target_slice.copy_from_slice(&bitmasks);
    }
    Ok(out_array.into())
}

#[cfg(feature = "cuda")]
#[pyfunction]
fn cuda_contact_frequency(
    py: Python<'_>,
    coords: PyReadonlyArray3<'_, f32>,
    cutoff: Option<f32>,
    device_id: Option<usize>,
) -> PyResult<Py<PyArray2<f32>>> {
    let cutoff = cutoff.unwrap_or(4.5);
    let device_id = device_id.unwrap_or(0);
    let view = coords.as_array();
    let num_frames = view.shape()[0];
    let num_res = view.shape()[1];

    let slice = coords.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Coordinate array must be contiguous: {:?}", e))
    })?;

    let engine = new_gpu_engine(device_id)?;

    let freqs = py.allow_threads(|| {
        engine.contact_frequency(slice, num_frames, num_res, cutoff)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA contact freq error: {:?}", e)))?;

    let out_array = PyArray2::<f32>::zeros_bound(py, [num_res, num_res], false);
    {
        let mut writer = out_array.readwrite();
        let target_slice = writer.as_slice_mut().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Buffer write error: {:?}", e))
        })?;
        target_slice.copy_from_slice(&freqs);
    }
    Ok(out_array.into())
}

#[pyfunction]
#[pyo3(signature = (coords_ref, coords_target, cutoff=15.0))]
fn calculate_lddt<'py>(
    py: Python<'py>,
    coords_ref: PyReadonlyArray2<'_, f32>,
    coords_target: PyReadonlyArray2<'_, f32>,
    cutoff: f32,
) -> PyResult<(f32, Bound<'py, PyArray1<f32>>)> {
    let c_ref = readonly2_to_vec3(&coords_ref)?;
    let c_tgt = readonly2_to_vec3(&coords_target)?;
    if c_ref.len() != c_tgt.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Reference and target coordinate arrays must have the same number of atoms",
        ));
    }

    let (global_lddt, per_res) = py.allow_threads(|| {
        analysis::calculate_lddt(&c_ref, &c_tgt, cutoff)
    });

    let py_per_res = PyArray1::from_vec_bound(py, per_res);
    Ok((global_lddt, py_per_res))
}

#[pyfunction]
#[pyo3(signature = (state_sequence, num_states, lag_time=1, temperature=300.0))]
fn markov_populations<'py>(
    py: Python<'py>,
    state_sequence: Vec<usize>,
    num_states: usize,
    lag_time: usize,
    temperature: f32,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    let res = py.allow_threads(|| {
        analysis::calculate_markov_populations(&state_sequence, num_states, lag_time, temperature)
    });

    let py_pop = PyArray1::from_vec_bound(py, res.populations);
    let py_dg = PyArray1::from_vec_bound(py, res.delta_g);

    let py_tmat = PyArray2::<f32>::zeros_bound(py, [res.num_states, res.num_states], false);
    if !res.transition_matrix.is_empty() {
        let mut writer = py_tmat.readwrite();
        let slice = writer.as_slice_mut().unwrap();
        slice.copy_from_slice(&res.transition_matrix);
    }

    Ok((py_pop, py_dg, py_tmat))
}

#[pyfunction]
#[pyo3(signature = (coords, n_components=2))]
fn trajectory_pca<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray3<'_, f32>,
    n_components: usize,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>)> {
    let view = coords.as_array();
    let (num_frames, num_atoms) = validate_traj3_shape(view.shape())?;

    let slice = coords.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Coordinate array must be contiguous: {:?}", e))
    })?;

    let k = n_components.min(num_frames).min(num_atoms * 3);
    let res = py.allow_threads(|| {
        analysis::calculate_trajectory_pca(slice, num_frames, num_atoms, k)
    });

    let py_proj = PyArray2::<f32>::zeros_bound(py, [num_frames, k], false);
    if !res.projections.is_empty() {
        let mut writer = py_proj.readwrite();
        let s = writer.as_slice_mut().unwrap();
        s.copy_from_slice(&res.projections);
    }

    let py_var = PyArray1::from_vec_bound(py, res.explained_variance_ratio);

    let py_comp = PyArray2::<f32>::zeros_bound(py, [k, num_atoms * 3], false);
    if !res.components.is_empty() {
        let mut writer = py_comp.readwrite();
        let s = writer.as_slice_mut().unwrap();
        s.copy_from_slice(&res.components);
    }

    Ok((py_proj, py_var, py_comp))
}

#[pyfunction]
#[pyo3(signature = (sequences, min_identity=0.7))]
fn cluster_msa<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
    min_identity: f32,
) -> PyResult<(Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>)> {
    let (labels, reps) = py.allow_threads(|| {
        msa::cluster_sequences(&sequences, min_identity)
    });
    let py_labels = PyArray1::from_vec_bound(py, labels);
    let py_reps = PyArray1::from_vec_bound(py, reps);
    Ok((py_labels, py_reps))
}

#[pyfunction]
fn calculate_pose_and_sc_rmsd(
    py: Python<'_>,
    ref_target_ca: PyReadonlyArray2<'_, f32>,
    ref_binder_ca: PyReadonlyArray2<'_, f32>,
    mob_target_ca: PyReadonlyArray2<'_, f32>,
    mob_binder_ca: PyReadonlyArray2<'_, f32>,
) -> PyResult<(f32, f32)> {
    let rt = readonly2_to_vec3(&ref_target_ca)?;
    let rb = readonly2_to_vec3(&ref_binder_ca)?;
    let mt = readonly2_to_vec3(&mob_target_ca)?;
    let mb = readonly2_to_vec3(&mob_binder_ca)?;
    if rt.len() != mt.len() || rb.len() != mb.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Target or binder CA coordinate length mismatch between reference and mobile",
        ));
    }
    Ok(py.allow_threads(|| biophysics::calculate_pose_and_sc_rmsd(&rt, &rb, &mt, &mb)))
}

#[pyfunction]
fn batch_pose_and_sc_rmsd<'py>(
    py: Python<'py>,
    ref_target_ca: PyReadonlyArray2<'_, f32>,
    ref_binder_ca: PyReadonlyArray2<'_, f32>,
    mob_targets_ca: PyReadonlyArray3<'_, f32>,
    mob_binders_ca: PyReadonlyArray3<'_, f32>,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<f32>>)> {
    let rt = readonly2_to_vec3(&ref_target_ca)?;
    let rb = readonly2_to_vec3(&ref_binder_ca)?;
    let mt_view = mob_targets_ca.as_array();
    let mb_view = mob_binders_ca.as_array();
    let (num_models, mt_atoms) = validate_traj3_shape(mt_view.shape())?;
    let (mb_models, mb_atoms) = validate_traj3_shape(mb_view.shape())?;
    if num_models != mb_models || mt_atoms != rt.len() || mb_atoms != rb.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Batch target/binder dimensions do not match reference CA lengths",
        ));
    }
    let mt_slice = mob_targets_ca.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("mob_targets_ca must be C-contiguous")
    })?;
    let mb_slice = mob_binders_ca.as_slice().map_err(|_| {
        pyo3::exceptions::PyValueError::new_err("mob_binders_ca must be C-contiguous")
    })?;

    let (sc, pose) = py.allow_threads(|| {
        biophysics::batch_pose_and_sc_rmsd(&rt, &rb, mt_slice, mb_slice, num_models)
    });
    Ok((PyArray1::from_vec_bound(py, sc), PyArray1::from_vec_bound(py, pose)))
}

#[pyfunction]
#[pyo3(signature = (coords, atom_names, res_names, chain_ids, res_ids, target_chain="A", binder_chain="B"))]
fn analyze_interface_physics<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'_, f32>,
    atom_names: Vec<String>,
    res_names: Vec<String>,
    chain_ids: Vec<String>,
    res_ids: Vec<i32>,
    target_chain: &str,
    binder_chain: &str,
) -> PyResult<(usize, usize, usize, usize, Bound<'py, PyArray1<usize>>)> {
    let c = readonly2_to_vec3(&coords)?;
    let n = c.len();
    if atom_names.len() != n || res_names.len() != n || chain_ids.len() != n || res_ids.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords, atom_names, res_names, chain_ids, and res_ids must have identical length",
        ));
    }
    let tc = target_chain.to_string();
    let bc = binder_chain.to_string();
    let res = py.allow_threads(|| {
        biophysics::analyze_interface_physics(&c, &atom_names, &res_names, &chain_ids, &res_ids, &tc, &bc)
    });
    let buns_arr = PyArray1::from_vec_bound(py, res.buns_indices);
    Ok((res.interface_contacts, res.polar_contacts, res.salt_bridges, res.buns_count, buns_arr))
}

#[pyfunction]
#[pyo3(signature = (coords, radii=None, probe_radius=1.4, n_points=96))]
fn calculate_sasa<'py>(
    py: Python<'py>,
    coords: PyReadonlyArray2<'_, f32>,
    radii: Option<PyReadonlyArray1<'_, f32>>,
    probe_radius: f32,
    n_points: usize,
) -> PyResult<(f32, Bound<'py, PyArray1<f32>>)> {
    let c = readonly2_to_vec3(&coords)?;
    let r_vec: Option<Vec<f32>> = radii.map(|r| r.as_array().to_vec());
    if let Some(ref rv) = r_vec {
        if rv.len() != c.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radii length must match number of atoms in coords",
            ));
        }
    }
    let (total, per_atom) = py.allow_threads(|| {
        biophysics::calculate_sasa(&c, r_vec.as_deref(), probe_radius, n_points)
    });
    Ok((total, PyArray1::from_vec_bound(py, per_atom)))
}

#[pyfunction]
#[pyo3(signature = (target_coords, binder_coords, probe_radius=1.4, n_points=96, target_radii=None, binder_radii=None))]
fn calculate_bsa(
    py: Python<'_>,
    target_coords: PyReadonlyArray2<'_, f32>,
    binder_coords: PyReadonlyArray2<'_, f32>,
    probe_radius: f32,
    n_points: usize,
    target_radii: Option<PyReadonlyArray1<'_, f32>>,
    binder_radii: Option<PyReadonlyArray1<'_, f32>>,
) -> PyResult<(f32, f32, f32, f32)> {
    let tc = readonly2_to_vec3(&target_coords)?;
    let bc = readonly2_to_vec3(&binder_coords)?;
    let tr: Option<Vec<f32>> = target_radii.map(|r| r.as_array().to_vec());
    let br: Option<Vec<f32>> = binder_radii.map(|r| r.as_array().to_vec());
    if tr.as_ref().map_or(false, |r| r.len() != tc.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("target_radii length must match target_coords"));
    }
    if br.as_ref().map_or(false, |r| r.len() != bc.len()) {
        return Err(pyo3::exceptions::PyValueError::new_err("binder_radii length must match binder_coords"));
    }
    Ok(py.allow_threads(|| {
        biophysics::calculate_bsa(&tc, &bc, tr.as_deref(), br.as_deref(), probe_radius, n_points)
    }))
}

#[pyfunction]
#[pyo3(signature = (coords, atom_names, res_names, chain_ids, target_chain="A", binder_chain="B", ph_low=6.0, ph_high=7.4, ionic_strength=0.15, dielectric=20.0))]
fn calculate_ph_switch_ddg(
    py: Python<'_>,
    coords: PyReadonlyArray2<'_, f32>,
    atom_names: Vec<String>,
    res_names: Vec<String>,
    chain_ids: Vec<String>,
    target_chain: &str,
    binder_chain: &str,
    ph_low: f32,
    ph_high: f32,
    ionic_strength: f32,
    dielectric: f32,
) -> PyResult<(f32, f32, f32)> {
    let c = readonly2_to_vec3(&coords)?;
    let n = c.len();
    if atom_names.len() != n || res_names.len() != n || chain_ids.len() != n {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "coords, atom_names, res_names, and chain_ids must have identical length",
        ));
    }
    let tc = target_chain.to_string();
    let bc = binder_chain.to_string();
    Ok(py.allow_threads(|| {
        biophysics::calculate_ph_switch_ddg(
            &c,
            &atom_names,
            &res_names,
            &chain_ids,
            &tc,
            &bc,
            ph_low,
            ph_high,
            ionic_strength,
            dielectric,
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (ca_coords, cutoff=15.0, gamma=1.0, n_modes=6))]
fn anm_modes<'py>(
    py: Python<'py>,
    ca_coords: PyReadonlyArray2<'_, f32>,
    cutoff: f32,
    gamma: f32,
    n_modes: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray2<f32>>, Bound<'py, PyArray1<f32>>)> {
    let c = readonly2_to_vec3(&ca_coords)?;
    let n = c.len();
    let res = py.allow_threads(|| biophysics::calculate_anm_modes(&c, cutoff, gamma, n_modes));
    let k = res.eigenvalues.len();
    let py_evals = PyArray1::from_vec_bound(py, res.eigenvalues);
    let py_bfac = PyArray1::from_vec_bound(py, res.bfactors);
    let py_evecs = PyArray2::<f32>::zeros_bound(py, [k, n * 3], false);
    if k > 0 && n > 0 {
        let mut writer = py_evecs.readwrite();
        writer.as_slice_mut().unwrap().copy_from_slice(&res.eigenvectors);
    }
    Ok((py_evals, py_evecs, py_bfac))
}

#[pyfunction]
fn screen_sequences_biophysics<'py>(
    py: Python<'py>,
    sequences: Vec<String>,
) -> PyResult<Bound<'py, PyDict>> {
    let res = py.allow_threads(|| biophysics::screen_sequences_biophysics(&sequences));
    let dict = PyDict::new_bound(py);
    dict.set_item("isoelectric_point", PyArray1::from_vec_bound(py, res.isoelectric_point))?;
    dict.set_item("gravy", PyArray1::from_vec_bound(py, res.gravy))?;
    dict.set_item("net_charge_ph74", PyArray1::from_vec_bound(py, res.net_charge_ph74))?;
    dict.set_item("camsol_score", PyArray1::from_vec_bound(py, res.camsol_score))?;
    dict.set_item("apr_count", PyArray1::from_vec_bound(py, res.apr_count))?;
    dict.set_item("liability_count", PyArray1::from_vec_bound(py, res.liability_count))?;
    dict.set_item("hqnst_fraction", PyArray1::from_vec_bound(py, res.hqnst_fraction))?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (queries, reference_db, k=15))]
fn sliding_kmer_novelty<'py>(
    py: Python<'py>,
    queries: Vec<String>,
    reference_db: Vec<String>,
    k: usize,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<usize>>)> {
    let (idents, hammings) = py.allow_threads(|| {
        biophysics::sliding_kmer_novelty(&queries, &reference_db, k)
    });
    Ok((
        PyArray1::from_vec_bound(py, idents),
        PyArray1::from_vec_bound(py, hammings),
    ))
}

#[pymodule]
fn strux_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Msa>()?;
    m.add_function(wrap_pyfunction!(parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cif, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cif_with_plddt, m)?)?;
    m.add_function(wrap_pyfunction!(parse_cif_full, m)?)?;
    m.add_function(wrap_pyfunction!(parse_dcd, m)?)?;
    m.add_function(wrap_pyfunction!(parse_xtc, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_raw, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_kabsch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_lddt, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_pose_and_sc_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(batch_pose_and_sc_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(analyze_interface_physics, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sasa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_bsa, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_ph_switch_ddg, m)?)?;
    m.add_function(wrap_pyfunction!(anm_modes, m)?)?;
    m.add_function(wrap_pyfunction!(screen_sequences_biophysics, m)?)?;
    m.add_function(wrap_pyfunction!(sliding_kmer_novelty, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_trajectory_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_pairwise_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsf, m)?)?;
    m.add_function(wrap_pyfunction!(find_interface_contacts, m)?)?;
    m.add_function(wrap_pyfunction!(trajectory_pca, m)?)?;
    m.add_function(wrap_pyfunction!(markov_populations, m)?)?;
    m.add_function(wrap_pyfunction!(cluster_msa, m)?)?;
    m.add_function(wrap_pyfunction!(parse_a3m, m)?)?;
    m.add_function(wrap_pyfunction!(parse_a3m_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_stockholm, m)?)?;
    m.add_function(wrap_pyfunction!(parse_stockholm_file, m)?)?;

    m.add_function(wrap_pyfunction!(cuda_is_available, m)?)?;
    #[cfg(feature = "cuda")]
    {
        m.add_function(wrap_pyfunction!(cuda_pairwise_rmsd, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_cluster_daura, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_contact_map_bitmask, m)?)?;
        m.add_function(wrap_pyfunction!(cuda_contact_frequency, m)?)?;
    }
    Ok(())
}


