// ponytail: pyo3 module glue, minimal code, fast array conversions.
use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

pub mod analysis;
pub mod msa;
pub mod pdb;
pub mod spatial;

#[cfg(feature = "cuda")]
pub mod cuda;


pub use msa::Msa;

#[pyfunction]
fn parse_pdb(py: Python<'_>, path: &str) -> PyResult<Py<PyArray3<f32>>> {
    let path_str = path.to_string();
    let frames = py.allow_threads(|| {
        pdb::parse_pdb_trajectory(path_str)
    }).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    
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
fn calculate_rg(py: Python<'_>, coords: PyReadonlyArray2<'_, f32>) -> f32 {
    let view = coords.as_array();
    py.allow_threads(|| {
        let n = view.shape()[0];
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push([view[[i, 0]], view[[i, 1]], view[[i, 2]]]);
        }
        analysis::calculate_rg(&v)
    })
}

#[pyfunction]
fn calculate_rmsd_raw(
    py: Python<'_>,
    coords1: PyReadonlyArray2<'_, f32>,
    coords2: PyReadonlyArray2<'_, f32>,
) -> f32 {
    let view1 = coords1.as_array();
    let view2 = coords2.as_array();
    py.allow_threads(|| {
        let n1 = view1.shape()[0];
        let mut v1 = Vec::with_capacity(n1);
        for i in 0..n1 {
            v1.push([view1[[i, 0]], view1[[i, 1]], view1[[i, 2]]]);
        }
        let n2 = view2.shape()[0];
        let mut v2 = Vec::with_capacity(n2);
        for i in 0..n2 {
            v2.push([view2[[i, 0]], view2[[i, 1]], view2[[i, 2]]]);
        }
        analysis::calculate_rmsd_raw(&v1, &v2)
    })
}

#[pyfunction]
fn calculate_rmsd_kabsch(
    py: Python<'_>,
    coords1: PyReadonlyArray2<'_, f32>,
    coords2: PyReadonlyArray2<'_, f32>,
) -> f32 {
    let view1 = coords1.as_array();
    let view2 = coords2.as_array();
    py.allow_threads(|| {
        let n1 = view1.shape()[0];
        let mut v1 = Vec::with_capacity(n1);
        for i in 0..n1 {
            v1.push([view1[[i, 0]], view1[[i, 1]], view1[[i, 2]]]);
        }
        let n2 = view2.shape()[0];
        let mut v2 = Vec::with_capacity(n2);
        for i in 0..n2 {
            v2.push([view2[[i, 0]], view2[[i, 1]], view2[[i, 2]]]);
        }
        analysis::calculate_rmsd_kabsch(&v1, &v2)
    })
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
    let shape = view.shape();
    let num_frames = shape[0];
    let num_atoms = shape[1];

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
    let shape = view.shape();
    let num_frames = shape[0];
    let num_atoms = shape[1];

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
                if i == j {
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
fn calculate_rmsf(py: Python<'_>, trajectory: PyReadonlyArray3<'_, f32>) -> Vec<f32> {
    let view = trajectory.as_array();
    py.allow_threads(|| {
        let shape = view.shape();
        let num_frames = shape[0];
        let num_atoms = shape[1];
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
    })
}

#[pyfunction]
fn find_interface_contacts(
    py: Python<'_>,
    target: PyReadonlyArray2<'_, f32>,
    binder: PyReadonlyArray2<'_, f32>,
    cutoff: f32,
    box_dims: Option<[f32; 3]>,
) -> Vec<(usize, usize)> {
    let target_view = target.as_array();
    let binder_view = binder.as_array();
    py.allow_threads(|| {
        let n_t = target_view.shape()[0];
        let mut t = Vec::with_capacity(n_t);
        for i in 0..n_t {
            t.push([target_view[[i, 0]], target_view[[i, 1]], target_view[[i, 2]]]);
        }
        let n_b = binder_view.shape()[0];
        let mut b = Vec::with_capacity(n_b);
        for i in 0..n_b {
            b.push([binder_view[[i, 0]], binder_view[[i, 1]], binder_view[[i, 2]]]);
        }
        analysis::find_interface_contacts(&t, &b, cutoff, box_dims)
    })
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

#[pyfunction]
fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cuda::GpuTrajectoryEngine::new(0).is_ok()
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

    let engine = cuda::GpuTrajectoryEngine::new(0)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init error: {:?}", e)))?;

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

    let engine = cuda::GpuTrajectoryEngine::new(0)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init error: {:?}", e)))?;

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

    let engine = cuda::GpuTrajectoryEngine::new(0)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA init error: {:?}", e)))?;

    let bitmasks = py.allow_threads(|| {
        engine.contact_map_bitmask(slice, num_frames, num_res, cutoff)
    }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("CUDA contact error: {:?}", e)))?;

    let num_u64_per_frame = (num_res * num_res + 63) / 64;
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

#[pymodule]
fn strux_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Msa>()?;
    m.add_function(wrap_pyfunction!(parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_raw, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_kabsch, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_trajectory_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_pairwise_rmsd, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsf, m)?)?;
    m.add_function(wrap_pyfunction!(find_interface_contacts, m)?)?;
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
    }
    Ok(())
}

