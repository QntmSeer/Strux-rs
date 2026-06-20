// ponytail: pyo3 module glue, minimal code, fast array conversions.
use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

pub mod analysis;
pub mod pdb;
pub mod spatial;

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

#[pymodule]
fn strux_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_pdb, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rg, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_raw, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsd_kabsch, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_rmsf, m)?)?;
    m.add_function(wrap_pyfunction!(find_interface_contacts, m)?)?;
    Ok(())
}
