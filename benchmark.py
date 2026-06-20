# ponytail: comprehensive benchmark comparing Rust to Python/numpy.
import os
import glob
import time
import numpy as np
import strux_rs

def create_trajectory():
    frames_dir = "../MD/data/cabs_traj_design9/frames"
    pdb_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.pdb")))
    if not pdb_files:
        print("Error: No frame PDB files found.")
        return None
    
    trajectory_file = "trajectory.pdb"
    with open(trajectory_file, "w") as outfile:
        for idx, pdb in enumerate(pdb_files):
            outfile.write(f"MODEL        {idx + 1}\n")
            with open(pdb, "r") as infile:
                for line in infile:
                    if line.startswith(("ATOM", "HETATM")):
                        outfile.write(line)
            outfile.write("ENDMDL\n")
    print(f"Created trajectory.pdb with {len(pdb_files)} models.")
    return trajectory_file

def python_parse_pdb(path):
    frames = []
    current_frame = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("MODEL"):
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
            elif line.startswith("ENDMDL"):
                if current_frame:
                    frames.append(current_frame)
                    current_frame = []
            elif line.startswith(("ATOM", "HETATM")):
                if len(line) >= 54:
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        current_frame.append([x, y, z])
                    except ValueError:
                        pass
    if current_frame:
        frames.append(current_frame)
    return np.array(frames, dtype=np.float32)

def python_calculate_rg(coords):
    mean = np.mean(coords, axis=0)
    return np.sqrt(np.mean(np.sum((coords - mean) ** 2, axis=1)))

def python_rmsd_raw(c1, c2):
    return np.sqrt(np.mean(np.sum((c1 - c2) ** 2, axis=1)))

def python_rmsd_kabsch(p, q):
    pc = p - np.mean(p, axis=0)
    qc = q - np.mean(q, axis=0)
    h = pc.T @ qc
    u, s, vt = np.linalg.svd(h)
    d = np.linalg.det(u @ vt)
    w = np.eye(3)
    w[2, 2] = np.sign(d)
    r = u @ w @ vt
    p_aligned = pc @ r
    return np.sqrt(np.mean(np.sum((p_aligned - qc) ** 2, axis=1)))

def python_calculate_rmsf(traj):
    mean = np.mean(traj, axis=0)
    return np.sqrt(np.mean(np.sum((traj - mean) ** 2, axis=-1), axis=0))

def python_contacts_naive(target, binder, cutoff):
    diff = target[:, np.newaxis, :] - binder[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    t_idxs, b_idxs = np.where(dists < cutoff)
    return set(zip(t_idxs, b_idxs))

def main():
    traj_path = create_trajectory()
    if not traj_path:
        return

    print("\n=== RUNNING BENCHMARKS ===")
    
    # 1. PARSING SPEED
    t0 = time.perf_counter()
    py_traj = python_parse_pdb(traj_path)
    t_py_parse = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rs_traj = strux_rs.parse_pdb(traj_path)
    t_rs_parse = time.perf_counter() - t0
    
    print(f"PDB Parsing: Python = {t_py_parse:.4f}s, Rust = {t_rs_parse:.4f}s | Speedup = {t_py_parse/t_rs_parse:.1f}x")
    assert np.allclose(py_traj, rs_traj, atol=1e-3), "Parsing mismatch!"
    
    # Extract frame 0
    c0 = py_traj[0]
    c1 = py_traj[1]
    
    # 2. RADIUS OF GYRATION
    t0 = time.perf_counter()
    rg_py = python_calculate_rg(c0)
    t_py_rg = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rg_rs = strux_rs.calculate_rg(c0)
    t_rs_rg = time.perf_counter() - t0
    
    print(f"Radius of Gyration: Python = {t_py_rg:.6f}s, Rust = {t_rs_rg:.6f}s | Speedup = {t_py_rg/t_rs_rg:.1f}x")
    print(f"  Values: Python = {rg_py:.4f} Å, Rust = {rg_rs:.4f} Å")
    assert np.isclose(rg_py, rg_rs, atol=1e-4), "Rg mismatch!"
    
    # 3. RMSD RAW
    t0 = time.perf_counter()
    rmsd_py = python_rmsd_raw(c0, c1)
    t_py_rmsd = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rmsd_rs = strux_rs.calculate_rmsd_raw(c0, c1)
    t_rs_rmsd = time.perf_counter() - t0
    
    print(f"RMSD Raw: Python = {t_py_rmsd:.6f}s, Rust = {t_rs_rmsd:.6f}s | Speedup = {t_py_rmsd/t_rs_rmsd:.1f}x")
    print(f"  Values: Python = {rmsd_py:.4f} Å, Rust = {rmsd_rs:.4f} Å")
    assert np.isclose(rmsd_py, rmsd_rs, atol=1e-4), "RMSD raw mismatch!"
    
    # 4. RMSD KABSCH
    t0 = time.perf_counter()
    kabsch_py = python_rmsd_kabsch(c0, c1)
    t_py_kabsch = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    kabsch_rs = strux_rs.calculate_rmsd_kabsch(c0, c1)
    t_rs_kabsch = time.perf_counter() - t0
    
    print(f"RMSD Kabsch: Python = {t_py_kabsch:.6f}s, Rust = {t_rs_kabsch:.6f}s | Speedup = {t_py_kabsch/t_rs_kabsch:.1f}x")
    print(f"  Values: Python = {kabsch_py:.4f} Å, Rust = {kabsch_rs:.4f} Å")
    assert np.isclose(kabsch_py, kabsch_rs, atol=1e-4), "Kabsch RMSD mismatch!"
    
    # 5. RMSF
    t0 = time.perf_counter()
    rmsf_py = python_calculate_rmsf(py_traj)
    t_py_rmsf = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    rmsf_rs = strux_rs.calculate_rmsf(py_traj)
    t_rs_rmsf = time.perf_counter() - t0
    
    print(f"RMSF: Python = {t_py_rmsf:.6f}s, Rust = {t_rs_rmsf:.6f}s | Speedup = {t_py_rmsf/t_rs_rmsf:.1f}x")
    assert np.allclose(rmsf_py, rmsf_rs, atol=1e-4), "RMSF mismatch!"
    
    # 6. INTERFACE CONTACTS (Naive vs. Rust Spatial Hashing)
    target = c0[:1023]
    binder = c0[1023:]
    cutoff = 5.0
    
    t0 = time.perf_counter()
    contacts_py = python_contacts_naive(target, binder, cutoff)
    t_py_contacts = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    contacts_rs = set(strux_rs.find_interface_contacts(target, binder, cutoff, None))
    t_rs_contacts = time.perf_counter() - t0
    
    print(f"Interface Contacts (Cutoff={cutoff}Å): Python = {t_py_contacts:.4f}s, Rust = {t_rs_contacts:.4f}s | Speedup = {t_py_contacts/t_rs_contacts:.1f}x")
    print(f"  Contacts detected: Python = {len(contacts_py)}, Rust = {len(contacts_rs)}")
    assert contacts_py == contacts_rs, "Contacts mismatch!"

    print("\nSUCCESS: All calculations matched within tolerance!")

if __name__ == "__main__":
    main()
