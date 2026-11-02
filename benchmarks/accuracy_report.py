"""Accuracy of strux-rs against reference tools on the bundled trajectory.pdb.

Writes benchmarks/accuracy.json and prints the Markdown table used in the README.
Every number in that table comes from this script; do not edit the table by hand.

    pip install strux-rs mdtraj freesasa
    python benchmarks/accuracy_report.py

The functions here are also imported by tests/test_accuracy.py.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PDB = ROOT / "trajectory.pdb"
OUT = ROOT / "benchmarks" / "accuracy.json"

# Bondi-style van der Waals radii (A), the common FreeSASA / NACCESS element set.
ELEMENT_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}


def load():
    import mdtraj as md

    traj = md.load(str(PDB))
    t = traj.atom_slice(traj.topology.select("not element H"))
    xyz = np.ascontiguousarray(t.xyz * 10.0, dtype=np.float32)  # nm -> A
    top = t.topology
    atoms = list(top.atoms)
    meta = {
        "names": [a.name for a in atoms],
        "resn": [a.residue.name for a in atoms],
        "chains": [chr(ord("A") + a.residue.chain.index) for a in atoms],
        "resid": [a.residue.resSeq for a in atoms],
        "radii": np.array([ELEMENT_RADII.get(a.element.symbol, 1.70) for a in atoms], dtype=np.float32),
        "ca": top.select("name CA"),
    }
    return t, xyz, meta


def kabsch(P, Q):
    """Rotation R and centroids so that (Q - cq) @ R.T + cp best fits P (float64, NumPy SVD)."""
    P = P.astype(np.float64)
    Q = Q.astype(np.float64)
    cp, cq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((Q - cq).T @ (P - cp))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cp, cq


def kabsch_rmsd(P, Q):
    R, cp, cq = kabsch(P, Q)
    return float(np.sqrt(np.mean(np.sum((P - ((Q - cq) @ R.T + cp)) ** 2, 1))))


def lddt_reference(ref, mod, cutoff=15.0):
    """Pooled global lDDT and per-atom lDDT, as in AlphaFold lddt() / OpenStructure."""
    ref = ref.astype(np.float64)
    mod = mod.astype(np.float64)
    dr = np.linalg.norm(ref[:, None] - ref[None], axis=-1)
    dm = np.linalg.norm(mod[:, None] - mod[None], axis=-1)
    mask = (dr < cutoff) & ~np.eye(len(ref), dtype=bool)
    dd = np.abs(dr - dm)
    score = 0.25 * sum((dd < th).astype(float) for th in (0.5, 1.0, 2.0, 4.0))
    per = np.where(mask.sum(1) > 0, (score * mask).sum(1) / np.maximum(mask.sum(1), 1), 1.0)
    return float((score * mask).sum() / mask.sum()), per


def freesasa_total(coords, radii, n_points):
    import freesasa

    p = freesasa.Parameters()
    p.setAlgorithm(freesasa.ShrakeRupley)
    p.setNPoints(int(n_points))
    p.setProbeRadius(1.4)
    return freesasa.calcCoord(coords.astype(float).ravel(), [float(r) for r in radii], p).totalArea()


def run():
    import mdtraj as md
    import strux_rs as sx

    t, xyz, m = load()
    f0 = xyz[0]
    out = {"system": {"file": PDB.name, "frames": int(t.n_frames), "heavy_atoms": int(t.n_atoms)}}

    with tempfile.TemporaryDirectory() as d:
        xtc, dcd = str(Path(d) / "t.xtc"), str(Path(d) / "t.dcd")
        t.save_xtc(xtc)
        t.save_dcd(dcd)
        ca = list(m["ca"])
        for fmt, path, reader in (("xtc", xtc, sx.parse_xtc), ("dcd", dcd, sx.parse_dcd)):
            ref = md.load(path, top=t.topology).xyz * 10.0
            out[f"{fmt}_max_abs_diff_A"] = float(np.max(np.abs(reader(path) - ref)))
            out[f"{fmt}_subset_max_abs_diff_A"] = float(np.max(np.abs(reader(path, atom_indices=ca) - ref[:, ca])))

    out["kabsch_rmsd_max_abs_diff_A"] = max(
        abs(sx.calculate_rmsd_kabsch(f0, xyz[i]) - kabsch_rmsd(f0, xyz[i])) for i in range(1, t.n_frames))
    rg_ref = md.compute_rg(t) * 10.0
    out["rg_max_abs_diff_A"] = float(max(abs(sx.calculate_rg(xyz[i]) - rg_ref[i]) for i in range(t.n_frames)))

    caA = [i for i in m["ca"] if m["chains"][i] == "A"]
    caB = [i for i in m["ca"] if m["chains"][i] == "B"]
    rt, rb = np.ascontiguousarray(f0[caA]), np.ascontiguousarray(f0[caB])
    mt, mb = np.ascontiguousarray(xyz[1:, caA]), np.ascontiguousarray(xyz[1:, caB])
    sc, pose = sx.batch_pose_and_sc_rmsd(rt, rb, mt, mb)
    ref_pose, ref_sc = [], []
    for k in range(len(mt)):
        R, cp, cq = kabsch(rt, mt[k])
        ref_pose.append(np.sqrt(np.mean(np.sum((rb - ((mb[k] - cq) @ R.T + cp)) ** 2, 1))))
        ref_sc.append(kabsch_rmsd(rb, mb[k]))
    out["pose_rmsd_max_abs_diff_A"] = float(np.max(np.abs(np.asarray(pose) - ref_pose)))
    out["binder_rmsd_max_abs_diff_A"] = float(np.max(np.abs(np.asarray(sc) - ref_sc)))

    r = m["radii"]
    s_sx, _ = sx.calculate_sasa(f0, r, 1.4, 1000)
    s_fs = freesasa_total(f0, r, 1000)
    out["sasa_rel_diff_vs_freesasa"] = float(abs(s_sx - s_fs) / s_fs)

    A = np.array([c == "A" for c in m["chains"]])
    B = ~A
    bsa_r, *_ = sx.calculate_bsa(f0[A], f0[B], 1.4, 1000, r[A], r[B])
    bsa_fs = freesasa_total(f0[A], r[A], 1000) + freesasa_total(f0[B], r[B], 1000) - freesasa_total(f0, r, 1000)
    bsa_default, *_ = sx.calculate_bsa(f0[A], f0[B], 1.4, 1000)
    out["bsa_element_radii_rel_diff_vs_freesasa"] = float(abs(bsa_r - bsa_fs) / bsa_fs)
    out["bsa_default_radius_rel_diff_vs_freesasa"] = float(abs(bsa_default - bsa_fs) / bsa_fs)

    g_err, p_err = [], []
    ref_ca = f0[m["ca"]]
    for i in range(1, t.n_frames):
        g, per = sx.calculate_lddt(ref_ca, np.ascontiguousarray(xyz[i][m["ca"]]), 15.0)
        g_ref, per_ref = lddt_reference(ref_ca, xyz[i][m["ca"]])
        g_err.append(abs(g - g_ref))
        p_err.append(float(np.max(np.abs(np.asarray(per) - per_ref))))
    out["lddt_global_max_abs_diff"] = float(max(g_err))
    out["lddt_per_residue_max_abs_diff"] = float(max(p_err))
    return out


ROWS = [
    ("XTC read, full and atom subset", "MDTraj", "max(xtc_max_abs_diff_A, xtc_subset_max_abs_diff_A)", "{:.1e} Å"),
    ("DCD read, full and atom subset", "MDTraj", "max(dcd_max_abs_diff_A, dcd_subset_max_abs_diff_A)", "{:.1e} Å"),
    ("Kabsch RMSD", "NumPy SVD (float64)", "kabsch_rmsd_max_abs_diff_A", "{:.1e} Å"),
    ("Target-anchored pose RMSD", "NumPy SVD (float64)", "pose_rmsd_max_abs_diff_A", "{:.1e} Å"),
    ("Binder-only RMSD", "NumPy SVD (float64)", "binder_rmsd_max_abs_diff_A", "{:.1e} Å"),
    ("Radius of gyration", "MDTraj", "rg_max_abs_diff_A", "{:.1e} Å"),
    ("SASA, element radii, 1000 pts", "FreeSASA", "sasa_rel_diff_vs_freesasa", "{:.2%}"),
    ("BSA, element radii", "FreeSASA", "bsa_element_radii_rel_diff_vs_freesasa", "{:.2%}"),
    ("BSA, default 1.70 Å radius", "FreeSASA, element radii", "bsa_default_radius_rel_diff_vs_freesasa", "{:.2%}"),
    ("lDDT, global (pooled pairs)", "AlphaFold / OpenStructure formula", "lddt_global_max_abs_diff", "{:.1e}"),
    ("lDDT, per residue", "same formula", "lddt_per_residue_max_abs_diff", "{:.1e}"),
]


def table(res):
    lines = ["| Quantity | Reference | Max difference |", "| :--- | :--- | :--- |"]
    for name, ref, key, fmt in ROWS:
        val = eval(key, {"max": max}, res) if key.startswith("max(") else res[key]
        lines.append(f"| {name} | {ref} | {fmt.format(val)} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import strux_rs

    res = run()
    res["strux_rs_version"] = getattr(strux_rs, "__version__", "unknown")
    res["python"] = sys.version.split()[0]
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(table(res))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
