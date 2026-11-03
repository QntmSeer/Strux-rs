#!/usr/bin/env python3
"""
Standardized, Reproducible End-to-End & Hardware Scaling Benchmark Suite for strux-rs v0.4.0.
Executes:
  - Suite 1: Real-World Therapeutic Discovery .cif Triage at Scale (Challenge 05 Boltz-2 / AF3 ensembles)
  - Suite 2: Explicit-Solvent .xtc & .dcd Solute-Filtered AMS Loop & GPU VRAM Scaling (up to 20,000 x 20,000 frames)
  - Suite 3: 100,000-Sequence MPNN Biophysics / k-mer Novelty Triage & Zero-Copy AF-Cluster MSA Pipeline
Outputs reproducible JSON, CSV, and Markdown reports with hardware provenance.
"""
import csv
import glob
import json
import os
import platform
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import strux_rs


def get_hardware_provenance() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
        "cpu_threads": os.cpu_count() or 1,
        "cpu_model": platform.processor() or "Unknown CPU",
        "gpu_name": "None",
        "gpu_vram_mib": 0,
        "gpu_driver": "None",
        "gpu_power_limit_w": 0.0,
    }
    try:
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version,power.limit",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        parts = [p.strip() for p in out.split(",")]
        info["gpu_name"] = parts[0]
        info["gpu_vram_mib"] = int(float(parts[1]))
        info["gpu_driver"] = parts[2]
        info["gpu_power_limit_w"] = float(parts[3]) if parts[3] not in ("[N/A]", "N/A") else 0.0
    except Exception:
        pass
    return info


def bench_fn(fn, warmup: int = 1, repeats: int = 5):
    for _ in range(warmup):
        out = fn()
    times_ms = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms, dtype=np.float64)
    return out, float(np.median(arr)), float(np.percentile(arr, 75) - np.percentile(arr, 25))


# --- Pure Python / Biopython Reference Baselines ---

def py_parse_cif_simple(path: str):
    coords, bfacs, chains, res_ids, atoms = [], [], [], [], []
    in_atom = False
    headers = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s.startswith("_atom_site."):
                in_atom = True
                headers.append(s.split(".", 1)[1].split()[0])
            elif in_atom and (s.startswith("#") or s.startswith("loop_") or s.startswith("_")):
                if coords:
                    break
                in_atom = False
                headers = []
            elif in_atom and s:
                toks = s.split()
                if len(toks) < len(headers):
                    continue
                row = dict(zip(headers, toks))
                if row.get("group_PDB") not in ("ATOM", "HETATM"):
                    continue
                if row.get("label_alt_id", ".") not in (".", "?", "A"):
                    continue
                if row.get("type_symbol", "C") in ("H", "D"):
                    continue
                coords.append([float(row["Cartn_x"]), float(row["Cartn_y"]), float(row["Cartn_z"])])
                bfacs.append(float(row.get("B_iso_or_equiv", 0.0)))
                chains.append(row.get("auth_asym_id", row.get("label_asym_id", "A")))
                res_ids.append(int(row.get("auth_seq_id", row.get("label_seq_id", "1"))))
                atoms.append(row.get("label_atom_id", "CA"))
    return np.array(coords, dtype=np.float32), np.array(bfacs, dtype=np.float32), chains, res_ids, atoms


def py_pose_and_sc_rmsd(ref_t, ref_b, mob_t_batch, mob_b_batch):
    def kabsch(P, Q):
        cp, cq = P.mean(axis=0), Q.mean(axis=0)
        P0, Q0 = P - cp, Q - cq
        H = Q0.T @ P0
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = cp - cq @ R.T
        return R, t

    B = mob_t_batch.shape[0]
    sc_out = np.zeros(B, dtype=np.float32)
    pose_out = np.zeros(B, dtype=np.float32)
    for b in range(B):
        R_t, t_t = kabsch(ref_t, mob_t_batch[b])
        aligned_b = mob_b_batch[b] @ R_t.T + t_t
        pose_out[b] = np.sqrt(np.mean(np.sum((ref_b - aligned_b) ** 2, axis=1)))
        R_b, t_b = kabsch(ref_b, mob_b_batch[b])
        self_b = mob_b_batch[b] @ R_b.T + t_b
        sc_out[b] = np.sqrt(np.mean(np.sum((ref_b - self_b) ** 2, axis=1)))
    return sc_out, pose_out


def py_interface_physics(coords, atom_names, res_names, chain_ids, res_ids):
    c_arr = np.asarray(coords, dtype=np.float32)
    chains = np.asarray(chain_ids)
    mask_a = chains == chains[0]
    mask_b = ~mask_a
    ca, cb = c_arr[mask_a], c_arr[mask_b]
    diff = ca[:, None, :] - cb[None, :, :]
    d2 = np.sum(diff * diff, axis=-1)
    contacts = int(np.sum(d2 <= 25.0))
    # All-atom polar distance check
    is_polar = np.array([a[:1] in ("N", "O") for a in atom_names])
    polar_idx = np.where(is_polar)[0]
    p_coords = c_arr[polar_idx]
    pd2 = np.sum((p_coords[:, None, :] - p_coords[None, :, :]) ** 2, axis=-1)
    hbonds = int(np.sum((pd2 <= 12.25) & (pd2 > 0.1))) // 2
    return contacts, hbonds


def py_shrake_rupley_sasa(coords, radii, n_points=96, probe=1.4):
    N = coords.shape[0]
    idx = np.arange(n_points, dtype=np.float32)
    phi = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - (idx / float(n_points - 1)) * 2.0
    r = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = phi * idx
    sphere = np.stack([np.cos(theta) * r, y, np.sin(theta) * r], axis=-1)
    eff_r = radii + probe
    sasa = np.zeros(N, dtype=np.float32)
    dmat = np.sqrt(np.sum((coords[:, None, :] - coords[None, :, :]) ** 2, axis=-1))
    for i in range(N):
        nbrs = np.where((dmat[i] < eff_r[i] + eff_r) & (np.arange(N) != i))[0]
        if len(nbrs) == 0:
            sasa[i] = 4.0 * np.pi * eff_r[i] ** 2
            continue
        pts = coords[i] + sphere * eff_r[i]
        diff = pts[:, None, :] - coords[nbrs][None, :, :]
        dist2 = np.sum(diff * diff, axis=-1)
        exposed = ~np.any(dist2 < (eff_r[nbrs] ** 2)[None, :], axis=1)
        sasa[i] = (4.0 * np.pi * eff_r[i] ** 2) * (np.sum(exposed) / float(n_points))
    return float(np.sum(sasa))


def py_ph_switch_ddg(coords, res_names, chain_ids, ph_low=6.0, ph_high=7.4, ionic=0.15):
    pka_map = {"ASP": (3.9, False), "GLU": (4.3, False), "HIS": (6.0, True), "LYS": (10.5, True), "ARG": (12.5, True)}
    idx, q_low, q_high = [], [], []
    for i, r in enumerate(res_names):
        if r in pka_map:
            pka, is_base = pka_map[r]
            ql = (1.0 / (1.0 + 10.0 ** (ph_low - pka))) if is_base else (-1.0 / (1.0 + 10.0 ** (pka - ph_low)))
            qh = (1.0 / (1.0 + 10.0 ** (ph_high - pka))) if is_base else (-1.0 / (1.0 + 10.0 ** (pka - ph_high)))
            idx.append(i)
            q_low.append(ql)
            q_high.append(qh)
    idx = np.array(idx)
    c = coords[idx]
    ch = np.array([chain_ids[i] for i in idx])
    ql = np.array(q_low)
    qh = np.array(q_high)
    kappa = np.sqrt(ionic) / 3.04
    d = np.maximum(np.sqrt(np.sum((c[:, None, :] - c[None, :, :]) ** 2, axis=-1)), 2.5)
    inter = ch[:, None] != ch[None, :]
    pref = (332.0637 / 8.0) * np.exp(-kappa * d) / d
    g_low = 0.5 * np.sum(pref * inter * (ql[:, None] * ql[None, :]))
    g_high = 0.5 * np.sum(pref * inter * (qh[:, None] * qh[None, :]))
    return float(g_low - g_high)


def py_anm_modes(ca_coords, cutoff=15.0, n_modes=20):
    N = ca_coords.shape[0]
    H = np.zeros((3 * N, 3 * N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            d = ca_coords[j] - ca_coords[i]
            r2 = float(np.dot(d, d))
            if 1e-6 < r2 <= cutoff * cutoff:
                sub = -np.outer(d, d) / r2
                H[3 * i : 3 * i + 3, 3 * j : 3 * j + 3] = sub
                H[3 * j : 3 * j + 3, 3 * i : 3 * i + 3] = sub
                H[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] -= sub
                H[3 * j : 3 * j + 3, 3 * j : 3 * j + 3] -= sub
    evals, evecs = np.linalg.eigh(H)
    return evals[6 : 6 + n_modes], evecs[:, 6 : 6 + n_modes]


def py_screen_sequences(seqs):
    kd = dict(zip("ACDEFGHIKLMNPQRSTVWY", [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
    out = np.zeros((len(seqs), 6), dtype=np.float32)
    for idx, s in enumerate(seqs):
        n = max(len(s), 1)
        gravy = sum(kd.get(c, 0.0) for c in s) / n
        liab = sum(1 for i in range(len(s) - 1) if s[i : i + 2] in ("NG", "NS", "NT", "DG", "DS", "DP"))
        out[idx, 0] = -gravy
        out[idx, 3] = gravy
        out[idx, 5] = liab
    return out


def py_kmer_novelty(queries, refs, k=15):
    ref_kmers = [r[i : i + k] for r in refs for i in range(max(0, len(r) - k + 1))]
    scores = np.zeros(len(queries), dtype=np.float32)
    for qi, q in enumerate(queries):
        best = 0.0
        for i in range(max(0, len(q) - k + 1)):
            qw = q[i : i + k]
            for rw in ref_kmers:
                m = sum(1 for a, b in zip(qw, rw) if a == b) / float(k)
                if m > best:
                    best = m
                    if best >= 1.0:
                        break
            if best >= 1.0:
                break
        scores[qi] = best
    return scores


def create_synthetic_boltz_cif(path: Path, n_target_res: int = 180, n_binder_res: int = 110, seed: int = 42):
    rng = np.random.default_rng(seed)
    lines = [
        "data_boltz2_complex",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_seq_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    res_pool = ["ALA", "ASP", "GLU", "HIS", "LYS", "ARG", "SER", "TYR", "LEU", "VAL"]
    atom_defs = [("N", "N", [-1.2, 0.3, 0.0]), ("CA", "C", [0.0, 0.0, 0.0]), ("C", "C", [1.3, 0.4, 0.0]), ("O", "O", [1.5, 1.6, 0.0])]
    atom_id = 1
    for chain_idx, (chain, n_res) in enumerate([("A", n_target_res), ("B", n_binder_res)]):
        for r in range(1, n_res + 1):
            rname = res_pool[(r + chain_idx * 7) % len(res_pool)]
            theta = r * 0.35
            base = np.array([
                12.0 * np.cos(theta) + chain_idx * 14.0,
                12.0 * np.sin(theta),
                r * 0.55 - (n_res * 0.27),
            ]) + rng.normal(0, 0.15, size=3)
            plddt = 88.5 - 10.0 * chain_idx
            for aname, elem, offset in atom_defs:
                pos = base + np.array(offset)
                lines.append(
                    f"ATOM {atom_id} {elem} {aname} . {rname} {chain} {r} ? "
                    f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 1.00 {plddt:.2f} {chain} {r} 1"
                )
                atom_id += 1
            if rname in ("ASP", "GLU"):
                pos = base + np.array([0.0, -1.8, -1.2])
                lines.append(
                    f"ATOM {atom_id} O OD1 . {rname} {chain} {r} ? "
                    f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 1.00 {plddt:.2f} {chain} {r} 1"
                )
                atom_id += 1
            elif rname in ("LYS", "ARG", "HIS"):
                an = "NZ" if rname == "LYS" else ("NH1" if rname == "ARG" else "ND1")
                pos = base + np.array([-0.5, -1.9, -1.1])
                lines.append(
                    f"ATOM {atom_id} N {an} . {rname} {chain} {r} ? "
                    f"{pos[0]:.3f} {pos[1]:.3f} {pos[2]:.3f} 1.00 {plddt:.2f} {chain} {r} 1"
                )
                atom_id += 1
    path.write_text("\n".join(lines) + "\n")


def main():
    hw = get_hardware_provenance()
    print("=" * 88)
    v_str = getattr(strux_rs, "__version__", "unknown")
    print(f"STRUX-RS v{v_str} STANDARDIZED BENCHMARK SUITE — {hw['hostname'].upper()}")
    print(f"CPU: {hw['cpu_model']} ({hw['cpu_threads']} threads) | GPU: {hw['gpu_name']} ({hw['gpu_vram_mib']} MiB, {hw['gpu_power_limit_w']}W)")
    print("=" * 88)

    records = []
    tmpdir = Path(tempfile.mkdtemp(prefix="strux_std_bench_"))

    try:
        # =========================================================================
        # SUITE 1: Real-World Therapeutic Discovery .cif Triage at Scale
        # =========================================================================
        print("\n[Suite 1] Real-World Therapeutic Discovery .cif Triage at Scale")
        real_cifs = sorted(glob.glob("/home/ubuntu/Challenge_05/results/stage4_boltz2/**/*.cif", recursive=True))
        if real_cifs:
            print(f"  -> Found {len(real_cifs)} real Challenge 05 Boltz-2 co-folded .cif structures on disk!")
            cif_sample_paths = [Path(p) for p in real_cifs[:50]]
            dataset_source = f"Challenge_05_Boltz2 ({len(real_cifs)} real .cif files)"
        else:
            print("  -> Generating 50 multi-chain Boltz-2 schema .cif structures (290 res, ~1,450 heavy atoms each)...")
            cif_sample_paths = []
            for i in range(50):
                p = tmpdir / f"boltz2_complex_{i:03d}.cif"
                create_synthetic_boltz_cif(p, seed=100 + i)
                cif_sample_paths.append(p)
            dataset_source = "Synthetic_Boltz2_MultiChain (50 .cif files, 290 res)"

        # 1A: Parse 1,000 .cif files (tiled from 50 unique files x 20)
        cif_1000 = [str(cif_sample_paths[i % len(cif_sample_paths)]) for i in range(1000)]
        _, t_strux_cif1k, iqr_strux_cif1k = bench_fn(
            lambda: [strux_rs.parse_cif_full(p) for p in cif_1000], warmup=1, repeats=3
        )
        # Baseline: Python dynamic _atom_site parser on 50 files, scaled to 1,000
        _, t_py_cif50, _ = bench_fn(
            lambda: [py_parse_cif_simple(p) for p in cif_1000[:50]], warmup=1, repeats=3
        )
        t_py_cif1k = t_py_cif50 * 20.0
        speedup_cif = t_py_cif1k / t_strux_cif1k
        print(f"  1A. Parse 1,000 .cif complexes (+pLDDT): strux-rs = {t_strux_cif1k:.2f} ms | Python = {t_py_cif1k:.1f} ms ({speedup_cif:.1f}x)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "parse_cif_full_1000_complexes",
            "strux_ms": round(t_strux_cif1k, 3),
            "strux_iqr_ms": round(iqr_strux_cif1k, 3),
            "baseline_ms": round(t_py_cif1k, 3),
            "baseline_name": "Python_MMCIF_Parser",
            "speedup": round(speedup_cif, 2),
        })

        # Extract a reference complex for 1B-1E
        coords0_raw, bfac0_raw, chains0, res_ids0_raw, res_names0, atoms0 = strux_rs.parse_cif_full(cif_1000[0])
        coords0 = coords0_raw[0] if coords0_raw.ndim == 3 else coords0_raw
        res_ids0 = list(res_ids0_raw)

        # 1B: Target-Anchored Pose_RMSD & scRMSD across 10,000 designs
        rng = np.random.default_rng(42)
        ref_t_ca = rng.standard_normal((180, 3), dtype=np.float32) * 12.0
        ref_b_ca = rng.standard_normal((110, 3), dtype=np.float32) * 8.0 + np.array([18.0, 0.0, 0.0], dtype=np.float32)
        mob_t_10k = np.broadcast_to(ref_t_ca, (10000, 180, 3)).copy() + rng.standard_normal((10000, 180, 3), dtype=np.float32) * 0.4
        mob_b_10k = np.broadcast_to(ref_b_ca, (10000, 110, 3)).copy() + rng.standard_normal((10000, 110, 3), dtype=np.float32) * 1.2

        (sc_rs, pose_rs), t_strux_pose10k, iqr_pose10k = bench_fn(
            lambda: strux_rs.batch_pose_and_sc_rmsd(ref_t_ca, ref_b_ca, mob_t_10k, mob_b_10k), warmup=2, repeats=5
        )
        (sc_py, pose_py), t_py_pose1k, _ = bench_fn(
            lambda: py_pose_and_sc_rmsd(ref_t_ca, ref_b_ca, mob_t_10k[:1000], mob_b_10k[:1000]), warmup=1, repeats=2
        )
        t_py_pose10k = t_py_pose1k * 10.0
        max_pose_err = float(np.max(np.abs(pose_rs[:1000] - pose_py)))
        speedup_pose = t_py_pose10k / t_strux_pose10k
        print(f"  1B. Target-Anchored Pose+scRMSD (10,000 designs): strux-rs = {t_strux_pose10k:.2f} ms | NumPy SVD = {t_py_pose10k:.1f} ms ({speedup_pose:.1f}x, err={max_pose_err:.2e} A)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "batch_pose_and_sc_rmsd_10000",
            "strux_ms": round(t_strux_pose10k, 3),
            "strux_iqr_ms": round(iqr_pose10k, 3),
            "baseline_ms": round(t_py_pose10k, 3),
            "baseline_name": "NumPy_SVD_Superimposer",
            "speedup": round(speedup_pose, 2),
        })

        # 1C: All-Atom Interface Physics (buns, salt bridges, H-bonds) across 1,000 complexes
        _, t_strux_phys1, iqr_phys1 = bench_fn(
            lambda: strux_rs.analyze_interface_physics(coords0, atoms0, res_names0, chains0, res_ids0), warmup=3, repeats=7
        )
        _, t_py_phys1, _ = bench_fn(
            lambda: py_interface_physics(coords0, atoms0, res_names0, chains0, res_ids0), warmup=1, repeats=5
        )
        t_strux_phys1k = t_strux_phys1 * 1000.0
        t_py_phys1k = t_py_phys1 * 1000.0
        speedup_phys = t_py_phys1k / t_strux_phys1k
        print(f"  1C. All-Atom Interface Physics + BUNS (1,000 complexes): strux-rs = {t_strux_phys1k:.1f} ms | Python O(N^2) = {t_py_phys1k:.1f} ms ({speedup_phys:.1f}x)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "analyze_interface_physics_1000_complexes",
            "strux_ms": round(t_strux_phys1k, 3),
            "strux_iqr_ms": round(iqr_phys1 * 1000.0, 3),
            "baseline_ms": round(t_py_phys1k, 3),
            "baseline_name": "Python_NumPy_Interface_Screener",
            "speedup": round(speedup_phys, 2),
        })

        # 1D: Shrake-Rupley SASA & BSA (96-point Fibonacci sphere)
        unique_chains = sorted(set(chains0))
        target_chain = unique_chains[0]
        binder_chain = unique_chains[1] if len(unique_chains) > 1 else unique_chains[0]
        chains_arr = np.array(chains0)
        target_coords = coords0[chains_arr == target_chain]
        binder_coords = coords0[chains_arr == binder_chain]
        if binder_coords.shape[0] == 0:
            half = coords0.shape[0] // 2
            target_coords, binder_coords = coords0[:half], coords0[half:]

        bsa_out, t_strux_bsa1, iqr_bsa1 = bench_fn(
            lambda: strux_rs.calculate_bsa(target_coords, binder_coords, 1.4, 96), warmup=2, repeats=5
        )
        radii_t = np.full(target_coords.shape[0], 1.7, dtype=np.float32)
        _, t_py_sasa1, _ = bench_fn(
            lambda: py_shrake_rupley_sasa(target_coords, radii_t, 96, 1.4), warmup=1, repeats=2
        )
        t_py_bsa1 = t_py_sasa1 * 2.0  # Complex + monomer SASA passes
        speedup_bsa = t_py_bsa1 / t_strux_bsa1
        print(f"  1D. Shrake-Rupley BSA ({target_coords.shape[0]}+{binder_coords.shape[0]} atoms): strux-rs = {t_strux_bsa1:.3f} ms | Python = {t_py_bsa1:.1f} ms ({speedup_bsa:.1f}x)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "calculate_bsa_shrake_rupley_per_complex",
            "strux_ms": round(t_strux_bsa1, 3),
            "strux_iqr_ms": round(iqr_bsa1, 3),
            "baseline_ms": round(t_py_bsa1, 3),
            "baseline_name": "Python_ShrakeRupley_96pt",
            "speedup": round(speedup_bsa, 2),
        })

        # 1E: Screened Debye-Hückel pH-Switch ddG & ANM Normal Modes
        _, t_strux_ph, iqr_ph = bench_fn(
            lambda: strux_rs.calculate_ph_switch_ddg(coords0, atoms0, res_names0, chains0, target_chain, binder_chain, 6.0, 7.4, 0.15, 8.0), warmup=5, repeats=10
        )
        _, t_py_ph, _ = bench_fn(
            lambda: py_ph_switch_ddg(coords0, res_names0, chains0, 6.0, 7.4, 0.15), warmup=2, repeats=5
        )
        speedup_ph = t_py_ph / t_strux_ph
        print(f"  1E-i. Debye-Hückel pH-Switch ddG (pH 6.0 vs 7.4): strux-rs = {t_strux_ph:.4f} ms | Python = {t_py_ph:.3f} ms ({speedup_ph:.1f}x)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "calculate_ph_switch_ddg",
            "strux_ms": round(t_strux_ph, 4),
            "strux_iqr_ms": round(iqr_ph, 4),
            "baseline_ms": round(t_py_ph, 4),
            "baseline_name": "Python_DebyeHuckel_Screener",
            "speedup": round(speedup_ph, 2),
        })

        ca_mask = np.array([a == "CA" for a in atoms0])
        ca_coords = coords0[ca_mask][:180]
        if ca_coords.shape[0] < 30:
            ca_coords = rng.standard_normal((150, 3), dtype=np.float32) * 15.0
        _, t_strux_anm, iqr_anm = bench_fn(
            lambda: strux_rs.anm_modes(ca_coords, 15.0, 20), warmup=1, repeats=3
        )
        _, t_py_anm, _ = bench_fn(
            lambda: py_anm_modes(ca_coords, 15.0, 20), warmup=1, repeats=2
        )
        speedup_anm = t_py_anm / t_strux_anm
        print(f"  1E-ii. ANM Normal Modes ({ca_coords.shape[0]} C-alpha): strux-rs = {t_strux_anm:.2f} ms | Python/NumPy = {t_py_anm:.2f} ms ({speedup_anm:.1f}x)")
        records.append({
            "suite": "Suite1_CIF_Triage",
            "benchmark": "anm_modes_hessian_eigensolver",
            "strux_ms": round(t_strux_anm, 3),
            "strux_iqr_ms": round(iqr_anm, 3),
            "baseline_ms": round(t_py_anm, 3),
            "baseline_name": "Python_ANM_Hessian_eigh",
            "speedup": round(speedup_anm, 2),
        })

        # =========================================================================
        # SUITE 2: Solute-Filtered .xtc / .dcd AMS Loop & GPU VRAM Scaling
        # =========================================================================
        print("\n[Suite 2] Explicit-Solvent .xtc & .dcd Solute-Filtered AMS Loop & GPU VRAM Scaling")
        from mdtraj.formats import XTCTrajectoryFile, DCDTrajectoryFile

        n_frames_io = 200
        n_total_atoms = 40000  # 90% water/ions (36,000 solvent + 4,000 protein atoms, 400 C-alpha)
        ca_indices = list(range(0, 4000, 10))  # 400 C-alpha atoms (1% of total box atoms!)
        solvated_traj_nm = (rng.standard_normal((n_frames_io, n_total_atoms, 3), dtype=np.float32) * 2.5)
        time_arr = np.arange(n_frames_io, dtype=np.float32) * 0.5
        step_arr = np.arange(n_frames_io, dtype=np.int32) * 100
        box_arr = np.tile(np.eye(3, dtype=np.float32) * 10.0, (n_frames_io, 1, 1))

        xtc_path = str(tmpdir / "solvated_40k.xtc")
        dcd_path = str(tmpdir / "solvated_40k.dcd")

        with XTCTrajectoryFile(xtc_path, "w") as f:
            f.write(solvated_traj_nm, time=time_arr, step=step_arr, box=box_arr)

        with DCDTrajectoryFile(dcd_path, "w") as f:
            f.write(solvated_traj_nm * 10.0)  # DCD in Angstroms

        full_xtc, t_xtc_full, _ = bench_fn(lambda: strux_rs.parse_xtc(xtc_path), warmup=1, repeats=3)
        sub_xtc, t_xtc_sub, iqr_xtc_sub = bench_fn(
            lambda: strux_rs.parse_xtc(xtc_path, atom_indices=ca_indices), warmup=1, repeats=5
        )
        full_dcd, t_dcd_full, _ = bench_fn(lambda: strux_rs.parse_dcd(dcd_path), warmup=1, repeats=3)
        sub_dcd, t_dcd_sub, iqr_dcd_sub = bench_fn(
            lambda: strux_rs.parse_dcd(dcd_path, atom_indices=ca_indices), warmup=1, repeats=5
        )
        assert np.allclose(full_xtc[:, ca_indices, :], sub_xtc, atol=1e-4)
        assert np.allclose(full_dcd[:, ca_indices, :], sub_dcd, atol=1e-4)

        ram_full_mb = full_xtc.nbytes / (1024 * 1024)
        ram_sub_mb = sub_xtc.nbytes / (1024 * 1024)
        print(
            f"  2A-i. Solvated .xtc ({n_frames_io}f x {n_total_atoms} atoms -> {len(ca_indices)} CA): "
            f"Full = {t_xtc_full:.1f} ms ({ram_full_mb:.1f} MB) | Solute-Filtered = {t_xtc_sub:.1f} ms ({ram_sub_mb:.2f} MB, {ram_full_mb/ram_sub_mb:.0f}x RAM reduction)"
        )
        print(
            f"  2A-ii. Solvated .dcd ({n_frames_io}f x {n_total_atoms} atoms -> {len(ca_indices)} CA): "
            f"Full = {t_dcd_full:.1f} ms ({ram_full_mb:.1f} MB) | Solute-Filtered = {t_dcd_sub:.1f} ms ({ram_sub_mb:.2f} MB, {t_dcd_full/t_dcd_sub:.1f}x faster)"
        )
        records.append({
            "suite": "Suite2_Trajectory_GPU",
            "benchmark": "parse_xtc_solute_filtered_40k_to_400",
            "strux_ms": round(t_xtc_sub, 3),
            "strux_iqr_ms": round(iqr_xtc_sub, 3),
            "baseline_ms": round(t_xtc_full, 3),
            "baseline_name": "Full_Solvated_XTC_Load",
            "speedup": round(t_xtc_full / t_xtc_sub, 2),
        })
        records.append({
            "suite": "Suite2_Trajectory_GPU",
            "benchmark": "parse_dcd_solute_filtered_40k_to_400",
            "strux_ms": round(t_dcd_sub, 3),
            "strux_iqr_ms": round(iqr_dcd_sub, 3),
            "baseline_ms": round(t_dcd_full, 3),
            "baseline_name": "Full_Solvated_DCD_Load",
            "speedup": round(t_dcd_full / t_dcd_sub, 2),
        })
        del full_xtc, full_dcd, solvated_traj_nm

        # 2B: GPU Pairwise QCP RMSD Scaling up to 20,000 x 20,000 frames (400M alignments)
        has_cuda = hasattr(strux_rs, "cuda_pairwise_rmsd")
        vram_mib = hw["gpu_vram_mib"]
        frame_tiers = [1000, 2500, 5000, 10000]
        if vram_mib >= 12000:
            frame_tiers.extend([15000, 20000])

        n_atoms_qcp = 250
        # Measure CPU 1000x1000 baseline once
        traj_1k = rng.standard_normal((1000, n_atoms_qcp, 3), dtype=np.float32) * 10.0
        _, t_cpu_1k, iqr_cpu_1k = bench_fn(lambda: strux_rs.cpu_pairwise_rmsd(traj_1k), warmup=1, repeats=3)
        print(f"  2B-CPU. Rayon CPU pairwise_rmsd (1,000 x 1,000, 250 atoms): {t_cpu_1k:.2f} ms")

        if has_cuda:
            for n_f in frame_tiers:
                traj_f = rng.standard_normal((n_f, n_atoms_qcp, 3), dtype=np.float32) * 10.0
                reps = 5 if n_f <= 5000 else 3
                dmat_gpu, t_gpu, iqr_gpu = bench_fn(
                    lambda: strux_rs.cuda_pairwise_rmsd(traj_f), warmup=1, repeats=reps
                )
                pairs_m = (n_f * n_f) / 1e6
                atom_pairs_b = (n_f * n_f * n_atoms_qcp) / 1e9
                throughput_b_atom_pairs_s = atom_pairs_b / (t_gpu / 1000.0)
                est_cpu_ms = t_cpu_1k * ((n_f / 1000.0) ** 2)
                speedup_vs_cpu = est_cpu_ms / t_gpu
                print(
                    f"  2B-GPU. cuda_pairwise_rmsd ({n_f:,} x {n_f:,} = {pairs_m:.1f}M pairs): "
                    f"GPU = {t_gpu:.2f} ms (IQR {iqr_gpu:.2f} ms) | {throughput_b_atom_pairs_s:.2f} B atom-pairs/s | {speedup_vs_cpu:.2f}x vs CPU"
                )
                records.append({
                    "suite": "Suite2_Trajectory_GPU",
                    "benchmark": f"cuda_pairwise_rmsd_{n_f}x{n_f}_250atoms",
                    "strux_ms": round(t_gpu, 3),
                    "strux_iqr_ms": round(iqr_gpu, 3),
                    "baseline_ms": round(est_cpu_ms, 3),
                    "baseline_name": f"Rayon_CPU_{hw['cpu_threads']}T",
                    "speedup": round(speedup_vs_cpu, 2),
                })
                del dmat_gpu, traj_f

            # 2C: Full GPU + Rayon AMS Feedback Loop (5,000 frames x 250 atoms)
            traj_ams = rng.standard_normal((5000, n_atoms_qcp, 3), dtype=np.float32) * 10.0
            ref_ams = traj_ams[0].copy()

            def run_ams_cycle():
                lddt = strux_rs.calculate_lddt(ref_ams, traj_ams[1], 15.0)
                evals, evecs, proj = strux_rs.trajectory_pca(traj_ams, 10)
                labels, centroids = strux_rs.cuda_cluster_daura(traj_ams, 2.5)
                cfreq = strux_rs.cuda_contact_frequency(traj_ams, 8.0, 0)
                n_states = min(int(np.max(labels)) + 1, 50)
                mod_labels = np.mod(labels.astype(np.uint32), max(n_states, 2))
                pops = strux_rs.markov_populations(mod_labels, max(n_states, 2), 1)
                return lddt, evals, pops, cfreq

            _, t_ams_5k, iqr_ams_5k = bench_fn(run_ams_cycle, warmup=1, repeats=3)
            print(f"  2C. Complete 5,000-Frame AMS Loop (lDDT + PCA + GPU Daura + GPU ContactFreq + MSM): {t_ams_5k:.2f} ms (IQR {iqr_ams_5k:.2f} ms)")
            records.append({
                "suite": "Suite2_Trajectory_GPU",
                "benchmark": "full_ams_loop_5000frames_250atoms",
                "strux_ms": round(t_ams_5k, 3),
                "strux_iqr_ms": round(iqr_ams_5k, 3),
                # No baseline was run, so no speedup is reported (timing only).
                "baseline_ms": None,
                "baseline_name": None,
                "speedup": None,
            })
            del traj_ams

        # =========================================================================
        # SUITE 3: 100,000-Sequence MPNN Biophysics / k-mer Novelty & AF-Cluster
        # =========================================================================
        print("\n[Suite 3] 100,000-Sequence MPNN Developability / k-mer Novelty & AF-Cluster MSA")
        real_seqs = []
        cand_csv = Path("/home/ubuntu/Challenge_05/results/stage2_potts/stage2_potts_candidates.csv")
        if cand_csv.exists():
            with open(cand_csv, "r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for k in ("seq", "sequence", "binder_seq", "Sequence"):
                        if k in row and row[k]:
                            real_seqs.append(row[k].strip())
                            break
        if not real_seqs:
            alphabet = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
            real_seqs = ["".join(alphabet[rng.integers(0, 20, size=120)]) for _ in range(1000)]

        seqs_100k = [real_seqs[i % len(real_seqs)] for i in range(100000)]
        _, t_strux_seq100k, iqr_seq100k = bench_fn(
            lambda: strux_rs.screen_sequences_biophysics(seqs_100k), warmup=1, repeats=5
        )
        _, t_py_seq5k, _ = bench_fn(lambda: py_screen_sequences(seqs_100k[:5000]), warmup=1, repeats=3)
        t_py_seq100k = t_py_seq5k * 20.0
        speedup_seq = t_py_seq100k / t_strux_seq100k
        print(f"  3A. screen_sequences_biophysics (100,000 MPNN sequences): strux-rs = {t_strux_seq100k:.2f} ms | Python = {t_py_seq100k:.1f} ms ({speedup_seq:.1f}x)")
        records.append({
            "suite": "Suite3_Sequence_MSA",
            "benchmark": "screen_sequences_biophysics_100000",
            "strux_ms": round(t_strux_seq100k, 3),
            "strux_iqr_ms": round(iqr_seq100k, 3),
            "baseline_ms": round(t_py_seq100k, 3),
            "baseline_name": "Python_Biophysics_Screener",
            "speedup": round(speedup_seq, 2),
        })

        # 3B: Sliding 15-mer novelty (5,000 queries vs 500 reference binders)
        queries_5k = seqs_100k[:5000]
        refs_500 = [s[::-1] for s in real_seqs[:500]]
        _, t_strux_kmer, iqr_kmer = bench_fn(
            lambda: strux_rs.sliding_kmer_novelty(queries_5k, refs_500, 15), warmup=1, repeats=3
        )
        _, t_py_kmer50, _ = bench_fn(
            lambda: py_kmer_novelty(queries_5k[:50], refs_500[:100], 15), warmup=1, repeats=1
        )
        t_py_kmer_full = t_py_kmer50 * 100.0 * 5.0
        speedup_kmer = t_py_kmer_full / t_strux_kmer
        print(f"  3B. sliding_kmer_novelty (5,000 queries x 500 refs, k=15): strux-rs = {t_strux_kmer:.2f} ms | Python = {t_py_kmer_full:.1f} ms ({speedup_kmer:.1f}x)")
        records.append({
            "suite": "Suite3_Sequence_MSA",
            "benchmark": "sliding_kmer_novelty_5000x500_k15",
            "strux_ms": round(t_strux_kmer, 3),
            "strux_iqr_ms": round(iqr_kmer, 3),
            "baseline_ms": round(t_py_kmer_full, 3),
            "baseline_name": "Python_Sliding_Kmer",
            "speedup": round(speedup_kmer, 2),
        })

        # 3C: AF-Cluster .a3m parse + tokens_np + cluster_msa (5,000 seqs x 256 cols)
        a3m_path = tmpdir / "deep_5k.a3m"
        base_msa_seq = (real_seqs[0] * 3)[:256]
        with open(a3m_path, "w") as f:
            for i in range(5000):
                f.write(f">seq_{i}\n{base_msa_seq}\n")

        def run_af_cluster():
            msa = strux_rs.parse_a3m_file(str(a3m_path))
            toks = msa.tokens_np
            labels, reps = strux_rs.cluster_msa(msa.sequences, min_identity=0.75)
            return toks.shape[0], len(reps)

        _, t_strux_msa, iqr_msa = bench_fn(run_af_cluster, warmup=1, repeats=5)
        print(f"  3C. Zero-Copy .a3m Parse + tokens_np + cluster_msa (5,000 x 256): {t_strux_msa:.2f} ms (IQR {iqr_msa:.2f} ms)")
        records.append({
            "suite": "Suite3_Sequence_MSA",
            "benchmark": "af_cluster_a3m_5000x256",
            "strux_ms": round(t_strux_msa, 3),
            "strux_iqr_ms": round(iqr_msa, 3),
            # No baseline was run, so no speedup is reported (timing only).
            "baseline_ms": None,
            "baseline_name": None,
            "speedup": None,
        })

        # Save outputs
        out_dir = Path(__file__).resolve().parent
        tag = hw["gpu_name"].lower().replace(" ", "_").replace("nvidia_", "")
        if tag == "none":
            tag = hw["hostname"].lower()
        json_path = out_dir / f"standardized_results_{tag}.json"
        csv_path = out_dir / f"standardized_results_{tag}.csv"

        payload = {
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "strux_rs_version": getattr(strux_rs, "__version__", "unknown"),
            "strux_rs_file": getattr(strux_rs, "__file__", "unknown"),
            "dataset_source": dataset_source,
            "hardware": hw,
            "benchmarks": records,
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["suite", "benchmark", "strux_ms", "strux_iqr_ms", "baseline_ms", "baseline_name", "speedup"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)

        print("\n" + "=" * 88)
        print(f"Saved JSON -> {json_path}")
        print(f"Saved CSV  -> {csv_path}")
        print("=" * 88)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
