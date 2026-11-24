"""Format robustness, box-vector parity, and triclinic/NPT imaging tests for strux-rs."""

from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest

import strux_rs

try:
    import mdtraj as md
except ImportError:
    md = None


def write_dcd_fixture(
    path: Path,
    frames: np.ndarray,
    unitcells: list[tuple[float, float, float, float, float, float]] | None = None,
    charmm_version: int = 24,
    truncate_bytes: int = 0,
) -> None:
    """Write a minimal CHARMM/NAMD/OpenMM or X-PLOR binary DCD file."""
    nframes, natoms, _ = frames.shape
    buf = bytearray()

    # Block 1: 84-byte header
    buf += struct.pack("<I", 84)
    buf += b"CORD"
    header_ints = [0] * 20
    header_ints[0] = nframes
    header_ints[19] = charmm_version
    if unitcells is not None:
        header_ints[10] = 1  # QCRYS at file offset 48..52
    elif charmm_version == 0:
        header_ints[9] = 1
        header_ints[10] = 1
    buf += struct.pack("<20i", *header_ints)
    buf += struct.pack("<I", 84)

    # Block 2: Title
    title_len = 4 + 80
    buf += struct.pack("<I", title_len)
    buf += struct.pack("<i", 1)
    buf += b" " * 80
    buf += struct.pack("<I", title_len)

    # Block 3: natoms
    buf += struct.pack("<IiI", 4, natoms, 4)

    for fi in range(nframes):
        if unitcells is not None:
            buf += struct.pack("<I", 48)
            buf += struct.pack("<6d", *unitcells[fi])
            buf += struct.pack("<I", 48)
        cbytes = natoms * 4
        for d in range(3):
            buf += struct.pack("<I", cbytes)
            buf += frames[fi, :, d].astype("<f4").tobytes()
            buf += struct.pack("<I", cbytes)

    if truncate_bytes > 0:
        buf = buf[:-truncate_bytes]
    path.write_bytes(bytes(buf))


def write_xtc_uncompressed_fixture(
    path: Path,
    magic: int,
    frames_nm: np.ndarray,
    boxes_nm: np.ndarray,
    truncate_bytes: int = 0,
) -> None:
    """Write a minimal uncompressed XTC file (natoms <= 9) with magic 1995 or 2023."""
    nframes, natoms, _ = frames_nm.shape
    assert natoms <= 9
    buf = bytearray()
    for fi in range(nframes):
        buf += struct.pack(">iiif", magic, natoms, fi, float(fi) * 10.0)
        buf += boxes_nm[fi].astype(">f4").tobytes()
        buf += struct.pack(">i", natoms)
        buf += frames_nm[fi].astype(">f4").tobytes()
    if truncate_bytes > 0:
        buf = buf[:-truncate_bytes]
    path.write_bytes(bytes(buf))


def test_dcd_openmm_charmm_and_no_unitcell(tmp_path: Path) -> None:
    frames = np.array(
        [
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5], [7.5, 8.5, 9.5]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]],
        ],
        dtype=np.float32,
    )
    # Rhombic dodecahedron NPT: [A, gamma, B, beta, alpha, C]
    ucs_deg = [
        (40.0, 90.0, 40.0, 60.0, 60.0, 40.0),
        (41.0, 90.0, 41.0, 60.0, 60.0, 41.0),
        (42.0, 90.0, 42.0, 60.0, 60.0, 42.0),
    ]
    ucs_cos = [
        (40.0, 0.0, 40.0, 0.5, 0.5, 40.0),
        (41.0, 0.0, 41.0, 0.5, 0.5, 41.0),
        (42.0, 0.0, 42.0, 0.5, 0.5, 42.0),
    ]

    p_openmm = tmp_path / "openmm.dcd"
    p_charmm = tmp_path / "charmm.dcd"
    p_nouc = tmp_path / "nouc.dcd"
    p_trunc = tmp_path / "trunc.dcd"

    write_dcd_fixture(p_openmm, frames, ucs_deg, charmm_version=24)
    write_dcd_fixture(p_charmm, frames, ucs_cos, charmm_version=24)
    write_dcd_fixture(p_nouc, frames, None, charmm_version=0)
    write_dcd_fixture(p_trunc, frames, ucs_deg, charmm_version=24, truncate_bytes=12)

    c_o, b_o = strux_rs.parse_dcd_with_box(str(p_openmm))
    c_c, b_c = strux_rs.parse_dcd_with_box(str(p_charmm))
    c_n, b_n = strux_rs.parse_dcd_with_box(str(p_nouc))
    c_t, b_t = strux_rs.parse_dcd_with_box(str(p_trunc))

    assert np.allclose(c_o, frames, atol=1e-4)
    assert np.allclose(c_c, frames, atol=1e-4)
    assert np.allclose(c_n, frames, atol=1e-4)
    assert b_n is None
    assert b_o is not None and b_c is not None
    assert b_o.shape == (3, 3, 3)
    assert np.allclose(b_o, b_c, atol=1e-4)

    expected_b0 = np.array(
        [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [20.0, 20.0, 40.0 / math.sqrt(2.0)]],
        dtype=np.float32,
    )
    assert np.allclose(b_o[0], expected_b0, atol=1e-4)

    # Truncated final frame keeps only 2 complete frames and 2 box matrices
    assert c_t.shape == (2, 3, 3)
    assert b_t is not None and b_t.shape == (2, 3, 3)

    # Atom subset, empty atom_indices consistency, and out-of-bounds check
    c_sub, b_sub = strux_rs.parse_dcd_with_box(str(p_openmm), atom_indices=[2, 0])
    assert np.allclose(c_sub, frames[:, [2, 0], :], atol=1e-4)
    assert b_sub is not None and np.allclose(b_sub, b_o, atol=1e-5)
    assert strux_rs.parse_dcd(str(p_openmm), atom_indices=[]).shape == (3, 0, 3)
    c_empty, b_empty = strux_rs.parse_dcd_with_box(str(p_openmm), atom_indices=[])
    assert c_empty.shape == (3, 0, 3) and b_empty is not None and b_empty.shape == (3, 3, 3)
    with pytest.raises(ValueError):
        strux_rs.parse_dcd_with_box(str(p_openmm), atom_indices=[99])


@pytest.mark.parametrize("magic", [1995, 2023])
def test_xtc_uncompressed_magic_and_box(tmp_path: Path, magic: int) -> None:
    frames_nm = np.array(
        [
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
            [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
            [[0.3, 0.4, 0.5], [0.6, 0.7, 0.8], [0.9, 1.0, 1.1]],
        ],
        dtype=np.float32,
    )
    boxes_nm = np.array(
        [
            [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [1.5, 1.5, 2.1213203]],
            [[3.1, 0.0, 0.0], [0.0, 3.1, 0.0], [1.55, 1.55, 2.192031]],
            [[3.2, 0.0, 0.0], [0.0, 3.2, 0.0], [1.6, 1.6, 2.2627418]],
        ],
        dtype=np.float32,
    )

    p = tmp_path / f"test_{magic}.xtc"
    write_xtc_uncompressed_fixture(p, magic, frames_nm, boxes_nm)
    coords, boxes = strux_rs.parse_xtc_with_box(str(p))

    assert np.allclose(coords, frames_nm * 10.0, atol=1e-4)
    assert np.allclose(boxes, boxes_nm * 10.0, atol=1e-4)

    p_trunc = tmp_path / f"trunc_{magic}.xtc"
    write_xtc_uncompressed_fixture(p_trunc, magic, frames_nm, boxes_nm, truncate_bytes=8)
    c_t, b_t = strux_rs.parse_xtc_with_box(str(p_trunc))
    assert c_t.shape == (2, 3, 3)
    assert b_t.shape == (2, 3, 3)

    with pytest.raises(ValueError):
        strux_rs.parse_xtc_with_box(str(p), atom_indices=[10])


@pytest.mark.skipif(md is None, reason="mdtraj not installed")
def test_mdtraj_parity_compressed_xtc_and_boxed_dcd(tmp_path: Path) -> None:
    """Cross-check compressed XTC (natoms > 9) and boxed DCD against MDTraj (< 1e-4 A)."""
    top = md.Topology()
    chain = top.add_chain()
    for r in range(5):
        res = top.add_residue("ALA", chain)
        for name in ("N", "CA", "C", "O"):
            top.add_atom(name, md.element.carbon, res)

    rng = np.random.default_rng(42)
    xyz_nm = rng.uniform(0.5, 2.5, size=(3, 20, 3)).astype(np.float32)
    # Triclinic rhombic dodecahedron in nm
    uv_nm = np.array(
        [
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 2.0, 4.0 / math.sqrt(2.0)]],
            [[4.1, 0.0, 0.0], [0.0, 4.1, 0.0], [2.05, 2.05, 4.1 / math.sqrt(2.0)]],
            [[4.2, 0.0, 0.0], [0.0, 4.2, 0.0], [2.1, 2.1, 4.2 / math.sqrt(2.0)]],
        ],
        dtype=np.float32,
    )
    t = md.Trajectory(xyz_nm, top)
    t.unitcell_vectors = uv_nm

    p_xtc = tmp_path / "mdtraj_20atoms.xtc"
    p_dcd = tmp_path / "mdtraj_20atoms.dcd"
    t.save_xtc(str(p_xtc))
    t.save_dcd(str(p_dcd))

    t_xtc = md.load_xtc(str(p_xtc), top=top)
    c_xtc, b_xtc = strux_rs.parse_xtc_with_box(str(p_xtc))
    assert np.max(np.abs(c_xtc - t_xtc.xyz * 10.0)) < 1e-4
    assert np.max(np.abs(b_xtc - t_xtc.unitcell_vectors * 10.0)) < 1e-4

    t_dcd = md.load_dcd(str(p_dcd), top=top)
    c_dcd, b_dcd = strux_rs.parse_dcd_with_box(str(p_dcd))
    assert b_dcd is not None
    assert np.max(np.abs(c_dcd - t_dcd.xyz * 10.0)) < 1e-4
    assert np.max(np.abs(b_dcd - t_dcd.unitcell_vectors * 10.0)) < 1e-3


def test_image_molecules_and_nojump_box_triclinic() -> None:
    a = np.array([40.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 40.0, 0.0], dtype=np.float32)
    c = np.array([20.0, 20.0, 40.0 / math.sqrt(2.0)], dtype=np.float32)
    h0 = np.stack([a, b, c], axis=0)
    h1 = h0 * np.float32(1.05)

    intact = np.array(
        [
            [10.0, 10.0, 10.0],
            [11.5, 10.0, 10.0],
            [13.0, 10.0, 10.0],
            [13.0, 14.0, 10.0],
            [14.5, 14.0, 10.0],
            [16.0, 14.0, 10.0],
        ],
        dtype=np.float32,
    )
    f0 = intact.copy()
    f0[2] += c
    f0[3:6] += a - c
    f0[5] += b

    f1 = intact.copy() + h1[2]
    f1[3:6] -= h1[0]

    coords = np.stack([f0, f1], axis=0)
    boxes = np.stack([h0, h1], axis=0)

    imaged = strux_rs.image_molecules(coords, boxes, [(0, 3), (3, 6)], anchor_mol=0)
    for fi in range(2):
        # Exact bond lengths (1.5 A) and interface contact (4.0 A) preserved
        assert abs(np.linalg.norm(imaged[fi, 1] - imaged[fi, 0]) - 1.5) < 1e-4
        assert abs(np.linalg.norm(imaged[fi, 2] - imaged[fi, 1]) - 1.5) < 1e-4
        assert abs(np.linalg.norm(imaged[fi, 4] - imaged[fi, 3]) - 1.5) < 1e-4
        assert abs(np.linalg.norm(imaged[fi, 5] - imaged[fi, 4]) - 1.5) < 1e-4
        assert abs(np.linalg.norm(imaged[fi, 3] - imaged[fi, 2]) - 4.0) < 1e-4

    nojump = strux_rs.unwrap_trajectory_nojump_box(imaged, boxes)
    assert np.allclose(nojump[0].mean(axis=0), nojump[1].mean(axis=0), atol=1e-4)

    # Rhombic dodecahedron Voronoi-corner checks + overlapping mol_ranges rejection
    d100 = np.array([[[100.0, 0.0, 0.0], [0.0, 100.0, 0.0], [50.0, 50.0, 100.0 / math.sqrt(2.0)]]], dtype=np.float32)
    c_bound = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 36.0]]], dtype=np.float32)
    img_bound = strux_rs.image_molecules(c_bound, d100, [(0, 1), (1, 2)], anchor_mol=0)
    assert abs(np.linalg.norm(img_bound[0, 1] - img_bound[0, 0]) - 36.0) < 1e-4

    c_corner = np.array([[[0.0, 0.0, 0.0], [67.5, 67.5, 45.0 / math.sqrt(2.0)]]], dtype=np.float32)
    img_corner = strux_rs.image_molecules(c_corner, d100, [(0, 1), (1, 2)], anchor_mol=0)
    assert abs(np.linalg.norm(img_corner[0, 1] - img_corner[0, 0]) - 46.1008) < 1e-2

    with pytest.raises(ValueError):
        strux_rs.image_molecules(coords, boxes, [(0, 4), (3, 6)], anchor_mol=0)
