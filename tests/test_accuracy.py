"""Accuracy against reference tools. Requires: pip install mdtraj freesasa pytest."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mdtraj")
pytest.importorskip("freesasa")

_spec = importlib.util.spec_from_file_location(
    "accuracy_report", Path(__file__).resolve().parents[1] / "benchmarks" / "accuracy_report.py")
acc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(acc)


@pytest.fixture(scope="module")
def res():
    return acc.run()


@pytest.mark.parametrize("key", [
    "xtc_max_abs_diff_A", "xtc_subset_max_abs_diff_A",  # XTC is stored at 1e-3 nm precision
    "dcd_max_abs_diff_A", "dcd_subset_max_abs_diff_A",
])
def test_trajectory_readers_match_mdtraj(res, key):
    assert res[key] < 1e-4


@pytest.mark.parametrize("key", [
    "kabsch_rmsd_max_abs_diff_A", "pose_rmsd_max_abs_diff_A", "binder_rmsd_max_abs_diff_A", "rg_max_abs_diff_A",
])
def test_geometry_matches_float64_reference(res, key):
    assert res[key] < 1e-3


def test_sasa_matches_freesasa(res):
    assert res["sasa_rel_diff_vs_freesasa"] < 0.01


def test_bsa_with_element_radii_matches_freesasa(res):
    assert res["bsa_element_radii_rel_diff_vs_freesasa"] < 0.02


def test_lddt_matches_pooled_definition(res):
    assert res["lddt_global_max_abs_diff"] < 1e-5
    assert res["lddt_per_residue_max_abs_diff"] < 1e-5


def test_lddt_global_is_pooled_not_per_residue_mean():
    """A model with one badly placed region separates the two definitions."""
    import strux_rs as sx

    rng = np.random.default_rng(0)
    ref = rng.normal(0, 8, (120, 3)).astype(np.float32)
    mod = ref + rng.normal(0, 1.5, ref.shape).astype(np.float32)
    mod[:30] += 6.0
    g, per = sx.calculate_lddt(ref, mod, 15.0)
    g_pooled, per_ref = acc.lddt_reference(ref, mod)
    assert abs(g - g_pooled) < 1e-5
    assert abs(g - float(np.mean(per_ref))) > 1e-3


def test_bsa_default_radius_unchanged_when_no_radii_given():
    import strux_rs as sx

    rng = np.random.default_rng(1)
    a = rng.normal(0, 6, (200, 3)).astype(np.float32)
    b = (rng.normal(0, 6, (150, 3)) + [9, 0, 0]).astype(np.float32)
    default = sx.calculate_bsa(a, b)
    explicit = sx.calculate_bsa(a, b, 1.4, 96, np.full(200, 1.70, np.float32), np.full(150, 1.70, np.float32))
    assert np.allclose(default, explicit, rtol=1e-6)


def test_interface_counts_match_stated_definition():
    import strux_rs as sx

    t, xyz, m = acc.load()
    f0 = xyz[0]
    contacts, polar, salt, buns, _ = sx.analyze_interface_physics(
        f0, m["names"], m["resn"], m["chains"], m["resid"], "A", "B")
    A = np.array([c == "A" for c in m["chains"]])
    pol = np.array([n[0] in "NO" for n in m["names"]])
    D = np.linalg.norm(f0[A][:, None] - f0[~A][None], axis=-1)
    assert contacts == int((D <= 4.5).sum())
    assert polar == int(((D <= 3.5) & pol[A][:, None] & pol[~A][None]).sum())
