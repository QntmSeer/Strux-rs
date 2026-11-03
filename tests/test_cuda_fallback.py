"""Without a usable NVIDIA GPU, the CUDA API must fail cleanly instead of panicking."""
import numpy as np
import pytest

import strux_rs as sx


def test_cuda_is_available_returns_bool():
    assert isinstance(sx.cuda_is_available(), bool)


@pytest.mark.skipif(sx.cuda_is_available(), reason="a usable GPU is present")
@pytest.mark.skipif(not hasattr(sx, "cuda_pairwise_rmsd"), reason="built without the cuda feature")
def test_cuda_functions_raise_runtime_error_without_gpu():
    with pytest.raises(RuntimeError, match="NVIDIA"):
        sx.cuda_pairwise_rmsd(np.zeros((3, 5, 3), dtype=np.float32))
    # CPU functions are unaffected
    assert sx.calculate_rg(np.random.rand(10, 3).astype(np.float32)) > 0


@pytest.mark.skipif(not sx.cuda_is_available(), reason="needs a usable NVIDIA GPU")
def test_cuda_pairwise_rmsd_matches_cpu_on_near_degenerate_inputs():
    """Uniform random clouds have near-degenerate top QCP eigenvalues; a fixed 12-step Newton
    loop under-reported RMSD by up to ~0.07 A there before the convergence check was added."""
    t = (np.random.default_rng(0).random((300, 250, 3)) * 10).astype(np.float32)
    assert np.abs(sx.cuda_pairwise_rmsd(t) - sx.cpu_pairwise_rmsd(t)).max() < 1e-3
