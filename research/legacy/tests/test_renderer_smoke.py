import os
import shutil

import pytest
import torch


pytestmark = pytest.mark.skipif(
    os.environ.get("LAFGS_RUN_CUDA_SMOKE") != "1" or not torch.cuda.is_available(),
    reason="set LAFGS_RUN_CUDA_SMOKE=1 on a CUDA host to run the release gate",
)


def test_gsplat_2dgs_cuda_rasterizer_is_executable():
    if shutil.which("ninja") is None:
        pytest.fail("ninja executable is not on PATH")

    from gsplat.rendering import rasterization_2dgs

    device = torch.device("cuda")
    means = torch.tensor([[0.0, 0.0, 2.0]], device=device)
    quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
    scales = torch.tensor([[0.1, 0.1, 0.001]], device=device)
    opacities = torch.ones(1, device=device)
    colors = torch.tensor([[[1.0, 0.0, 0.0]]], device=device)
    viewmats = torch.eye(4, device=device)[None]
    intrinsics = torch.tensor(
        [[[100.0, 0.0, 16.0], [0.0, 100.0, 16.0], [0.0, 0.0, 1.0]]],
        device=device,
    )
    rendered, alpha, *_ = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        width=32,
        height=32,
    )
    assert rendered.shape == (1, 32, 32, 3)
    assert alpha.shape == (1, 32, 32, 1)
    assert torch.isfinite(rendered).all()
    assert torch.isfinite(alpha).all()
    assert float(alpha.sum()) > 0.0
