from pathlib import Path

import pytest
import torch

from reproduction.native_hooks import mask_rgb, transform_native


def test_native_query_mask_preserves_only_shared_valid_pixels():
    image = torch.ones(3, 2, 2)
    valid = torch.tensor([[True, False], [True, True]])
    masks = {"q.png": (valid, torch.ones_like(valid), torch.ones_like(valid))}
    result = mask_rgb("q.png", image, masks, {"q.png"})
    assert torch.equal(result[:, 0, 1], torch.zeros(3))
    assert torch.equal(result[:, 1, 1], torch.ones(3))


@pytest.mark.parametrize(
    "method,entry", [("STDLoc", "stdloc.py"), ("ULFLoc", "ulfloc.py")]
)
def test_native_timing_adapter_keeps_scientific_definitions(method, entry):
    path = Path(__file__).parents[1] / "assets/third_party" / method / entry
    if not path.exists():
        pytest.skip("optional native source snapshot is not installed")
    transformed = transform_native(path.read_text(), method)
    compile(transformed, str(path), "exec")
    assert "_MATCHED.finish" in transformed
