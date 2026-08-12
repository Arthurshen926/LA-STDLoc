import torch

import localization.frontend as frontend_module
from localization.frontend import NativeSuperPointFrontend
from map_learning.context_metric import (
    MapConsistentContextAdapter,
    dense_context_tokens,
)


class _FakeSuperPoint:
    def __init__(self, sparse, dense):
        self.sparse = sparse
        self.dense = dense

    def to(self, _device):
        return self

    def eval(self):
        return self

    def detectAndComputeWithDense(self, _image, top_k):
        assert top_k == 2
        return [self.sparse], (self.dense[None], torch.zeros(1, 1, 16, 16))


def test_context_frontend_reuses_one_sparse_dense_forward(monkeypatch):
    torch.manual_seed(8)
    keypoints = torch.tensor([[3.0, 4.0], [11.0, 12.0]])
    raw = torch.nn.functional.normalize(torch.randn(2, 256), dim=1)
    dense = torch.nn.functional.normalize(torch.randn(256, 2, 2), dim=0)
    sparse = {
        "keypoints": keypoints,
        "keypoint_scores": torch.tensor([0.9, 0.8]),
        "descriptors": raw,
    }
    fake = _FakeSuperPoint(sparse, dense)
    monkeypatch.setattr(frontend_module, "SuperPoint", lambda: fake)
    adapter = MapConsistentContextAdapter(
        hidden_dim=8,
        context_kernels=(3,),
        maximum_residual_norm=0.05,
    )
    with torch.no_grad():
        adapter.context_head[-1].weight.normal_()
        adapter.context_head[-1].bias.normal_()

    frontend = NativeSuperPointFrontend(
        device="cpu", keypoint_count=2, nms_radius=4, context_adapter=adapter
    )
    assert fake.nms_radius == 4
    output = frontend(torch.randn(3, 16, 16))

    tokens = dense_context_tokens(
        dense,
        keypoints,
        (16, 16),
        valid_mask=torch.ones(2, 2, dtype=torch.bool),
        kernels=(3,),
    )
    expected, _ = adapter(raw, tokens)
    torch.testing.assert_close(output.descriptors, expected)
    assert output.image_hw == (16, 16)
