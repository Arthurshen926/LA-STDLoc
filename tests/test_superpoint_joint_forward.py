import types

import torch

from encoders.sp_encoder.export_image_embeddings import SuperPoint


def test_combined_superpoint_api_runs_dense_encoder_once():
    model = SuperPoint()
    calls = {"count": 0}

    def fake_dense(self, image):
        calls["count"] += 1
        descriptors = torch.zeros(1, 256, 2, 2)
        descriptors[:, 0] = 1.0
        scores = torch.zeros(1, 16, 16)
        scores[0, 8, 8] = 1.0
        return descriptors, scores

    model._dense_outputs = types.MethodType(fake_dense, model)
    sparse, (dense, scores) = model.detectAndComputeWithDense(
        torch.zeros(1, 3, 16, 16), top_k=8
    )
    assert calls["count"] == 1
    assert len(sparse) == 1
    assert dense.shape == (1, 256, 2, 2)
    assert scores.shape == (1, 1, 16, 16)
