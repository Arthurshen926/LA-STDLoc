from __future__ import annotations

import torch

from features.superpoint import SuperPoint, render_validity_mask_from_alpha


def _model() -> SuperPoint:
    model = object.__new__(SuperPoint)
    torch.nn.Module.__init__(model)
    model.nms_radius = 1
    model.max_num_keypoints = None
    model.detection_threshold = 0.0
    model.remove_borders = 0
    return model


def test_all_valid_mask_is_exact_legacy_sparse_extraction() -> None:
    model = _model()
    dense = torch.nn.functional.normalize(torch.rand(1, 4, 2, 2), dim=1)
    scores = torch.rand(1, 16, 16)
    legacy = model._sparse_from_dense(dense, scores, top_k=20)[0]
    guarded = model._sparse_from_dense(
        dense, scores, top_k=20, validity_mask=torch.ones_like(scores).bool()
    )[0]
    for key in ("keypoints", "keypoint_scores", "descriptors"):
        assert torch.equal(legacy[key], guarded[key])


def test_invalid_peak_cannot_suppress_valid_neighbor() -> None:
    model = _model()
    dense = torch.nn.functional.normalize(torch.rand(1, 4, 1, 1), dim=1)
    scores = torch.zeros(1, 8, 8)
    scores[0, 3, 3] = 0.99
    scores[0, 3, 4] = 0.8
    valid = torch.ones_like(scores).bool()
    valid[0, 3, 3] = False
    rows = model._sparse_from_dense(dense, scores, validity_mask=valid)[0]
    assert [4.0, 3.0] in rows["keypoints"].tolist()
    assert [3.0, 3.0] not in rows["keypoints"].tolist()


def test_alpha_neighborhood_is_eroded_before_nms() -> None:
    alpha = torch.ones(1, 5, 5)
    alpha[0, 2, 2] = 0.1
    valid = render_validity_mask_from_alpha(
        alpha, minimum_alpha=0.5, neighborhood_radius=1
    )
    assert int((~valid).sum()) == 9


def test_alpha_channel_last_from_gsplat_is_supported() -> None:
    alpha = torch.ones((1, 3, 4, 1))
    alpha[0, 1, 2, 0] = 0.0
    valid = render_validity_mask_from_alpha(
        alpha, minimum_alpha=0.05, neighborhood_radius=0
    )
    assert valid.shape == (1, 3, 4)
    assert valid[0, 1, 2].item() is False
