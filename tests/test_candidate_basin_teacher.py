import torch
import torch.nn.functional as F

from scripts.build_lafgs_candidate_basin_teacher import (
    _family_topk,
    _pose_level,
)


def test_candidate_family_topk_reports_winning_mode_and_unique_anchor():
    query = F.normalize(torch.tensor([[0.0, 1.0], [1.0, 0.0]]), dim=1)
    bank = F.normalize(
        torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.6, 0.8]]), dim=1
    )
    family = {
        "prototype_features": torch.tensor([[0.0, 1.0], [0.1, 1.0]]),
        "prototype_anchor_indices": torch.tensor([0, 0]),
        "prototype_bias": torch.tensor([0.0, -0.1]),
        "prototype_temperature": torch.ones(2),
    }
    scores, anchors, modes, _ = _family_topk(query, bank, family, 2)
    assert anchors[0, 0].item() == 0
    assert modes[0, 0].item() == 0
    assert anchors[0].unique().numel() == 2
    assert modes[1, 0].item() == -1
    assert scores.shape == (2, 2)


def test_pose_level_is_hierarchical():
    assert _pose_level({"valid": False}) == 0
    assert _pose_level({"valid": True, "te_cm": 40.0, "re_deg": 4.0}) == 1
    assert _pose_level({"valid": True, "te_cm": 12.0, "re_deg": 1.0}) == 2
    assert _pose_level({"valid": True, "te_cm": 4.0, "re_deg": 4.0}) == 3
