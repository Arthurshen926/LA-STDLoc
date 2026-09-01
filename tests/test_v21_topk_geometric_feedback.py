from __future__ import annotations

import pytest
import torch

from map_learning.v21_topk_geometric_feedback import (
    default_config,
    finalize_evaluations,
    select_topk_geometry_rows,
)


def test_topk_geometry_preserves_inliers_and_selects_projected_outlier() -> None:
    # Identity pose and K make x/z,y/z project directly to pixels.
    xyz = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )
    result = select_topk_geometry_rows(
        keypoints=torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        topk_anchor_rows=torch.tensor(
            [[0] * 64, [1, 2] + [1] * 62], dtype=torch.long
        ),
        topk_scores=torch.tensor(
            [[1.0] * 64, [1.0, 0.95] + [0.0] * 62]
        ),
        baseline_anchor_rows=torch.tensor([0, 1]),
        baseline_scores=torch.tensor([1.0, 1.0]),
        baseline_inlier_rows=torch.tensor([0]),
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=default_config(),
    )
    assert result["anchor_rows"].tolist() == [0, 2]
    assert result["changed_query_rows"].tolist() == [1]
    assert result["selected_candidate_ranks"].tolist() == [2]


def test_topk_geometry_rejects_candidate_below_score_drop() -> None:
    rows = torch.zeros((1, 64), dtype=torch.long)
    rows[0, 1] = 1
    scores = torch.zeros((1, 64))
    scores[0, 0] = 1.0
    scores[0, 1] = 0.89
    result = select_topk_geometry_rows(
        keypoints=torch.tensor([[1.0, 0.0]]),
        topk_anchor_rows=rows,
        topk_scores=scores,
        baseline_anchor_rows=torch.tensor([0]),
        baseline_scores=torch.tensor([1.0]),
        baseline_inlier_rows=torch.empty(0, dtype=torch.long),
        anchor_xyz=torch.tensor([[10.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        intrinsic=torch.eye(3),
        baseline_pose_w2c=torch.eye(4),
        config=default_config(),
    )
    assert result["changed_query_rows"].numel() == 0


def test_finalizer_requires_disjoint_forward_roles(monkeypatch) -> None:
    base = {
        "configuration": default_config(),
        "baseline_contract": {"frozen": True},
        "inputs": {
            "stable_map": {"path": "/map", "sha256": "a" * 64},
            "split_manifest": {"path": "/split", "sha256": "b" * 64},
        },
        "summary": {
            "query_count": 1,
            "baseline_r5_success_count": 0,
            "candidate_r5_success_count": 1,
            "paired_r5_gain_count": 1,
            "paired_r5_loss_count": 0,
            "catastrophe_count": 0,
        },
        "records": [{"query_index": 7}],
    }
    payloads = []
    for role in ("adaptation", "control", "confirmation"):
        payloads.append({**base, "evaluation_role": role})
    monkeypatch.setattr(
        "map_learning.v21_topk_geometric_feedback.validate_evaluation",
        lambda _: None,
    )
    sources = [
        {"path": f"/{role}", "sha256": str(index) * 64, "size_bytes": 1}
        for index, role in enumerate(("adaptation", "control", "confirmation"), 1)
    ]
    with torch.no_grad(), pytest.raises(ValueError, match="overlap"):
        finalize_evaluations(payloads, sources)
