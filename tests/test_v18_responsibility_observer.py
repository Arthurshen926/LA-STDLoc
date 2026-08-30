import torch

import map_learning.v18_responsibility_observer as observer
from map_learning.v18_provenance_truth import TRUTH_UNIQUE


def test_responsibility_decomposes_row_anchor_and_metric_actions(monkeypatch) -> None:
    def fake_pose(*, keypoints, anchor_rows, **_kwargs):
        error = float((torch.as_tensor(anchor_rows) == 3).sum())
        return {
            "translation_error_cm": error * 5.0,
            "rotation_error_deg": 0.0,
            "task_error": error,
            "inlier_count": int(torch.as_tensor(keypoints).shape[0]),
        }

    monkeypatch.setattr(observer, "standard_pose_replay", fake_pose)
    result = observer.decompose_correspondence_responsibility(
        keypoints=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]]
        ),
        candidate_anchor_rows=torch.tensor(
            [[3, 0], [1, 2], [2, 1], [0, 1], [1, 0]]
        ),
        candidate_scores=torch.tensor(
            [[0.9, 0.8], [0.9, 0.7], [0.9, 0.7], [0.9, 0.7], [0.9, 0.7]]
        ),
        truth={
            "row_count": 5,
            "truth_status": torch.full((5,), TRUTH_UNIQUE),
            "truth_offsets": torch.arange(6),
            "truth_anchor_rows": torch.tensor([0, 1, 2, 0, 1]),
            "status_counts": {"UNIQUE": 5},
        },
        anchor_xyz=torch.zeros(4, 3),
        intrinsic=torch.eye(3),
        pose_w2c=torch.eye(4),
        minimum_task_gain=0.1,
    )
    assert result["wrong_decisive_row_count"] == 1
    assert result["row_suppressible_query_rows"].tolist() == [0]
    assert result["anchor_suppressible_anchor_rows"].tolist() == [3]
    assert result["metric_controllable_query_rows"].tolist() == [0]
    assert result["full_truth_oracle_task_gain"] == 1.0
    assert result["loo_used"] is False
