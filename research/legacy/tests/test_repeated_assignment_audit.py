import torch

from map_learning.metric import SharedLowRankMetric
from map_learning.repeated_assignment_audit import audit_repeated_assignments


def test_repeated_assignment_audit_separates_rank_headroom_and_anchor_kind():
    state = {
        "anchor_ids": torch.arange(4),
        "anchor_xyz": torch.tensor(
            [
                [0.0, 0.0, 2.0],
                [0.1, 0.0, 2.0],
                [0.2, 0.0, 2.0],
                [0.3, 0.0, 2.0],
            ]
        ),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]]
        ),
        "anchor_type": torch.tensor([1, 0, 1, 0]),
    }
    teacher = {
        "anchor_count": 4,
        "query_names": ["q"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1, 2]),
                "positive_offsets": torch.tensor([0, 1, 2, 2]),
                "positive_indices": torch.tensor([1, 2]),
                "ambiguous_offsets": torch.tensor([0, 0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
    }
    cache = {
        "q": {
            "native_descriptors": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.2, 0.8]]
            ),
            "native_keypoints": torch.zeros(3, 2),
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
        }
    }
    metric = SharedLowRankMetric(descriptor_dim=2, rank=1)
    report = audit_repeated_assignments(
        state=state,
        metric=metric,
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        topks=(1, 2),
        oracle_query_indices=[],
        progress_interval=0,
    )

    assert report["row_counts"]["replayed"] == 3
    assert report["row_counts"]["positive_eligible"] == 2
    assert report["positive_recall_at_k"]["1"]["hit_count"] == 1
    assert report["positive_recall_at_k"]["2"]["hit_count"] == 2
    assert report["positive_recall_at_k"]["1"]["positive_eligible_recall"] == 0.5
    assert report["false_top1_recoverable_at_k"]["2"]["fraction"] == 1.0
    assert report["positive_recall_at_k_by_anchor_kind"]["track_core"]["1"][
        "eligible_recall"
    ] == 1.0
    assert report["positive_recall_at_k_by_anchor_kind"]["gaussian_reserve"]["2"][
        "eligible_recall"
    ] == 1.0
    assert report["winner_breakdown"]["track_core"]["winner_count"] == 2
    assert report["winner_breakdown"]["gaussian_reserve"]["winner_count"] == 1
    assert report["false_attractors"]["top"][0]["anchor_index"] == 0
    assert report["oracle_pose_summaries"]["current"]["query_count"] == 0


def test_repeated_assignment_audit_applies_detector_rank_prefix():
    state = {
        "anchor_ids": torch.arange(2),
        "anchor_xyz": torch.zeros(2, 3),
        "anchor_features": torch.eye(2),
        "anchor_type": torch.tensor([1, 0]),
    }
    teacher = {
        "anchor_count": 2,
        "query_names": ["q"],
        "records": [
            {
                "query_rows": torch.tensor([1, 7]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
    }
    cache = {
        "q": {
            "native_descriptors": torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 1.0],
                ]
            ),
            "native_keypoints": torch.zeros(8, 2),
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
        }
    }
    report = audit_repeated_assignments(
        state=state,
        metric=SharedLowRankMetric(descriptor_dim=2, rank=1),
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        topks=(1,),
        oracle_query_indices=[],
        deployment_row_limit=4,
        progress_interval=0,
    )
    assert report["row_counts"]["replayed"] == 1
    assert report["positive_recall_at_k"]["1"]["hit_count"] == 1
