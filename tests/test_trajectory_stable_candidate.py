import torch

from localization_training.shared_metric import SharedLowRankMetric
from localization_training.trajectory_stable_candidate import (
    TrajectoryStableCandidateConfig,
    build_trajectory_stable_candidate_teacher,
)
from scripts.train_lafgs_v7_online_metric import (
    _trajectory_stable_promotion_loss,
)


def _record(query_index, name):
    return {
        "query_index": query_index,
        "query_name": name,
        "query_rows": torch.tensor([0]),
        "positive_offsets": torch.tensor([0, 1]),
        "positive_indices": torch.tensor([1]),
        "ambiguous_offsets": torch.tensor([0, 0]),
        "ambiguous_indices": torch.empty(0, dtype=torch.long),
    }


def test_candidate_teacher_only_accepts_cross_trajectory_repeated_repairs():
    names = ["seq1/frame00001.png", "seq2/frame00001.png"]
    bank = torch.nn.functional.normalize(
        torch.tensor([[1.0, 0.0, 0.0], [0.99, 0.10, 0.0]]), dim=1
    )
    state = {
        "anchor_xyz": torch.zeros(2, 3),
        "anchor_features": bank,
        "coarse_dependency_group_ids": torch.tensor([7, 8]),
    }
    positives = {
        "anchor_count": 2,
        "query_names": names,
        "records": [_record(index, name) for index, name in enumerate(names)],
    }
    dynamic = {
        "anchor_count": 2,
        "query_names": names,
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "top1_anchor_indices": torch.tensor([0]),
                "gt_reprojection_errors_px": torch.tensor([20.0]),
                "ransac_inlier_mask": torch.tensor([True]),
            }
            for _ in names
        ],
    }
    cache = {
        name: {"native_descriptors": torch.tensor([[1.0, 0.0, 0.0]])}
        for name in names
    }
    metric = SharedLowRankMetric(descriptor_dim=3, rank=2, max_residual_norm=0.05)
    teacher = build_trajectory_stable_candidate_teacher(
        state=state,
        metric=metric,
        positives=positives,
        dynamic=dynamic,
        cache=cache,
        query_bins={names[0]: 0, names[1]: 1},
        config=TrajectoryStableCandidateConfig(),
        device=torch.device("cpu"),
    )
    assert teacher["summary"]["stable_relation_count"] == 1
    assert teacher["summary"]["accepted_row_count"] == 2
    for record in teacher["records"]:
        assert record["promotion_positive_anchor"].tolist() == [1]
        assert record["promotion_negative_anchor"].tolist() == [0]
        assert record["promotion_types"].tolist() == [1]


def test_promotion_loss_rewards_target_above_false_attractor():
    positive = torch.tensor([1])
    negative = torch.tensor([0])
    weights = torch.tensor([2.0])
    bad, count = _trajectory_stable_promotion_loss(
        torch.tensor([[0.8, 0.5]]), positive, negative, weights, margin=0.02
    )
    good, _ = _trajectory_stable_promotion_loss(
        torch.tensor([[0.5, 0.8]]), positive, negative, weights, margin=0.02
    )
    assert count == 1
    assert good < bad
