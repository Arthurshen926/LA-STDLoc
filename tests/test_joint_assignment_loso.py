import torch

from localization_training.local_assignment import OneOfKAssignmentHead
from scripts.train_joint_assignment_loso import (
    _forward_record_batch,
    _record_tensors,
    _trajectory_block_split,
)


def _graph(count=20):
    return {
        "scene": "ShopFacade",
        "records": [
            {
                "query_name": f"seq2/frame{index:05d}.png",
                "trajectory": "seq2",
            }
            for index in range(count)
        ],
    }


def test_single_trajectory_uses_nonempty_contiguous_temporal_holdout():
    calibration, training, summary = _trajectory_block_split(_graph(), 2026)
    assert len(calibration) == 4
    assert len(training) == 16
    assert not calibration & training
    assert len(calibration | training) == 20
    selected = sorted(int(name[10:15]) for name in calibration)
    assert selected == list(range(selected[0], selected[0] + 4))
    assert summary[0]["block_count"] == 5


def test_temporal_block_split_is_deterministic():
    first = _trajectory_block_split(_graph(23), 2026)
    second = _trajectory_block_split(_graph(23), 2026)
    assert first == second


def test_scene_balanced_batch_preserves_per_query_boundaries():
    head = OneOfKAssignmentHead(hidden_dim=4, feature_dim=5)
    records = []
    for row_count in (2, 3):
        records.append(
            {
                "features": torch.randn(row_count, 4, 5),
                "positive_mask": torch.zeros(row_count, 4, dtype=torch.bool),
                "candidate_target_weights": torch.zeros(row_count, 4),
                "row_weights": torch.ones(row_count),
            }
        )
    outputs = _forward_record_batch(head, records, torch.device("cpu"))
    assert [len(output[3]) for output in outputs] == [2, 3]
    assert [len(output[4]) for output in outputs] == [2, 3]


def test_record_tensors_are_staged_only_once():
    record = {
        "features": torch.randn(2, 4, 5).half(),
        "positive_mask": torch.zeros(2, 4, dtype=torch.bool),
        "candidate_target_weights": torch.ones(2, 4).half(),
        "row_weights": torch.ones(2).half(),
    }
    first = _record_tensors(record, torch.device("cpu"))
    second = _record_tensors(record, torch.device("cpu"))
    assert all(left.data_ptr() == right.data_ptr() for left, right in zip(first, second))
    assert first[0].dtype == torch.float32
    assert first[1].dtype == torch.bool


@torch.no_grad()
def test_record_tensors_accept_explicit_current_cuda_device():
    if not torch.cuda.is_available():
        return
    record = {
        "features": torch.randn(2, 4, 5).half(),
        "positive_mask": torch.zeros(2, 4, dtype=torch.bool),
        "candidate_target_weights": torch.ones(2, 4).half(),
        "row_weights": torch.ones(2).half(),
    }
    device = torch.device("cuda", torch.cuda.current_device())
    assert all(value.device == device for value in _record_tensors(record, device))
