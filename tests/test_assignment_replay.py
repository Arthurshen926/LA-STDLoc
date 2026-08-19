from types import SimpleNamespace

import numpy as np
import torch

import topology.assignment_replay as assignment_replay


class _IdentityMetric(torch.nn.Module):
    def forward(self, descriptors):
        return descriptors, None


def _fixture():
    state = {
        "anchor_ids": torch.tensor([0, 1]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
    }
    teacher = {
        "anchor_count": 2,
        "query_names": ["q"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([1, 0]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
    }
    cache = {
        "queries": {
            "q": {
                "native_descriptors": torch.tensor([[1.0, 0.0], [0.99, -0.1]]),
                "native_keypoints": torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
                "native_K": torch.eye(3),
                "pose_w2c": torch.eye(4),
            }
        }
    }
    return state, teacher, cache


def _empty_pose(*args, **kwargs):
    return SimpleNamespace(
        pose_w2c=np.eye(4),
        inliers=np.empty(0, dtype=np.int64),
        diagnostics={"iterations": 7},
    )


def test_shared_topk_replays_top1_and_assignment(monkeypatch):
    state, teacher, cache = _fixture()
    monkeypatch.setattr(
        assignment_replay,
        "load_shared_metric",
        lambda *args, **kwargs: _IdentityMetric(),
    )
    monkeypatch.setattr(assignment_replay, "solve_absolute_pose", _empty_pose)
    sidecar = assignment_replay.materialize_mapping_topk(
        state=state,
        metric_state_path="unused.pt",
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        anchor_bank_updater=lambda *_: None,
        topk=2,
    )
    top1 = assignment_replay.replay_mapping_topk(
        sidecar=sidecar,
        state=state,
        teacher=teacher,
        assignment_topk=0,
        ransac_reprojection_px=12.0,
        clean_reprojection_px=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
        seed=2026,
        pose_workers=2,
    )
    assigned = assignment_replay.replay_mapping_topk(
        sidecar=sidecar,
        state=state,
        teacher=teacher,
        assignment_topk=2,
        assignment_dustbin_score=-1.0,
        ransac_reprojection_px=12.0,
        clean_reprojection_px=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
        seed=2026,
        pose_workers=2,
    )
    assert top1["summary"]["raw_gt_precision_percent"] == 50.0
    assert assigned["summary"]["raw_gt_precision_percent"] == 100.0
    assert assigned["summary"]["assignment_reassigned_query_rows"] == 1
    assert assigned["queries"][0]["correspondences"] == 2


def test_regret_bound_turns_weak_reassignment_into_unmatched(monkeypatch):
    state, teacher, cache = _fixture()
    monkeypatch.setattr(
        assignment_replay,
        "load_shared_metric",
        lambda *args, **kwargs: _IdentityMetric(),
    )
    monkeypatch.setattr(assignment_replay, "solve_absolute_pose", _empty_pose)
    sidecar = assignment_replay.materialize_mapping_topk(
        state=state,
        metric_state_path="unused.pt",
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        anchor_bank_updater=lambda *_: None,
        topk=2,
    )
    replay = assignment_replay.replay_mapping_topk(
        sidecar=sidecar,
        state=state,
        teacher=teacher,
        assignment_topk=2,
        assignment_dustbin_score=-1.0,
        assignment_maximum_regret=0.1,
        ransac_reprojection_px=12.0,
        clean_reprojection_px=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
        seed=2026,
        pose_workers=1,
    )
    assert replay["summary"]["assignment_unmatched_query_rows"] == 1
    assert replay["queries"][0]["correspondences"] == 1


def test_sidecar_rejects_test_scope(monkeypatch):
    state, teacher, cache = _fixture()
    monkeypatch.setattr(
        assignment_replay,
        "load_shared_metric",
        lambda *args, **kwargs: _IdentityMetric(),
    )
    sidecar = assignment_replay.materialize_mapping_topk(
        state=state,
        metric_state_path="unused.pt",
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        anchor_bank_updater=lambda *_: None,
        topk=2,
    )
    sidecar["uses_test_queries"] = True
    try:
        assignment_replay.replay_mapping_topk(
            sidecar=sidecar,
            state=state,
            teacher=teacher,
            assignment_topk=0,
            ransac_reprojection_px=12.0,
            clean_reprojection_px=4.0,
            task_translation_m=0.05,
            task_rotation_deg=5.0,
            seed=2026,
        )
    except ValueError as error:
        assert "test queries" in str(error)
    else:
        raise AssertionError("test-scoped sidecar was accepted")
