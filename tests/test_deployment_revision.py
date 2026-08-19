from types import SimpleNamespace

import numpy as np
import pytest
import torch

import topology.deployment_revision as deployment_revision

from topology.deployment_revision import (
    collect_deployment_statistics,
    select_revision,
    subset_map_and_metric,
    subset_teacher,
)


def _teacher():
    return {
        "anchor_count": 3,
        "query_names": ["q"],
        "records": [
            {
                "query_index": 0,
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
                "ambiguous_offsets": torch.tensor([0, 0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
        ],
        "diagnostics": {
            "positive_rows": 2,
            "strong_pair_count": 2,
            "ambiguous_pair_count": 0,
        },
    }


def test_revision_prunes_harmful_noncritical_anchor_without_losing_rank(tmp_path):
    counters = {
        name: torch.zeros(3, dtype=torch.float64)
        for name in (
            "winner_count",
            "correct_winner_count",
            "false_attractor_count",
            "ambiguous_winner_count",
            "clean_inlier_count",
            "harmful_inlier_count",
            "counterfactual_clean_gain",
            "information_deletion_loss",
            "tail_nonimproving_winner_count",
        )
    }
    counters["false_attractor_count"][2] = 4
    counters["harmful_inlier_count"][2] = 2
    counters["counterfactual_clean_gain"][2] = 3
    pruned, report = select_revision(
        _teacher(),
        {"counters": counters},
        matching_rows_target=2,
        maximum_prune_fraction=0.5,
    )
    assert pruned.tolist() == [2]
    assert report["matching_constraint"]["unmet_query_count"] == 0


def test_revision_subsets_teacher_and_map_consistently(tmp_path):
    teacher = _teacher()
    keep = torch.tensor([True, False, True])
    revised_teacher = subset_teacher(
        teacher,
        keep,
        tmp_path / "map.pt",
        source_anchor_type=torch.tensor([1, 0, 0]),
    )
    assert revised_teacher["anchor_count"] == 2
    assert revised_teacher["records"][0]["positive_indices"].tolist() == [0]
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.randn(3, 3),
        "anchor_features": torch.randn(3, 4),
        "anchor_type": torch.tensor([1, 0, 0]),
        "track_cluster_ids": torch.tensor([10, -1, -1]),
        "base_anchor_count": 2,
        "micro_anchor_count": 1,
        "requested_micro_anchor_budget": 1,
        "canonical_anchor_count": 3,
        "track_centric_reconstruction": {
            "track_indices": torch.tensor([10]),
            "base_canonical_rows": torch.tensor([7, 8]),
        },
    }
    metric = {"landmark_indices": torch.arange(3)}
    revised_map, revised_metric = subset_map_and_metric(
        state, metric, keep, output_map=tmp_path / "map.pt"
    )
    assert revised_map["anchor_ids"].tolist() == [0, 1]
    assert revised_map["track_centric_reconstruction"]["track_indices"].tolist() == [10]
    assert revised_metric["landmark_indices"].tolist() == [0, 1]
    assert revised_metric["map_path"] == str((tmp_path / "map.pt").resolve())


def test_revision_subsets_track_only_map_without_selector_metadata(tmp_path):
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.randn(3, 3),
        "anchor_features": torch.randn(3, 4),
        "anchor_type": torch.ones(3, dtype=torch.long),
        "source_primitive_ids": torch.full((3,), -1, dtype=torch.long),
        "track_cluster_ids": torch.arange(3),
        "base_anchor_count": 0,
        "micro_anchor_count": 3,
        "canonical_anchor_count": 3,
    }
    metric = {"landmark_indices": torch.arange(3)}
    revised_map, revised_metric = subset_map_and_metric(
        state,
        metric,
        torch.tensor([True, False, True]),
        output_map=tmp_path / "map.pt",
    )
    assert revised_map["anchor_ids"].tolist() == [0, 1]
    assert revised_map["anchor_type"].tolist() == [1, 1]
    assert revised_map["source_primitive_ids"].tolist() == [-1, -1]
    assert "track_centric_reconstruction" not in revised_map
    assert revised_map["base_anchor_count"] == 0
    assert revised_map["micro_anchor_count"] == 2
    assert revised_metric["landmark_indices"].tolist() == [0, 1]


def test_revision_teacher_refuses_to_remove_track_core(tmp_path):
    with pytest.raises(ValueError, match="Track Core"):
        subset_teacher(
            _teacher(),
            torch.tensor([False, True, True]),
            tmp_path / "map.pt",
            source_anchor_type=torch.tensor([1, 1, 0]),
        )


class _IdentityMetric(torch.nn.Module):
    def forward(self, descriptors):
        return descriptors, None


def _pose_w2c(*, center_x_m: float, rotation_deg: float) -> np.ndarray:
    angle = np.deg2rad(rotation_deg)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    center = np.asarray([center_x_m, 0.0, 0.0], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = -(rotation @ center)
    return pose


def test_mapping_statistics_report_rotation_tails_and_joint_recall(monkeypatch):
    query_names = ["q0", "q1", "q2"]
    record = {
        "query_rows": torch.tensor([0]),
        "positive_offsets": torch.tensor([0, 1]),
        "positive_indices": torch.tensor([0]),
        "ambiguous_offsets": torch.tensor([0, 0]),
        "ambiguous_indices": torch.empty(0, dtype=torch.long),
    }
    teacher = {
        "anchor_count": 1,
        "query_names": query_names,
        "records": [dict(record, query_index=index) for index in range(3)],
    }
    cached_query = {
        "native_descriptors": torch.tensor([[1.0, 0.0]]),
        "native_keypoints": torch.tensor([[0.0, 0.0]]),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
    }
    cache = {"queries": {name: cached_query for name in query_names}}
    estimates = iter(
        [
            _pose_w2c(center_x_m=0.04, rotation_deg=4.0),
            _pose_w2c(center_x_m=0.06, rotation_deg=1.0),
            _pose_w2c(center_x_m=0.01, rotation_deg=6.0),
        ]
    )
    monkeypatch.setattr(
        deployment_revision,
        "load_shared_metric",
        lambda *args, **kwargs: _IdentityMetric(),
    )
    monkeypatch.setattr(
        deployment_revision,
        "solve_absolute_pose",
        lambda *args, **kwargs: SimpleNamespace(
            pose_w2c=next(estimates),
            inliers=np.empty(0, dtype=np.int64),
            diagnostics={"iterations": 7},
        ),
    )

    statistics = collect_deployment_statistics(
        state={
            "anchor_ids": torch.tensor([0]),
            "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0]]),
            "anchor_features": torch.tensor([[1.0, 0.0]]),
        },
        metric_state_path="unused.pt",
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        ransac_reprojection_px=12.0,
        clean_reprojection_px=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
        seed=2026,
        collect_anchor_statistics=False,
    )

    assert [row["te_cm"] for row in statistics["queries"]] == pytest.approx(
        [4.0, 6.0, 1.0]
    )
    assert [row["ae_deg"] for row in statistics["queries"]] == pytest.approx(
        [4.0, 1.0, 6.0]
    )
    summary = statistics["summary"]
    assert summary["query_count"] == 3
    assert summary["median_te_cm"] == pytest.approx(4.0)
    assert summary["p90_te_cm"] == pytest.approx(np.percentile([4.0, 6.0, 1.0], 90))
    assert summary["median_ae_deg"] == pytest.approx(4.0)
    assert summary["p90_ae_deg"] == pytest.approx(np.percentile([4.0, 1.0, 6.0], 90))
    assert summary["p95_ae_deg"] == pytest.approx(np.percentile([4.0, 1.0, 6.0], 95))
    assert summary["recall_5cm_5deg_percent"] == pytest.approx(100.0 / 3.0)
    assert summary["raw_gt_precision_percent"] == 100.0


def test_mapping_statistics_replays_capacity_assignment_with_fallback(monkeypatch):
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
    captured = {}
    monkeypatch.setattr(
        deployment_revision,
        "load_shared_metric",
        lambda *args, **kwargs: _IdentityMetric(),
    )

    def solve(points_2d, points_3d, *args, **kwargs):
        captured["points_3d"] = np.asarray(points_3d)
        return SimpleNamespace(
            pose_w2c=np.eye(4),
            inliers=np.empty(0, dtype=np.int64),
            diagnostics={"iterations": 1},
        )

    monkeypatch.setattr(deployment_revision, "solve_absolute_pose", solve)
    statistics = collect_deployment_statistics(
        state={
            "anchor_ids": torch.tensor([0, 1]),
            "anchor_xyz": torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]]),
            "anchor_features": torch.tensor([[1.0, 0.0], [0.8, 0.6]]),
        },
        metric_state_path="unused.pt",
        teacher=teacher,
        query_cache=cache,
        device=torch.device("cpu"),
        ransac_reprojection_px=12.0,
        clean_reprojection_px=4.0,
        task_translation_m=0.05,
        task_rotation_deg=5.0,
        seed=2026,
        collect_anchor_statistics=False,
        assignment_topk=2,
        assignment_dustbin_score=-1.0,
    )
    assert captured["points_3d"].tolist() == [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    assert statistics["summary"]["raw_gt_precision_percent"] == 100.0
    assert statistics["summary"]["assignment_reassigned_query_rows"] == 1
    assert statistics["summary"]["assignment_top1_collisions"] == 1
