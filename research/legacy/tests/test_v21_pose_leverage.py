from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from map_learning.v21_pose_leverage import (
    ASSIGNMENT_INDETERMINATE,
    COVERAGE_LIMITED,
    DESCRIPTOR_CONTROLLABLE,
    GAUSSIAN_GEOMETRY_SUPPORTED,
    GEOMETRY_LIMITED,
    PROTECTION_ONLY,
    REPROJECTION_UPPER_BOUND,
    SOLVER_LIMITED,
    TRACK_CONSENSUS_DIAGNOSTIC,
    TRACK_DIAGNOSTIC_COVERAGE_LIMITED,
    analyze_pose_recovery_query,
    build_legal_positive_csr,
    build_track_consensus_diagnostic_csr,
    select_equivalence_unique_legal_pairs,
    summarize_pose_recovery,
)
from map_learning.v21_correspondence_truth import (
    STATUS_AMBIGUOUS,
    STATUS_EQUIVALENT,
    STATUS_NAMES,
    STATUS_NO_TRUTH,
    STATUS_UNIQUE,
)
from scripts.audit_v21_pose_recovery_oracle import (
    _resolve_baseline_plant,
    _support_geometry_kwargs,
    _validate_complete_cache_shards,
)


def _threshold_solver(required: int):
    def solve(_keypoints, points, _intrinsics, **_kwargs):
        corrected = int((torch.as_tensor(points)[:, 2] > 1.5).sum())
        pose = torch.eye(4)
        if corrected < required:
            pose[0, 3] = 1.0
        return SimpleNamespace(
            pose_w2c=pose.numpy(),
            inliers=np.arange(len(points), dtype=np.int64),
        )

    return solve


def _robust_target_solver(*, robust_available: bool):
    def solve(_keypoints, points, _intrinsics, **_kwargs):
        corrected = int((torch.as_tensor(points)[:, 2] > 1.5).sum())
        pose = torch.eye(4)
        if corrected < 2:
            pose[0, 3] = 0.10
        elif corrected < 4 or not robust_available:
            pose[0, 3] = 0.045
        return SimpleNamespace(
            pose_w2c=pose.numpy(),
            inliers=np.arange(len(points), dtype=np.int64),
        )

    return solve


def _square_scene() -> dict:
    keypoints = torch.tensor(
        [[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]]
    )
    wrong = torch.tensor(
        [
            [101.0, 101.0, 1.0],
            [104.0, 101.0, 1.0],
            [101.0, 104.0, 1.0],
            [104.0, 104.0, 1.0],
        ]
    )
    # z=2 is also a convenient marker for the discontinuous synthetic solver.
    positive = torch.tensor(
        [[2.0, 2.0, 2.0], [8.0, 2.0, 2.0], [2.0, 8.0, 2.0], [8.0, 8.0, 2.0]]
    )
    return {
        "keypoints": keypoints,
        "anchor_xyz": torch.cat((wrong, positive)),
        "winners": torch.arange(4),
        "intrinsic": torch.eye(3),
        "pose": torch.eye(4),
        "equivalence": torch.arange(8),
    }


def _sampled_support(scene: dict) -> dict:
    count = int(scene["keypoints"].shape[0])
    return {
        "gaussian_depth_at_keypoints": torch.full((count,), 2.0),
        "gaussian_alpha_at_keypoints": torch.ones(count),
        "gaussian_valid_keypoint_mask": torch.ones(count, dtype=torch.bool),
        "gaussian_relative_depth_spread_3x3": torch.zeros(count),
        "gaussian_local_valid_fraction_3x3": torch.ones(count),
    }


def _track_record(anchor_lists: list[list[int]], statuses: list[int] | None = None):
    counts = torch.tensor([len(value) for value in anchor_lists], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    anchors = torch.tensor(
        [anchor for values in anchor_lists for anchor in values], dtype=torch.long
    )
    if statuses is None:
        statuses = [
            STATUS_NO_TRUTH
            if len(values) == 0
            else STATUS_UNIQUE
            if len(values) == 1
            else STATUS_EQUIVALENT
            for values in anchor_lists
        ]
    count = len(anchor_lists)
    return {
        "query_index": 17,
        "image_name": "query.png",
        "image_sha256": "a" * 64,
        "sequence_id": "seq",
        "frame_index": 1,
        "block_id": "block",
        "role": "adaptation",
        "source_record_sha256": "b" * 64,
        "pose_w2c_sha256": "c" * 64,
        "keypoint_count": count,
        "keypoints_sha256": "d" * 64,
        "descriptors_sha256": "e" * 64,
        "tier_name": "tier_c",
        "requested_action": "planner_priority",
        "action_authorized": False,
        "geometry_valid": torch.ones(count, dtype=torch.bool),
        "source_v19_invalid": torch.zeros(count, dtype=torch.bool),
        "projection_candidate_count": torch.ones(count, dtype=torch.long),
        "diagnostic_truth_status": torch.tensor(statuses, dtype=torch.int8),
        "diagnostic_positive_offsets": offsets,
        "diagnostic_positive_anchor_rows": anchors,
        "truth_status": torch.full((count,), STATUS_NO_TRUTH, dtype=torch.int8),
        "truth_status_names": STATUS_NAMES,
        "positive_offsets": torch.zeros(count + 1, dtype=torch.long),
        "positive_anchor_rows": torch.empty(0, dtype=torch.long),
        "negative_anchor_rows": None,
        "ambiguous_or_unlabelled_are_negative": False,
    }


def _analyze(scene: dict, **kwargs) -> dict:
    return analyze_pose_recovery_query(
        query_index=17,
        keypoints=scene["keypoints"],
        winner_anchor_rows=scene["winners"],
        anchor_xyz=scene["anchor_xyz"],
        intrinsic=scene["intrinsic"],
        ground_truth_w2c=scene["pose"],
        equivalence_class_ids=scene["equivalence"],
        positive_reprojection_px=0.1,
        ransac_reprojection_px=4.0,
        maximum_minimal_set_size=4,
        **kwargs,
    )


def test_legal_positive_csr_uses_depth_alpha_and_preserves_equivalence() -> None:
    keypoints = torch.tensor([[1.0, 1.0], [3.0, 3.0]])
    xyz = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],  # same projection, wrong Gaussian depth
            [2.0, 2.0, 2.0],
        ]
    )
    result = build_legal_positive_csr(
        keypoints=keypoints,
        anchor_xyz=xyz,
        intrinsic=torch.eye(3),
        ground_truth_w2c=torch.eye(4),
        equivalence_class_ids=torch.tensor([7, 8, 7]),
        row_valid_mask=torch.tensor([True, False]),
        gaussian_depth_at_keypoints=torch.tensor([2.0, 2.0]),
        gaussian_alpha_at_keypoints=torch.ones(2),
        gaussian_valid_keypoint_mask=torch.tensor([True, False]),
        gaussian_relative_depth_spread_3x3=torch.tensor([0.01, float("inf")]),
        gaussian_local_valid_fraction_3x3=torch.tensor([1.0, 0.0]),
        reprojection_threshold_px=0.1,
        depth_absolute_m=0.1,
        depth_relative=0.0,
    )
    assert result["positive_evidence_mode"] == GAUSSIAN_GEOMETRY_SUPPORTED
    assert result["geometry_supported_candidate"] is True
    assert result["deployable_positive_authorized"] is False
    assert result["positive_offsets"].tolist() == [0, 2, 2]
    assert result["positive_anchor_rows"].tolist() == [0, 2]
    assert result["positive_equivalence_class_ids"].tolist() == [7, 7]
    assert result["negative_anchor_rows"].numel() == 0
    assert result["unlabeled_rows_are_negative"] is False


def test_equivalence_unique_pairs_use_maximum_cardinality_not_greedy() -> None:
    legal = {
        "row_count": 2,
        "positive_offsets": torch.tensor([0, 2, 3]),
        "positive_anchor_rows": torch.tensor([0, 1, 0]),
        "positive_equivalence_class_ids": torch.tensor([0, 1, 0]),
        "positive_reprojection_error_px": torch.tensor([0.0, 1.0, 0.5]),
    }
    result = select_equivalence_unique_legal_pairs(
        legal_positive_csr=legal,
        equivalence_class_ids=torch.tensor([0, 1]),
    )
    assert result["query_rows"].tolist() == [0, 1]
    assert result["anchor_rows"].tolist() == [1, 0]


def test_track_diagnostic_csr_uses_only_unique_and_equivalent_rows() -> None:
    record = _track_record(
        [[0], [1, 2], [], []],
        [STATUS_UNIQUE, STATUS_EQUIVALENT, STATUS_AMBIGUOUS, STATUS_NO_TRUTH],
    )
    result = build_track_consensus_diagnostic_csr(
        keypoints=torch.tensor([[1.0, 1.0], [3.0, 3.0], [5.0, 5.0], [7.0, 7.0]]),
        anchor_xyz=torch.tensor(
            [[2.0, 2.0, 2.0], [6.0, 6.0, 2.0], [6.0, 6.0, 2.0]]
        ),
        intrinsic=torch.eye(3),
        ground_truth_w2c=torch.eye(4),
        track_consensus_record=record,
        equivalence_class_ids=torch.tensor([10, 20, 20]),
    )
    assert result["positive_evidence_mode"] == TRACK_CONSENSUS_DIAGNOSTIC
    assert result["positive_offsets"].tolist() == [0, 1, 3, 3, 3]
    assert result["positive_anchor_rows"].tolist() == [0, 1, 2]
    assert result["source_decisive_row_count"] == 2
    assert result["source_ambiguous_or_no_truth_row_count"] == 2
    assert result["deployable_positive_authorized"] is False
    assert result["negative_anchor_rows"].numel() == 0


def test_track_diagnostic_exact_recovery_remains_planner_only() -> None:
    scene = _square_scene()
    record = _track_record([[4], [5], [6], [7]])
    recovered = _analyze(
        scene,
        positive_source=TRACK_CONSENSUS_DIAGNOSTIC,
        track_consensus_record=record,
        solver=_threshold_solver(2),
    )
    assert recovered["route"] == DESCRIPTOR_CONTROLLABLE
    assert recovered["positive_source"] == TRACK_CONSENSUS_DIAGNOSTIC
    assert recovered["controller_authorized"] is False
    assert recovered["recovery_bundle"]["exact_delta_r5"] == 1
    assert "no_map_or_metric_action_authority" in recovered["authorization_reason"]

    unavailable = _analyze(
        scene,
        positive_source=TRACK_CONSENSUS_DIAGNOSTIC,
        track_consensus_record=record,
        solver=_threshold_solver(99),
    )
    assert unavailable["route"] == TRACK_DIAGNOSTIC_COVERAGE_LIMITED
    assert unavailable["controller_authorized"] is False
    assert "no_global_solver_exclusion" in unavailable["authorization_reason"]


def test_track_diagnostic_rejects_a_second_geometry_positive_source() -> None:
    scene = _square_scene()
    with pytest.raises(ValueError, match="exactly one positive source"):
        _analyze(
            scene,
            positive_source=TRACK_CONSENSUS_DIAGNOSTIC,
            track_consensus_record=_track_record([[4], [5], [6], [7]]),
            **_sampled_support(scene),
            solver=_threshold_solver(2),
        )


def test_exact_oracle_finds_inclusion_minimal_recovery_and_necessity() -> None:
    scene = _square_scene()
    result = _analyze(
        scene,
        **_sampled_support(scene),
        topk_candidate_anchor_rows=scene["winners"][:, None],
        solver=_threshold_solver(2),
    )
    assert result["route"] == DESCRIPTOR_CONTROLLABLE
    assert result["controller_authorized"] is False
    assert "no_identity_authority" in result["authorization_reason"]
    assert result["topk"]["legal_positive_recall_row_count"] == 0
    assert result["topk"]["topk_is_authorization_source"] is False
    assert result["one_assignment_lower_bound"]["r5_success"] is True
    assert (
        result["one_assignment_lower_bound"]["semantics"]
        == "one_deterministic_assignment_not_all_legal_assignments"
    )
    bundle = result["recovery_bundle"]
    assert bundle["query_rows"].numel() == 2
    assert bundle["anchor_rows"].numel() == 2
    assert bundle["exact_delta_r5"] == 1
    assert bundle["exact_delta_task"] > 0.0
    assert bundle["inclusion_minimal"] is True
    assert all(
        row["removal_loses_recovery"] and row["exact_r5_loss_if_removed"] == 1
        for row in bundle["row_necessity"]
    )


def test_reprojection_only_can_diagnose_but_never_authorizes_action() -> None:
    scene = _square_scene()
    result = _analyze(scene, solver=_threshold_solver(2))
    assert result["route"] == DESCRIPTOR_CONTROLLABLE
    assert result["legal_positive_csr"]["positive_evidence_mode"] == (
        REPROJECTION_UPPER_BOUND
    )
    assert result["controller_authorized"] is False
    assert "upper_bound_only" in result["authorization_reason"]
    assert result["recovery_bundle"]["exact_delta_r5"] == 1


def test_stricter_bundle_target_does_not_change_standard_r5_reporting() -> None:
    scene = _square_scene()
    result = _analyze(
        scene,
        bundle_target_translation_cm=4.0,
        bundle_target_rotation_deg=4.0,
        solver=_robust_target_solver(robust_available=True),
    )
    bundle = result["recovery_bundle"]
    assert result["baseline"]["r5_success"] is False
    assert result["one_assignment_lower_bound"]["r5_success"] is True
    assert result["one_assignment_lower_bound"]["bundle_target_success"] is True
    assert bundle["query_rows"].numel() == 4
    assert bundle["bundle_target_success"] is True
    assert bundle["standard_r5_success"] is True
    assert bundle["exact_delta_r5"] == 1
    assert all(
        row["removal_loses_bundle_target"]
        and row["exact_bundle_target_loss_if_removed"] == 1
        and row["exact_r5_loss_if_removed"] == 0
        for row in bundle["row_necessity"]
    )
    assert result["bundle_target"]["changes_standard_r5_definition"] is False


def test_standard_recovery_without_strict_target_emits_no_fragile_bundle() -> None:
    scene = _square_scene()
    result = _analyze(
        scene,
        bundle_target_translation_cm=4.0,
        bundle_target_rotation_deg=4.0,
        solver=_robust_target_solver(robust_available=False),
    )
    assert result["route"] == DESCRIPTOR_CONTROLLABLE
    assert result["one_assignment_lower_bound"]["r5_success"] is True
    assert result["one_assignment_lower_bound"]["bundle_target_success"] is False
    assert result["recovery_bundle"] is None
    assert "stricter_bundle_target_unavailable" in result["authorization_reason"]


def test_success_query_emits_protection_only_and_no_repair_bundle() -> None:
    scene = _square_scene()
    scene["winners"] = torch.arange(4, 8)
    result = _analyze(
        scene,
        **_sampled_support(scene),
        solver=_threshold_solver(1),
    )
    assert result["route"] == PROTECTION_ONLY
    assert result["controller_authorized"] is False
    assert result["recovery_bundle"] is None
    assert result["one_assignment_lower_bound"] is None
    assert result["protection"]["query_rows"].numel() == 0
    assert result["protection"]["anchor_rows"].numel() == 0
    assert result["protection"]["pose_inlier_query_rows"].numel() == 0
    assert result["protection"]["geometry_candidate_query_rows"].tolist() == [
        0,
        1,
        2,
        3,
    ]
    assert result["protection"]["identity_certified_positive_count"] == 0


def test_no_legal_positive_is_coverage_limited_and_never_a_negative_label() -> None:
    scene = _square_scene()
    result = _analyze(
        scene,
        row_valid_mask=torch.zeros(4, dtype=torch.bool),
        **_sampled_support(scene),
        solver=_threshold_solver(99),
    )
    assert result["route"] == COVERAGE_LIMITED
    assert result["controller_authorized"] is False
    assert result["legal_positive_csr"]["legal_positive_edge_count"] == 0
    assert result["correction_candidates"]["candidate_rows"].numel() == 0
    assert result["negative_anchor_rows"].numel() == 0
    assert result["unlabeled_rows_are_negative"] is False


def test_failed_full_oracle_routes_geometry_and_solver_limits_separately() -> None:
    scene = _square_scene()
    solver_limited = _analyze(
        scene,
        **_sampled_support(scene),
        solver=_threshold_solver(99),
    )
    assert solver_limited["route"] == SOLVER_LIMITED
    assert solver_limited["legal_only_diagnostic"]["geometry"]["degenerate"] is False

    line = _square_scene()
    line["keypoints"] = torch.tensor(
        [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]
    )
    line["anchor_xyz"][4:] = torch.tensor(
        [[2.0, 2.0, 2.0], [4.0, 2.0, 2.0], [6.0, 2.0, 2.0], [8.0, 2.0, 2.0]]
    )
    geometry_limited = _analyze(
        line,
        **_sampled_support(line),
        solver=_threshold_solver(99),
    )
    assert geometry_limited["route"] == GEOMETRY_LIMITED
    assert geometry_limited["legal_only_diagnostic"]["geometry"]["degenerate"] is True


def test_equivalent_current_winner_is_not_mined_as_a_repair() -> None:
    scene = _square_scene()
    scene["equivalence"] = torch.tensor([4, 5, 6, 7, 4, 5, 6, 7])
    result = _analyze(
        scene,
        **_sampled_support(scene),
        solver=_threshold_solver(99),
    )
    assert result["correction_candidates"]["candidate_rows"].numel() == 0
    assert result["controller_authorized"] is False
    assert result["recovery_bundle"] is None


def test_failed_ambiguous_assignment_is_indeterminate_not_solver_limited() -> None:
    scene = _square_scene()
    scene["anchor_xyz"] = torch.cat((scene["anchor_xyz"], scene["anchor_xyz"][4:]))
    scene["equivalence"] = torch.arange(12)
    result = _analyze(
        scene,
        **_sampled_support(scene),
        solver=_threshold_solver(99),
    )
    assert result["route"] == ASSIGNMENT_INDETERMINATE
    assert result["correction_candidates"]["assignment_search_exhaustive"] is False


def test_oracle_plant_threshold_is_read_from_cache_without_default_drift() -> None:
    real_threshold = 11.954342673385437
    contract = {"reprojection_error_px": real_threshold, "seed": 2026}
    assert _resolve_baseline_plant(
        contract, requested_reprojection_px=None, requested_seed=None
    ) == (real_threshold, 2026)
    with pytest.raises(ValueError, match="threshold differs"):
        _resolve_baseline_plant(
            contract,
            requested_reprojection_px=11.954343111400277,
            requested_seed=None,
        )


def test_cli_requires_complete_frontend_shards_and_forwards_sampled_support() -> None:
    registry = {"registry_sha256": "a" * 64}
    split = {"path": "/split.pt", "sha256": "b" * 64}
    stable = {"path": "/map.pt", "sha256": "c" * 64}

    def entry(index: int):
        payload = {
            "shard_index": index,
            "shard_count": 2,
            "shard_registry": registry,
            "inputs": {"split_manifest": split, "stable_map": stable},
        }
        return (Path(f"/cache-{index}.pt"), str(index) * 64, payload)

    with pytest.raises(ValueError, match="full registry"):
        _validate_complete_cache_shards([entry(0)])
    assert [
        value[2]["shard_index"]
        for value in _validate_complete_cache_shards([entry(1), entry(0)])
    ] == [0, 1]

    sampled = {
        "gaussian_depth_at_keypoints": torch.tensor([2.0]),
        "gaussian_alpha_at_keypoints": torch.tensor([1.0]),
        "gaussian_support_valid": torch.tensor([True]),
        "gaussian_relative_depth_spread_3x3": torch.tensor([0.0]),
        "gaussian_local_valid_fraction_3x3": torch.tensor([1.0]),
    }
    forwarded = _support_geometry_kwargs(sampled)
    assert forwarded["gaussian_valid_keypoint_mask"].tolist() == [True]
    assert "gaussian_depth_at_keypoints" in forwarded


def test_summary_preserves_legacy_geometry_v1_but_never_infers_track() -> None:
    scene = _square_scene()
    legacy = _analyze(scene, solver=_threshold_solver(99))
    legacy.pop("positive_source")
    summary = summarize_pose_recovery([legacy])
    assert summary["all_action_authority_is_exact_poselib"] is True
    assert "track_consensus_diagnostic_query_count" not in summary
    assert "all_pose_recovery_claims_use_exact_poselib" not in summary

    track = _analyze(
        scene,
        positive_source=TRACK_CONSENSUS_DIAGNOSTIC,
        track_consensus_record=_track_record([[4], [5], [6], [7]]),
        solver=_threshold_solver(99),
    )
    track.pop("positive_source")
    with pytest.raises(ValueError, match="legacy source inference"):
        summarize_pose_recovery([track])
