import torch

from localization_training.micro_anchors import (
    build_add_only_materialized_anchor_map,
    compute_track_coverage_gain,
    protected_micro_anchor_descriptor_loss,
    robust_fuse_track_descriptors,
    truncate_materialized_anchor_map,
)
from stdloc import load_materialized_anchor_map


def test_materialized_anchor_schema_allows_repeated_source_primitives(tmp_path):
    path = tmp_path / "anchors.pt"
    torch.save(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_ids": torch.tensor([0, 1, 2]),
            "source_primitive_ids": torch.tensor([4, 4, 7]),
            "anchor_features": torch.tensor(
                [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]
            ),
            "anchor_xyz": torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.2, 1.0]]
            ),
        },
        path,
    )
    loaded = load_materialized_anchor_map(
        path, point_count=8, expected_feature_dim=2
    )
    assert loaded["anchor_ids"].tolist() == [0, 1, 2]
    assert loaded["source_primitive_ids"].tolist() == [4, 4, 7]
    assert torch.allclose(
        torch.linalg.norm(loaded["anchor_features"], dim=1),
        torch.ones(3),
    )


def test_track_descriptor_fusion_balances_view_bins_and_trims_outlier():
    descriptors = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.98, 0.02],
            [-1.0, 0.0],
        ]
    )
    fused = robust_fuse_track_descriptors(
        descriptors,
        torch.tensor([0, 0, 1, 2]),
        trim_fraction=1.0 / 3.0,
    )
    assert fused[0] > 0.99
    assert fused[1].abs() < 0.02


def test_add_only_builder_preserves_base_rows_and_adds_level_a_track():
    base_state = {
        "landmark_indices": torch.tensor([10]),
        "landmark_features": torch.tensor([[0.0, 1.0]]),
        "landmark_xyz": torch.tensor([[5.0, 5.0, 2.0]]),
    }
    payload = {
        "schema": "lafgs_track_first_payload",
        "query_names": ["q0", "q1", "q2"],
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "track_index": torch.tensor([0, 0, 0]),
            "query_index": torch.tensor([0, 1, 2]),
            "keypoint_index": torch.tensor([0, 0, 0]),
            "confidence": torch.ones(3),
        },
        "track_geometry": {
            "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
            "triangulation_high_confidence": torch.tensor([True]),
            "track_confidence_level": torch.tensor([2], dtype=torch.int8),
            "triangulation_distinct_view_bin_count": torch.tensor([3]),
            "triangulation_observation_count": torch.tensor([3]),
            "triangulation_reprojection_median_px": torch.tensor([0.0]),
            "triangulation_covariance_trace": torch.tensor([1e-5]),
        },
        "assignment": {"track_landmark_index": torch.tensor([0])},
    }
    cache = {"queries": {}}
    K = torch.eye(3)
    pose = torch.eye(4)
    for index in range(3):
        cache["queries"][f"q{index}"] = {
            "native_K": K,
            "pose_w2c": pose,
            "native_keypoints": torch.tensor([[0.0, 0.0]]),
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_depth": torch.full((4, 4), 2.0),
            "pixel_center_offset": 0.0,
        }
    state, diagnostics = build_add_only_materialized_anchor_map(
        base_state=base_state,
        payload=payload,
        query_cache=cache,
        budget=1,
    )
    assert state["anchor_ids"].tolist() == [0, 1]
    assert state["source_primitive_ids"].tolist() == [10, 10]
    assert state["track_cluster_ids"].tolist() == [-1, 0]
    assert torch.equal(
        state["anchor_features"][0], base_state["landmark_features"][0]
    )
    assert diagnostics["selected_micro_anchor_count"] == 1

    control, control_diagnostics = build_add_only_materialized_anchor_map(
        base_state=base_state,
        payload=payload,
        query_cache=cache,
        budget=0,
    )
    assert control["anchor_ids"].tolist() == [0]
    assert control["source_primitive_ids"].tolist() == [10]
    assert control_diagnostics["selected_micro_anchor_count"] == 0


def test_protected_descriptor_loss_rewards_target_and_preserves_guard():
    candidate = torch.nn.Parameter(
        torch.nn.functional.normalize(
            torch.tensor([[1.0, 0.2], [0.1, 1.0]]), dim=1
        )
    )
    initial = candidate.detach().clone()
    positive = torch.nn.functional.normalize(
        torch.tensor([[0.8, 0.6], [0.2, 1.0]]), dim=1
    )
    targets = torch.tensor([0, 1])
    positive_old_best = torch.tensor([0.7, 0.8])
    guard = torch.tensor([[1.0, 0.0]])
    guard_old_best = torch.tensor([0.99])
    optimizer = torch.optim.Adam([candidate], lr=0.03)
    with torch.no_grad():
        _, before = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate,
            positive_descriptors=positive,
            positive_targets=targets,
            positive_old_best=positive_old_best,
            guard_descriptors=guard,
            guard_old_best=guard_old_best,
            initial_features=initial,
            guard_weight=2.0,
            trust_weight=0.01,
        )
    for _ in range(40):
        optimizer.zero_grad()
        loss, _ = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate,
            positive_descriptors=positive,
            positive_targets=targets,
            positive_old_best=positive_old_best,
            guard_descriptors=guard,
            guard_old_best=guard_old_best,
            initial_features=initial,
            guard_weight=2.0,
            trust_weight=0.01,
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        _, after = protected_micro_anchor_descriptor_loss(
            candidate_features=candidate,
            positive_descriptors=positive,
            positive_targets=targets,
            positive_old_best=positive_old_best,
            guard_descriptors=guard,
            guard_old_best=guard_old_best,
            initial_features=initial,
            guard_weight=2.0,
            trust_weight=0.01,
        )
    assert after["loss"] < before["loss"]
    assert after["guard_loss"] <= before["guard_loss"] + 1e-6
    assert after["positive_loss"] < 0.005


def test_coverage_gain_respects_raster_visibility():
    payload = {
        "query_names": ["q0"],
        "query_bins": torch.tensor([0]),
        "tracks": {
            "track_index": torch.tensor([0]),
            "query_index": torch.tensor([0]),
            "keypoint_index": torch.tensor([0]),
        },
        "track_geometry": {
            "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0]])
        },
    }
    query_cache = {
        "q0": {
            "native_K": torch.eye(3),
            "pose_w2c": torch.eye(4),
            "native_keypoints": torch.tensor([[0.0, 0.0]]),
            "native_depth": torch.full((2, 2), 2.0),
            "pixel_center_offset": 0.0,
        }
    }
    base_xyz = torch.tensor([[0.0, 0.0, 2.0]])

    center_visible = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
    )
    raster_hidden = compute_track_coverage_gain(
        payload=payload,
        query_cache=query_cache,
        base_xyz=base_xyz,
        visibility_cache={"q0": torch.tensor([False])},
    )

    assert center_visible["represented_observations"].tolist() == [1]
    assert center_visible["coverage_gain"].tolist() == [0]
    assert raster_hidden["represented_observations"].tolist() == [0]
    assert raster_hidden["coverage_gain"].tolist() == [1]
    assert bool(raster_hidden["raster_visibility_enabled"])


def test_truncate_materialized_anchor_map_preserves_csr_alignment():
    state = {
        "schema": "lafgs_materialized_anchor_map",
        "base_anchor_count": 2,
        "micro_anchor_count": 2,
        "anchor_ids": torch.arange(4),
        "source_primitive_ids": torch.tensor([1, 2, 3, 4]),
        "track_cluster_ids": torch.tensor([-1, -1, 10, 11]),
        "anchor_xyz": torch.zeros(4, 3),
        "anchor_features": torch.zeros(4, 2),
        "anchor_type": torch.tensor([0, 0, 1, 2]),
        "track_cluster_member_offsets": torch.tensor([0, 0, 0, 2, 3]),
        "track_cluster_member_ids": torch.tensor([20, 21, 22]),
        "source_group_offsets": torch.tensor([0, 1, 2, 4, 5]),
        "source_group_primitive_ids": torch.tensor([1, 2, 3, 30, 4]),
        "micro_anchor_quality": {
            "coverage_gain": torch.tensor([7, 5]),
        },
    }

    truncated = truncate_materialized_anchor_map(state, 1)

    assert truncated["anchor_ids"].tolist() == [0, 1, 2]
    assert truncated["micro_anchor_count"] == 1
    assert truncated["track_cluster_member_offsets"].tolist() == [0, 0, 0, 2]
    assert truncated["track_cluster_member_ids"].tolist() == [20, 21]
    assert truncated["source_group_offsets"].tolist() == [0, 1, 2, 4]
    assert truncated["source_group_primitive_ids"].tolist() == [1, 2, 3, 30]
    assert truncated["micro_anchor_quality"]["coverage_gain"].tolist() == [7]
