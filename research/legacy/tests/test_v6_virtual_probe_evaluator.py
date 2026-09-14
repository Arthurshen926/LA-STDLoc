from types import SimpleNamespace

import torch

from map_learning.v6_virtual_probe_evaluator import evaluate_fixed_map_virtual_probes


def _identity_solver(keypoints, _points, _intrinsics, **_kwargs):
    return SimpleNamespace(
        pose_w2c=torch.eye(4).numpy(),
        inliers=list(range(len(keypoints))),
    )


def test_virtual_probe_evaluator_keeps_map_fixed_and_reports_oracle() -> None:
    xyz = torch.tensor(
        [
            [-0.2, -0.2, 2.0],
            [0.2, -0.2, 2.0],
            [-0.2, 0.2, 2.0],
            [0.2, 0.2, 2.0],
        ]
    )
    intrinsics = torch.tensor(
        [[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]
    )
    projected = torch.tensor(
        [[9.0, 9.0], [11.0, 9.0], [9.0, 11.0], [11.0, 11.0]]
    )
    state = {"anchor_xyz": xyz, "anchor_features": torch.eye(4)}
    cache = {
        "schema": "lafgs_v6_fixed_map_observer_probe_cache",
        "version": 2,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "virtual_probes_added_to_map": False,
        "virtual_probes_added_to_anchor_observations": False,
        "query_names": ["virtual/0000/clean", "virtual/0001/clean"],
        "inputs": {
            "source_map_sha256": "a" * 64,
            "probe_plan_sha256": "b" * 64,
            "gaussian_ply_sha256": "c" * 64,
        },
        "queries": {
            name: {
                "native_keypoints": projected - 0.5,
                "native_descriptors": torch.eye(4),
                "native_scores": torch.ones(4),
                "native_K": intrinsics,
                "pose_w2c": torch.eye(4),
                "native_input_hw": torch.tensor([20, 20]),
                "native_alpha": torch.ones((20, 20)),
                "native_depth": torch.full((20, 20), 2.0),
                "native_valid_keypoint_mask": torch.ones(4, dtype=torch.bool),
                "pixel_center_offset": 0.5,
                "probe_index": index,
                "sensor_variant": "clean",
                "probe_kind": "interpolation",
                "pose_family": index,
            }
            for index, name in enumerate(
                ("virtual/0000/clean", "virtual/0001/clean")
            )
        },
    }
    result = evaluate_fixed_map_virtual_probes(
        state,
        cache,
        map_sha256="a" * 64,
        probe_cache_sha256="d" * 64,
        positive_radius_px=0.6,
        ransac_reprojection_px=4.0,
        device="cpu",
        solver=_identity_solver,
    )
    record = result["records"][0]
    assert result["fixed_map_plant"] is True
    assert result["virtual_probes_added_to_map"] is False
    assert record["pose_valid_winner_count"] == 4
    assert record["pose_valid_unique_anchor_count"] == 4
    assert record["oracle_available"] is True
    assert result["summary"]["recall_5cm_5deg_percent"] == 100.0
    assert result["control_split"]["training_probe_indices"] == [0]
    assert result["control_split"]["validation_probe_indices"] == [1]
    assert result["control_split"]["validation_used_by_controller"] is False
    assert record["controller_route"] == "nominal_success"
    assert result["frontend_correspondence_ceiling"]["controller_route_counts"][
        "nominal_success"
    ] == 2
    assert len(record["winner_anchor_ids"]) == 4
    assert len(record["descriptor_triplet_pose_weights"]) == len(
        record["descriptor_triplets"]
    )
