from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest
import torch

from evidence.virtual_render_planner import (
    PlannerPolicy,
    generate_candidate_poses,
    greedy_capped_coverage,
    validate_mapping_inputs,
)
from scripts.plan_sufficiency_guided_virtual_rendering import run


def test_capped_coverage_is_monotone_and_has_diminishing_returns():
    cells = [torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([2])]
    demand = torch.ones(3)
    family = torch.arange(3)
    selected1, trace1 = greedy_capped_coverage(
        cells, demand, family, budget=1, parallax=torch.zeros(3),
        appearance=torch.zeros(3), artifact_risk=torch.zeros(3),
    )
    selected2, trace2 = greedy_capped_coverage(
        cells, demand, family, budget=2, parallax=torch.zeros(3),
        appearance=torch.zeros(3), artifact_risk=torch.zeros(3),
    )
    assert selected1.tolist() == [0]
    assert selected2.tolist() == [0, 1]
    assert trace2[0]["coverage_gain"] == 2.0
    assert trace2[1]["coverage_gain"] == 1.0
    assert trace2[-1]["remaining_demand"] <= trace1[-1]["remaining_demand"]


def test_duplicate_pose_family_cannot_supply_two_triangulation_evidences():
    selected, trace = greedy_capped_coverage(
        [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])],
        torch.ones(3),
        torch.tensor([7, 7, 9]),
        budget=3,
        maximum_per_family=1,
        artifact_risk=torch.zeros(3),
    )
    assert selected.tolist() == [0, 2]
    assert [row["pose_family"] for row in trace] == [7, 9]


def test_candidate_pool_is_bounded_and_contains_declared_geometry_operations():
    poses = torch.eye(4).repeat(4, 1, 1)
    poses[:, 0, 3] = torch.tensor([0.0, -0.3, -0.6, -0.9])
    policy = PlannerPolicy(maximum_candidates=64, maximum_artifact_risk=0.7)
    result = generate_candidate_poses(
        poses, torch.tensor([[0.0, 0.0, 3.0], [1.0, 0.0, 4.0]]), policy
    )
    assert result["pose_w2c"].shape[0] <= 64
    kinds = set(result["kind"])
    assert {"se3_interpolation", "small_translation", "small_rotation",
            "boundary_expansion", "reverse_view", "deficit_directed"} <= kinds
    for parent in range(4):
        rows = [index for index, kind in enumerate(result["kind"])
                if kind in {"small_translation", "small_rotation"}
                and int(result["parent_camera_index"][index]) == parent]
        assert len(set(result["pose_family"][rows].tolist())) == 1
    assert bool((result["artifact_risk"] <= policy.maximum_artifact_risk).all())


def test_planner_rejects_test_or_source_rgb_evidence():
    track = {"rendered_rgb_only": True}
    with pytest.raises(ValueError, match="mapping-only"):
        validate_mapping_inputs(
            {"uses_test_queries": True, "uses_source_mapping_rgb": False}, track
        )
    with pytest.raises(ValueError, match="rendered-RGB"):
        validate_mapping_inputs(
            {"uses_test_queries": False, "uses_source_mapping_rgb": True}, track
        )


def _synthetic_inputs(tmp_path: Path):
    names = ["opaque-a", "opaque-b", "opaque-c"]
    queries = {}
    poses = []
    K = torch.tensor([[8.0, 0.0, 4.0], [0.0, 8.0, 4.0], [0.0, 0.0, 1.0]])
    for index, x in enumerate((-0.3, 0.0, 0.3)):
        pose = torch.eye(4)
        pose[0, 3] = -x
        poses.append(pose)
        queries[names[index]] = {
            "native_K": K,
            "pose_w2c": pose,
            "native_input_hw": [8, 8],
            "native_depth": torch.full((8, 8), 3.0),
            "native_alpha": torch.ones(8, 8),
        }
    query = {
        "schema": "lafgs_rendered_rgb_only_sparse_mapping_cache",
        "uses_test_queries": False,
        "uses_source_mapping_rgb": False,
        "queries": queries,
    }
    track = {
        "schema": "lafgs_track_first_payload",
        "rendered_rgb_only": True,
        "query_names": names,
        "query_bins": torch.tensor([0, 1, 2]),
        "tracks": {
            "track_index": torch.tensor([0, 0, 0]),
            "query_index": torch.tensor([0, 1, 2]),
            "keypoint_index": torch.tensor([0, 0, 0]),
        },
        "track_geometry": {
            "triangulated": torch.tensor([True]),
            "triangulated_xyz": torch.tensor([[0.0, 0.0, 3.0]]),
            "track_confidence_level": torch.tensor([2]),
        },
    }
    selected = {
        "track_cluster_ids": torch.tensor([0]),
    }
    paths = [tmp_path / name for name in ("query.pt", "track.pt", "map.pt")]
    for path, value in zip(paths, (query, track, selected)):
        torch.save(value, path)
    return paths


def test_cli_materializes_schema_without_rendering_or_test_selection(tmp_path):
    query, track, selected_map = _synthetic_inputs(tmp_path)
    output = tmp_path / "plan.pt"
    summary = run(Namespace(
        query_cache=query,
        track_payload=track,
        selected_map=selected_map,
        output=output,
        view_budget=4,
        maximum_candidates=32,
        voxel_size_m=0.5,
        surface_stride=2,
    ))
    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["schema"] == "lafgs_sufficiency_guided_virtual_render_plan"
    assert payload["mapping_only"] is True
    assert payload["uses_test_queries"] is False
    assert payload["renders_images"] is False
    assert payload["default_pipeline_enabled"] is False
    assert payload["gt_visible_diagnostic"] is None
    assert summary["candidate_count"] <= 32
    assert int(payload["coverage_field"]["stable_observation_count"].max()) <= 3
    chosen_families = payload["candidates"]["pose_family"][
        payload["selected_candidate_indices"]
    ]
    assert chosen_families.unique().numel() == chosen_families.numel()
