from pathlib import Path
import json

import pytest
import torch

from common.hashing import sha256_file
from scripts.materialize_rendered_track_training import materialize
from scripts.evaluate_rendered_track_crossfit import (
    _crossfit_groups,
    _fold_component_bank,
    _subset_state,
)
from scripts.materialize_rendered_track_fullchain_inputs import (
    materialize as materialize_fullchain_inputs,
)
from scripts.train_rendered_track_crossfit import _training_inputs


def _inputs(tmp_path: Path):
    names = ["seq-02/a.png", "seq-03/b.png", "seq-05/c.png"]
    queries = {}
    poses = []
    for index, name in enumerate(names):
        pose = torch.eye(4)
        pose[0, 3] = -0.1 * index
        poses.append(pose)
        queries[name] = {
            "native_keypoints": torch.tensor([[50.0 - 5.0 * index, 50.0]]),
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_K": torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            "pose_w2c": pose,
            "native_input_hw": torch.tensor([100, 100]),
            "pixel_center_offset": 0.0,
        }
    cache = {
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": queries,
    }
    track_payload = {
        "query_names": names,
        "query_bins": torch.arange(3),
        "tracks": {
            "track_index": torch.zeros(3, dtype=torch.long),
            "query_index": torch.arange(3),
            "keypoint_index": torch.zeros(3, dtype=torch.long),
        },
    }
    anchor_map = {
        "anchor_ids": torch.tensor([0]),
        "anchor_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
        "anchor_features": torch.tensor([[1.0, 0.0]]),
        "track_cluster_ids": torch.tensor([0]),
    }
    paths = {}
    for key, value in (
        ("cache", cache),
        ("track", track_payload),
        ("map", anchor_map),
    ):
        paths[key] = tmp_path / f"{key}.pt"
        torch.save(value, paths[key])
    return paths


def test_rendered_track_training_materializes_geometry_teacher(tmp_path):
    paths = _inputs(tmp_path)
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "out",
        strong_radius_px=2.0,
        ambiguous_radius_px=8.0,
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    payload = torch.load(
        report["outputs"]["track_payload"],
        map_location="cpu",
        weights_only=False,
    )
    assert teacher["diagnostics"]["positive_rows"] == 3
    assert teacher["diagnostics"]["exact_track_positive_count"] == 3
    assert teacher["query_cache"] == str(paths["cache"].resolve())
    assert teacher["query_cache_sha256"] == sha256_file(paths["cache"])
    assert all(
        record["positive_indices"].tolist() == [0] for record in teacher["records"]
    )
    assert payload["training_sequence_names"] == ["seq-02", "seq-03", "seq-05"]
    assert payload["query_bins"].tolist() == [0, 1, 2]
    assert report["uses_source_mapping_rgb"] is False
    assert report["uses_test_queries"] is False


def test_rendered_track_training_rejects_source_rgb_or_test_cache(tmp_path):
    for field in ("uses_source_mapping_rgb", "uses_test_queries"):
        case = tmp_path / field
        case.mkdir()
        paths = _inputs(case)
        cache = torch.load(paths["cache"], map_location="cpu", weights_only=False)
        cache[field] = True
        torch.save(cache, paths["cache"])
        try:
            materialize(
                anchor_map_path=paths["map"],
                track_payload_path=paths["track"],
                query_cache_path=paths["cache"],
                output_dir=case / "out",
                strong_radius_px=2.0,
                ambiguous_radius_px=8.0,
            )
        except ValueError as error:
            assert "rendered-RGB-only" in str(error) or "test queries" in str(error)
        else:
            raise AssertionError("forbidden training source must fail closed")


def test_rendered_track_training_binds_mapping_calibration(tmp_path):
    paths = _inputs(tmp_path)
    calibration = tmp_path / "scene_calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": {
                    "uses_test_queries": False,
                    "uses_source_mapping_rgb": False,
                    "mapping_source": "gaussian_render",
                },
                "parameters": {
                    "positive_radius_px": 1.25,
                    "negative_radius_px": 5.0,
                    "evidence_depth_abs_tolerance_m": 0.04,
                },
            }
        )
    )
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "bound",
        strong_radius_px=1.25,
        ambiguous_radius_px=5.0,
        depth_abs_tolerance_m=0.04,
        scene_calibration_path=calibration,
        expected_scene_calibration_sha256=sha256_file(calibration),
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    assert teacher["scene_calibration"] == str(calibration.resolve())
    assert teacher["scene_calibration_sha256"] == sha256_file(calibration)
    assert teacher["config"]["threshold_lineage"] == "mapping_scene_calibration"

    bad = tmp_path / "bad"
    with pytest.raises(ValueError, match="positive_radius_px"):
        materialize(
            anchor_map_path=paths["map"],
            track_payload_path=paths["track"],
            query_cache_path=paths["cache"],
            output_dir=bad,
            strong_radius_px=2.0,
            ambiguous_radius_px=5.0,
            depth_abs_tolerance_m=0.04,
            scene_calibration_path=calibration,
            expected_scene_calibration_sha256=sha256_file(calibration),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("uses_source_mapping_rgb", True),
        ("uses_source_mapping_rgb", None),
        ("mapping_source", "mapping_rgb"),
        ("mapping_source", None),
    ],
)
def test_rendered_track_training_rejects_non_render_calibration(tmp_path, field, value):
    paths = _inputs(tmp_path)
    sources = {
        "uses_test_queries": False,
        "uses_source_mapping_rgb": False,
        "mapping_source": "gaussian_render",
    }
    if value is None:
        sources.pop(field)
    else:
        sources[field] = value
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "sources": sources,
                "parameters": {
                    "positive_radius_px": 2.0,
                    "negative_radius_px": 8.0,
                    "evidence_depth_abs_tolerance_m": 0.05,
                },
            }
        )
    )
    with pytest.raises(ValueError, match="source-image-free"):
        materialize(
            anchor_map_path=paths["map"],
            track_payload_path=paths["track"],
            query_cache_path=paths["cache"],
            output_dir=tmp_path / "out",
            strong_radius_px=2.0,
            ambiguous_radius_px=8.0,
            scene_calibration_path=calibration,
            expected_scene_calibration_sha256=sha256_file(calibration),
        )


def test_uncertified_exact_track_observation_is_ambiguous_not_positive(tmp_path):
    paths = _inputs(tmp_path)
    payload = torch.load(paths["track"], map_location="cpu", weights_only=False)
    payload["support_repair"] = {"schema": "lafgs_rendered_track_support_repair"}
    payload["tracks"]["identity_positive_certified"] = torch.tensor([True, False, True])
    torch.save(payload, paths["track"])
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "graded",
        strong_radius_px=2.0,
        ambiguous_radius_px=8.0,
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    assert teacher["records"][0]["positive_indices"].tolist() == [0]
    assert teacher["records"][1]["positive_indices"].numel() == 0
    assert teacher["records"][1]["exact_identity_indices"].tolist() == [0]
    assert teacher["records"][1]["ambiguous_indices"].tolist() == [0]
    assert teacher["diagnostics"]["weak_exact_ambiguous_count"] == 1


def test_rendered_track_training_honors_alpha_rows_and_depth_visibility(tmp_path):
    paths = _inputs(tmp_path)
    cache = torch.load(paths["cache"], map_location="cpu", weights_only=False)
    for record in cache["queries"].values():
        record["native_valid_keypoint_mask"] = torch.tensor([True])
        record["native_rendered_alpha"] = torch.ones((100, 100))
        record["native_rendered_depth"] = torch.full((100, 100), 3.0)
    first = next(iter(cache["queries"].values()))
    first["native_valid_keypoint_mask"] = torch.tensor([False])
    torch.save(cache, paths["cache"])
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "visibility",
        strong_radius_px=2.0,
        ambiguous_radius_px=8.0,
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    # Exact multi-view identity is retained even when Gaussian expected depth
    # or alpha disagrees; the disagreement is audited rather than promoted to
    # geometric truth.
    assert teacher["records"][0]["query_rows"].numel() == 1
    assert all(
        record["positive_indices"].tolist() == [0] for record in teacher["records"]
    )
    assert teacher["diagnostics"]["masked_query_row_count"] == 0
    assert teacher["diagnostics"]["depth_visibility_rejected_anchor_count"] == 3
    assert teacher["diagnostics"]["exact_depth_disagreement_audit_count"] == 3
    assert teacher["config"]["uses_rendered_depth"] is True
    assert teacher["config"]["uses_rendered_alpha"] is True


def test_projection_compatible_anchor_is_ignored_not_identity_positive(tmp_path):
    paths = _inputs(tmp_path)
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    state["anchor_ids"] = torch.tensor([0, 1])
    state["anchor_xyz"] = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    state["anchor_features"] = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    state["track_cluster_ids"] = torch.tensor([0, 1])
    torch.save(state, paths["map"])
    report = materialize(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        output_dir=tmp_path / "compatible",
        strong_radius_px=2.0,
        ambiguous_radius_px=8.0,
    )
    teacher = torch.load(
        report["outputs"]["teacher"], map_location="cpu", weights_only=False
    )
    for record in teacher["records"]:
        assert record["positive_indices"].tolist() == [0]
        assert record["exact_identity_indices"].tolist() == [0]
        assert record["support_compatible_indices"].tolist() == [1]
        assert record["ambiguous_indices"].tolist() == [1]


def test_crossfit_training_inputs_exclude_held_mapping_sequence():
    teacher = {
        "query_names": ["seq-02/a", "seq-03/b", "seq-05/c"],
        "records": [
            {
                "query_index": index,
                "query_name": name,
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([0]),
                "ambiguous_offsets": torch.tensor([0, 0]),
                "ambiguous_indices": torch.empty(0, dtype=torch.long),
            }
            for index, name in enumerate(("seq-02/a", "seq-03/b", "seq-05/c"))
        ],
        "diagnostics": {
            "query_count": 3,
            "positive_rows": 3,
            "strong_pair_count": 3,
            "ambiguous_pair_count": 0,
        },
    }
    revised, graph, payload = _training_inputs(
        held_sequence="seq-03",
        crossfit_groups=["seq-02", "seq-03", "seq-05"],
        fold_teacher=teacher,
        full_payload={},
    )
    assert revised["query_names"] == ["seq-02/a", "seq-05/c"]
    assert graph["query_names"] == revised["query_names"]
    assert payload["query_names"] == revised["query_names"]
    assert payload["query_bins"].tolist() == [0, 1]
    assert payload["uses_test_queries"] is False
    assert revised["crossfit"]["held_mapping_sequence"] == "seq-03"


def test_single_mapping_trajectory_uses_contiguous_blocked_crossfit():
    names = [f"seq2/frame{index:05d}.png" for index in range(7)]
    groups, folds = _crossfit_groups(names, 3)
    assert folds == ["blocked_00", "blocked_01", "blocked_02"]
    assert groups == [
        "blocked_00",
        "blocked_00",
        "blocked_00",
        "blocked_01",
        "blocked_01",
        "blocked_02",
        "blocked_02",
    ]


def test_crossfit_fold_map_uses_support_only_retriangulated_geometry():
    state = {
        "anchor_ids": torch.arange(3),
        "anchor_xyz": torch.full((3, 3), -1.0),
        "anchor_features": torch.eye(3),
        "track_cluster_ids": torch.tensor([4, 5, 6]),
        "anchor_position_covariance": torch.zeros(3, 3, 3),
    }
    keep = torch.tensor([True, False, True])
    features = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    geometry = {
        "triangulated_xyz": torch.tensor(
            [[10.0, 0.0, 1.0], [20.0, 0.0, 1.0], [30.0, 0.0, 1.0]]
        ),
        "triangulation_covariance_matrix": torch.stack(
            [torch.eye(3), 2 * torch.eye(3), 3 * torch.eye(3)]
        ),
    }
    output = _subset_state(state, keep, features, geometry)
    assert output["anchor_xyz"].tolist() == [
        [10.0, 0.0, 1.0],
        [30.0, 0.0, 1.0],
    ]
    assert output["anchor_position_covariance"][:, 0, 0].tolist() == [1.0, 3.0]
    assert output["track_cluster_ids"].tolist() == [4, 6]
    assert output["rendered_track_crossfit_geometry"]["support_only"] is True


def test_crossfit_rebuilds_components_from_frozen_pairs_without_held_camera():
    names = [f"seq-{index:02d}/frame.png" for index in range(4)]
    queries = {}
    for index, name in enumerate(names):
        pose = torch.eye(4)
        pose[0, 3] = -0.1 * index
        queries[name] = {
            "native_keypoints": torch.tensor([[50.0 - 5.0 * index, 50.0]]),
            "native_descriptors": torch.tensor([[1.0, 0.0]]),
            "native_scores": torch.tensor([1.0]),
            "native_K": torch.tensor(
                [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
            ),
            "pose_w2c": pose,
            "pixel_center_offset": 0.0,
        }
    pairs = [(left, right) for left in range(4) for right in range(left + 1, 4)]
    payload = {
        "query_names": names,
        "query_bins": torch.arange(4),
        "pair_sidecar": {
            "pair": {
                "left_query_index": torch.tensor([pair[0] for pair in pairs]),
                "right_query_index": torch.tensor([pair[1] for pair in pairs]),
            }
        },
        "support_repair": {
            "maximum_children_per_source_track": 3,
            "source_track_index_at_keypoint": [torch.tensor([9]) for _ in names],
            "frozen_support_pair_matches": {
                "schema": "lafgs_rendered_track_frozen_support_pair_matches",
                "offsets": torch.arange(len(pairs) + 1),
                "source_keypoint_indices": torch.zeros(len(pairs), dtype=torch.long),
                "target_keypoint_indices": torch.zeros(len(pairs), dtype=torch.long),
                "confidence": torch.ones(len(pairs)),
            },
            "track_build_contract": {
                "pair_policy": "nearest",
                "pair_neighbors": 3,
                "minimum_baseline_m": 0.0,
                "maximum_baseline_m": 5.0,
                "maximum_axis_angle_deg": 180.0,
                "minimum_similarity": 0.65,
                "minimum_margin": 0.01,
                "maximum_epipolar_error_px": 2.0,
                "epipolar_candidate_topk": 1,
                "minimum_track_views": 3,
                "maximum_observations": 32,
                "minimum_view_bins": 2,
                "huber_delta_px": 2.0,
                "triangulation_iterations": 3,
                "minimum_parallax_deg": 1.0,
                "parallax_quantile": 0.75,
                "maximum_reprojection_px": 2.0,
                "maximum_condition_number": 1e6,
            },
        },
    }
    state = {
        "anchor_ids": torch.tensor([0]),
        "parent_source_track_ids": torch.tensor([9]),
    }
    keep, features, geometry, diagnostics = _fold_component_bank(
        state=state,
        payload=payload,
        query_cache={"queries": queries},
        held_sequence="seq-00",
        crossfit_groups=["seq-00", "seq-01", "seq-02", "seq-03"],
        trim_fraction=0.0,
    )
    assert keep.tolist() == [True]
    torch.testing.assert_close(features, torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(
        geometry["triangulated_xyz"][0],
        torch.tensor([0.0, 0.0, 2.0]),
        atol=1e-4,
        rtol=0,
    )
    assert diagnostics["fold_specific_component_rebuild"] is True
    assert diagnostics["frozen_support_pair_count"] == 3


def test_fullchain_inputs_resolve_pruned_rows_to_track_ids(tmp_path):
    paths = _inputs(tmp_path)
    state = torch.load(paths["map"], map_location="cpu", weights_only=False)
    state.update(
        {
            "schema": "lafgs_materialized_anchor_map",
            "anchor_type": torch.ones(1, dtype=torch.long),
        }
    )
    torch.save(state, paths["map"])
    payload = torch.load(paths["track"], map_location="cpu", weights_only=False)
    payload.update(
        {
            "schema": "lafgs_track_first_payload",
            "rendered_rgb_only": True,
            "track_geometry": {"triangulated": torch.tensor([True])},
        }
    )
    torch.save(payload, paths["track"])
    capacity = {
        "schema": "lafgs_rendered_track_train_only_capacity_selection",
        "version": 1,
        "uses_test_queries": False,
        "inputs": {"anchor_map": str(paths["map"].resolve())},
        "input_sha256": {"anchor_map": sha256_file(paths["map"])},
        "pruned_anchor_rows": [0],
    }
    capacity_path = tmp_path / "capacity.json"
    capacity_path.write_text(json.dumps(capacity))
    result = materialize_fullchain_inputs(
        anchor_map_path=paths["map"],
        track_payload_path=paths["track"],
        query_cache_path=paths["cache"],
        capacity_report_path=capacity_path,
        output_dir=tmp_path / "fullchain",
    )
    empty = torch.load(
        result["outputs"]["empty_canonical_map"],
        map_location="cpu",
        weights_only=False,
    )
    exclusions = torch.load(
        result["outputs"]["track_exclusions"],
        map_location="cpu",
        weights_only=False,
    )
    assert empty["base_anchor_count"] == 0
    assert empty["anchor_xyz"].shape == (0, 3)
    assert exclusions["excluded_track_ids"].tolist() == [0]
