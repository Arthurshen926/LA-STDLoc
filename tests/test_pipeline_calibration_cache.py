from __future__ import annotations

import json
import hashlib

import pytest
import torch

from common.hashing import sha256_file
from map_learning import pipeline
from scripts.audit_pair_payload_lineage import audit_pair_payload
from scripts.materialize_pair_factor_calibration import (
    materialize_pair_factor_calibration,
)


def test_exact_scene_calibration_sidecar_avoids_query_cache_reload(
    tmp_path, monkeypatch
) -> None:
    query_cache = tmp_path / "query_cache.pt"
    track_payload = tmp_path / "tracks.pt"
    query_cache.touch()
    track_payload.touch()
    policy = {"matching_rows_fraction": 0.04735}
    cached = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": {"query_count": 10},
        "parameters": {"metric_steps": 8},
        "policy": policy,
        "sources": {
            "query_cache": str(query_cache.resolve()),
            "track_payload": str(track_payload.resolve()),
            "uses_test_queries": False,
        },
    }
    path = tmp_path / "scene_calibration.json"
    path.write_text(json.dumps(cached))

    def unexpected_calibration(*args, **kwargs):
        raise AssertionError("the exact calibration sidecar must be reused")

    monkeypatch.setattr(pipeline, "calibrate_scene", unexpected_calibration)
    assert (
        pipeline._load_or_compute_scene_calibration(
            query_cache=query_cache,
            track_payload=track_payload,
            policy=policy,
            cached_path=path,
        )
        == cached
    )


def test_stale_scene_calibration_sidecar_is_recomputed(tmp_path, monkeypatch) -> None:
    query_cache = tmp_path / "query_cache.pt"
    track_payload = tmp_path / "tracks.pt"
    query_cache.touch()
    track_payload.touch()
    path = tmp_path / "scene_calibration.json"
    path.write_text(
        json.dumps(
            {
                "schema": "lafgs_mapping_only_scene_calibration",
                "version": 2,
                "policy": {"value": "stale"},
                "sources": {
                    "query_cache": str(query_cache.resolve()),
                    "track_payload": str(track_payload.resolve()),
                    "uses_test_queries": False,
                },
            }
        )
    )
    expected = {"parameters": {"metric_steps": 12}}
    monkeypatch.setattr(pipeline, "calibrate_scene", lambda *args, **kwargs: expected)
    assert (
        pipeline._load_or_compute_scene_calibration(
            query_cache=query_cache,
            track_payload=track_payload,
            policy={"value": "current"},
            cached_path=path,
        )
        == expected
    )


def test_pair_factor_calibration_rebinds_paths_without_numeric_drift(
    tmp_path, monkeypatch
) -> None:
    query_cache = tmp_path / "query_cache.pt"
    query_cache.write_bytes(b"frozen-cache")
    factor_path = tmp_path / "factor.pt"
    factor = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 1024,
        "mapping_nms_radius": 4,
        "density_factor_mutated": False,
        "descriptor_factor_mutated": False,
        "selector_factor_mutated": False,
        "pair_policy": "parallax_diverse",
        "pair_policy_parameters": {"candidate_pool_per_camera": 48},
        "query_names": ["q0", "q1"],
        "query_names_sha256": hashlib.sha256(b"q0\nq1\n").hexdigest(),
        "query_bins": torch.tensor([0, 1]),
        "tracks": {
            "track_level": torch.tensor([2]),
            "track_index": torch.tensor([0]),
        },
        "track_geometry": {"triangulated": torch.tensor([True])},
        "pair_sidecar": {
            "schema": "lafgs_mapping_track_pair_sidecar",
            "version": 1,
            "triangulation_attached": True,
            "policy": {
                "name": "parallax_diverse",
                "exact_pair_budget": 1,
                "uses_test_queries": False,
                "uses_descriptors_for_selection": False,
                "overlap_constraint_applied": True,
            },
            "pair": {
                "left_query_index": torch.tensor([0]),
                "right_query_index": torch.tensor([1]),
                "baseline_m": torch.tensor([0.1]),
                "axis_angle_deg": torch.tensor([1.0]),
                "mapping_point_joint_visibility_count": torch.tensor([8]),
                "mapping_point_overlap_jaccard": torch.tensor([0.8]),
                "mapping_point_parallax_median_deg": torch.tensor([2.0]),
                "raw_match_count": torch.tensor([3]),
                "descriptor_accepted_before_epipolar_count": torch.tensor([2]),
                "epipolar_accepted_top1_count": torch.tensor([1]),
                "cycle_supported_edge_count": torch.tensor([1]),
                "conflict_rejected_edge_count": torch.tensor([0]),
                "final_component_edge_count": torch.tensor([1]),
                "triangulated_track_count": torch.tensor([1]),
                "actual_triangulation_parallax_median_deg": torch.tensor([2.0]),
                "final_track_offsets": torch.tensor([0, 1]),
                "final_track_indices": torch.tensor([0]),
            },
        },
    }
    torch.save(factor, factor_path)
    base_state_path = tmp_path / "base_state.pt"
    base_state = {"landmark_indices": torch.tensor([9])}
    torch.save(base_state, base_state_path)
    assignment_parameters = {
        "topk": 4,
        "minimum_consensus_rate": 0.35,
        "minimum_views": 2,
        "group_maximum_landmarks": 4,
        "group_minimum_relative_mass": 0.25,
        "group_minimum_consensus_rate": 0.1,
        "depth_absolute_tolerance_m": 0.05,
        "depth_relative_tolerance": 0.02,
    }
    manifest_path = tmp_path / "bootstrap_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "arguments": {
                    "native_keypoint_count": 1024,
                    "geometry_teacher_provenance_topk": 4,
                    "geometry_teacher_provenance_min_consensus_rate": 0.35,
                    "geometry_teacher_provenance_min_views": 2,
                    "geometry_teacher_provenance_group_max_landmarks": 4,
                    "geometry_teacher_provenance_group_min_relative_mass": 0.25,
                    "geometry_teacher_provenance_group_min_consensus_rate": 0.1,
                    "geometry_teacher_provenance_depth_abs_tolerance_m": 0.05,
                    "geometry_teacher_provenance_depth_rel_tolerance": 0.02,
                },
                "inputs": {
                    "query_cache_path": {
                        "path": str(query_cache.resolve()),
                        "sha256": sha256_file(query_cache),
                    }
                },
            }
        )
    )
    factor["input_lineage"] = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
        },
        "query_cache": {
            "path": str(query_cache.resolve()),
            "sha256": sha256_file(query_cache),
        },
        "frozen_track_payload": {
            "path": str((tmp_path / "frozen.pt").resolve()),
            "sha256": "0" * 64,
        },
        "equivalent_query_cache_rebind": None,
    }
    torch.save(factor, factor_path)
    track_payload = tmp_path / "variant_tracks.pt"
    payload = {
        "schema": "lafgs_track_first_payload",
        "version": 1,
        "query_names": factor["query_names"],
        "query_bins": factor["query_bins"],
        "landmark_indices": base_state["landmark_indices"],
        "tracks": factor["tracks"],
        "track_geometry": factor["track_geometry"],
        "pair_sidecar": factor["pair_sidecar"],
        "assignment": {
            "track_landmark_index": torch.tensor([0]),
            "track_assignment_cost": torch.tensor([0.2]),
            "landmark_best_track_index": torch.tensor([0]),
            "track_landmark_offsets": torch.tensor([0, 1]),
            "track_landmark_indices": torch.tensor([0]),
            "track_landmark_costs": torch.tensor([0.2]),
        },
        "provenance": {
            "schema": "lafgs_replayed_track_provenance_assignment",
            "version": 1,
            "uses_test_queries": False,
            "source_factor": str(factor_path.resolve()),
            "source_factor_sha256": sha256_file(factor_path),
            "base_state": str(base_state_path.resolve()),
            "base_state_sha256": sha256_file(base_state_path),
            "query_cache": str(query_cache.resolve()),
            "query_cache_sha256": sha256_file(query_cache),
            "expected_query_cache_sha256": sha256_file(query_cache),
            "frozen_bootstrap_manifest": str(manifest_path.resolve()),
            "frozen_bootstrap_manifest_sha256": sha256_file(manifest_path),
            "assignment_algorithm": ("frozen_2dgs_splat_provenance_exact_replay"),
            "assignment_parameters": assignment_parameters,
            "factor_input_lineage": factor["input_lineage"],
        },
    }
    torch.save(payload, track_payload)
    audit = audit_pair_payload(
        payload,
        factor,
        base_state,
        payload_path=track_payload,
        factor_path=factor_path,
        base_state_path=base_state_path,
        query_cache_path=query_cache,
        expected_query_cache_sha256=sha256_file(query_cache),
        frozen_bootstrap_manifest_path=manifest_path,
        expected_frozen_bootstrap_manifest_sha256=sha256_file(manifest_path),
        expected_mapping_keypoints=1024,
        expected_nms_radius=4,
        expected_pair_budget=1,
    )
    assert audit["valid"]
    audit_path = tmp_path / "payload_lineage_audit.json"
    audit_path.write_text(json.dumps(audit))
    policy = {"matching_rows_fraction": 0.04735}
    parent = {
        "schema": "lafgs_mapping_only_scene_calibration",
        "version": 2,
        "statistics": {"query_count": 10},
        "parameters": {"metric_steps": 8, "positive_radius_px": 0.5},
        "policy": policy,
        "sources": {
            "query_cache": str(query_cache.resolve()),
            "track_payload": str((tmp_path / "old_tracks.pt").resolve()),
            "uses_test_queries": False,
        },
    }
    parent_path = tmp_path / "parent_calibration.json"
    parent_path.write_text(json.dumps(parent))
    rebound = materialize_pair_factor_calibration(
        parent_path=parent_path,
        query_cache_path=query_cache,
        track_payload_path=track_payload,
        payload_lineage_audit_path=audit_path,
        expected_parent_calibration_sha256=sha256_file(parent_path),
        expected_query_cache_sha256=sha256_file(query_cache),
        expected_payload_lineage_audit_sha256=sha256_file(audit_path),
        expected_mapping_keypoints=1024,
        expected_nms_radius=4,
        expected_pair_budget=1,
    )
    assert rebound["parameters"] == parent["parameters"]
    assert rebound["policy"] == parent["policy"]
    assert rebound["lineage"]["mode"] == "frozen_numeric_pair_factor"
    assert rebound["lineage"]["expected_parent_calibration_sha256"] == (
        sha256_file(parent_path)
    )
    assert rebound["lineage"]["expected_payload_lineage_audit_sha256"] == (
        sha256_file(audit_path)
    )
    assert rebound["sources"]["track_payload"] == str(track_payload.resolve())
    rebound_path = tmp_path / "pair_factor_calibration.json"
    rebound_path.write_text(json.dumps(rebound))

    def unexpected_calibration(*args, **kwargs):
        raise AssertionError("pair-factor calibration must be reused exactly")

    monkeypatch.setattr(pipeline, "calibrate_scene", unexpected_calibration)
    assert (
        pipeline._load_or_compute_scene_calibration(
            query_cache=query_cache,
            track_payload=track_payload,
            policy=policy,
            cached_path=rebound_path,
        )
        == rebound
    )

    with pytest.raises(ValueError, match="Parent calibration SHA-256"):
        materialize_pair_factor_calibration(
            parent_path=parent_path,
            query_cache_path=query_cache,
            track_payload_path=track_payload,
            payload_lineage_audit_path=audit_path,
            expected_parent_calibration_sha256="0" * 64,
            expected_query_cache_sha256=sha256_file(query_cache),
            expected_payload_lineage_audit_sha256=sha256_file(audit_path),
            expected_mapping_keypoints=1024,
            expected_nms_radius=4,
            expected_pair_budget=1,
        )

    rejected_audit_path = tmp_path / "rejected_payload_lineage_audit.json"
    rejected_audit_path.write_text(json.dumps({**audit, "valid": False}))
    with pytest.raises(ValueError, match="valid parallax-diverse contract"):
        materialize_pair_factor_calibration(
            parent_path=parent_path,
            query_cache_path=query_cache,
            track_payload_path=track_payload,
            payload_lineage_audit_path=rejected_audit_path,
            expected_parent_calibration_sha256=sha256_file(parent_path),
            expected_query_cache_sha256=sha256_file(query_cache),
            expected_payload_lineage_audit_sha256=sha256_file(rejected_audit_path),
            expected_mapping_keypoints=1024,
            expected_nms_radius=4,
            expected_pair_budget=1,
        )
