import json

import torch
import pytest

from scripts.audit_pair_payload_lineage import (
    _mapping_keypoint_contract,
    _pair_sidecar_contract,
)
from scripts.replay_track_provenance_assignment import _validate_factor_contract
from scripts.run_track_pair_factor import (
    _build_report,
    _factor_payload,
    _validate_expected_factor_contract,
)


def _track_payload():
    geometry = {
        "triangulated_xyz": torch.tensor([[0.0, 0.0, 2.0]]),
        "triangulated": torch.tensor([True]),
        "triangulation_high_confidence": torch.tensor([True]),
        "triangulation_observation_count": torch.tensor([3]),
        "triangulation_distinct_view_count": torch.tensor([3]),
        "triangulation_distinct_view_bin_count": torch.tensor([2]),
        "triangulation_reprojection_median_px": torch.tensor([0.1]),
        "triangulation_reprojection_p90_px": torch.tensor([0.2]),
        "triangulation_parallax_deg": torch.tensor([2.0]),
        "triangulation_condition_number": torch.tensor([10.0]),
        "triangulation_covariance_trace": torch.tensor([1e-4]),
        "track_confidence_level": torch.tensor([2]),
    }
    tracks = {
        "track_index": torch.tensor([0, 0, 0]),
        "query_index": torch.tensor([0, 1, 2]),
        "keypoint_index": torch.tensor([0, 0, 0]),
        "confidence": torch.ones(3),
        "track_level": torch.tensor([2]),
    }
    return {"tracks": tracks, "track_geometry": geometry}


def test_completed_track_factor_builds_and_writes_json_report(tmp_path):
    frozen = _track_payload()
    result = {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "mapping_keypoint_factor": 1024,
        "pair_policy": "nearest",
        "query_names": ["q0", "q1", "q2"],
        **_track_payload(),
    }
    sidecar = {
        "pair": {
            "left_query_index": torch.tensor([0]),
            "right_query_index": torch.tensor([1]),
            "baseline_m": torch.tensor([0.1]),
            "axis_angle_deg": torch.tensor([1.0]),
            "mapping_point_overlap_jaccard": torch.tensor([0.8]),
            "mapping_point_parallax_median_deg": torch.tensor([2.0]),
            "actual_triangulation_parallax_median_deg": torch.tensor([2.1]),
            "raw_match_count": torch.tensor([3]),
        }
    }
    report = _build_report(
        result=result,
        frozen=frozen,
        sidecar=sidecar,
        keypoint_counts=torch.tensor([1024, 1024, 1024]),
        scene_point_count=16,
        pair_budget=1,
        manifest_path=tmp_path / "manifest.json",
        query_cache_path=tmp_path / "query_cache.pt",
        frozen_track_payload_path=tmp_path / "frozen.pt",
    )
    output = tmp_path / "report.json"
    output.write_text(json.dumps(report) + "\n")
    loaded = json.loads(output.read_text())
    assert loaded["nearest_reproduces_all_frozen_counts"] is True
    assert loaded["track"]["broad_eligible_track_count"] == 1
    assert loaded["pair"]["raw_match_count"]["total"] == 3


def test_track_factor_never_materializes_provenance_assignment():
    track = _track_payload()
    factor = _factor_payload(
        mapping_keypoints=1024,
        pair_policy="parallax_diverse",
        query_names=["q0", "q1", "q2"],
        query_bins=torch.tensor([0, 1, 2]),
        tracks=track["tracks"],
        track_geometry=track["track_geometry"],
        pair_sidecar={"schema": "lafgs_mapping_track_pair_sidecar"},
        diagnostics={"track_count": 1},
    )
    assert "assignment" not in factor
    assert factor["descriptor_factor_mutated"] is False
    assert factor["density_factor_mutated"] is False
    assert factor["selector_factor_mutated"] is False


def _greatcourt_pair_factor() -> dict:
    pair_budget = 5254
    pair_indices = torch.triu_indices(104, 104, offset=1)[:, :pair_budget]
    pair = {
        "left_query_index": pair_indices[0],
        "right_query_index": pair_indices[1],
        "baseline_m": torch.ones(pair_budget),
        "axis_angle_deg": torch.ones(pair_budget),
        "mapping_point_joint_visibility_count": torch.full(
            (pair_budget,), 8, dtype=torch.long
        ),
        "mapping_point_overlap_jaccard": torch.ones(pair_budget),
        "mapping_point_parallax_median_deg": torch.ones(pair_budget),
        "raw_match_count": torch.zeros(pair_budget, dtype=torch.long),
        "descriptor_accepted_before_epipolar_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "epipolar_accepted_top1_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "cycle_supported_edge_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "conflict_rejected_edge_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "final_component_edge_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "triangulated_track_count": torch.zeros(
            pair_budget, dtype=torch.long
        ),
        "actual_triangulation_parallax_median_deg": torch.ones(pair_budget),
        "final_track_offsets": torch.zeros(pair_budget + 1, dtype=torch.long),
        "final_track_indices": torch.zeros(0, dtype=torch.long),
    }
    return {
        "schema": "lafgs_pair_policy_track_factor",
        "version": 1,
        "uses_test_queries": False,
        "mapping_keypoint_factor": 2048,
        "descriptor_factor_mutated": False,
        "density_factor_mutated": False,
        "selector_factor_mutated": False,
        "pair_policy": "parallax_diverse",
        "query_names": [f"q{index}" for index in range(104)],
        "pair_sidecar": {
            "schema": "lafgs_mapping_track_pair_sidecar",
            "version": 1,
            "triangulation_attached": True,
            "policy": {
                "name": "parallax_diverse",
                "exact_pair_budget": pair_budget,
                "uses_test_queries": False,
                "uses_descriptors_for_selection": False,
                "overlap_constraint_applied": True,
            },
            "pair": pair,
        },
    }


def test_k2048_pair_budget_5254_is_explicit_and_fail_closed():
    expected = _validate_expected_factor_contract(
        expected_mapping_keypoints=2048,
        expected_pair_budget=5254,
        manifest={"native_keypoint_count": 2048},
        query_cache_payload={
            "signature_payload": {"native_sparse_keypoint_count": 2048}
        },
        frozen_track_payload={
            "diagnostics": {"track_camera_pair_candidate_count": 5254}
        },
    )
    assert expected == (2048, 5254)
    with pytest.raises(ValueError, match="nearest-pair budget"):
        _validate_expected_factor_contract(
            expected_mapping_keypoints=2048,
            expected_pair_budget=5255,
            manifest={"native_keypoint_count": 2048},
            query_cache_payload={
                "signature_payload": {"native_sparse_keypoint_count": 2048}
            },
            frozen_track_payload={
                "diagnostics": {"track_camera_pair_candidate_count": 5254}
            },
        )

    factor = _greatcourt_pair_factor()
    _validate_factor_contract(
        factor,
        expected_mapping_keypoints=2048,
        expected_pair_budget=5254,
    )
    assert _mapping_keypoint_contract(
        factor, expected_mapping_keypoints=2048
    )["mapping_keypoints_expected"]
    pair_checks = _pair_sidecar_contract(
        factor, expected_pair_budget=5254
    )
    assert pair_checks
    assert all(pair_checks.values())
    with pytest.raises(ValueError, match="mapping K"):
        _validate_factor_contract(
            factor,
            expected_mapping_keypoints=1024,
            expected_pair_budget=5254,
        )
