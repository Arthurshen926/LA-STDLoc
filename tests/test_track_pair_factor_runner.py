import json

import torch

from scripts.run_track_pair_factor import _build_report, _factor_payload


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
