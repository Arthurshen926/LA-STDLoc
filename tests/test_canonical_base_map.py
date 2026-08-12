import json
import sys
from pathlib import Path

import torch

from evidence.tracks import (
    build_add_only_materialized_anchor_map,
    build_canonical_base_anchor_map,
)
from topology import candidates


def _base_state() -> dict:
    return {
        "landmark_features": torch.tensor(
            [[3.0, 4.0], [0.0, 2.0]], dtype=torch.float64
        ),
        "landmark_xyz": torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        ),
        "landmark_indices": torch.tensor([11, 29], dtype=torch.int32),
    }


def _track_payload() -> dict:
    return {
        "schema": "lafgs_track_first_payload",
        "query_names": ["unused"],
        "query_bins": torch.tensor([0]),
        "tracks": {
            "track_index": torch.tensor([0, 1]),
            "query_index": torch.tensor([0, 0]),
            "keypoint_index": torch.tensor([0, 1]),
            "confidence": torch.ones(2),
        },
        "track_geometry": {
            "triangulated_xyz": torch.tensor(
                [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]
            ),
            "triangulation_high_confidence": torch.tensor([True, True]),
            "track_confidence_level": torch.tensor([2, 2], dtype=torch.int8),
            "triangulation_distinct_view_bin_count": torch.tensor([3, 1]),
            "triangulation_observation_count": torch.tensor([4, 2]),
            "triangulation_reprojection_median_px": torch.tensor([0.2, 0.4]),
            "triangulation_covariance_trace": torch.tensor([0.01, 0.02]),
        },
        "assignment": {
            "track_landmark_index": torch.tensor([0, 1]),
        },
    }


def _assert_exact(left, right) -> None:
    assert type(left) is type(right)
    if isinstance(left, dict):
        assert list(left) == list(right)
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype
        assert left.shape == right.shape
        assert left.stride() == right.stride()
        assert torch.equal(left, right)
    else:
        assert left == right


def test_pure_base_map_is_field_exact_with_legacy_zero_budget() -> None:
    coverage = {
        "coverage_gain": torch.tensor([3, 0]),
        "valid_observations": torch.tensor([4, 2]),
    }
    legacy, legacy_diagnostics = build_add_only_materialized_anchor_map(
        base_state=_base_state(),
        payload=_track_payload(),
        query_cache={},
        budget=0,
        minimum_coverage_gain=2,
        minimum_distinct_view_bins=3,
        minimum_separation_m=0.02,
        descriptor_trim_fraction=0.1,
        radius_px=3.0,
        coverage=coverage,
    )
    fast, fast_diagnostics = build_canonical_base_anchor_map(
        base_state=_base_state(),
        minimum_coverage_gain=2,
        minimum_distinct_view_bins=3,
        minimum_separation_m=0.02,
        descriptor_trim_fraction=0.1,
        radius_px=3.0,
    )

    _assert_exact(fast, legacy)
    assert legacy_diagnostics["eligible_track_count"] == 1
    assert fast_diagnostics["eligible_track_count"] is None
    assert fast_diagnostics["eligibility_evaluated"] is False
    for key in (
        "base_anchor_count",
        "selected_micro_anchor_count",
        "selected_source_primitive_count",
        "selected_multi_anchor_source_count",
        "coverage_gain_sum",
        "coverage_gain_mean",
    ):
        assert fast_diagnostics[key] == legacy_diagnostics[key]


def test_zero_budget_cli_does_not_load_track_payload_or_query_cache(
    tmp_path: Path, monkeypatch
) -> None:
    base_path = tmp_path / "base.pt"
    track_path = tmp_path / "tracks.pt"
    query_path = tmp_path / "queries.pt"
    output = tmp_path / "canonical.pt"
    torch.save(_base_state(), base_path)
    track_path.write_bytes(b"must not be loaded")
    query_path.write_bytes(b"must not be loaded")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "topology.candidates",
            "--base_state",
            str(base_path),
            "--track_payload",
            str(track_path),
            "--query_cache",
            str(query_path),
            "--output",
            str(output),
            "--budget",
            "0",
        ],
    )

    candidates.main()

    state = torch.load(output, map_location="cpu", weights_only=False)
    expected, _ = build_canonical_base_anchor_map(base_state=_base_state())
    state.pop("provenance")
    _assert_exact(state, expected)
    report = json.loads(output.with_suffix(".json").read_text())
    assert report["eligible_track_count"] is None
    assert report["eligibility_evaluated"] is False

