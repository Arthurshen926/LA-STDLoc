from copy import deepcopy

import pytest
import torch

from scripts.audit_rendered_track_artifact_cache_equivalence import (
    audit_artifact_cache_equivalence,
)


def _record() -> dict:
    return {
        "native_keypoints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "native_descriptors": torch.eye(2),
        "native_scores": torch.tensor([0.9, 0.8]),
        "native_K": torch.eye(3),
        "pose_w2c": torch.eye(4),
        "native_input_hw": torch.tensor([8, 12]),
        "native_valid_keypoint_mask": torch.tensor([True, True]),
        "native_alpha_at_keypoints": torch.tensor([0.8, 0.7]),
        "native_depth_at_keypoints": torch.tensor([1.0, 2.0]),
        "native_rendered_alpha": torch.ones(2, 2),
        "native_rendered_depth": torch.ones(2, 2),
        "native_appearance_dispersion": torch.tensor([0.1, 0.2]),
        "native_appearance_reliability": torch.tensor([0.9, 0.8]),
        "source": "gaussian_rendered_rgb_appearance_ensemble",
    }


def _artifacts() -> tuple[dict, dict, dict]:
    source_record = _record()
    refreshed_record = deepcopy(source_record)
    refreshed_record.update(
        {
            "source": "gaussian_rendered_rgb_artifact_stability_r1",
            "native_appearance_reliability": torch.tensor([0.7, 0.6]),
            "native_artifact_reliability": torch.tensor([0.8, 0.75]),
            "native_artifact_exposure": torch.tensor([0.1, 0.2]),
            "native_raw_clean_descriptor_cosine": torch.tensor([0.95, 0.9]),
            "native_raw_clean_detector_score_stability": torch.tensor([0.9, 0.8]),
            "native_raw_clean_position_stability": torch.tensor([1.0, 0.7]),
            "native_raw_clean_position_displacement_px": torch.tensor([0.0, 1.0]),
        }
    )
    shared = {
        "version": 1,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "uses_rendered_depth": True,
        "uses_gaussian_geometry_for_triangulation": False,
        "mapping_query_count": 1,
        "full_mapping_query_count": 1,
        "source_mapping_indices": torch.tensor([0]),
        "configuration": {"sentinel": 3},
        "timing_seconds": {"total": 1.0},
        "uses_rendered_alpha": True,
        "appearance_ensemble": {"method": "fixed"},
    }
    source = {
        **deepcopy(shared),
        "schema": "lafgs_rendered_rgb_appearance_ensemble_cache",
        "queries": {"q0": source_record},
    }
    refreshed = {
        **deepcopy(shared),
        "schema": "lafgs_rendered_rgb_artifact_stability_cache",
        "queries": {"q0": refreshed_record},
        "artifact_stability": {
            "topology_frozen": True,
            "descriptors_remain_v14_appearance_descriptors": True,
        },
    }
    track = {"query_names": ["q0"], "rendered_rgb_only": True}
    return source, refreshed, track


def test_artifact_cache_equivalence_accepts_only_reliability_annotations():
    source, refreshed, track = _artifacts()
    audit = audit_artifact_cache_equivalence(source, refreshed, track)
    assert audit["localization_query_rows_bitwise_exact"] is True
    assert audit["calibration_numeric_reuse_authorized"] is True
    assert audit["query_count"] == 1


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("native_descriptors", torch.tensor([[0.0, 1.0], [1.0, 0.0]])),
        ("native_keypoints", torch.tensor([[2.0, 2.0], [3.0, 4.0]])),
        ("native_scores", torch.tensor([0.8, 0.8])),
    ],
)
def test_artifact_cache_equivalence_rejects_localization_row_tamper(field, replacement):
    source, refreshed, track = _artifacts()
    refreshed["queries"]["q0"][field] = replacement
    with pytest.raises(ValueError, match="changed frozen field"):
        audit_artifact_cache_equivalence(source, refreshed, track)


def test_artifact_cache_equivalence_rejects_shape_coercion_and_extra_fields():
    source, refreshed, track = _artifacts()
    refreshed["queries"]["q0"]["native_artifact_reliability"] = torch.ones(2, 1)
    with pytest.raises(ValueError, match="exact.*vector"):
        audit_artifact_cache_equivalence(source, refreshed, track)

    source, refreshed, track = _artifacts()
    refreshed["queries"]["q0"]["unregistered"] = torch.ones(2)
    with pytest.raises(ValueError, match="unauthorized field"):
        audit_artifact_cache_equivalence(source, refreshed, track)


def test_artifact_cache_equivalence_rejects_track_registry_or_scope_change():
    source, refreshed, track = _artifacts()
    track["query_names"] = ["different"]
    with pytest.raises(ValueError, match="Track payload query registry"):
        audit_artifact_cache_equivalence(source, refreshed, track)

    source, refreshed, track = _artifacts()
    refreshed["uses_test_queries"] = True
    with pytest.raises(ValueError, match="mapping-only"):
        audit_artifact_cache_equivalence(source, refreshed, track)
