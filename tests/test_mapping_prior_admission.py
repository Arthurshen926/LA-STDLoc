import torch

from common.v6_contracts import ANCHOR_CANDIDATE_SCHEMA
from map_learning.mapping_prior_admission import audit_mapping_prior_admission


def _candidates(identity, geometry):
    return {
        "schema": ANCHOR_CANDIDATE_SCHEMA,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "identity_reliability": torch.tensor(identity),
        "geometry_reliability": torch.tensor(geometry),
    }


def test_prior_admission_requires_count_and_fraction():
    report = audit_mapping_prior_admission(
        _candidates([0.9, 0.9, 0.1, 0.1], [0.9, 0.9, 0.9, 0.9]),
        high_identity_reliability=0.7,
        high_geometry_reliability=0.7,
        minimum_high_reliability_count=2,
        minimum_high_reliability_fraction=0.5,
    )
    assert report["status"] == "PASS"
    assert report["admitted"] is True
    assert report["localization_outcomes_consumed"] is False


def test_prior_admission_rejects_uniformly_weak_map():
    report = audit_mapping_prior_admission(
        _candidates([0.69] * 1000, [0.9] * 1000),
        high_identity_reliability=0.7,
        high_geometry_reliability=0.7,
        minimum_high_reliability_count=128,
        minimum_high_reliability_fraction=0.001,
    )
    assert report["status"] == "REJECT_UNSAFE_PRIOR"
    assert report["admitted"] is False
    assert report["map_mutated"] is False
