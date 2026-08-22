import inspect

import pytest

from common.v6_contracts import ASSOCIATION_GRAPH_SCHEMA
from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_reconstruction import reconstruct_projective_anchors


def test_reconstruction_rejects_legacy_association() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError):
        reconstruct_projective_anchors(provider, {"schema": "legacy"})


def test_association_scope_must_be_mapping_only() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError):
        reconstruct_projective_anchors(
            provider,
            {
                "schema": ASSOCIATION_GRAPH_SCHEMA,
                "uses_source_mapping_rgb": False,
                "uses_test_queries": True,
            },
        )


def test_base_reconstruction_does_not_require_cross_global_pose_bins() -> None:
    parameter = inspect.signature(reconstruct_projective_anchors).parameters[
        "minimum_view_bins"
    ]
    assert parameter.default == 1
