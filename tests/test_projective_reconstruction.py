import inspect

import pytest
import torch

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


def test_stable_group_materialization_preserves_legacy_row_order() -> None:
    group = torch.tensor([2, 0, 1, 2, 0, 2, 1])
    order = torch.argsort(group, stable=True)
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), torch.bincount(group).cumsum(0))
    )
    for index in range(3):
        legacy = torch.nonzero(group == index, as_tuple=False).reshape(-1)
        packed = order[offsets[index] : offsets[index + 1]]
        assert torch.equal(packed, legacy)
