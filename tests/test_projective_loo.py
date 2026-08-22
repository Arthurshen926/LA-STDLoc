import pytest
import torch

from evidence.observation_provider import GaussianRenderObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap


def test_loo_rejects_non_projective_map() -> None:
    provider = GaussianRenderObservationProvider(
        {"uses_source_mapping_rgb": False, "queries": {"q": {}}},
        validate_all=False,
    )
    with pytest.raises(ValueError, match="registries"):
        LeaveOneQueryOutProjectiveMap(
            {"v6_mapping_query_names": [], "anchor_ids": torch.tensor([0])},
            provider,
        )
