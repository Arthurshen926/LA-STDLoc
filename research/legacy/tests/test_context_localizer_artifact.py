import pytest
import torch

from localization.localizer import load_context_descriptor_artifact
from map_learning.context_metric import MapConsistentContextAdapter


def _write_artifact(path, anchor_ids):
    compatible = MapConsistentContextAdapter(
        hidden_dim=8,
        context_kernels=(3,),
        maximum_residual_norm=0.1,
    )
    config = compatible.export_config()
    torch.save(
        {
            "schema": "lafgs_map_consistent_context_descriptor",
            "version": 1,
            "uses_test_queries": False,
            "anchor_indices": torch.tensor([0, 2]),
            "anchor_ids": torch.as_tensor(anchor_ids),
            "anchor_features": torch.randn(2, 256),
            "adapter_config": config,
            "adapter_state_dict": compatible.state_dict(),
        },
        path,
    )


def test_context_artifact_loads_with_strict_base_map_alignment(tmp_path):
    path = tmp_path / "context.pt"
    _write_artifact(path, [10, 12])

    indices, ids, features, adapter = load_context_descriptor_artifact(
        path,
        base_anchor_ids=torch.tensor([10, 11, 12]),
        device=torch.device("cpu"),
    )

    assert indices.tolist() == [0, 2]
    assert ids.tolist() == [10, 12]
    assert features.shape == (2, 256)
    torch.testing.assert_close(features.norm(dim=1), torch.ones(2))
    assert adapter.context_mode == "multi_scale_global"


def test_context_artifact_rejects_misaligned_anchor_ids(tmp_path):
    path = tmp_path / "context.pt"
    _write_artifact(path, [10, 99])

    with pytest.raises(ValueError, match="do not align"):
        load_context_descriptor_artifact(
            path,
            base_anchor_ids=torch.tensor([10, 11, 12]),
            device=torch.device("cpu"),
        )
