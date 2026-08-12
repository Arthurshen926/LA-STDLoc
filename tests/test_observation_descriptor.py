import torch

from topology.observation_descriptor import (
    SCHEMA,
    materialize_observation_descriptor_audit,
    robust_observation_fusion,
)


def _registry():
    return {
        "schema": "lafgs_evidence_grounded_anchor_registry",
        "version": 1,
        "anchor_ids": torch.arange(3),
        "anchor_features": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ),
        "anchor_type": torch.tensor([1, 0, 0]),
        "observation_offsets": torch.tensor([0, 4, 5, 5]),
        "observation_query_indices": torch.tensor([0, 1, 2, 3, 0]),
        "observation_keypoint_indices": torch.tensor([0, 0, 0, 0, 1]),
        "query_names": [
            "seq-00/a.png",
            "seq-00/b.png",
            "seq-01/c.png",
            "seq-01/d.png",
        ],
        "query_group_ids": torch.tensor([0, 0, 1, 2]),
    }


def _cache():
    values = (
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.99, 0.01]],
        [[0.98, 0.02]],
        [[-1.0, 0.0]],
    )
    return {
        "signature": "mapping-only-signature",
        "queries": {
            name: {
                "native_descriptors": torch.tensor(value),
                "native_scores": torch.ones(len(value)),
            }
            for name, value in zip(_registry()["query_names"], values)
        },
    }


def test_trimmed_medoid_rejects_descriptor_outlier():
    fused, diagnostics = robust_observation_fusion(
        torch.tensor([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02], [-1.0, 0.0]]),
        torch.arange(4),
        torch.tensor([0, 0, 1, 2]),
        torch.tensor([0, 0, 1, 1]),
        torch.ones(4),
        trim_fraction=0.34,
    )
    assert fused[0] > 0.99
    assert diagnostics["stratum_count"] == 3
    assert diagnostics["retained_stratum_count"] == 2


def test_audit_materializes_parallel_bank_and_support_flags():
    registry = _registry()
    original = registry["anchor_features"].clone()
    result = materialize_observation_descriptor_audit(
        registry, _cache(), trim_fraction=0.34
    )
    assert result["schema"] == SCHEMA
    assert result["uses_test_queries"] is False
    assert result["audit_only"] is True
    assert result["deployment_descriptor_mutated"] is False
    torch.testing.assert_close(registry["anchor_features"], original)
    assert result["descriptor_valid_mask"].tolist() == [True, True, False]
    assert result["valid_observation_count"].tolist() == [4, 1, 0]
    assert result["distinct_view_group_count"].tolist() == [3, 1, 0]
    assert result["distinct_trajectory_count"].tolist() == [2, 1, 0]
    assert result["report"]["zero_observation_count"] == 1
    assert result["report"]["single_observation_count"] == 1
    assert result["observation_descriptor"][0, 0] > 0.99


def test_audit_rejects_out_of_range_keypoint_row():
    registry = _registry()
    registry["observation_keypoint_indices"][0] = 100
    try:
        materialize_observation_descriptor_audit(registry, _cache())
    except ValueError as error:
        assert "invalid keypoint" in str(error)
    else:
        raise AssertionError("invalid keypoint row was accepted")
