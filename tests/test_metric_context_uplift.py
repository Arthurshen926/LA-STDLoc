import torch
import torch.nn.functional as F

from map_learning.metric import SharedLowRankMetric
from map_learning.metric_context_uplift import (
    MetricPreservingContextUplift,
    build_metric_uplift_bank,
    load_frozen_metric_state,
)


def test_metric_context_uplift_is_exact_a1_identity_at_initialization():
    torch.manual_seed(11)
    adapter = MetricPreservingContextUplift(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_angle_rad=0.05,
    )
    base = F.normalize(torch.randn(6, 4), dim=1)
    context = torch.randn(6, 2, 4)

    uplifted, angle_vector, gate = adapter(base, context)

    assert torch.equal(uplifted, base)
    assert torch.equal(angle_vector, torch.zeros_like(angle_vector))
    torch.testing.assert_close(gate, torch.full_like(gate, 0.5))


def test_metric_context_uplift_is_tangent_and_angularly_bounded():
    torch.manual_seed(12)
    adapter = MetricPreservingContextUplift(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_angle_rad=0.05,
    )
    with torch.no_grad():
        adapter.context_head[-1].weight.normal_()
        adapter.context_head[-1].bias.normal_()
    base = F.normalize(torch.randn(7, 4), dim=1)

    uplifted, angle_vector, _ = adapter(base, torch.randn(7, 2, 4))

    assert bool((angle_vector.norm(dim=1) < 0.05).all())
    torch.testing.assert_close(
        (angle_vector * base).sum(dim=1),
        torch.zeros(7),
        atol=1e-6,
        rtol=0.0,
    )
    torch.testing.assert_close(
        uplifted.norm(dim=1), torch.ones(7), atol=1e-6, rtol=0.0
    )


def test_zero_uplift_bank_preserves_learned_anchor_descriptors_exactly():
    adapter = MetricPreservingContextUplift(
        descriptor_dim=4,
        hidden_dim=8,
        context_kernels=(3,),
        maximum_angle_rad=0.05,
    )
    metric = SharedLowRankMetric(descriptor_dim=4, rank=2).eval()
    base = F.normalize(torch.randn(2, 4), dim=1)
    teacher = {
        "anchor_count": 2,
        "query_names": ["q"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
            }
        ],
    }
    query_cache = {
        "queries": {
            "q": {
                "native_descriptors": F.normalize(torch.randn(2, 4), dim=1),
                "native_keypoints": torch.tensor([[1.0, 1.0], [9.0, 1.0]]),
                "native_input_hw": [8, 16],
                "feature_map": F.normalize(torch.randn(4, 1, 2), dim=0),
                "valid_mask": torch.ones(1, 2, dtype=torch.bool),
            }
        }
    }

    uplifted, report = build_metric_uplift_bank(
        adapter=adapter,
        metric=metric,
        base_anchor_bank=base,
        teacher=teacher,
        query_cache=query_cache,
        support_query_indices=[0],
        anchor_indices=torch.tensor([0, 1]),
        expected_view_counts=torch.tensor([1, 1]),
        device=torch.device("cpu"),
    )

    assert torch.equal(uplifted, base)
    assert report["observation_angle_rad_maximum"] == 0.0
    assert report["map_angle_rad_maximum"] == 0.0


def test_frozen_metric_loader_rejects_anchor_misalignment():
    metric = SharedLowRankMetric(descriptor_dim=4, rank=2)
    state = {
        "schema": "lafgs_shared_metric_state",
        "landmark_indices": torch.tensor([7, 8]),
        "metric_config": metric.export_config(),
        "metric_state_dict": metric.state_dict(),
    }

    loaded = load_frozen_metric_state(
        state,
        anchor_ids=torch.tensor([7, 8]),
        device=torch.device("cpu"),
    )
    assert not any(parameter.requires_grad for parameter in loaded.parameters())

    try:
        load_frozen_metric_state(
            state,
            anchor_ids=torch.tensor([8, 7]),
            device=torch.device("cpu"),
        )
    except ValueError as error:
        assert "does not align" in str(error)
    else:
        raise AssertionError("misaligned metric state was accepted")
