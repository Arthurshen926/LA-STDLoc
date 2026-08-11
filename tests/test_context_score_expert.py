import torch
import torch.nn.functional as F

from map_learning.context_score_expert import (
    ContextScoreExpert,
    build_context_score_bank,
    concatenate_dual_expert_descriptors,
    expected_clean_inlier_loss,
    expected_clean_pose_information_loss,
    protocol_name,
)
from map_learning.metric import SharedLowRankMetric


def test_context_score_expert_outputs_unit_codes():
    torch.manual_seed(21)
    expert = ContextScoreExpert(
        descriptor_dim=4,
        code_dim=3,
        hidden_dim=8,
        context_kernels=(3,),
        context_mode="local_only",
    )
    base = F.normalize(torch.randn(6, 4), dim=1)

    codes = expert(base, torch.randn(6, 2, 4))

    assert codes.shape == (6, 3)
    torch.testing.assert_close(
        codes.norm(dim=1), torch.ones(6), atol=1e-6, rtol=0.0
    )


def test_shared_global_expert_returns_one_code_for_the_whole_image():
    torch.manual_seed(24)
    expert = ContextScoreExpert(
        descriptor_dim=4,
        code_dim=3,
        hidden_dim=8,
        context_kernels=(3,),
        context_mode="global_only",
        input_scope="shared_global",
    )
    base = F.normalize(torch.randn(5, 4), dim=1)
    tokens = torch.randn(5, 2, 4)
    tokens[:, -1] = tokens[0, -1]

    codes = expert(base, tokens)

    torch.testing.assert_close(codes, codes[:1].expand_as(codes))


def test_shared_global_query_gate_is_image_shared_and_bounded():
    torch.manual_seed(25)
    expert = ContextScoreExpert(
        descriptor_dim=4,
        code_dim=3,
        hidden_dim=8,
        context_kernels=(3,),
        context_mode="global_only",
        input_scope="shared_global",
        learned_query_gate=True,
    )
    base = F.normalize(torch.randn(5, 4), dim=1)
    tokens = torch.randn(5, 2, 4)
    tokens[:, -1] = tokens[0, -1]

    gated_codes, gates = expert.query(base, tokens)

    torch.testing.assert_close(gates, torch.full_like(gates, 0.5))
    torch.testing.assert_close(gated_codes, gated_codes[:1].expand_as(gated_codes))
    torch.testing.assert_close(
        gated_codes.norm(dim=1), gates[:, 0], atol=1e-6, rtol=0.0
    )


def test_dual_descriptor_is_normalized_and_exactly_preserves_additive_score():
    torch.manual_seed(22)
    base_query = F.normalize(torch.randn(5, 4), dim=1)
    base_map = F.normalize(torch.randn(7, 4), dim=1)
    context_query = F.normalize(torch.randn(5, 3), dim=1)
    context_map = F.normalize(torch.randn(7, 3), dim=1)
    context_map *= torch.linspace(0.0, 1.0, 7)[:, None]
    weight = 0.05

    joint_query = concatenate_dual_expert_descriptors(
        base_query,
        context_query,
        context_weight=weight,
        map_side=False,
    )
    joint_map = concatenate_dual_expert_descriptors(
        base_map,
        context_map,
        context_weight=weight,
        map_side=True,
    )

    torch.testing.assert_close(
        joint_query.norm(dim=1), torch.ones(5), atol=1e-6, rtol=0.0
    )
    torch.testing.assert_close(
        joint_map.norm(dim=1), torch.ones(7), atol=1e-6, rtol=0.0
    )
    expected = (
        base_query @ base_map.T + weight * (context_query @ context_map.T)
    ) / (1.0 + weight)
    torch.testing.assert_close(
        joint_query @ joint_map.T, expected, atol=1e-6, rtol=1e-6
    )


def test_zero_context_weight_is_exact_a1_descriptor_identity():
    torch.manual_seed(23)
    base = F.normalize(torch.randn(4, 5), dim=1)
    context = F.normalize(torch.randn(4, 3), dim=1)

    joint = concatenate_dual_expert_descriptors(
        base,
        context,
        context_weight=0.0,
        map_side=False,
    )

    assert torch.equal(joint, base)


def test_expected_clean_inlier_loss_rewards_legal_probability_mass():
    bank = F.normalize(torch.eye(3), dim=1)
    positives = torch.tensor([[0], [1]])
    ignored = torch.full((2, 1), -1)
    good = F.normalize(torch.tensor([[4.0, 1.0, 0.0], [0.0, 4.0, 1.0]]), dim=1)
    bad = F.normalize(torch.tensor([[1.0, 4.0, 0.0], [0.0, 1.0, 4.0]]), dim=1)

    good_loss, good_mass = expected_clean_inlier_loss(
        good, bank, positives, ignored, topk=3, temperature=0.1
    )
    bad_loss, bad_mass = expected_clean_inlier_loss(
        bad, bank, positives, ignored, topk=3, temperature=0.1
    )

    assert good_loss < bad_loss
    assert good_mass.mean() > bad_mass.mean()


def test_pose_information_loss_rewards_geometrically_complete_clean_matches():
    bank = torch.eye(9)
    positives = torch.arange(8)[:, None]
    ignored = torch.full((8, 1), -1)
    xyz = torch.tensor(
        [
            [-1.0, -1.0, 4.0],
            [1.0, -1.0, 4.0],
            [-1.0, 1.0, 4.0],
            [1.0, 1.0, 4.0],
            [-2.0, 0.0, 6.0],
            [2.0, 0.0, 6.0],
            [0.0, -2.0, 8.0],
            [0.0, 2.0, 8.0],
            [0.0, 0.0, 5.0],
        ]
    )
    complete = torch.zeros(8, 9)
    complete[torch.arange(8), torch.arange(8)] = 4.0
    complete[:, 8] = 1.0
    degenerate = complete.clone()
    degenerate[4:, torch.arange(4, 8)] = 1.0
    degenerate[4:, 8] = 4.0
    complete = F.normalize(complete, dim=1).requires_grad_()
    degenerate = F.normalize(degenerate, dim=1)
    intrinsic = torch.tensor(
        [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]]
    )

    complete_loss, complete_diagnostics = expected_clean_pose_information_loss(
        complete,
        bank,
        positives,
        ignored,
        xyz,
        torch.eye(4),
        intrinsic,
        topk=9,
        temperature=0.1,
    )
    degenerate_loss, degenerate_diagnostics = expected_clean_pose_information_loss(
        degenerate,
        bank,
        positives,
        ignored,
        xyz,
        torch.eye(4),
        intrinsic,
        topk=9,
        temperature=0.1,
    )
    complete_loss.backward()

    assert complete_loss < degenerate_loss
    assert (
        complete_diagnostics["information_retention"]
        > degenerate_diagnostics["information_retention"]
    )
    assert complete.grad is not None
    assert bool(torch.isfinite(complete.grad).all())


def test_context_bank_norm_is_cross_view_concentration():
    expert = ContextScoreExpert(
        descriptor_dim=4,
        code_dim=2,
        hidden_dim=8,
        context_kernels=(3,),
        context_mode="zero",
    )
    metric = SharedLowRankMetric(descriptor_dim=4, rank=2).eval()
    teacher = {
        "anchor_count": 2,
        "query_names": ["q0", "q1"],
        "records": [
            {
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
            },
            {
                "query_rows": torch.tensor([0, 1]),
                "positive_offsets": torch.tensor([0, 1, 2]),
                "positive_indices": torch.tensor([0, 1]),
            },
        ],
    }
    query_cache = {
        "queries": {
            name: {
                "native_descriptors": F.normalize(torch.randn(2, 4), dim=1),
            }
            for name in teacher["query_names"]
        }
    }

    bank, report = build_context_score_bank(
        expert=expert,
        metric=metric,
        teacher=teacher,
        query_cache=query_cache,
        support_query_indices=[0, 1],
        anchor_indices=torch.tensor([0, 1]),
        expected_view_counts=torch.tensor([2, 2]),
        device=torch.device("cpu"),
    )

    assert bank.shape == (2, 2)
    assert bool((bank.norm(dim=1) <= 1.0 + 1e-6).all())
    assert 0.0 <= report["map_concentration_mean"] <= 1.0
    assert protocol_name(0.05) == "context_lambda_0p05"
