import torch

from map_learning.alias_teacher import (
    alias_group_ranking_loss,
    build_recurrent_alias_teacher,
    protected_clean_margin_loss,
)
from map_learning.metric import SharedLowRankMetric
from map_learning.trainer import (
    alias_repair_anchor_indices,
    bounded_anchor_bank,
    resolve_density_prefixes,
)


def test_recurrent_alias_requires_independent_groups_and_local_collision():
    false = {
        0: {0: 3, 1: 3},
        1: {0: 3, 1: 3},
        2: {0: 4, 1: 4},
    }
    teacher = build_recurrent_alias_teacher(
        false,
        torch.tensor([0, 1, 0]),
        anchor_count=8,
        minimum_distinct_groups=2,
        minimum_queries=2,
        minimum_occurrences=4,
        minimum_rows_per_query=2,
    )
    assert teacher.active_anchors.tolist() == [3]
    assert teacher.row_anchors == {0: {0: 3, 1: 3}, 1: {0: 3, 1: 3}}
    assert teacher.diagnostics["active_alias_row_count"] == 4


def test_alias_group_loss_repairs_false_winner_against_legal_positive():
    query = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    bank = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.8, 0.2]])
    positives = torch.tensor([[0, -1], [2, -1]])
    loss, diagnostics = alias_group_ranking_loss(
        query,
        bank,
        positives,
        alias_anchors=torch.tensor([1, 1]),
        alias_weights=torch.ones(2),
        margin=0.05,
        temperature=0.04,
    )
    assert float(loss) > 0.0
    assert diagnostics["alias_groups"] == 1
    loss.backward()
    assert torch.isfinite(query.grad).all()


def test_recurrent_alias_can_require_same_solver_harmful_assignment():
    false = {
        0: {0: 3, 1: 3, 2: 7},
        1: {0: 3, 1: 3},
        2: {0: 3, 1: 3},
    }
    harmful = {
        0: {0: 3, 1: 4, 2: 7},
        1: {0: 3, 1: 3},
        2: {0: 3, 1: 3},
    }
    teacher = build_recurrent_alias_teacher(
        false,
        torch.tensor([0, 1, 2]),
        anchor_count=8,
        minimum_distinct_groups=2,
        minimum_queries=2,
        minimum_occurrences=4,
        minimum_rows_per_query=2,
        solver_harmful_assignments=harmful,
    )
    assert teacher.active_anchors.tolist() == [3]
    assert teacher.row_anchors == {1: {0: 3, 1: 3}, 2: {0: 3, 1: 3}}
    assert teacher.diagnostics["observed_false_assignment_count"] == 7
    assert teacher.diagnostics["solver_conditioned_false_assignment_count"] == 6
    assert teacher.diagnostics["solver_conditioned_alias"] is True


def test_protected_clean_margin_is_one_sided():
    query = torch.tensor([[1.0, 0.0]])
    bank = torch.tensor([[1.0, 0.0], [0.8, 0.2]])
    safe, _ = protected_clean_margin_loss(
        query, bank, torch.tensor([0]), torch.tensor([0.1])
    )
    violated, diagnostics = protected_clean_margin_loss(
        query, bank, torch.tensor([0]), torch.tensor([0.3])
    )
    assert float(safe) == 0.0
    assert float(violated) > 0.0
    assert diagnostics["protected_clean_violations"] == 1


def test_density_prefixes_use_native_row_rank_and_include_full_density():
    records = [
        {
            "cache_rows": torch.tensor([0, 7, 11, 15]),
        }
    ]
    assert resolve_density_prefixes(records, 16, (1.0, 0.5, 0.75)) == (
        8,
        12,
        16,
    )


def test_anchor_specific_residual_is_bounded_and_does_not_change_query_metric():
    metric = SharedLowRankMetric(descriptor_dim=3, rank=2, max_residual_norm=0.05)
    raw = torch.nn.functional.normalize(torch.randn(4, 3), dim=1)
    parameter = torch.full_like(raw, 10.0, requires_grad=True)
    bank, shared_residual, anchor_residual = bounded_anchor_bank(
        metric, raw, parameter, 0.04
    )
    assert torch.allclose(torch.linalg.norm(bank, dim=1), torch.ones(4))
    assert float(torch.linalg.norm(anchor_residual, dim=1).max()) <= 0.040001
    assert torch.equal(shared_residual, torch.zeros_like(shared_residual))
    bank.sum().backward()
    assert torch.isfinite(parameter.grad).all()


def test_alias_pair_repair_includes_false_winner_and_legal_positives():
    teacher = build_recurrent_alias_teacher(
        {0: {3: 4, 7: 4}, 1: {2: 4, 9: 4}},
        torch.tensor([0, 1]),
        anchor_count=8,
        minimum_distinct_groups=2,
        minimum_queries=2,
        minimum_occurrences=4,
        minimum_rows_per_query=2,
    )
    records = [
        {
            "cache_rows": torch.tensor([3, 7]),
            "positives": torch.tensor([[1, 2, -1], [2, -1, -1]]),
        },
        {
            "cache_rows": torch.tensor([2, 9]),
            "positives": torch.tensor([[5, -1, -1], [6, -1, -1]]),
        },
    ]
    selected, diagnostics = alias_repair_anchor_indices(
        teacher, records, include_positives=True
    )
    assert selected.tolist() == [1, 2, 4, 5, 6]
    assert diagnostics["alias_false_trainable_anchor_count"] == 1
    assert diagnostics["alias_positive_trainable_anchor_count"] == 4
