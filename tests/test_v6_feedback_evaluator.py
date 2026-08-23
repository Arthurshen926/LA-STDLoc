import torch

from map_learning.v6_feedback_evaluator import (
    _descriptor_training_query_mask,
    _descriptor_training_query_masks,
    _layer_edges,
    _maximum_matching,
    _positive_score_statistics,
    _reconstruction_target_query_mask,
    _selection_training_query_mask,
)


def test_descriptor_training_query_mask_fails_closed_for_legacy_residual() -> None:
    legacy = {"anchor_descriptor_residual": torch.tensor([[0.0, 0.1]])}
    assert _descriptor_training_query_mask(legacy, 3).tolist() == [True, True, True]
    current = {
        "anchor_descriptor_residual": torch.tensor([[0.0, 0.1]]),
        "v6_descriptor_distillation": {
            "selected_query_indices": torch.tensor([1])
        },
    }
    assert _descriptor_training_query_mask(current, 3).tolist() == [False, True, False]
    legacy_report = {"v6_descriptor_distillation": {"version": 1}}
    assert _descriptor_training_query_mask(legacy_report, 3).tolist() == [True] * 3
    assert _descriptor_training_query_masks({}, 3)[
        "descriptor_dependency_present"
    ] is False


def test_explicit_training_split_is_distinct_from_gradient_reuse() -> None:
    state = {
        "v6_descriptor_distillation": {
            "training_query_indices": torch.tensor([0, 1]),
            "selected_query_indices": torch.tensor([1]),
            "training_query_registry_explicit": True,
        }
    }
    masks = _descriptor_training_query_masks(state, 3)
    assert masks["training_split"].tolist() == [True, True, False]
    assert masks["gradient_reused"].tolist() == [False, True, False]
    assert masks["training_registry_explicit"] is True


def test_topology_training_dependency_masks_fail_closed() -> None:
    state = {
        "v6_reconstruction_distillation": {
            "target_query_indices": torch.tensor([1])
        },
        "v6_selection_distillation": {
            "training_query_indices": torch.tensor([0, 2])
        },
    }
    assert _reconstruction_target_query_mask(state, 3).tolist() == [False, True, False]
    legacy_reconstruction = {
        "provenance": {"v6_reconstruction_feedback_sha256": "f" * 64}
    }
    assert _reconstruction_target_query_mask(
        legacy_reconstruction, 3
    ).tolist() == [True, True, True]
    assert _selection_training_query_mask(state, 3).tolist() == [True, False, True]
    assert _selection_training_query_mask(
        {"v6_selection_distillation": {}}, 3
    ).tolist() == [True, True, True]


def test_maximum_matching_prevents_duplicate_anchor_votes() -> None:
    count, pairs = _maximum_matching([[0, 1], [0], [1, 2]])
    assert count == 3
    assert len({anchor for _, anchor in pairs}) == 3


def test_spatial_positive_edges_match_dense_distance_oracle() -> None:
    generator = torch.Generator().manual_seed(91)
    keypoints = torch.rand((37, 2), generator=generator) * 100
    projected = torch.rand((211, 2), generator=generator) * 100
    visible = torch.arange(0, 211, 2, dtype=torch.long)
    radius = 7.0
    actual = _layer_edges(keypoints, projected, visible, radius)
    distance = torch.cdist(keypoints, projected[visible])
    expected = [
        visible[torch.nonzero(distance[row] <= radius).reshape(-1)].tolist()
        for row in range(keypoints.shape[0])
    ]
    assert actual == expected


def test_positive_statistics_match_stable_argsort_with_ties() -> None:
    scores = torch.tensor(
        [
            [0.7, 0.9, 0.9, -0.1, 0.8, 0.9],
            [0.2, 0.1, 0.3, 0.3, -0.5, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    positives_by_row = [[2, 4, 5], [], [3, 5]]
    result = _positive_score_statistics(scores, positives_by_row, chunk_size=1)
    assert set(result) == {0, 2}
    for row, positives in enumerate(positives_by_row):
        if not positives:
            continue
        order = torch.argsort(scores[row], descending=True, stable=True)
        oracle_rank = min(
            int(torch.nonzero(order == anchor, as_tuple=False)[0]) + 1
            for anchor in positives
        )
        wrong = scores[row].clone()
        wrong[torch.tensor(positives)] = -torch.inf
        positive, best_wrong, rank, best_anchor, best_wrong_anchor = result[row]
        positive_scores = scores[row, positives]
        expected_positive = positive_scores.max()
        expected_anchor = min(
            anchor
            for anchor in positives
            if scores[row, anchor] == expected_positive
        )
        assert positive == float(expected_positive)
        assert best_wrong == float(wrong.max())
        assert best_wrong_anchor == int(torch.argmax(wrong))
        assert rank == oracle_rank
        assert best_anchor == expected_anchor
