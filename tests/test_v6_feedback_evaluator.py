import torch
import pytest

from map_learning.v6_feedback_evaluator import (
    _aligned_keypoint_surface_depth,
    _anchor_unique_pose_rows,
    _descriptor_training_query_mask,
    _descriptor_training_query_masks,
    _exact_identity_anchor_by_query,
    _fixed_hypothesis_counterfactual_pose_weights,
    _layer_edges,
    _legal_descriptor_pair_is_clean,
    _maximum_matching,
    _partition_identity_edges,
    _positive_score_statistics,
    _reconstruction_training_query_mask,
    _reconstruction_target_query_mask,
    _selection_training_query_mask,
    _visible_spatial_rank,
)


class _IdentityObservations:
    names = ("q",)

    def __len__(self):
        return 1

    @staticmethod
    def build_view(_index):
        return type(
            "View", (), {"descriptors": torch.zeros((2, 4)), "image_name": "q"}
        )()


def test_pose_valid_depth_falls_back_to_dense_render_raster() -> None:
    view = type(
        "View",
        (),
        {
            "keypoint_depth": None,
            "depth": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "keypoints": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "keypoint_validity": None,
            "valid_mask": torch.tensor([[True, True], [True, False]]),
            "keypoint_alpha": None,
            "alpha": torch.tensor([[0.9, 0.9], [0.9, 0.9]]),
        },
    )()
    depth, source = _aligned_keypoint_surface_depth(view, alpha_minimum=0.05)
    assert source == "sampled_native_depth_raster_at_sparse_keypoints"
    assert depth[0] == 1.0
    assert torch.isnan(depth[1])


def test_pose_valid_depth_prefers_aligned_sparse_column_and_masks_alpha() -> None:
    view = type(
        "View",
        (),
        {
            "keypoint_depth": torch.tensor([2.0, 3.0]),
            "depth": torch.full((2, 2), 9.0),
            "keypoints": torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            "keypoint_validity": torch.tensor([True, True]),
            "valid_mask": None,
            "keypoint_alpha": torch.tensor([0.9, 0.01]),
            "alpha": None,
        },
    )()
    depth, source = _aligned_keypoint_surface_depth(view, alpha_minimum=0.05)
    assert source == "native_depth_at_keypoints"
    assert depth[0] == 2.0
    assert torch.isnan(depth[1])


def test_descriptor_training_query_mask_fails_closed_for_legacy_residual() -> None:
    legacy = {"anchor_descriptor_residual": torch.tensor([[0.0, 0.1]])}
    assert _descriptor_training_query_mask(legacy, 3).tolist() == [True, True, True]
    current = {
        "anchor_descriptor_residual": torch.tensor([[0.0, 0.1]]),
        "v6_descriptor_distillation": {"selected_query_indices": torch.tensor([1])},
    }
    assert _descriptor_training_query_mask(current, 3).tolist() == [False, True, False]
    legacy_report = {"v6_descriptor_distillation": {"version": 1}}
    assert _descriptor_training_query_mask(legacy_report, 3).tolist() == [True] * 3
    assert (
        _descriptor_training_query_masks({}, 3)["descriptor_dependency_present"]
        is False
    )


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
        "v6_reconstruction_distillation": {"target_query_indices": torch.tensor([1])},
        "v6_selection_distillation": {"training_query_indices": torch.tensor([0, 2])},
    }
    assert _reconstruction_target_query_mask(state, 3).tolist() == [False, True, False]
    assert _reconstruction_training_query_mask(state, 3).tolist() == [True] * 3
    split_reconstruction = {
        "v6_reconstruction_distillation": {
            "target_query_indices": torch.tensor([1]),
            "training_query_indices": torch.tensor([0, 1]),
            "eligible_support_query_indices": torch.tensor([0, 1]),
            "training_query_registry_explicit": True,
        }
    }
    assert _reconstruction_training_query_mask(
        split_reconstruction, 3
    ).tolist() == [True, True, False]
    legacy_reconstruction = {
        "provenance": {"v6_reconstruction_feedback_sha256": "f" * 64}
    }
    assert _reconstruction_target_query_mask(legacy_reconstruction, 3).tolist() == [
        True,
        True,
        True,
    ]
    initial_completion = {
        "anchor_candidate_kind": ["depth_proposed_projective_completion"]
    }
    assert _reconstruction_target_query_mask(initial_completion, 3).tolist() == [
        False,
        False,
        False,
    ]
    assert _reconstruction_training_query_mask(initial_completion, 3).tolist() == [
        False,
        False,
        False,
    ]
    assert _selection_training_query_mask(state, 3).tolist() == [True, False, True]
    assert _selection_training_query_mask(
        {"v6_selection_distillation": {}}, 3
    ).tolist() == [True, True, True]


def test_maximum_matching_prevents_duplicate_anchor_votes() -> None:
    count, pairs = _maximum_matching([[0, 1], [0], [1, 2]])
    assert count == 3
    assert len({anchor for _, anchor in pairs}) == 3


def test_visibility_rank_counts_image_cells_not_anchor_rows() -> None:
    projected = torch.tensor([[1.0, 1.0], [20.0, 20.0], [80.0, 20.0]])
    assert (
        _visible_spatial_rank(
            projected,
            torch.arange(3),
            image_hw=(100, 100),
        )
        == 2
    )


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
            anchor for anchor in positives if scores[row, anchor] == expected_positive
        )
        assert positive == float(expected_positive)
        assert best_wrong == float(wrong.max())
        assert best_wrong_anchor == int(torch.argmax(wrong))
        assert rank == oracle_rank
        assert best_anchor == expected_anchor


def test_identity_partition_uses_nearby_nonidentity_only_as_ignore() -> None:
    partition = _partition_identity_edges(
        torch.tensor([2, 3, 4, -1]),
        [[1, 2], [1], [4, 5], [6]],
        torch.tensor([True, True, True, True, False, True, True]),
    )
    assert partition["exact"] == [[2], [], [], []]
    assert partition["ambiguous"] == [[1], [1], [5], [6]]
    assert partition["incompatible"] == [[], [3], [], []]
    assert partition["inactive"] == [[], [], [4], []]
    assert partition["ignored"] == [[1], [1, 3], [5], [6]]


def test_positive_statistics_excludes_ambiguous_anchors_from_negative() -> None:
    scores = torch.tensor([[0.8, 0.95, 0.9, 0.7]])
    result = _positive_score_statistics(
        scores,
        [[0]],
        ignored_by_row=[[1, 2]],
    )
    positive, negative, rank, positive_anchor, negative_anchor = result[0]
    assert abs(positive - 0.8) < 1e-6
    assert abs(negative - 0.7) < 1e-6
    assert rank == 1
    assert positive_anchor == 0
    assert negative_anchor == 3
    # Raw global Top-1 is ignored Anchor 1, but the exact positive already
    # beats every legal negative and is therefore a clean protection pair.
    assert _legal_descriptor_pair_is_clean(positive, negative) is True
    assert _legal_descriptor_pair_is_clean(0.5, 0.5) is True


def test_counterfactual_pose_weight_measures_actual_winner_consensus_flip() -> None:
    xyz = torch.tensor(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [3.0, 0.0, 1.0]]
    )
    current = torch.eye(4)
    current[0, 3] = -1.0
    kwargs = {
        "triplets": torch.tensor([[0, 0, 1, 0]]),
        "keypoints": torch.tensor([[0.0, 0.0]]),
        "xyz": xyz,
        "intrinsics": torch.eye(3),
        "current_pose_w2c": current,
        "ground_truth_pose_w2c": torch.eye(4),
        "reprojection_error_px": 0.25,
    }
    weight = _fixed_hypothesis_counterfactual_pose_weights(
        winners=torch.tensor([1]), **kwargs
    )
    assert weight.tolist() == [1.0]
    # A legal pair that does not replace the deployed Top-1 is not a pose
    # counterfactual action, even if the two fixed hypotheses disagree on it.
    ignored = _fixed_hypothesis_counterfactual_pose_weights(
        winners=torch.tensor([0]), **kwargs
    )
    assert ignored.tolist() == [0.0]
    half = _fixed_hypothesis_counterfactual_pose_weights(
        winners=torch.tensor([2]),
        **{**kwargs, "triplets": torch.tensor([[0, 0, 2, 0]])},
    )
    assert half.tolist() == [0.5]


def test_pose_information_keeps_lowest_residual_row_per_anchor() -> None:
    rows, residuals = _anchor_unique_pose_rows(
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([5, 5, 7, 7]),
        keypoints=torch.tensor([[1.0, 0.0], [0.2, 0.0], [4.0, 0.0], [4.0, 0.0]]),
        projected=torch.tensor([[0.0, 0.0]] * 5 + [[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]]),
        winner_scores=torch.tensor([0.9, 0.7, 0.6, 0.8]),
    )
    assert rows.tolist() == [1, 3]
    assert torch.allclose(residuals, torch.tensor([0.2, 0.0], dtype=torch.float64))


def test_exact_identity_lineage_fails_closed_on_missing_or_duplicate_rows() -> None:
    observations = _IdentityObservations()
    with pytest.raises(ValueError, match="require projective Anchor observations"):
        _exact_identity_anchor_by_query(
            {"anchor_ids": torch.arange(2), "v6_mapping_query_names": ["q"]},
            observations,
        )
    duplicate = {
        "anchor_ids": torch.arange(2),
        "v6_mapping_query_names": ["q"],
        "projective_anchor_observations": {
            "schema": "lafgs_projective_anchor_observations",
            "version": 1,
            "observation_offsets": torch.tensor([0, 1, 2]),
            "query_indices": torch.tensor([0, 0]),
            "keypoint_indices": torch.tensor([0, 0]),
        },
    }
    with pytest.raises(ValueError, match="assigned to multiple Anchors"):
        _exact_identity_anchor_by_query(duplicate, observations)
