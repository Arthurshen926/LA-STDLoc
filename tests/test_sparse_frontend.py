import torch


def test_sparse_frontend_selects_sorted_nms_keypoints():
    from localization_training.sparse_frontend import select_keypoints

    heatmap = torch.zeros(1, 5, 5)
    heatmap[0, 1, 3] = 0.8
    heatmap[0, 3, 1] = 0.9
    keypoint_ids, scores = select_keypoints(heatmap, count=2, nms_radius=0)

    assert keypoint_ids.tolist() == [8, 16]
    assert torch.allclose(scores, torch.tensor([0.8, 0.9]))


def test_matchability_reranks_without_moving_keypoint_nms_proposals():
    from localization_training.sparse_frontend import rank_keypoint_proposals

    keypoint = torch.zeros(1, 5, 5)
    keypoint[0, 1, 1] = 0.9
    keypoint[0, 1, 2] = 0.8
    keypoint[0, 3, 3] = 0.7
    matchability = torch.ones_like(keypoint)
    matchability[0, 1, 1] = 0.01
    matchability[0, 1, 2] = 1.0
    matchability[0, 3, 3] = 1.0

    ranked = rank_keypoint_proposals(keypoint, matchability, nms_radius=1)

    assert ranked[0, 1, 1] > 0
    assert ranked[0, 1, 2] == 0
    assert ranked[0, 3, 3] > ranked[0, 1, 1]


def test_matchability_reranker_rejects_shape_mismatch():
    import pytest

    from localization_training.sparse_frontend import rank_keypoint_proposals

    with pytest.raises(ValueError, match="same shape"):
        rank_keypoint_proposals(torch.ones(1, 2, 2), torch.ones(1, 3, 2), 1)


def test_sparse_frontend_topk_and_mnn_use_same_score_matrix():
    from localization_training.sparse_frontend import match_score_matrix

    score = torch.tensor([[0.9, 0.8], [0.85, 0.1]])
    topk = match_score_matrix(score, mode="topk", topk=2, threshold=0.0)
    mnn = match_score_matrix(score, mode="mnn", threshold=0.0)

    assert topk.keypoint_idx.tolist() == [0, 0, 1, 1]
    assert topk.landmark_idx.tolist() == [0, 1, 0, 1]
    assert mnn.keypoint_idx.tolist() == [0]
    assert mnn.landmark_idx.tolist() == [0]


def test_sparse_frontend_normalization_matches_cosine_similarity():
    from localization_training.sparse_frontend import build_score_matrix

    query = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    landmarks = torch.tensor([[4.0, 0.0], [0.0, 5.0]])
    similarity, score = build_score_matrix(query, landmarks, normalize=True)

    assert torch.allclose(similarity, torch.eye(2))
    assert torch.equal(similarity, score)


def test_sparse_frontend_landmark_dedup_keeps_highest_score_and_first_tie():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        deduplicate_landmark_matches,
    )

    matches = SparseMatchResult(
        keypoint_idx=torch.tensor([0, 1, 2, 3, 4]),
        landmark_idx=torch.tensor([2, 1, 2, 1, 2]),
        scores=torch.tensor([0.4, 0.7, 0.9, 0.7, 0.9]),
    )
    unique = deduplicate_landmark_matches(matches)

    assert unique.keypoint_idx.tolist() == [1, 2]
    assert unique.landmark_idx.tolist() == [1, 2]
    assert torch.allclose(unique.scores, torch.tensor([0.7, 0.9]))


def test_sparse_frontend_landmark_limit_keeps_two_best_per_landmark():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        limit_matches_per_landmark,
    )

    matches = SparseMatchResult(
        keypoint_idx=torch.arange(6),
        landmark_idx=torch.tensor([0, 1, 0, 0, 1, 0]),
        scores=torch.tensor([0.1, 0.2, 0.8, 0.7, 0.9, 0.6]),
    )
    limited = limit_matches_per_landmark(matches, 2)

    assert limited.keypoint_idx.tolist() == [1, 2, 3, 4]
    assert limited.landmark_idx.tolist() == [1, 0, 0, 1]
    assert torch.allclose(limited.scores, torch.tensor([0.2, 0.8, 0.7, 0.9]))


def test_sparse_frontend_keypoint_limit_reranks_topk_landmark_hypotheses():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        limit_matches_per_keypoint,
    )

    matches = SparseMatchResult(
        keypoint_idx=torch.tensor([0, 0, 0, 1, 1, 1]),
        landmark_idx=torch.tensor([4, 5, 6, 7, 8, 9]),
        scores=torch.tensor([0.1, 0.9, 0.8, 0.4, 0.3, 0.7]),
    )
    limited = limit_matches_per_keypoint(matches, 1)

    assert limited.keypoint_idx.tolist() == [0, 1]
    assert limited.landmark_idx.tolist() == [5, 9]
    assert torch.allclose(limited.scores, torch.tensor([0.9, 0.7]))


def test_sparse_pair_context_is_finite_and_preserves_similarity_gradient():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        build_pair_context_features,
    )

    similarity = torch.tensor(
        [[0.9, 0.8, 0.1], [0.85, 0.2, 0.1]],
        requires_grad=True,
    )
    matches = SparseMatchResult(
        keypoint_idx=torch.tensor([0, 1]),
        landmark_idx=torch.tensor([0, 0]),
        scores=torch.tensor([0.9, 0.85]),
    )
    features = build_pair_context_features(
        similarity,
        torch.tensor([0.7, 0.6]),
        matches,
        context_topk=3,
    )
    features[:, 0].sum().backward()

    assert features.shape == (2, 6)
    assert torch.isfinite(features).all()
    assert torch.allclose(features[:, 5], torch.log(torch.tensor(2.0)) / torch.log(torch.tensor(16.0)))
    assert similarity.grad is not None
    assert similarity.grad[0, 0] == 1
    assert similarity.grad[1, 0] == 1


def test_sparse_candidate_selection_refills_floor_after_landmark_quota():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        select_match_candidates,
    )

    matches = SparseMatchResult(
        keypoint_idx=torch.arange(6),
        landmark_idx=torch.tensor([0, 0, 0, 1, 2, 3]),
        scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
    )
    selected = select_match_candidates(
        matches,
        threshold=0.75,
        max_matches_per_landmark=2,
        min_match_count=4,
    )

    assert selected.keypoint_idx.tolist() == [0, 1, 3, 4]
    assert selected.landmark_idx.tolist() == [0, 0, 1, 2]


def test_sparse_candidate_selection_only_refills_below_trigger():
    from localization_training.sparse_frontend import (
        SparseMatchResult,
        select_match_candidates,
    )

    matches = SparseMatchResult(
        keypoint_idx=torch.arange(5),
        landmark_idx=torch.arange(5),
        scores=torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5]),
    )
    untouched = select_match_candidates(
        matches,
        threshold=0.65,
        min_match_count=5,
        refill_trigger_count=2,
    )
    refilled = select_match_candidates(
        matches,
        threshold=0.75,
        min_match_count=5,
        refill_trigger_count=3,
    )

    assert untouched.keypoint_idx.tolist() == [0, 1, 2]
    assert refilled.keypoint_idx.tolist() == [0, 1, 2, 3, 4]
