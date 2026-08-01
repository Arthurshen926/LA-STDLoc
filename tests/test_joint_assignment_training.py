import torch

from localization_training.joint_assignment_training import (
    ambiguous_mask_from_teacher_record,
    beta_smoothed_anchor_features,
    positive_mask_from_teacher_record,
    select_balanced_training_rows,
    select_counterfactual_replacement_rows,
    mask_stale_dynamic_harmful_evidence,
    temporal_view_bin_support_from_positive_teacher,
    trajectory_support_from_positive_teacher,
    weighted_multi_positive_assignment_loss,
)


def test_beta_statistics_are_finite_and_do_not_trust_unobserved_anchors():
    features = beta_smoothed_anchor_features(
        {
            "attempts": torch.tensor([0.0, 8.0]),
            "clean": torch.tensor([0.0, 6.0]),
            "clean_inlier": torch.tensor([0.0, 4.0]),
            "harmful_inlier": torch.tensor([0.0, 1.0]),
        },
        torch.tensor([0, 3]),
        torch.tensor([0, 5]),
    )
    assert features.shape == (2, 6)
    assert torch.isfinite(features).all()
    assert features[0, :3].tolist() == [0.5, 0.5, 0.5]
    assert features[1, 0] > features[0, 0]


def test_trajectory_support_counts_distinct_sequences_not_frames():
    teacher = {
        "query_names": ["seq0/a.png", "seq0/b.png", "seq1/a.png"],
        "records": [
            {"positive_indices": torch.tensor([0, 1])},
            {"positive_indices": torch.tensor([0])},
            {"positive_indices": torch.tensor([0, 2])},
        ],
    }
    assert trajectory_support_from_positive_teacher(teacher, 3).tolist() == [2, 1, 1]


def test_temporal_support_counts_distinct_view_bins_within_one_sequence():
    teacher = {
        "query_names": [
            "seq0/frame00000.png",
            "seq0/frame00001.png",
            "seq0/frame00002.png",
            "seq0/frame00003.png",
        ],
        "records": [
            {"positive_indices": torch.tensor([0, 1])},
            {"positive_indices": torch.tensor([0])},
            {"positive_indices": torch.tensor([0, 2])},
            {"positive_indices": torch.tensor([0])},
        ],
    }
    support = temporal_view_bin_support_from_positive_teacher(
        teacher, 3, bins_per_trajectory=2
    )
    assert support.tolist() == [2, 1, 1]


def test_packed_positive_teacher_matches_topk_candidates():
    record = {
        "positive_offsets": torch.tensor([0, 2, 2, 3]),
        "positive_indices": torch.tensor([1, 4, 2]),
    }
    candidates = torch.tensor([[4, 3], [1, 2], [0, 2]])
    assert positive_mask_from_teacher_record(record, candidates).tolist() == [
        [True, False],
        [False, False],
        [False, True],
    ]


def test_ambiguous_teacher_matches_candidates_without_promoting_them():
    record = {
        "ambiguous_offsets": torch.tensor([0, 1, 2]),
        "ambiguous_indices": torch.tensor([3, 7]),
    }
    candidates = torch.tensor([[3, 4], [5, 7]])
    assert ambiguous_mask_from_teacher_record(record, candidates).tolist() == [
        [True, False],
        [False, True],
    ]


def test_balanced_rows_keep_all_positive_and_harmful_measurements():
    positive = torch.tensor(
        [[True, False], [False, False], [False, True], [False, False]]
    )
    scores = torch.tensor([[0.9, 0.8], [0.8, 0.79], [0.7, 0.6], [0.5, 0.1]])
    selected = select_balanced_training_rows(
        positive,
        scores,
        harmful_inlier_mask=torch.tensor([False, True, False, False]),
        maximum_rows=3,
    )
    assert set(selected.tolist()) == {0, 1, 2}


def test_balanced_rows_exclude_ambiguous_only_rows_even_if_stale_harmful():
    positive = torch.tensor(
        [[True, False], [False, False], [False, False], [False, False]]
    )
    scores = torch.tensor([[0.9, 0.8], [0.9, 0.89], [0.8, 0.79], [0.5, 0.1]])
    selected = select_balanced_training_rows(
        positive,
        scores,
        harmful_inlier_mask=torch.tensor([False, True, False, False]),
        ignored_row_mask=torch.tensor([False, True, False, False]),
        maximum_rows=3,
    )
    assert 1 not in selected.tolist()
    assert 0 in selected.tolist()


def test_counterfactual_rows_require_harmful_wrong_top1_and_legal_repair():
    positive = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, True],
        ]
    )
    harmful = torch.tensor([True, True, True, False])
    scores = torch.tensor(
        [[0.9, 0.8, 0.7], [0.9, 0.6, 0.5], [0.8, 0.7, 0.6], [0.7, 0.69, 0.6]]
    )
    assert select_counterfactual_replacement_rows(
        positive, harmful, scores, maximum_rows=4
    ).tolist() == [1]


def test_stale_dynamic_harmful_labels_are_masked_after_identity_switch():
    harmful, changed = mask_stale_dynamic_harmful_evidence(
        torch.tensor([True, True, False]),
        torch.tensor([4, 7, 9]),
        torch.tensor([4, 8, 9]),
    )
    assert changed.tolist() == [False, True, False]
    assert harmful.tolist() == [True, False, False]


def test_weighted_assignment_protects_clean_top1_and_supports_exact_veto():
    logits = torch.tensor([[0.0, 2.0], [2.0, 0.0]], requires_grad=True)
    null = torch.tensor([-5.0, 5.0], requires_grad=True)
    positive = torch.tensor([[True, True], [False, True]])
    target_weights = torch.tensor([[1.0, 4.0], [0.0, 0.0]])
    loss, diagnostics = weighted_multi_positive_assignment_loss(
        logits,
        null,
        positive,
        candidate_target_weights=target_weights,
        protect_clean_top1=True,
    )
    loss.backward()
    assert diagnostics["protected_top1_rows"] == 1
    assert diagnostics["positive_rows"] == 1
    assert diagnostics["null_rows"] == 1
    assert torch.isfinite(loss)


def test_exact_pose_preferences_form_a_nonnegative_multi_positive_nll():
    logits = torch.zeros(1, 3, requires_grad=True)
    loss, _ = weighted_multi_positive_assignment_loss(
        logits,
        torch.tensor([-10.0]),
        torch.tensor([[True, True, False]]),
        candidate_target_weights=torch.tensor([[2.0, 0.5, 0.0]]),
        protect_clean_top1=False,
        null_loss_weight=0.0,
    )
    loss.backward()
    assert float(loss) >= 0.0
    assert logits.grad[0, 0] < logits.grad[0, 1]
