import torch
import numpy as np
import pytest

from localization_training.local_assignment import (
    JOINT_ASSIGNMENT_V1_FEATURE_NAMES,
    OneOfKAssignmentHead,
    build_one_of_k_features,
    rerank_one_of_k,
    validate_joint_assignment_state_contract,
)
from stdloc import (
    one_of_k_assignment_gt_diagnostics,
    validate_one_of_k_topk_contract,
)


def test_one_of_k_reranker_never_duplicates_query_rows():
    feature_map = torch.zeros(2, 5, 5)
    feature_map[0] = 1.0
    feature_map[:, 2, 3] = torch.tensor([0.0, 1.0])
    landmark = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    output = rerank_one_of_k(
        feature_map,
        keypoint_xy=torch.tensor([[1.5, 1.5], [1.5, 1.5]]),
        topk_landmark_idx=torch.tensor([[0, 1], [0, 2]]),
        topk_scores=torch.tensor([[0.5, 0.49], [0.5, 0.49]]),
        landmark_descriptors=landmark,
        image_hw=(5, 5),
        radius=1,
        step_px=1,
        local_peak_weight=1.0,
        local_margin_weight=0.0,
        local_entropy_weight=0.0,
        offset_weight=0.0,
    )
    assert output.landmark_idx.shape == (2,)
    assert output.keep.shape == (2,)
    assert output.candidate_logits.shape == (2, 2)


def test_one_of_k_null_threshold_rejects_uncertain_row():
    output = rerank_one_of_k(
        torch.ones(2, 3, 3),
        keypoint_xy=torch.tensor([[1.0, 1.0]]),
        topk_landmark_idx=torch.tensor([[0, 1]]),
        topk_scores=torch.tensor([[0.1, 0.09]]),
        landmark_descriptors=torch.eye(2),
        image_hw=(3, 3),
        radius=0,
        local_peak_weight=0.0,
        local_margin_weight=0.0,
        local_entropy_weight=0.0,
        offset_weight=0.0,
        null_score_threshold=0.2,
    )
    assert not bool(output.keep.item())


def test_assignment_head_null_can_be_disabled():
    class AlwaysNull(torch.nn.Module):
        def forward(self, features):
            return features[:, :, 0], torch.full(
                (features.shape[0],), 100.0, device=features.device
            )

    output = rerank_one_of_k(
        torch.tensor([[[1.0]], [[0.0]]]),
        keypoint_xy=torch.tensor([[0.0, 0.0]]),
        topk_landmark_idx=torch.tensor([[0, 1]]),
        topk_scores=torch.tensor([[0.8, 0.2]]),
        landmark_descriptors=torch.eye(2),
        image_hw=(1, 1),
        radius=0,
        assignment_head=AlwaysNull(),
        use_assignment_null=False,
    )
    assert output.null_selected.tolist() == [False]
    assert output.keep.tolist() == [True]


def test_bounded_assignment_residual_cannot_exceed_raw_cosine_cap():
    head = OneOfKAssignmentHead(
        hidden_dim=4,
        feature_dim=5,
        bounded_residual_max=0.02,
        logit_temperature=1.0,
    )
    for parameter in head.candidate.parameters():
        torch.nn.init.constant_(parameter, 100.0)
    features = torch.zeros(2, 3, 5)
    features[:, :, 0] = torch.tensor([0.8, 0.7, 0.6])
    logits, _ = head(features)
    residual = logits - features[:, :, 0]
    assert bool((residual.abs() <= 0.020001).all())


def test_pooled_full_null_head_consumes_the_complete_candidate_context():
    head = OneOfKAssignmentHead(
        hidden_dim=4,
        feature_dim=7,
        null_feature_mode="pooled_full",
    )
    candidate, null = head(torch.randn(3, 4, 7))
    assert candidate.shape == (3, 4)
    assert null.shape == (3,)
    assert head.export_config()["null_feature_mode"] == "pooled_full"


def test_cross_scene_head_row_normalizes_local_evidence_but_keeps_raw_skip():
    head = OneOfKAssignmentHead(
        hidden_dim=4,
        feature_dim=5,
        bounded_residual_max=0.0,
        normalize_candidate_features=True,
    )
    features = torch.randn(3, 4, 5)
    candidate, _ = head(features)
    assert candidate.shape == (3, 4)
    assert head.export_config()["normalize_candidate_features"] is True


def test_cross_scene_head_is_invariant_to_unrelated_query_rows():
    torch.manual_seed(4)
    head = OneOfKAssignmentHead(
        hidden_dim=8,
        feature_dim=7,
        bounded_residual_max=0.05,
        null_feature_mode="pooled_full",
        normalize_candidate_features=True,
    )
    reference = torch.randn(2, 4, 7)
    unrelated = torch.randn(9, 4, 7) * 100.0
    candidate_reference, null_reference = head(reference)
    candidate_combined, null_combined = head(
        torch.cat((reference, unrelated), dim=0)
    )
    assert torch.allclose(candidate_reference, candidate_combined[:2])
    assert torch.allclose(null_reference, null_combined[:2])


def test_ambiguity_gate_forces_high_margin_global_top1():
    class SwapHead(torch.nn.Module):
        def forward(self, features):
            logits = torch.tensor(
                [[0.0, 10.0], [0.0, 10.0]], device=features.device
            )
            return logits, torch.full((2,), -100.0, device=features.device)

    output = rerank_one_of_k(
        torch.ones(2, 3, 3),
        keypoint_xy=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        topk_landmark_idx=torch.tensor([[0, 1], [0, 1]]),
        topk_scores=torch.tensor([[0.8, 0.5], [0.51, 0.50]]),
        landmark_descriptors=torch.eye(2),
        image_hw=(3, 3),
        radius=0,
        assignment_head=SwapHead(),
        ambiguity_margin_threshold=0.05,
    )
    assert output.selected_position.tolist() == [0, 1]
    assert output.ambiguous.tolist() == [False, True]
    assert output.candidate_logits.tolist() == [[0.0, 10.0], [0.0, 10.0]]
    assert output.assignment_score.tolist() == [0.0, 10.0]
    assert torch.allclose(
        output.selected_global_score,
        torch.tensor([0.8, 0.50]),
    )
    assert torch.equal(output.scores, output.selected_global_score)


def test_ambiguity_margin_uses_best_alternative_not_candidate_position_one():
    class SwapHead(torch.nn.Module):
        def forward(self, features):
            return torch.tensor(
                [[0.0, 0.0, 10.0]], device=features.device
            ), torch.full((1,), -100.0, device=features.device)

    output = rerank_one_of_k(
        torch.ones(3, 1, 1),
        keypoint_xy=torch.tensor([[0.0, 0.0]]),
        topk_landmark_idx=torch.tensor([[0, 1, 2]]),
        topk_scores=torch.tensor([[0.8, 0.7, 0.9]]),
        landmark_descriptors=torch.eye(3),
        image_hw=(1, 1),
        radius=0,
        assignment_head=SwapHead(),
        ambiguity_margin_threshold=0.05,
    )
    assert torch.allclose(output.global_margin, torch.tensor([-0.1]))
    assert output.ambiguous.tolist() == [True]
    assert output.selected_position.tolist() == [2]


def test_assignment_identity_correction_does_not_reorder_by_head_logit():
    class OppositeScaleHead(torch.nn.Module):
        def forward(self, features):
            return torch.tensor(
                [[0.0, 10.0], [100.0, 0.0]], device=features.device
            ), torch.full((2,), -100.0, device=features.device)

    output = rerank_one_of_k(
        torch.ones(2, 3, 3),
        keypoint_xy=torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        topk_landmark_idx=torch.tensor([[0, 1], [0, 1]]),
        topk_scores=torch.tensor([[0.9, 0.4], [0.8, 0.3]]),
        landmark_descriptors=torch.eye(2),
        image_hw=(3, 3),
        radius=0,
        assignment_head=OppositeScaleHead(),
    )
    assert output.selected_position.tolist() == [1, 0]
    assert output.assignment_score.tolist() == [10.0, 100.0]
    assert torch.allclose(output.scores, torch.tensor([0.4, 0.8]))


def test_assignment_gt_diagnostics_identify_beneficial_and_harmful_swap():
    diagnostics = one_of_k_assignment_gt_diagnostics(
        keypoint_xy=np.array([[9.5, 9.5], [19.5, 9.5]]),
        topk_landmark_idx=np.array([[0, 1], [2, 3]]),
        selected_position=np.array([1, 1]),
        null_selected=np.array([False, False]),
        landmark_xyz=np.array(
            [[1.0, 0.0, 1.0], [0.0, 0.0, 1.0],
             [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        K=np.array([[10.0, 0.0, 10.0], [0.0, 10.0, 10.0], [0.0, 0.0, 1.0]]),
        gt_pose_w2c=np.eye(4),
    )
    assert diagnostics[
        "sparse_diag_native_rerank_gt_beneficial_swap_rate"
    ] == 0.5
    assert diagnostics[
        "sparse_diag_native_rerank_gt_harmful_swap_rate"
    ] == 0.5


def test_null_rejection_respects_global_cap_and_grid_floor():
    class AlwaysNull(torch.nn.Module):
        def forward(self, features):
            return features[:, :, 0], torch.full(
                (features.shape[0],), 100.0, device=features.device
            )

    output = rerank_one_of_k(
        torch.ones(2, 4, 4),
        keypoint_xy=torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [2.0, 2.0], [3.0, 3.0]]
        ),
        topk_landmark_idx=torch.tensor([[0, 1]] * 4),
        topk_scores=torch.tensor([[0.8, 0.2]] * 4),
        landmark_descriptors=torch.eye(2),
        image_hw=(4, 4),
        radius=0,
        assignment_head=AlwaysNull(),
        max_null_fraction=0.5,
        null_grid_rows=2,
        null_grid_cols=2,
        null_min_kept_per_grid=1,
    )
    assert int(output.null_selected.sum()) <= 2
    assert int(output.keep.sum()) >= 2


def test_topk_mismatch_requires_explicit_candidate_only_ablation():
    with np.testing.assert_raises_regex(ValueError, "top-K mismatch"):
        validate_one_of_k_topk_contract(4, 8)
    assert validate_one_of_k_topk_contract(
        4,
        8,
        allow_candidate_only_mismatch=True,
        use_learned_null=False,
    )
    with np.testing.assert_raises_regex(ValueError, "learned null"):
        validate_one_of_k_topk_contract(
            4,
            8,
            allow_candidate_only_mismatch=True,
            use_learned_null=True,
        )


def test_joint_assignment_context_is_query_normalized_and_aligned():
    feature_map = torch.zeros(2, 2, 2)
    feature_map[0] = 1.0
    features = build_one_of_k_features(
        feature_map,
        keypoint_xy=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
        topk_landmark_idx=torch.tensor([[0, 1], [0, 2]]),
        topk_scores=torch.tensor([[0.9, 0.7], [0.8, 0.6]]),
        landmark_descriptors=torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        ),
        image_hw=(2, 2),
        radius=0,
        context_version=1,
        keypoint_scores=torch.tensor([0.9, 0.1]),
        landmark_xyz=torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 2.0]]
        ),
        source_groups=torch.tensor([3, 4, 3]),
        dependency_groups=torch.tensor([7, 8, 9]),
    )
    assert features.shape == (2, 2, 17)
    assert torch.isfinite(features).all()
    assert features[0, 0, 14] > features[0, 1, 14]


def test_joint_assignment_context_requires_native_detector_scores():
    with np.testing.assert_raises_regex(ValueError, "requires keypoint scores"):
        build_one_of_k_features(
            torch.ones(2, 1, 1),
            keypoint_xy=torch.tensor([[0.0, 0.0]]),
            topk_landmark_idx=torch.tensor([[0, 1]]),
            topk_scores=torch.tensor([[0.9, 0.8]]),
            landmark_descriptors=torch.eye(2),
            image_hw=(1, 1),
            radius=0,
            context_version=1,
            landmark_xyz=torch.ones(2, 3),
            source_groups=torch.arange(2),
            dependency_groups=torch.arange(2),
        )


def _joint_assignment_state():
    return {
        "schema": "lafgs_joint_assignment_loso",
        "version": 4,
        "config": {
            "context_version": 1,
            "context_feature_names": list(JOINT_ASSIGNMENT_V1_FEATURE_NAMES),
        },
        "head_config": {"feature_dim": len(JOINT_ASSIGNMENT_V1_FEATURE_NAMES)},
        "landmark_statistics": torch.zeros(7, 6),
    }


def test_joint_assignment_runtime_contract_requires_exact_feature_order():
    state = _joint_assignment_state()
    assert validate_joint_assignment_state_contract(state)
    state["version"] = 5
    assert validate_joint_assignment_state_contract(state)
    state["config"]["context_feature_names"][0:2] = reversed(
        state["config"]["context_feature_names"][0:2]
    )
    with pytest.raises(ValueError, match="feature contract mismatch"):
        validate_joint_assignment_state_contract(state)


def test_joint_assignment_runtime_contract_rejects_statistic_width_mismatch():
    state = _joint_assignment_state()
    state["landmark_statistics"] = torch.zeros(7, 5)
    with pytest.raises(ValueError, match="landmark statistics"):
        validate_joint_assignment_state_contract(state)


def test_local_candidate_features_use_the_frozen_query_metric():
    class SwapChannels(torch.nn.Module):
        def forward(self, value):
            return value.flip(-1), torch.zeros_like(value)

    feature_map = torch.tensor([[[1.0]], [[0.0]]])
    common = dict(
        keypoint_xy=torch.tensor([[0.0, 0.0]]),
        topk_landmark_idx=torch.tensor([[0, 1]]),
        topk_scores=torch.tensor([[0.6, 0.5]]),
        landmark_descriptors=torch.eye(2),
        image_hw=(1, 1),
        radius=0,
    )
    raw = build_one_of_k_features(feature_map, **common)
    metric = build_one_of_k_features(
        feature_map, query_metric=SwapChannels(), **common
    )
    assert raw[0, :, 1].tolist() == [1.0, 0.0]
    assert metric[0, :, 1].tolist() == [0.0, 1.0]
