import torch
import numpy as np

from localization_training.local_assignment import (
    OneOfKAssignmentHead,
    rerank_one_of_k,
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
