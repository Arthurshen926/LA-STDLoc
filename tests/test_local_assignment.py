import torch

from localization_training.local_assignment import rerank_one_of_k


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
