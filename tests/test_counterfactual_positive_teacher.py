import torch

from localization_training.counterfactual_positive_teacher import (
    build_anchor_cross_trajectory_support,
    choose_candidate_targets,
)


def test_cross_trajectory_support_counts_unique_trajectories():
    teacher = {
        "query_names": ["seq1/a.png", "seq1/b.png", "seq2/c.png"],
        "records": [
            {
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([2]),
            },
            {
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 1]),
                "positive_indices": torch.tensor([2]),
            },
            {
                "query_rows": torch.tensor([0]),
                "positive_offsets": torch.tensor([0, 2]),
                "positive_indices": torch.tensor([1, 2]),
            },
        ],
    }
    observations, trajectories = build_anchor_cross_trajectory_support(
        teacher, 4
    )
    assert observations.tolist() == [0, 1, 3, 0]
    assert trajectories.tolist() == [0, 1, 2, 0]


def test_candidate_target_variants_are_not_collapsed_to_score_best():
    choices = choose_candidate_targets(
        candidate_ids=torch.tensor([10, 11, 12]),
        scores=torch.tensor([0.9, 0.8, 0.7]),
        reprojection_errors=torch.tensor([1.8, 0.2, 0.9]),
        trajectory_support=torch.tensor([1, 2, 5]),
        bias_gain_m2=torch.tensor([-1.0, 0.1, 0.2]),
        translation_logdet_gain=torch.tensor([0.3, 0.0, 0.4]),
        strict_counterfactual=torch.tensor([False, True, True]),
    )
    assert choices == {
        "score_best": 0,
        "reprojection_best": 1,
        "track_stable": 2,
        "counterfactual_pose_best": 2,
    }
