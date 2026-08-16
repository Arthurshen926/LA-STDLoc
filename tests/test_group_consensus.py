from __future__ import annotations

import numpy as np
import pytest
import torch

from localization.group_consensus import (
    build_standard_and_group_diverse_hypotheses,
    classify_hypothesis_oracle,
    correlation_groups_from_map,
    group_diverse_minimal_sets,
    score_hypothesis_residuals,
    select_group_capped_hypothesis,
    select_standard_hypothesis,
)


def test_group_capped_score_defeats_repeated_false_consensus() -> None:
    # Hypothesis 0 has twelve raw inliers but only three independent groups.
    # Hypothesis 1 has four raw inliers from four independent groups.
    groups = np.asarray([0] * 10 + [1, 2, 3, 4], dtype=np.int64)
    residuals = np.asarray(
        [
            [0.2] * 10 + [0.3, 0.4, 9.0, 9.0],
            [9.0] * 10 + [0.3, 0.4, 0.5, 0.6],
        ]
    )
    scores = score_hypothesis_residuals(residuals, groups, threshold_px=2.0)
    assert select_standard_hypothesis(scores) == 0
    assert select_group_capped_hypothesis(scores) == 1
    assert scores.standard_inlier_count.tolist() == [12, 4]
    assert scores.group_inlier_count.tolist() == [3, 4]


def test_group_diverse_minimal_sets_never_repeat_a_group() -> None:
    groups = np.repeat(np.arange(8), 3)
    samples = group_diverse_minimal_sets(
        groups, sample_size=4, sample_count=50, seed=2026
    )
    assert samples.shape == (50, 4)
    assert len({tuple(row) for row in samples.tolist()}) == 50
    for sample in samples:
        assert np.unique(groups[sample]).size == 4
    assert group_diverse_minimal_sets(
        np.asarray([0, 0, 1, 1]), sample_size=4, sample_count=10, seed=1
    ).shape == (0, 4)


def _oracle_fixture():
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]])
    points = np.asarray(
        [
            [(index % 4) * 0.2 - 0.3, (index // 4) * 0.2 - 0.2, 4.0 + 0.1 * index]
            for index in range(14)
        ]
    )
    correct = np.eye(4)
    wrong = np.eye(4)
    wrong[0, 3] = 0.5

    def project(pose, xyz):
        camera = xyz @ pose[:3, :3].T + pose[:3, 3]
        pixel = camera @ intrinsic.T
        return pixel[:, :2] / pixel[:, 2:]

    points_2d = np.concatenate(
        (project(wrong, points[:10]), project(correct, points[10:])), axis=0
    )
    groups = np.asarray([0] * 10 + [1, 2, 3, 4], dtype=np.int64)
    return intrinsic, points, points_2d, groups, correct, wrong


def test_hypothesis_oracle_separates_scorer_and_sampler_headroom() -> None:
    intrinsic, points, points_2d, groups, correct, wrong = _oracle_fixture()
    scorer = classify_hypothesis_oracle(
        standard_hypotheses_w2c=np.stack((wrong, correct)),
        group_diverse_hypotheses_w2c=np.empty((0, 4, 4)),
        points_2d=points_2d,
        points_3d=points,
        intrinsic=intrinsic,
        group_ids=groups,
        ground_truth_w2c=correct,
        reprojection_threshold_px=2.0,
    )
    assert scorer["category"] == "A_GROUP_SCORER_HEADROOM"
    assert scorer["standard_winner_correct"] is False
    assert scorer["group_capped_winner_correct"] is True
    assert scorer["authorizes_deployment_solver_change"] is False

    sampler = classify_hypothesis_oracle(
        standard_hypotheses_w2c=np.stack((wrong,)),
        group_diverse_hypotheses_w2c=np.stack((correct,)),
        points_2d=points_2d,
        points_3d=points,
        intrinsic=intrinsic,
        group_ids=groups,
        ground_truth_w2c=correct,
        reprojection_threshold_px=2.0,
    )
    assert sampler["category"] == "B_GROUP_DIVERSE_SAMPLING_HEADROOM"


def test_map_correlation_group_resolution_is_fail_closed() -> None:
    state = {
        "anchor_ids": torch.arange(5),
        "anchor_correlation_group_ids": torch.tensor([2, 2, -1, 3, -1]),
    }
    groups = correlation_groups_from_map(state, np.asarray([0, 1, 2, 4]))
    assert groups[0] == groups[1]
    assert groups[2] != groups[3]
    assert np.all(groups >= 0)
    with pytest.raises(ValueError):
        correlation_groups_from_map(state, np.asarray([5]))


def test_real_pnp_hypothesis_builder_is_bounded_and_group_diverse() -> None:
    intrinsic, points, points_2d, groups, _, _ = _oracle_fixture()
    standard, diverse = build_standard_and_group_diverse_hypotheses(
        points_2d,
        points,
        intrinsic,
        groups,
        sample_count=8,
        seed=2026,
    )
    assert standard.ndim == diverse.ndim == 3
    assert standard.shape[1:] == diverse.shape[1:] == (4, 4)
    assert standard.shape[0] <= 32
    assert diverse.shape[0] <= 32
