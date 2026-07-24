import numpy as np
import pytest
import torch

from localization_training.hard_candidate_teacher import (
    HardCandidateTeacherCache,
    _batched_camera_centers_after_left_updates,
    derive_hard_candidate_targets,
    hard_candidate_preservation_loss,
)


def _teacher_inputs():
    K = torch.tensor(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]
    )
    xyz = torch.tensor(
        [
            [-0.5, -0.2, 5.0],
            [0.4, -0.2, 5.0],
            [-0.3, 0.3, 5.0],
            [0.3, 0.3, 5.0],
            [0.1, 0.1, 5.0],
            [0.2, -0.1, 5.0],
        ]
    )
    projected = torch.stack(
        [
            K[0, 0] * xyz[:, 0] / xyz[:, 2] + K[0, 2],
            K[1, 1] * xyz[:, 1] / xyz[:, 2] + K[1, 2],
        ],
        dim=1,
    )
    return {
        "keypoint_xy": projected - 0.5,
        "keypoint_ids": torch.arange(xyz.shape[0]),
        "candidate_keypoint_idx": torch.arange(xyz.shape[0]),
        "candidate_landmark_idx": torch.arange(xyz.shape[0]),
        "candidate_scores": torch.tensor([0.9, 0.8, 0.7, 0.6, 0.95, 0.1]),
        "deployment_mask": torch.ones(xyz.shape[0], dtype=torch.bool),
        "gt_correct_mask": torch.tensor([True, True, True, True, False, False]),
        "landmark_xyz": xyz,
        "K": K,
        "pose_gt_w2c": torch.eye(4),
    }


def _fake_solver(p2d, p3d, K, **kwargs):
    del p2d, p3d, K, kwargs
    return np.eye(4, dtype=np.float64), np.array([0, 1, 2, 3, 4], dtype=np.int64)


def test_discrete_teacher_labels_useful_and_harmful_ransac_inliers():
    targets = derive_hard_candidate_targets(
        **_teacher_inputs(),
        max_useful=8,
        max_harmful=8,
        pose_solver=_fake_solver,
    )
    assert targets.useful_mask.tolist() == [True, True, True, True, False, False]
    assert targets.harmful_mask.tolist() == [False, False, False, False, True, False]
    assert targets.diagnostics["hard_teacher_ransac_inlier_count"] == 5.0
    assert targets.diagnostics["hard_teacher_ransac_inlier_gt_precision"] == pytest.approx(0.8)
    assert targets.diagnostics["hard_teacher_valid"] == 1.0


def test_discrete_teacher_keeps_measurement_band_neutral():
    inputs = _teacher_inputs()
    inputs["gt_neutral_mask"] = torch.tensor(
        [False, False, False, False, True, False]
    )
    targets = derive_hard_candidate_targets(
        **inputs,
        max_useful=8,
        max_harmful=8,
        pose_solver=_fake_solver,
    )
    assert not bool(targets.harmful_mask.any())
    assert targets.diagnostics["hard_teacher_neutral_ransac_inlier_count"] == 1.0


def test_protected_useful_set_enforces_surface_group_cap():
    targets = derive_hard_candidate_targets(
        **_teacher_inputs(),
        max_useful=8,
        max_harmful=8,
        useful_grid_rows=2,
        useful_grid_cols=2,
        useful_depth_bins=2,
        useful_surface_voxel_m=100.0,
        useful_max_per_surface_group=1,
        pose_solver=_fake_solver,
    )
    assert int(targets.useful_mask.sum().item()) == 1
    assert (
        targets.diagnostics["hard_teacher_protected_max_per_surface_group"]
        == 1.0
    )


def test_hard_preservation_loss_moves_useful_up_and_harmful_down():
    targets = derive_hard_candidate_targets(
        **_teacher_inputs(),
        max_useful=8,
        max_harmful=8,
        pose_solver=_fake_solver,
    )
    logits = torch.zeros(6, requires_grad=True)
    loss, diagnostics = hard_candidate_preservation_loss(
        logits, targets, temperature=0.1, margin=0.05, score_target=0.5
    )
    loss.backward()
    assert diagnostics["hard_teacher_loss_pairwise"] > 0.0
    assert logits.grad[0] < 0.0
    assert logits.grad[4] > 0.0


def test_exact_hard_loss_skips_useful_only_updates_without_harmful_edge():
    targets = derive_hard_candidate_targets(
        **_teacher_inputs(),
        max_useful=8,
        max_harmful=8,
        harmful_mode="translation_delete",
        harmful_min_translation_delete_gain_m=1e6,
        pose_solver=_fake_solver,
    )
    assert not bool(targets.harmful_mask.any())
    logits = torch.zeros(6, requires_grad=True)
    loss, diagnostics = hard_candidate_preservation_loss(
        logits,
        targets,
        require_harmful=True,
    )
    loss.backward()

    assert diagnostics["hard_teacher_loss_skipped_no_harmful"] == 1.0
    assert loss.item() == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_hard_teacher_cache_refreshes_by_visit_and_remaps_edge_keys():
    calls = []

    def counted_solver(*args, **kwargs):
        calls.append(1)
        return _fake_solver(*args, **kwargs)

    inputs = _teacher_inputs()
    cache = HardCandidateTeacherCache(
        refresh_visits=2,
        max_useful=8,
        max_harmful=8,
        pose_solver=counted_solver,
    )
    first = cache.build("query", **inputs)
    second = cache.build("query", **inputs)
    third = cache.build("query", **inputs)
    assert len(calls) == 2
    assert first.useful_mask.tolist() == second.useful_mask.tolist()
    assert third.diagnostics["hard_teacher_refreshed"] == 1.0
    assert cache.diagnostics()["hard_teacher_cache_hits"] == 1.0


def test_hard_teacher_cache_refreshes_every_visit_when_requested():
    calls = []

    def counted_solver(*args, **kwargs):
        calls.append(1)
        return _fake_solver(*args, **kwargs)

    cache = HardCandidateTeacherCache(
        refresh_visits=1,
        max_useful=8,
        max_harmful=8,
        pose_solver=counted_solver,
    )
    inputs = _teacher_inputs()
    cache.build("query", **inputs)
    cache.build("query", **inputs)
    assert len(calls) == 2
    assert cache.diagnostics()["hard_teacher_cache_hits"] == 0.0


def test_translation_delete_harmful_teacher_filters_false_consensus_by_bias_gain():
    inputs = _teacher_inputs()
    xyz = torch.tensor(
        [
            [-0.5, -0.2, 4.0],
            [0.4, -0.2, 5.0],
            [-0.3, 0.3, 6.0],
            [0.3, 0.3, 7.0],
            [0.1, 0.1, 5.0],
            [0.2, -0.1, 6.0],
        ]
    )
    K = inputs["K"]
    projected = torch.stack(
        [
            K[0, 0] * xyz[:, 0] / xyz[:, 2] + K[0, 2],
            K[1, 1] * xyz[:, 1] / xyz[:, 2] + K[1, 2],
        ],
        dim=1,
    )
    inputs["landmark_xyz"] = xyz
    inputs["keypoint_xy"] = projected - 0.5
    # The fake RANSAC solver accepts this GT-wrong edge.  Removing it should
    # reduce the local translation bias, so it is retained as a hard negative.
    inputs["keypoint_xy"][4] += torch.tensor([12.0, -8.0])

    def solver_with_clean_false_inlier(p2d, p3d, K, **kwargs):
        del p2d, p3d, K, kwargs
        # Index 5 is also labelled false, but has a clean correspondence and
        # must not be punished merely because RANSAC accepted it.
        return np.eye(4, dtype=np.float64), np.arange(6, dtype=np.int64)

    targets = derive_hard_candidate_targets(
        **inputs,
        max_useful=8,
        max_harmful=8,
        harmful_mode="translation_delete",
        pose_solver=solver_with_clean_false_inlier,
    )
    assert targets.harmful_mask.tolist() == [False, False, False, False, True, False]
    assert targets.diagnostics["hard_teacher_harmful_translation_delete_evaluable"] == 1.0
    assert targets.diagnostics["hard_teacher_harmful_bias_improving_count"] == 1.0
    assert targets.diagnostics["hard_teacher_harmful_translation_delete_gain_mean"] > 0.0
    assert (
        targets.diagnostics[
            "hard_teacher_selected_harmful_translation_delete_gain_mean"
        ]
        > 0.0
    )


def test_exact_pose_delete_labels_only_false_inlier_with_actual_pose_gain():
    inputs = _teacher_inputs()

    def replay_solver(p2d, p3d, K, **kwargs):
        del p2d, K, kwargs
        pose = np.eye(4, dtype=np.float64)
        # Landmark four is the false inlier. Its removal changes the actual
        # PnP replay from a 10 cm pose to the GT pose.
        if np.any(np.isclose(p3d[:, 0], 0.1)):
            pose[0, 3] = 0.1
        return pose, np.arange(p3d.shape[0], dtype=np.int64)

    targets = derive_hard_candidate_targets(
        **inputs,
        max_useful=8,
        max_harmful=8,
        harmful_mode="exact_pose_delete",
        exact_replay_max_candidates=8,
        exact_replay_min_pose_gain_cm=1.0,
        exact_replay_selection_threshold=-float("inf"),
        pose_solver=replay_solver,
    )

    assert targets.harmful_mask.tolist() == [False, False, False, False, True, False]
    assert targets.diagnostics["hard_teacher_exact_replay_graph_aligned"] == 1.0
    assert targets.diagnostics["hard_teacher_exact_replay_evaluated_count"] == 2.0
    assert targets.diagnostics["hard_teacher_exact_replay_positive_count"] == 1.0
    assert targets.diagnostics["hard_teacher_exact_replay_pose_gain_cm_max"] == pytest.approx(10.0)


def test_exact_pose_delete_replays_quota_refill_before_pnp():
    inputs = _teacher_inputs()
    inputs.update(
        {
            "candidate_landmark_idx": torch.tensor([0, 0, 1, 2, 3, 4]),
            "candidate_scores": torch.tensor([0.99, 0.80, 0.70, 0.60, 0.50, 0.40]),
            "deployment_mask": torch.tensor([True, False, True, True, True, True]),
            "gt_correct_mask": torch.tensor([False, True, True, True, True, True]),
        }
    )
    first_keypoint_x = float(inputs["keypoint_xy"][0, 0].item() + 0.5)

    def refill_solver(p2d, p3d, K, **kwargs):
        del p3d, K, kwargs
        pose = np.eye(4, dtype=np.float64)
        # The original top-scoring duplicate produces a bad pose. Replaying
        # its deletion must admit the second duplicate before PnP is solved.
        if np.any(np.isclose(p2d[:, 0], first_keypoint_x)):
            pose[0, 3] = 0.1
        return pose, np.arange(p2d.shape[0], dtype=np.int64)

    targets = derive_hard_candidate_targets(
        **inputs,
        max_useful=8,
        max_harmful=8,
        harmful_mode="exact_pose_delete",
        exact_replay_max_candidates=8,
        exact_replay_min_pose_gain_cm=1.0,
        exact_replay_selection_threshold=-float("inf"),
        exact_replay_max_matches_per_landmark=1,
        pose_solver=refill_solver,
    )

    assert targets.harmful_mask.tolist() == [True, False, False, False, False, False]
    assert targets.diagnostics["hard_teacher_exact_replay_graph_aligned"] == 1.0
    assert targets.diagnostics["hard_teacher_exact_replay_refill_count"] == 1.0


def test_batched_left_update_camera_centers_match_se3_exp():
    from localization_training.pose_refiner import camera_center_from_w2c, se3_exp

    dtype = torch.float64
    pose = se3_exp(
        torch.tensor([0.3, -0.2, 0.4, 0.02, -0.03, 0.01], dtype=dtype)
    )
    updates = torch.tensor(
        [
            [0.01, -0.02, 0.03, 0.001, -0.002, 0.003],
            [-0.04, 0.02, 0.01, -0.04, 0.03, -0.02],
        ],
        dtype=dtype,
    )
    expected = torch.stack(
        [camera_center_from_w2c(se3_exp(update) @ pose) for update in updates]
    )
    actual = _batched_camera_centers_after_left_updates(pose, updates)
    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)
