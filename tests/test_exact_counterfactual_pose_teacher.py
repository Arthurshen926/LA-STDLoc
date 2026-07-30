import numpy as np

from localization_training.exact_counterfactual_pose_teacher import (
    ExactCounterfactualConfig,
    improves_lexicographically,
    outcome_order_key,
    solve_counterfactual_pose,
)


def _outcome(**updates):
    value = {
        "valid": True,
        "correct_basin": True,
        "strict_translation_success": False,
        "translation_error_cm": 8.0,
        "rotation_error_degrees": 1.0,
        "harmful_consensus_count": 2,
        "geometry_diversity": 5.0,
    }
    value.update(updates)
    return value


def test_lexicographic_target_prioritizes_strict_crossing_before_te():
    baseline = _outcome()
    crossing = _outcome(
        strict_translation_success=True,
        translation_error_cm=4.9,
        rotation_error_degrees=4.0,
        harmful_consensus_count=8,
    )
    lower_te_without_crossing = _outcome(translation_error_cm=6.0)
    assert improves_lexicographically(crossing, baseline)
    assert outcome_order_key(crossing) > outcome_order_key(
        lower_te_without_crossing
    )


def test_lexicographic_target_uses_harmful_consensus_after_pose_error():
    baseline = _outcome()
    cleaner = _outcome(harmful_consensus_count=1)
    assert improves_lexicographically(cleaner, baseline)


def test_fixed_seed_poselib_counterfactual_is_deterministic():
    points3d = np.asarray(
        [
            [-1.0, -0.8, 5.0],
            [1.2, -0.7, 5.5],
            [-0.9, 0.9, 6.0],
            [1.0, 0.8, 6.5],
            [0.1, -1.2, 7.0],
            [-1.3, 0.2, 7.5],
            [1.4, 0.1, 8.0],
            [0.0, 1.3, 8.5],
        ],
        dtype=np.float64,
    )
    intrinsics = np.asarray(
        [[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]
    )
    points2d = np.stack(
        (
            intrinsics[0, 0] * points3d[:, 0] / points3d[:, 2]
            + intrinsics[0, 2],
            intrinsics[1, 1] * points3d[:, 1] / points3d[:, 2]
            + intrinsics[1, 2],
        ),
        axis=1,
    )
    config = ExactCounterfactualConfig(
        maximum_iterations=1000,
        minimum_iterations=10,
    )
    kwargs = {
        "points2d": points2d,
        "points3d": points3d,
        "intrinsics": intrinsics,
        "ground_truth_w2c": np.eye(4),
        "dependency_groups": np.arange(len(points3d)),
        "source_groups": np.arange(len(points3d)),
        "config": config,
    }
    first = solve_counterfactual_pose(**kwargs)
    second = solve_counterfactual_pose(**kwargs)
    assert first["valid"]
    assert first["inlier_count"] == second["inlier_count"]
    assert first["hypotheses"] == second["hypotheses"]
    assert np.isclose(
        first["translation_error_cm"],
        second["translation_error_cm"],
        atol=1e-8,
    )
