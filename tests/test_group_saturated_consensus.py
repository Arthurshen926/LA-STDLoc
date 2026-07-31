import numpy as np

from localization_training.group_saturated_consensus import (
    build_surface_component_groups,
    image_cell_normalization,
    saturated_group_support,
    score_pose,
    soft_msac_support,
)
from localization_training.dependency_pose_sampler import (
    GROUP_SATURATED_SOLVER_VERSION,
    compiled_backend_available,
    compiled_group_saturated_solver_version,
    solve_group_saturated_absolute_pose,
)
from scripts.select_lafgs_group_consensus_queries import select_queries
from scripts.summarize_lafgs_group_consensus_oracle import aggregate


def test_group_cap_prevents_repeated_matches_from_dominating():
    correct = np.asarray([1.0, 1.0, 1.0, 1.0])
    repeated_false = np.asarray([0.9] * 8)
    correct_groups = np.arange(4)
    false_groups = np.zeros(8, dtype=np.int64)
    assert repeated_false.sum() > correct.sum()
    assert saturated_group_support(
        correct, correct_groups, cap=2
    ) > saturated_group_support(repeated_false, false_groups, cap=2)


def test_soft_msac_support_is_truncated_quadratic():
    support = soft_msac_support([0.0, 5.0, 10.0, np.inf], threshold=10.0)
    np.testing.assert_allclose(support, [1.0, 0.75, 0.0, 0.0])


def test_image_cell_weights_are_bounded_and_normalized():
    weights = image_cell_normalization([0, 0, 0, 0, 1])
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[0]
    assert weights.min() >= 0.25
    assert weights.max() <= 4.0


def test_surface_components_join_coplanar_neighbors_only():
    xyz = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.6, 0.0, 0.0],
            [1.2, 0.0, 0.0],
            [0.0, 0.0, 0.6],
        ]
    )
    normals = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ]
    )
    groups, diagnostics = build_surface_component_groups(
        xyz,
        normals,
        voxel_size=0.5,
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    assert diagnostics["component_count"] == 2


def test_query_selection_uses_tail_and_stratified_controls():
    names = [f"frame{i:02d}.png" for i in range(20)]
    rows = {
        seed: {
            name: {"te_cm": float(index), "ae_deg": 0.1}
            for index, name in enumerate(names)
        }
        for seed in (1, 2, 3)
    }
    selected = select_queries(
        names,
        rows,
        failure_count=4,
        control_count=5,
    )
    failures = [row for row in selected if row["role"] == "failure"]
    controls = [row for row in selected if row["role"] == "control"]
    assert {row["image_name"] for row in failures} == set(names[-4:])
    assert len(controls) == 5
    assert len({row["image_name"] for row in selected}) == 9


def test_cross_scene_gate_requires_three_scenes_and_safe_greatcourt():
    def report(scene, failure_delta, recovered, regressed):
        summary = {
            "all": {
                "recovered_count": recovered,
                "regressed_count": regressed,
            },
            "failure": {"win_rate_delta": failure_delta},
            "control": {
                "win_rate_delta": 0.0,
                "recovered_count": 0,
                "regressed_count": 0,
            },
        }
        return {"scene": scene, "variant_summary": {"candidate": summary}}

    payload = aggregate(
        [
            report("KingsCollege", 0.1, 2, 0),
            report("ShopFacade", 0.1, 1, 0),
            report("StMarysChurch", 0.1, 2, 1),
            report("GreatCourt", 0.0, 0, 0),
        ]
    )
    assert payload["gate_pass"]


def test_compiled_group_solver_matches_python_score_and_recovers_pose():
    if not compiled_backend_available():
        return
    assert (
        compiled_group_saturated_solver_version()
        == GROUP_SATURATED_SOLVER_VERSION
    )
    rng = np.random.default_rng(12)
    K = np.asarray(
        [[700.0, 0.0, 400.0], [0.0, 700.0, 300.0], [0.0, 0.0, 1.0]]
    )
    xyz = np.column_stack(
        (
            rng.uniform(-3.0, 3.0, 120),
            rng.uniform(-2.0, 2.0, 120),
            rng.uniform(5.0, 15.0, 120),
        )
    )
    points2d = np.column_stack(
        (
            700.0 * xyz[:, 0] / xyz[:, 2] + 400.0,
            700.0 * xyz[:, 1] / xyz[:, 2] + 300.0,
        )
    )
    points2d += rng.normal(0.0, 0.4, points2d.shape)
    surface_groups = np.repeat(np.arange(30), 4)
    pose, inliers, diagnostics = solve_group_saturated_absolute_pose(
        points2d,
        xyz,
        K,
        surface_groups=surface_groups,
        group_cap=8.0,
        reprojection_error=4.0,
        max_iterations=1000,
        min_iterations=100,
        seed=3,
    )
    score, _, _ = score_pose(
        points2d,
        xyz,
        K,
        pose,
        4.0,
        group_ids=surface_groups,
        cap=8.0,
    )
    np.testing.assert_allclose(
        diagnostics["group_score"], score, rtol=0.0, atol=1e-6
    )
    assert (
        diagnostics["implementation_version"]
        == "group_saturated_poselib_parity_v2"
    )
    assert diagnostics["dynamic_trial_multiplier"] == 3.0
    assert diagnostics["local_refinements"] > 0
    assert inliers.size >= 110
    np.testing.assert_allclose(pose, np.eye(4), atol=5e-3)


def test_progressive_dependency_order_keeps_metadata_and_inliers_aligned(
    monkeypatch,
):
    import localization_training.dependency_pose_sampler as sampler
    from utils.pose_utils import solve_pose

    def fake_solver(points2d, points3d, K, **kwargs):
        np.testing.assert_array_equal(
            kwargs["dependency_groups"], [11, 13, 12, 10]
        )
        np.testing.assert_array_equal(
            kwargs["surface_groups"], [21, 23, 22, 20]
        )
        return np.eye(4), np.asarray([0]), {
            "iterations": 1,
            "diverse_samples": 1,
            "fallback_samples": 0,
            "local_refinements": 0,
            "rescue_used": False,
            "backend": "test",
        }

    monkeypatch.setattr(
        sampler, "solve_dependency_absolute_pose", fake_solver
    )
    _, inliers = solve_pose(
        np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float64),
        np.asarray(
            [[0, 0, 4], [1, 0, 4], [0, 1, 4], [1, 1, 4]],
            dtype=np.float64,
        ),
        np.eye(3),
        solver="poselib_dependency",
        scores=np.asarray([0.0, 3.0, 1.0, 2.0]),
        progressive_sampling=True,
        dependency_groups=np.asarray([10, 11, 12, 13]),
        image_cells=np.asarray([0, 1, 2, 3]),
        surface_groups=np.asarray([20, 21, 22, 23]),
    )
    np.testing.assert_array_equal(inliers, [1])
