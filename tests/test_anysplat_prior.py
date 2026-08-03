import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from prior_reconstruction.anysplat import (
    SimilarityTransform,
    colmap_qvec_to_rotation,
    covariance_to_scale_rotation,
    fit_similarity_robust,
    fit_similarity_from_camera_poses,
    probability_to_logit,
    select_trajectory_windows,
    spatial_confidence_coreset,
    transform_gaussian_moments,
    write_graphdeco_dc_ply,
)


def test_mapping_window_selection_is_trajectory_complete_and_deterministic():
    names = [f"seq1/frame{i:05d}.png" for i in range(8)] + [
        f"seq2/frame{i:05d}.png" for i in range(5)
    ]
    centers = {
        name: np.array([index, int(name[3]), 0.0])
        for index, name in enumerate(names)
    }
    first = select_trajectory_windows(names, centers, 3)
    second = select_trajectory_windows(reversed(names), centers, 3)
    assert first == second
    assert [window["trajectory"] for window in first] == ["seq1", "seq2"]
    assert [window["selected_view_count"] for window in first] == [3, 3]

    segmented = select_trajectory_windows(names, centers, 3, segment_size=4)
    assert [window["window_id"] for window in segmented] == [
        "seq1_000", "seq1_001", "seq2_000", "seq2_001"
    ]
    assert [window["available_view_count"] for window in segmented] == [4, 4, 4, 1]

    complete = select_trajectory_windows(
        names[:8], centers, 4, segment_size=4, complete_coverage=True
    )
    selected = [name for window in complete for name in window["image_names"]]
    assert [window["selected_view_count"] for window in complete] == [4, 4]
    assert sorted(selected) == sorted(names[:8])


def test_colmap_quaternion_conversion_uses_wxyz_convention():
    qvec = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])
    expected = Rotation.from_euler("z", np.pi / 2).as_matrix()
    assert np.allclose(colmap_qvec_to_rotation(qvec), expected)


def test_robust_sim3_recovers_mapping_frame_with_one_outlier():
    rng = np.random.default_rng(3)
    source = rng.normal(size=(20, 3))
    rotation = Rotation.from_euler("xyz", [0.3, -0.2, 0.4]).as_matrix()
    expected = SimilarityTransform(2.5, rotation, np.array([4.0, -3.0, 1.0]))
    target = expected.transform_points(source)
    target[-1] += 20.0
    fitted, inliers, residuals = fit_similarity_robust(source, target)
    assert inliers.sum() == 19
    assert np.allclose(fitted.scale, expected.scale, atol=1e-8)
    assert np.allclose(fitted.rotation, expected.rotation, atol=1e-8)
    assert np.allclose(fitted.translation, expected.translation, atol=1e-8)
    assert residuals[:-1].max() < 1e-8


def test_pose_sim3_uses_orientations_to_resolve_collinear_camera_path():
    source = np.stack((np.linspace(0, 5, 12), np.zeros(12), np.zeros(12)), axis=1)
    global_rotation = Rotation.from_euler("xyz", [0.6, -0.2, 0.4]).as_matrix()
    source_rotation = Rotation.from_euler(
        "z", np.linspace(-0.4, 0.4, 12)
    ).as_matrix()
    target_rotation = global_rotation[None] @ source_rotation
    expected = SimilarityTransform(3.0, global_rotation, np.array([2.0, -1.0, 4.0]))
    target = expected.transform_points(source)
    fitted, center_inliers, center_residual, orientation_inliers, rotation_residual = (
        fit_similarity_from_camera_poses(
            source,
            target,
            source_rotation,
            target_rotation,
        )
    )
    assert center_inliers.all()
    assert orientation_inliers.all()
    assert center_residual.max() < 1e-8
    assert rotation_residual.max() < 1e-8
    assert np.allclose(fitted.rotation, global_rotation, atol=1e-8)


def test_covariance_transform_and_graphdeco_factorization_are_equivalent():
    rng = np.random.default_rng(4)
    rotations = Rotation.random(12, random_state=rng).as_matrix()
    scales = rng.uniform(0.01, 0.2, size=(12, 3))
    covariance = rotations @ (np.eye(3)[None] * scales[:, None, :] ** 2) @ np.swapaxes(rotations, 1, 2)
    transform = SimilarityTransform(
        3.0,
        Rotation.from_euler("z", 0.7).as_matrix(),
        np.array([1.0, 2.0, 3.0]),
    )
    _, transformed = transform_gaussian_moments(np.zeros((12, 3)), covariance, transform)
    log_scales, quaternion = covariance_to_scale_rotation(transformed)
    recovered_rotation = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]]).as_matrix()
    recovered = recovered_rotation @ (np.eye(3)[None] * np.exp(log_scales)[:, None, :] ** 2) @ np.swapaxes(recovered_rotation, 1, 2)
    assert np.allclose(recovered, transformed, atol=1e-8)


def test_ply_converts_probability_opacity_to_graphdeco_logit(tmp_path):
    path = tmp_path / "point_cloud.ply"
    means = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    covariance = np.diag([0.01, 0.04, 0.09])[None]
    write_graphdeco_dc_ply(path, means, covariance, np.zeros((1, 3)), np.array([0.2]))
    vertex = PlyData.read(path)["vertex"].data
    assert np.allclose(vertex["opacity"], probability_to_logit(np.array([0.2])))
    assert set(vertex.dtype.names) == {
        "x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
        "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
    }


def test_spatial_confidence_coreset_is_deterministic_and_budgeted():
    means = np.stack((np.arange(20), np.zeros(20), np.zeros(20)), axis=1)
    covariance = np.repeat(np.eye(3)[None] * 0.01, 20, axis=0)
    opacity = np.linspace(0.1, 0.9, 20)
    first = spatial_confidence_coreset(means, covariance, opacity, 7)
    second = spatial_confidence_coreset(means, covariance, opacity, 7)
    assert len(first) == 7
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 7
