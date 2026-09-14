import numpy as np

from topology.reserve_geometry_refinement import (
    fit_surface_bounded_point,
    quaternion_wxyz_to_matrix,
    temporal_threeway_split,
)


def test_quaternion_identity_uses_wxyz_convention():
    assert np.allclose(
        quaternion_wxyz_to_matrix(np.asarray([1.0, 0.0, 0.0, 0.0])),
        np.eye(3),
    )


def test_temporal_threeway_split_is_disjoint_and_complete():
    names = [f"seq/frame{index:05d}.png" for index in range(18)]
    fit, validation, gate, report = temporal_threeway_split(names, block_count=9)
    assert not (set(fit) & set(validation))
    assert not (set(fit) & set(gate))
    assert not (set(validation) & set(gate))
    assert sorted(fit + validation + gate) == list(range(18))
    assert report["uses_test_queries"] is False


def test_surface_bounded_point_fit_recovers_point_without_leaving_disk():
    true = np.asarray([0.025, -0.015, 2.0])
    initial = np.asarray([0.0, 0.0, 2.0])
    poses = []
    intrinsics = []
    observations = []
    K = np.asarray([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    for center_x in (-0.2, 0.0, 0.2, 0.4):
        pose = np.eye(4)
        pose[0, 3] = -center_x
        camera = pose[:3, :3] @ true + pose[:3, 3]
        uv_h = K @ camera
        poses.append(pose)
        intrinsics.append(K)
        observations.append(uv_h[:2] / uv_h[2])
    point, covariance, report = fit_surface_bounded_point(
        initial_xyz=initial,
        surface_center=initial,
        surface_basis=np.eye(3),
        local_bounds=np.asarray([0.05, 0.05, 0.005]),
        poses_w2c=np.stack(poses),
        intrinsics=np.stack(intrinsics),
        observations_uv=np.stack(observations),
        prior_sigma=np.asarray([0.02, 0.02, 0.002]),
        huber_px=2.0,
    )
    assert report["success"]
    assert np.linalg.norm(point[:2] - true[:2]) < 0.003
    assert abs(point[2] - 2.0) <= 0.005
    assert covariance.shape == (3, 3)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-9
