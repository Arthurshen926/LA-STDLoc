from __future__ import annotations

import numpy as np

from visualization.paper_figures import _project, _quaternion_matrix
from visualization.publication import PCAProjection, robust_limits


def test_pca_projection_is_deterministic_and_centered() -> None:
    xyz = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0]]
    )
    first = PCAProjection.fit(xyz)
    second = PCAProjection.fit(xyz)
    assert np.allclose(first.axes, second.axes)
    assert np.allclose(first.transform(first.center[None]), 0.0)


def test_projection_uses_world_to_camera_and_pixel_intrinsics() -> None:
    points = np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]])
    intrinsic = np.asarray([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0, 0, 1]])
    projected = _project(points, intrinsic, np.eye(4))
    assert np.allclose(projected, [[50.0, 40.0], [100.0, 40.0]])


def test_quaternion_identity_and_robust_limits() -> None:
    assert np.allclose(_quaternion_matrix(np.asarray([1.0, 0.0, 0.0, 0.0])), np.eye(3))
    low, high = robust_limits(np.asarray([[0.0, 0.0], [1.0, 2.0]]), percent=0.0)
    assert np.all(low < [0.0, 0.0])
    assert np.all(high > [1.0, 2.0])
