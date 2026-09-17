import importlib.util
from pathlib import Path
import numpy as np

spec = importlib.util.spec_from_file_location('center_control',
    Path(__file__).resolve().parents[1] / 'tools/gaussian_center_control.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_front_center_wins_pixel_collision():
    xyz = np.array([[1., 1., 2.], [2., 2., 4.], [0., 0., -1.]], dtype=np.float32)
    result = module.projected_centers(xyz, np.eye(3), np.eye(4), 10, 10,
                                      np.array([[0., 0.], [9., 9.]]))
    np.testing.assert_allclose(result[0], xyz[0])
    assert np.isnan(result[1]).all()


def test_projection_uses_world_to_camera_and_half_pixel():
    xyz = np.array([[0., 0., 2.]], dtype=np.float32)
    pose = np.eye(4)
    pose[0, 3] = 1.
    K = np.array([[2., 0., 3.], [0., 2., 3.], [0., 0., 1.]])
    result = module.projected_centers(xyz, K, pose, 10, 10, np.array([[3.5, 2.5]]), .01)
    np.testing.assert_allclose(result[0], xyz[0])
