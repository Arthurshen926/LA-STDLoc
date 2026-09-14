import numpy as np
from anygsloc.data.colmap import Camera
from anygsloc.data.geometric_validity import rectification_support


def test_identity_rectification_preserves_black_pixels_too():
    c = Camera(0, 'SIMPLE_PINHOLE', 64, 48, np.array([50.,32.,24.]))
    assert rectification_support(c).all()


def test_positive_radial_distortion_excludes_border_not_center():
    c = Camera(0, 'SIMPLE_RADIAL', 64, 48, np.array([50.,32.,24.,.3]))
    mask = rectification_support(c)
    assert mask[24,32]
    assert not mask[0,0]
    assert 0 < mask.float().mean() < 1
