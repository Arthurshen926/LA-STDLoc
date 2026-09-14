import cv2
import numpy as np
import pytest
from priors.image_domain import prepare_pixels


def fixture():
    return np.random.default_rng(4).integers(0,256,(57,111,3),dtype=np.uint8), np.array([[100.,0,55.5],[0,100.,28.5],[0,0,1.]]), np.array([.05,0,0,0.])


def test_nonzero_distortion_requires_explicit_domain():
    with pytest.raises(ValueError,match='image domain'): prepare_pixels(*fixture())


def test_rectified_pixels_are_not_undistorted_again():
    pixels,K,d = fixture(); out,newK = prepare_pixels(pixels,K,d,domain='rectified')
    assert np.array_equal(out,pixels)
    assert np.array_equal(newK,K)


def test_resized_intrinsics_follow_actual_rounded_extent():
    pixels,K,d = fixture(); out,newK = prepare_pixels(pixels,K,d,domain='rectified',width=56,height=28)
    expected=K.copy(); expected[0]*=56/111; expected[1]*=28/57
    assert np.array_equal(newK,expected)
    assert np.array_equal(out,cv2.resize(pixels,(56,28),interpolation=cv2.INTER_AREA))
    assert newK[0,2] == pytest.approx(28) and newK[1,2] == pytest.approx(14)


def test_raw_rectification_obeys_half_pixel_contract():
    pixels,K,d = fixture(); out,newK = prepare_pixels(pixels,K,d,domain='distorted')
    opencvK=K.copy();opencvK[:2,2]-=.5
    mx,my=cv2.initUndistortRectifyMap(opencvK,d,None,opencvK,(111,57),cv2.CV_32FC1)
    assert np.array_equal(out,cv2.remap(pixels,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
    assert np.array_equal(newK,K)


@pytest.mark.parametrize('extent', [{'width':0}, {'height':0}, {'width':-1}, {'height':-1}])
def test_nonpositive_explicit_extent_is_not_silently_defaulted(extent):
    with pytest.raises(ValueError, match='positive target extent'):
        prepare_pixels(*fixture(), domain='rectified', **extent)
