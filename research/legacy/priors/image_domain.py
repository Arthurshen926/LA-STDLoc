"""Explicit image-domain conversion for external Gaussian prior inputs."""
import cv2
import numpy as np


def resolve_image_domain(distortion, domain):
    if domain not in {None, 'rectified', 'distorted'}:
        raise ValueError('input_image_domain must be rectified or distorted')
    if domain is None:
        if np.any(np.asarray(distortion) != 0):
            raise ValueError('nonzero camera distortion does not identify the image domain; explicitly set input_image_domain=rectified for already corrected pixels or distorted for raw pixels')
        return 'rectified'
    return domain


def prepare_pixels(image, matrix, distortion, *, domain=None, width=None, height=None):
    """COLMAP half-center calibration, OpenCV grid conversion, then area resize.

    Already rectified pixels are never undistorted again. Exact image extents,
    not a nominal downscale factor, determine resized intrinsics.
    """
    domain = resolve_image_domain(distortion, domain)
    h, w = image.shape[:2]
    width, height = int(w if width is None else width), int(h if height is None else height)
    if width <= 0 or height <= 0: raise ValueError('positive target extent required')
    K = np.asarray(matrix, dtype=np.float64)
    if K.shape != (3,3) or not np.isfinite(K).all() or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError('invalid intrinsics')
    prepared = image
    if domain == 'distorted' and np.any(np.asarray(distortion) != 0):
        opencv_K = K.copy(); opencv_K[:2,2] -= .5
        mx, my = cv2.initUndistortRectifyMap(opencv_K, np.asarray(distortion), None, opencv_K, (w,h), cv2.CV_32FC1)
        prepared = cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    if (width,height) != (w,h):
        prepared = cv2.resize(prepared,(width,height),interpolation=cv2.INTER_AREA)
    target = K.copy(); target[0] *= width/w; target[1] *= height/h
    return prepared, target
