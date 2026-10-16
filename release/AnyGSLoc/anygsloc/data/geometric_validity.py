"""Valid sampling support for the audited OpenCV same-K rectification path.

Never reads RGB values, semantic labels, masks.pkl, or a segmentation model.
This is not a general rectification adapter: other output camera transforms
must provide a different explicitly validated mapping.
"""

import cv2
import numpy as np
import torch

from anygsloc.data.colmap import (
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


def rectification_support(intrinsic):
    p = intrinsic.params
    if intrinsic.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
        fx = fy = p[0]
        cx, cy = p[1:3]
        distortion = np.array(
            [p[3] if intrinsic.model == "SIMPLE_RADIAL" else 0, 0, 0, 0],
            dtype=np.float32,
        )
    elif intrinsic.model in ("PINHOLE", "OPENCV"):
        fx, fy, cx, cy = p[:4]
        distortion = np.array(
            p[4:] if intrinsic.model == "OPENCV" else [0, 0, 0, 0], dtype=np.float32
        )
    else:
        raise ValueError(f"unsupported rectification model: {intrinsic.model}")
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    w, h = intrinsic.width, intrinsic.height
    if not np.any(distortion):
        return torch.ones((h, w), dtype=torch.bool)
    x, y = cv2.initUndistortRectifyMap(K, distortion, None, K, (w, h), cv2.CV_32FC1)
    # Require the bilinear sampling footprint to remain inside the source image.
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0)
        & (x <= w - 1)
        & (y >= 0)
        & (y <= h - 1)
    )
    return torch.from_numpy(valid)


class SameKRectificationSupport:
    def __init__(self, dataset):
        path = dataset.root / "sparse/0/cameras.bin"
        self.intrinsics = (
            read_intrinsics_binary(str(path))
            if path.exists()
            else read_intrinsics_text(str(path.with_suffix(".txt")))
        )
        images = dataset.root / "sparse/0/images.bin"
        extrinsics = (
            read_extrinsics_binary(str(images))
            if images.exists()
            else read_extrinsics_text(str(images.with_suffix(".txt")))
        )
        self.by_name = {v.name: v.camera_id for v in extrinsics.values()}

    def __call__(self, camera):
        intrinsic = self.intrinsics[self.by_name[camera.image_name]]
        if (camera.width, camera.height) != (intrinsic.width, intrinsic.height):
            raise ValueError(
                "processed image dimensions differ from rectification contract"
            )
        return rectification_support(intrinsic)
