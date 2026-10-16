"""Frozen mapping-camera-derived query density and angular PnP threshold."""

import math


def resolve_scene_calibration(cameras):
    cameras = list(cameras)
    if not cameras:
        raise ValueError("mapping cameras are required; no test-derived fallback")
    areas = sorted(int(c.width) * int(c.height) for c in cameras)
    keypoints = max(1024, min(2048, round(2048 * areas[len(areas) // 2] / 2073600.0)))
    focal = sorted(
        math.sqrt(
            (float(c.width) / (2.0 * math.tan(float(c.fov_x) / 2.0)))
            * (float(c.height) / (2.0 * math.tan(float(c.fov_y) / 2.0)))
        )
        for c in cameras
    )
    return dict(
        keypoint_count=keypoints,
        reprojection_error_px=max(2.0, 0.007176912978323976 * focal[len(focal) // 2]),
        mapping_camera_count=len(cameras),
        source="mapping_camera_density_and_focal_v1",
    )
