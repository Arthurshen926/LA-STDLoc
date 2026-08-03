"""Camera tensors shared by prior rasterization and feature extraction."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from data.datasets import CameraRecord, ColmapDataset


def _world_to_view(
    rotation_c2w: np.ndarray,
    translation_w2c: np.ndarray,
    translate: np.ndarray | None = None,
    scale: float = 1.0,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation_c2w).T
    transform[:3, 3] = np.asarray(translation_w2c)
    camera_to_world = np.linalg.inv(transform)
    offset = np.zeros(3) if translate is None else np.asarray(translate)
    camera_to_world[:3, 3] = (camera_to_world[:3, 3] + offset) * float(scale)
    return np.linalg.inv(camera_to_world).astype(np.float32)


def _projection(z_near: float, z_far: float, fov_x: float, fov_y: float) -> torch.Tensor:
    tan_y = math.tan(fov_y / 2.0)
    tan_x = math.tan(fov_x / 2.0)
    top, right = tan_y * z_near, tan_x * z_near
    matrix = torch.zeros((4, 4), dtype=torch.float32)
    matrix[0, 0] = z_near / right
    matrix[1, 1] = z_near / top
    matrix[3, 2] = 1.0
    matrix[2, 2] = z_far / (z_far - z_near)
    matrix[2, 3] = -(z_far * z_near) / (z_far - z_near)
    return matrix


class Camera(nn.Module):
    """GraphDeco-compatible camera surface without scene-training behavior."""

    def __init__(
        self,
        record: CameraRecord,
        image: torch.Tensor,
        *,
        uid: int,
        data_device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.uid = int(uid)
        self.colmap_id = int(uid)
        self.R = np.asarray(record.rotation_c2w)
        self.T = np.asarray(record.translation_w2c)
        self.FoVx = float(record.fov_x)
        self.FoVy = float(record.fov_y)
        self.image_name = record.image_name
        self.data_device = torch.device(data_device)
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_height = int(self.original_image.shape[1])
        self.image_width = int(self.original_image.shape[2])
        self.znear = 0.01
        self.zfar = 100.0
        self.world_view_transform = torch.from_numpy(
            _world_to_view(self.R, self.T)
        ).transpose(0, 1)
        self.projection_matrix = _projection(
            self.znear, self.zfar, self.FoVx, self.FoVy
        ).transpose(0, 1)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def load_camera(
    dataset: ColmapDataset,
    record: CameraRecord,
    *,
    uid: int,
    resolution: int | float = 1,
    data_device: str | torch.device = "cpu",
) -> Camera:
    image = dataset.load_image(record)
    if float(resolution) != 1.0:
        if float(resolution) <= 0.0:
            raise ValueError("Only positive explicit resolution scales are supported")
        height = round(image.shape[1] / float(resolution))
        width = round(image.shape[2] / float(resolution))
        image = torch.nn.functional.interpolate(
            image[None], size=(height, width), mode="bilinear", align_corners=False
        )[0]
    return Camera(record, image, uid=uid, data_device=data_device)
