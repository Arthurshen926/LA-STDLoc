"""Frozen mainline kernel extracted from data.datasets."""

from __future__ import annotations


from dataclasses import dataclass


import math


from pathlib import Path


import pickle


import numpy as np


from PIL import Image


import torch


import torch.nn.functional as F


from anygsloc.data.colmap import (
    qvec2rotmat,
    read_extrinsics_binary,
    read_extrinsics_text,
    read_intrinsics_binary,
    read_intrinsics_text,
)


def _focal_to_fov(focal: float, pixels: int) -> float:
    if not math.isfinite(float(focal)) or float(focal) <= 0 or int(pixels) <= 0:
        raise ValueError(
            "camera focal length and image extent must be finite and positive"
        )
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def validate_centered_camera(intrinsic) -> None:
    """The current FOV-only raster/PnP interface cannot retain a shifted center.

    Reject incompatible calibration instead of silently localizing with another
    camera. This does not rectify pixels or infer that distorted RGB is valid.
    Legacy Cambridge processed RGB has a separately audited rectification path.
    """
    params = np.asarray(intrinsic.params, dtype=np.float64)
    if intrinsic.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
        expected = 3 if intrinsic.model == "SIMPLE_PINHOLE" else 4
        principal = params[1:3]
    elif intrinsic.model in {"PINHOLE", "OPENCV"}:
        expected = 4 if intrinsic.model == "PINHOLE" else 8
        principal = params[2:4]
    else:
        raise ValueError(f"unsupported camera model: {intrinsic.model}")
    if params.shape != (expected,) or not np.isfinite(params).all():
        raise ValueError("camera parameters have an invalid shape or nonfinite values")
    center = np.array([intrinsic.width / 2.0, intrinsic.height / 2.0])
    if not np.allclose(principal, center, atol=1e-6, rtol=0):
        raise ValueError(
            "non-centered principal point is unsupported by the FOV-only renderer/PnP; rectify images and calibration together before loading"
        )


@dataclass(frozen=True)
class CameraRecord:
    image_name: str
    image_path: Path
    fov_x: float
    fov_y: float
    rotation_c2w: np.ndarray
    translation_w2c: np.ndarray
    width: int
    height: int

    @property
    def pose_w2c(self) -> np.ndarray:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.rotation_c2w.T
        pose[:3, 3] = self.translation_w2c
        # Preserve the reference loader's C2W round trip and FP32 cast.
        return np.linalg.inv(np.linalg.inv(pose)).astype(np.float32)


class ColmapDataset:
    def __init__(self, root: str | Path, *, images: str = "processed", load_masks: bool = True) -> None:
        self.root = Path(root).expanduser().resolve()
        self.images = str(images)
        self._test_names = self._load_test_names()
        self.cameras = self._load_cameras()
        self._by_name = {camera.image_name: camera for camera in self.cameras}
        if len(self._by_name) != len(self.cameras):
            raise ValueError("camera image names must be unique")
        missing_test = self._test_names - self._by_name.keys()
        if missing_test:
            raise ValueError(
                f"test split references unregistered images: {sorted(missing_test)[:3]}"
            )
        self._masks = self._load_masks() if load_masks else None

    def _load_test_names(self) -> frozenset[str]:
        cambridge = self.root / "dataset_test.txt"
        scene_list = self.root / "sparse/0/list_test.txt"
        if cambridge.is_file():
            names = [
                line.split()[0]
                for line in cambridge.read_text().splitlines()
                if line.strip()
                and not line.lstrip().startswith("#")
                and Path(line.split()[0]).suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp"}
            ]
        elif scene_list.is_file():
            names = [line.strip() for line in scene_list.read_text().splitlines()]
        else:
            names = []
        if len(names) != len(set(names)):
            raise ValueError("test split contains duplicate image names")
        return frozenset(names)

    def _load_cameras(self) -> list[CameraRecord]:
        sparse = self.root / "sparse/0"
        try:
            extrinsics = read_extrinsics_binary(str(sparse / "images.bin"))
            intrinsics = read_intrinsics_binary(str(sparse / "cameras.bin"))
        except Exception:
            extrinsics = read_extrinsics_text(str(sparse / "images.txt"))
            intrinsics = read_intrinsics_text(str(sparse / "cameras.txt"))
        records = []
        for intrinsic in intrinsics.values():
            validate_centered_camera(intrinsic)
        for key in extrinsics:
            extrinsic = extrinsics[key]
            intrinsic = intrinsics[extrinsic.camera_id]
            if intrinsic.model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL"}:
                focal_x = focal_y = intrinsic.params[0]
            elif intrinsic.model in {"PINHOLE", "OPENCV"}:
                focal_x, focal_y = intrinsic.params[:2]
            else:
                raise ValueError(f"unsupported camera model: {intrinsic.model}")
            records.append(
                CameraRecord(
                    image_name=extrinsic.name,
                    image_path=self.root / self.images / extrinsic.name,
                    fov_x=_focal_to_fov(focal_x, intrinsic.width),
                    fov_y=_focal_to_fov(focal_y, intrinsic.height),
                    rotation_c2w=qvec2rotmat(extrinsic.qvec).T,
                    translation_w2c=np.asarray(extrinsic.tvec),
                    width=int(intrinsic.width),
                    height=int(intrinsic.height),
                )
            )
        return sorted(records, key=lambda camera: camera.image_name)

    def _load_masks(self):
        for path in (
            self.root / self.images / "masks.pkl",
            self.root / "masks.pkl",
        ):
            if path.is_file():
                with path.open("rb") as handle:
                    return pickle.load(handle)
        return None

    def split(self, name: str) -> list[CameraRecord]:
        if name == "all":
            return list(self.cameras)
        if name == "test":
            return [c for c in self.cameras if c.image_name in self._test_names]
        if name == "mapping":
            return [c for c in self.cameras if c.image_name not in self._test_names]
        raise ValueError("split must be one of: all, mapping, test")

    def camera(self, image_name: str) -> CameraRecord:
        return self._by_name[image_name]

    @staticmethod
    def load_image(camera: CameraRecord) -> torch.Tensor:
        with Image.open(camera.image_path) as image:
            # Palette/CMYK channels are not RGB channels. Preserve the legacy
            # RGB, RGBA and grayscale numeric paths used by frozen evaluations.
            if image.mode in {"P", "CMYK", "LA", "1"}:
                image = image.convert("RGB")
            elif image.mode not in {"RGB", "RGBA", "L"}:
                raise ValueError(
                    f"unsupported image mode {image.mode}; explicit 8-bit RGB conversion required"
                )
            value = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        if value.ndim == 2:
            value = value[..., None]
        value = value.permute(2, 0, 1)[:3]
        # Preserve the original loader's interpolation call at resolution=1.
        return F.interpolate(
            value[None],
            size=(camera.height, camera.width),
            mode="bilinear",
            align_corners=False,
        )[0].clamp(0.0, 1.0)

    def valid_mask(self, camera: CameraRecord) -> torch.Tensor | None:
        if self._masks is None or camera.image_name not in self._masks:
            return None
        channels = self._masks[camera.image_name]
        if len(channels) < 3:
            raise ValueError("valid mask requires object, sky, and distortion channels")
        masks = []
        for channel in channels[:3]:
            value = torch.as_tensor(channel, dtype=torch.float32, device="cpu")
            while value.ndim > 2:
                value = value.squeeze(0)
            masks.append(
                F.interpolate(
                    value[None, None],
                    size=(camera.height, camera.width),
                    mode="nearest",
                )[0, 0].bool()
            )
        return masks[0] & masks[1] & masks[2]
