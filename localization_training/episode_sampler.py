import os
from dataclasses import dataclass

import torch

from localization_training.pose_refiner import se3_exp


def apply_pose_noise(pose_w2c, xi):
    """Apply a left-multiplied SE(3) perturbation to a world-to-camera pose."""
    pose_w2c = pose_w2c.to(dtype=xi.dtype, device=xi.device)
    return se3_exp(xi) @ pose_w2c


def _randn_like_dim(dim, dtype, device, generator=None):
    return torch.randn(dim, dtype=dtype, device=device, generator=generator)


def sample_noise_from_distribution(error_distribution, quantile=0.5, generator=None, device=None, dtype=torch.float32):
    """Sample a 6D pose perturbation from cached sparse error magnitudes."""
    translation = torch.as_tensor(error_distribution.get("translation", [0.0]), dtype=dtype, device=device)
    rotation_deg = torch.as_tensor(error_distribution.get("rotation_deg", [0.0]), dtype=dtype, device=device)
    q = float(max(0.0, min(1.0, quantile)))
    t_mag = torch.quantile(translation, q) if translation.numel() > 0 else torch.tensor(0.0, dtype=dtype, device=device)
    r_mag = torch.quantile(rotation_deg, q) if rotation_deg.numel() > 0 else torch.tensor(0.0, dtype=dtype, device=device)
    t_dir = _randn_like_dim(3, dtype, translation.device, generator)
    r_dir = _randn_like_dim(3, dtype, translation.device, generator)
    t_dir = t_dir / torch.linalg.norm(t_dir).clamp_min(1e-8)
    r_dir = r_dir / torch.linalg.norm(r_dir).clamp_min(1e-8)
    rot_rad = torch.deg2rad(r_mag)
    return torch.cat([t_dir * t_mag, r_dir * rot_rad])


@dataclass
class LocalizationEpisode:
    image_name: str
    pose_gt_w2c: torch.Tensor
    pose_init_w2c: torch.Tensor
    source: str
    sparse_meta: dict


class SparsePoseCache:
    def __init__(self, path):
        self.path = path
        self.items = {}

    def load(self):
        if os.path.exists(self.path):
            self.items = torch.load(self.path, map_location="cpu")
        return self

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(self.items, self.path)

    def update(self, image_name, pose_w2c, inliers=0, ae=None, te=None, failed=False):
        self.items[image_name] = {
            "pose_w2c": pose_w2c.detach().cpu(),
            "inliers": int(inliers),
            "ae": None if ae is None else float(ae),
            "te": None if te is None else float(te),
            "failed": bool(failed),
        }

    def get(self, image_name, default=None):
        return self.items.get(image_name, default)

    def error_distribution(self):
        translations = []
        rotations = []
        for item in self.items.values():
            if item.get("te") is not None:
                translations.append(float(item["te"]) / 100.0)
            if item.get("ae") is not None:
                rotations.append(float(item["ae"]))
        return {
            "translation": torch.tensor(translations or [0.0], dtype=torch.float32),
            "rotation_deg": torch.tensor(rotations or [0.0], dtype=torch.float32),
        }


class EpisodeSampler:
    def __init__(self, sparse_pose_cache=None, error_distribution=None, query_mode="noise", noise_quantile=0.5):
        self.sparse_pose_cache = sparse_pose_cache
        self.error_distribution = error_distribution
        self.query_mode = query_mode
        self.noise_quantile = noise_quantile

    def sample(self, query_camera, generator=None):
        pose_gt = query_camera.world_view_transform.transpose(0, 1).detach().cpu()
        sparse_item = self.sparse_pose_cache.get(query_camera.image_name) if self.sparse_pose_cache else None
        if self.query_mode in {"sparse", "mixed"} and sparse_item is not None and not sparse_item.get("failed", False):
            return LocalizationEpisode(query_camera.image_name, pose_gt, sparse_item["pose_w2c"].float(), "sparse", sparse_item)
        dist = self.error_distribution
        if dist is None and self.sparse_pose_cache is not None:
            dist = self.sparse_pose_cache.error_distribution()
        if dist is None:
            dist = {"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])}
        xi = sample_noise_from_distribution(dist, self.noise_quantile, generator=generator)
        return LocalizationEpisode(query_camera.image_name, pose_gt, apply_pose_noise(pose_gt.float(), xi), "noise", {})
