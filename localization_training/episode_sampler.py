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


def _sample_observed_magnitude(values, generator=None):
    if values.numel() == 0:
        return values.new_tensor(0.0)
    idx = torch.randint(values.numel(), (1,), generator=generator).item()
    return values[idx]


def sample_noise_from_distribution(
    error_distribution,
    quantile=0.5,
    generator=None,
    device=None,
    dtype=torch.float32,
    sampling="quantile",
):
    """Sample a 6D pose perturbation from cached sparse error magnitudes."""
    translation = torch.as_tensor(error_distribution.get("translation", [0.0]), dtype=dtype, device=device)
    rotation_deg = torch.as_tensor(error_distribution.get("rotation_deg", [0.0]), dtype=dtype, device=device)
    if sampling == "empirical":
        if translation.numel() > 0 and translation.numel() == rotation_deg.numel():
            idx = torch.randint(translation.numel(), (1,), generator=generator).item()
            t_mag = translation[idx]
            r_mag = rotation_deg[idx]
        else:
            t_mag = _sample_observed_magnitude(translation, generator=generator)
            r_mag = _sample_observed_magnitude(rotation_deg, generator=generator)
    elif sampling == "quantile":
        q = float(max(0.0, min(1.0, quantile)))
        t_mag = torch.quantile(translation, q) if translation.numel() > 0 else torch.tensor(0.0, dtype=dtype, device=device)
        r_mag = torch.quantile(rotation_deg, q) if rotation_deg.numel() > 0 else torch.tensor(0.0, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown pose noise sampling mode: {sampling}")
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


def _camera_sequence_key(camera):
    image_name = str(getattr(camera, "image_name", ""))
    normalized = image_name.replace("\\", "/")
    if "/" in normalized:
        return normalized.split("/", 1)[0]
    return ""


def _query_count_for_split(camera_count, query_ratio):
    ratio = float(max(0.0, min(1.0, query_ratio)))
    query_count = int(round(camera_count * ratio))
    return max(1, min(camera_count - 1, query_count))


def _split_by_query_ids(cameras, query_ids):
    support = [camera for idx, camera in enumerate(cameras) if idx not in query_ids]
    query = [camera for idx, camera in enumerate(cameras) if idx in query_ids]
    return support, query


def _sequence_block_query_ids(cameras, query_count, generator):
    sequence_to_indices = {}
    for idx, camera in enumerate(cameras):
        sequence_to_indices.setdefault(_camera_sequence_key(camera), []).append(idx)
    if len(sequence_to_indices) < 2:
        return None

    sequence_keys = list(sequence_to_indices.keys())
    order = torch.randperm(len(sequence_keys), generator=generator).tolist()
    selected = set()
    for seq_pos in order:
        candidate = sequence_to_indices[sequence_keys[seq_pos]]
        if len(selected) + len(candidate) >= len(cameras):
            continue
        selected.update(candidate)
        if len(selected) >= query_count:
            break
    if not selected:
        first = sequence_to_indices[sequence_keys[order[0]]]
        if len(first) < len(cameras):
            selected.update(first)
    return selected if selected else None


def _temporal_block_query_ids(camera_count, query_count, generator):
    max_start = max(0, camera_count - query_count)
    start = torch.randint(max_start + 1, (1,), generator=generator).item()
    return set(range(start, start + query_count))


def split_support_query_cameras(cameras, query_ratio=0.2, seed=0, mode="random"):
    cameras = list(cameras)
    if len(cameras) < 2:
        return cameras, cameras.copy()
    query_count = _query_count_for_split(len(cameras), query_ratio)
    generator = torch.Generator().manual_seed(int(seed))
    if mode == "random":
        query_ids = set(torch.randperm(len(cameras), generator=generator)[:query_count].tolist())
    elif mode == "sequence_block":
        query_ids = _sequence_block_query_ids(cameras, query_count, generator)
        if query_ids is None:
            query_ids = set(torch.randperm(len(cameras), generator=generator)[:query_count].tolist())
    elif mode == "temporal_block":
        query_ids = _temporal_block_query_ids(len(cameras), query_count, generator)
    else:
        raise ValueError(f"Unknown support/query split mode: {mode}")
    return _split_by_query_ids(cameras, query_ids)


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

    def update(
        self,
        image_name,
        pose_w2c,
        inliers=0,
        ae=None,
        te=None,
        failed=False,
        dense_pose_w2c=None,
        dense_inliers=None,
        dense_ae=None,
        dense_te=None,
    ):
        self.items[image_name] = {
            "pose_w2c": pose_w2c.detach().cpu(),
            "inliers": int(inliers),
            "ae": None if ae is None else float(ae),
            "te": None if te is None else float(te),
            "failed": bool(failed),
        }
        if dense_pose_w2c is not None:
            self.items[image_name]["dense_pose_w2c"] = dense_pose_w2c.detach().cpu()
        if dense_inliers is not None:
            self.items[image_name]["dense_inliers"] = int(dense_inliers)
        if dense_ae is not None:
            self.items[image_name]["dense_ae"] = float(dense_ae)
        if dense_te is not None:
            self.items[image_name]["dense_te"] = float(dense_te)

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
    def __init__(
        self,
        sparse_pose_cache=None,
        error_distribution=None,
        query_mode="noise",
        noise_quantile=0.5,
        mixed_sparse_probability=0.5,
        noise_sampling="empirical",
    ):
        self.sparse_pose_cache = sparse_pose_cache
        self.error_distribution = error_distribution
        self.query_mode = query_mode
        self.noise_quantile = noise_quantile
        self.mixed_sparse_probability = float(max(0.0, min(1.0, mixed_sparse_probability)))
        self.noise_sampling = noise_sampling

    def sample(self, query_camera, generator=None):
        pose_gt = query_camera.world_view_transform.transpose(0, 1).detach().cpu()
        sparse_item = self.sparse_pose_cache.get(query_camera.image_name) if self.sparse_pose_cache else None
        valid_sparse = sparse_item is not None and not sparse_item.get("failed", False)
        use_sparse = self.query_mode == "sparse" and valid_sparse
        if self.query_mode == "mixed" and valid_sparse:
            use_sparse = torch.rand((), generator=generator).item() < self.mixed_sparse_probability
        if use_sparse:
            return LocalizationEpisode(query_camera.image_name, pose_gt, sparse_item["pose_w2c"].float(), "sparse", sparse_item)
        dist = self.error_distribution
        if dist is None and self.sparse_pose_cache is not None:
            dist = self.sparse_pose_cache.error_distribution()
        if dist is None:
            dist = {"translation": torch.tensor([0.0]), "rotation_deg": torch.tensor([0.0])}
        xi = sample_noise_from_distribution(
            dist,
            self.noise_quantile,
            generator=generator,
            sampling=self.noise_sampling,
        )
        return LocalizationEpisode(query_camera.image_name, pose_gt, apply_pose_noise(pose_gt.float(), xi), "noise", {})
