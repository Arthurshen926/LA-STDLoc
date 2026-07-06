import math
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


@dataclass
class SyntheticView:
    image_name: str
    world_view_transform: torch.Tensor
    FoVx: float
    FoVy: float
    source: str
    alpha: float
    coverage: float
    difficulty: float
    train_index_a: int = -1
    train_index_b: int = -1
    sampler_mode: str = "adjacent_interpolate"
    anchor_index: int = -1
    nearest_train_distance: float = 0.0
    nearest_train_angle_deg: float = 0.0
    spatial_offset_distance: float = 0.0


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


def _pose_w2c_from_camera(camera):
    return camera.world_view_transform.transpose(0, 1).detach().cpu().float()


def _camera_center_from_w2c(pose_w2c):
    return torch.linalg.inv(pose_w2c)[:3, 3]


def _camera_centers(cameras):
    return torch.stack([_camera_center_from_w2c(_pose_w2c_from_camera(camera)) for camera in cameras], dim=0)


def _project_rotation(matrix):
    u, _, vh = torch.linalg.svd(matrix)
    rotation = u @ vh
    if torch.linalg.det(rotation) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def _axis_angle_rotation(axis, angle_rad, dtype=torch.float32, device=None):
    axis = torch.as_tensor(axis, dtype=dtype, device=device)
    axis = axis / torch.linalg.norm(axis).clamp_min(1e-8)
    x, y, z = axis
    zero = torch.zeros((), dtype=dtype, device=device)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    eye = torch.eye(3, dtype=dtype, device=device)
    sin = math.sin(float(angle_rad))
    cos = math.cos(float(angle_rad))
    return eye + sin * k + (1.0 - cos) * (k @ k)


def _median_nearest_camera_distance(centers):
    if centers.shape[0] < 2:
        return centers.new_tensor(1.0)
    distances = torch.cdist(centers, centers)
    distances = distances + torch.eye(centers.shape[0], dtype=distances.dtype, device=distances.device) * 1e9
    nearest = distances.min(dim=1).values
    valid = nearest[nearest < 1e8]
    if valid.numel() == 0:
        return centers.new_tensor(1.0)
    return torch.median(valid).clamp_min(1e-4)


def _local_track_direction(centers, index):
    count = centers.shape[0]
    if count < 2:
        return centers.new_tensor([1.0, 0.0, 0.0])
    if index <= 0:
        direction = centers[1] - centers[0]
    elif index >= count - 1:
        direction = centers[-1] - centers[-2]
    else:
        direction = centers[index + 1] - centers[index - 1]
    if torch.linalg.norm(direction) < 1e-8:
        deltas = centers - centers[index]
        norms = torch.linalg.norm(deltas, dim=1)
        norms[index] = 1e9
        nearest = int(torch.argmin(norms).item())
        direction = centers[nearest] - centers[index]
    return direction / torch.linalg.norm(direction).clamp_min(1e-8)


def _lateral_direction(c2w, track_direction):
    up = c2w[:3, 1]
    lateral = torch.cross(up, track_direction, dim=0)
    if torch.linalg.norm(lateral) < 1e-8:
        right = c2w[:3, 0]
        lateral = right - torch.dot(right, track_direction) * track_direction
    if torch.linalg.norm(lateral) < 1e-8:
        fallback = track_direction.new_tensor([0.0, 0.0, 1.0])
        lateral = torch.cross(fallback, track_direction, dim=0)
    return lateral / torch.linalg.norm(lateral).clamp_min(1e-8)


def interpolate_pose_w2c(pose_a_w2c, pose_b_w2c, alpha):
    """Interpolate two W2C poses in camera-center space and project rotation to SO(3)."""
    dtype = torch.float32
    device = pose_a_w2c.device if isinstance(pose_a_w2c, torch.Tensor) else torch.device("cpu")
    pose_a_w2c = torch.as_tensor(pose_a_w2c, device=device, dtype=dtype)
    pose_b_w2c = torch.as_tensor(pose_b_w2c, device=device, dtype=dtype)
    alpha = float(max(0.0, min(1.0, alpha)))
    c2w_a = torch.linalg.inv(pose_a_w2c)
    c2w_b = torch.linalg.inv(pose_b_w2c)
    c2w = torch.eye(4, device=device, dtype=dtype)
    c2w[:3, 3] = (1.0 - alpha) * c2w_a[:3, 3] + alpha * c2w_b[:3, 3]
    blended_rotation = (1.0 - alpha) * c2w_a[:3, :3] + alpha * c2w_b[:3, :3]
    c2w[:3, :3] = _project_rotation(blended_rotation)
    return torch.linalg.inv(c2w)


def sample_interpolated_novel_view(cameras, generator=None, alpha_min=0.35, alpha_max=0.65):
    """Sample a low-risk synthetic pose between adjacent real cameras."""
    cameras = list(cameras)
    if len(cameras) < 2:
        raise ValueError("At least two cameras are required for interpolated novel views.")
    hi = max(1, len(cameras) - 1)
    pair_idx = torch.randint(hi, (1,), generator=generator).item()
    cam_a = cameras[pair_idx]
    cam_b = cameras[pair_idx + 1]
    lo = float(max(0.0, min(1.0, alpha_min)))
    hi_alpha = float(max(lo, min(1.0, alpha_max)))
    if hi_alpha == lo:
        alpha = lo
    else:
        alpha = lo + torch.rand((), generator=generator).item() * (hi_alpha - lo)
    pose = interpolate_pose_w2c(_pose_w2c_from_camera(cam_a), _pose_w2c_from_camera(cam_b), alpha)
    difficulty = 2.0 * min(alpha, 1.0 - alpha)
    name_a = str(getattr(cam_a, "image_name", pair_idx))
    name_b = str(getattr(cam_b, "image_name", pair_idx + 1))
    return SyntheticView(
        image_name=f"synthetic/{name_a}__{name_b}__{alpha:.3f}",
        world_view_transform=pose.transpose(0, 1),
        FoVx=float((float(cam_a.FoVx) + float(cam_b.FoVx)) * 0.5),
        FoVy=float((float(cam_a.FoVy) + float(cam_b.FoVy)) * 0.5),
        source="synthetic_interpolate",
        alpha=alpha,
        coverage=difficulty,
        difficulty=difficulty,
        train_index_a=int(pair_idx),
        train_index_b=int(pair_idx + 1),
        sampler_mode="adjacent_interpolate",
        anchor_index=int(pair_idx),
    )


def sample_spatial_offset_novel_view(
    cameras,
    generator=None,
    min_offset_ratio=1.0,
    max_offset_ratio=3.0,
    yaw_deg=20.0,
    height_offset_ratio=0.15,
):
    """Sample a local off-trajectory synthetic pose around a real camera anchor."""
    cameras = list(cameras)
    if not cameras:
        raise ValueError("At least one camera is required for spatial-offset novel views.")
    anchor_idx = torch.randint(len(cameras), (1,), generator=generator).item()
    anchor = cameras[anchor_idx]
    pose_w2c = _pose_w2c_from_camera(anchor)
    c2w = torch.linalg.inv(pose_w2c)
    centers = _camera_centers(cameras)
    local_scale = _median_nearest_camera_distance(centers)
    track_direction = _local_track_direction(centers, int(anchor_idx))
    lateral = _lateral_direction(c2w, track_direction)
    up = c2w[:3, 1] / torch.linalg.norm(c2w[:3, 1]).clamp_min(1e-8)

    lo = float(max(0.0, min_offset_ratio))
    hi = float(max(lo, max_offset_ratio))
    ratio = lo if hi == lo else lo + torch.rand((), generator=generator).item() * (hi - lo)
    side = -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
    height = 0.0
    if float(height_offset_ratio) > 0.0:
        height = (torch.rand((), generator=generator).item() * 2.0 - 1.0) * float(height_offset_ratio)
    offset = lateral * (side * local_scale * ratio) + up * (local_scale * height)

    max_yaw = abs(float(yaw_deg))
    yaw = 0.0 if max_yaw == 0.0 else (torch.rand((), generator=generator).item() * 2.0 - 1.0) * max_yaw
    yaw_rotation = _axis_angle_rotation(c2w[:3, 1], math.radians(yaw), dtype=c2w.dtype, device=c2w.device)

    synthetic_c2w = c2w.clone()
    synthetic_c2w[:3, 3] = c2w[:3, 3] + offset
    synthetic_c2w[:3, :3] = _project_rotation(yaw_rotation @ c2w[:3, :3])
    synthetic_w2c = torch.linalg.inv(synthetic_c2w)

    nearest_distance = torch.linalg.norm(centers - synthetic_c2w[:3, 3], dim=1).min().item()
    normalized_distance = float(nearest_distance / float(local_scale.item()))
    anchor_name = str(getattr(anchor, "image_name", anchor_idx))
    return SyntheticView(
        image_name=f"synthetic_spatial_offset/{anchor_name}__offset{ratio:.3f}__yaw{yaw:.1f}",
        world_view_transform=synthetic_w2c.transpose(0, 1),
        FoVx=float(anchor.FoVx),
        FoVy=float(anchor.FoVy),
        source="synthetic_spatial_offset",
        alpha=0.0,
        coverage=normalized_distance,
        difficulty=normalized_distance,
        train_index_a=int(anchor_idx),
        train_index_b=-1,
        sampler_mode="spatial_offset",
        anchor_index=int(anchor_idx),
        nearest_train_distance=float(nearest_distance),
        nearest_train_angle_deg=abs(float(yaw)),
        spatial_offset_distance=float(torch.linalg.norm(offset).item()),
    )


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
        exclude_sparse_failure_stages=False,
    ):
        self.sparse_pose_cache = sparse_pose_cache
        self.error_distribution = error_distribution
        self.query_mode = query_mode
        self.noise_quantile = noise_quantile
        self.mixed_sparse_probability = float(max(0.0, min(1.0, mixed_sparse_probability)))
        self.noise_sampling = noise_sampling
        self.exclude_sparse_failure_stages = bool(exclude_sparse_failure_stages)

    def sample(self, query_camera, generator=None):
        pose_gt = query_camera.world_view_transform.transpose(0, 1).detach().cpu()
        cache_key = getattr(query_camera, "teacher_cache_key", None) or getattr(query_camera, "image_name", "")
        sparse_item = self.sparse_pose_cache.get(cache_key) if self.sparse_pose_cache else None
        if sparse_item is None and cache_key != getattr(query_camera, "image_name", "") and self.sparse_pose_cache:
            sparse_item = self.sparse_pose_cache.get(query_camera.image_name)
        bad_stage = self.exclude_sparse_failure_stages and str((sparse_item or {}).get("failure_stage", "")) in {
            "sparse_failure",
            "dense_rescues_sparse",
        }
        valid_sparse = sparse_item is not None and not sparse_item.get("failed", False) and not bad_stage
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
        return LocalizationEpisode(
            query_camera.image_name,
            pose_gt,
            apply_pose_noise(pose_gt.float(), xi),
            "noise",
            sparse_item or {},
        )
