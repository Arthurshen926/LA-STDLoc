"""Native SuperPoint -> exact Top-1 -> confidence filtering -> one PoseLib call."""

import math
import time

import torch
import torch.nn.functional as F

from anygsloc.artifacts import validate
from anygsloc.config import FROZEN
from anygsloc.features.superpoint import SuperPoint
from anygsloc.online.matching import global_cosine_top1, retain_high_score_matches
from anygsloc.online.pose import camera_intrinsics, solve_absolute_pose


class Localizer:
    def __init__(
        self,
        map_path,
        *,
        keypoint_count,
        reprojection_error_px,
        mask_policy,
        device="cuda",
        seed=2026,
    ):
        self.config = FROZEN.resolved(
            keypoint_count=keypoint_count,
            reprojection_error_px=reprojection_error_px,
            mask_policy=mask_policy,
        )
        self.device = torch.device(device)
        self.seed = int(seed)
        state = validate(torch.load(map_path, map_location="cpu", weights_only=True))
        self.anchor_ids = state["anchor_ids"]
        self.anchor_xyz = state["anchor_xyz"].to(self.device)
        self.anchor_features = F.normalize(
            state["anchor_features"].to(self.device), dim=1
        )
        # Preserve historical Fast's second map normalization exactly.
        self.anchor_features = F.normalize(self.anchor_features.float(), dim=1)
        self.model = SuperPoint().to(self.device).eval()
        self.model.nms_radius = FROZEN.nms_radius
        # Preserve the old Fast path's reusable host buffers; one instance is
        # serial and must not be called concurrently from multiple threads.
        self._host = None
        if self.device.type == "cuda":
            self._host = (
                torch.empty((keypoint_count, 2), pin_memory=True),
                torch.empty((keypoint_count, 3), pin_memory=True),
            )

    @torch.inference_mode()
    def localize(self, image, *, fov_x, fov_y, valid_mask=None):
        self.config["actual_cpu_threads"] = torch.get_num_threads()
        if not all(math.isfinite(v) and 0 < v < math.pi for v in (fov_x, fov_y)):
            raise ValueError("expected calibrated centered-pinhole FOV in radians")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        image = torch.as_tensor(image, device=self.device).float()
        if image.ndim == 3:
            image = image[None]
        if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
            raise ValueError("expected one RGB tensor")
        if (valid_mask is not None) != (
            self.config["mask_policy"] in {"explicit_valid_mask", "geometric_validity"}
        ):
            raise ValueError("mask input differs from declared policy")
        h, w = image.shape[-2:]
        mask = None
        if valid_mask is not None:
            mask = torch.as_tensor(valid_mask, device=self.device).float()
            if mask.ndim != 2:
                raise ValueError("expected 2D valid mask")
            if mask.shape != (h, w):
                mask = F.interpolate(mask[None, None], size=(h, w), mode="nearest")[
                    0, 0
                ]
            mask = mask.bool()
            image = image * mask[None, None].to(image.dtype)
        sparse = self.model.detectAndCompute(
            image, top_k=self.config["keypoint_count"], subpixel_refinement=False
        )[0]
        xy = sparse["keypoints"]
        desc = F.normalize(sparse["descriptors"], dim=1)
        if mask is not None:
            uv = xy.round().long()
            keep = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            good = torch.zeros_like(keep)
            good[keep] = mask[uv[keep, 1], uv[keep, 0]]
            xy = xy[good]
            desc = desc[good]
        # Preserve BOTH normalizations in the historical zero-residual metric.
        desc = F.normalize(F.normalize(desc, dim=-1), dim=-1)
        matches = global_cosine_top1(
            desc,
            self.anchor_features,
            chunk_size=FROZEN.match_chunk_size,
            anchor_descriptors_normalized=True,
        )
        matches = retain_high_score_matches(
            matches,
            retention_fraction=FROZEN.retention_fraction,
            minimum_count=FROZEN.minimum_matches,
        )
        selected2 = xy[matches.keypoint_indices]
        selected3 = self.anchor_xyz[matches.anchor_indices]
        if self._host is not None:
            count = len(selected2)
            host2, host3 = self._host[0][:count], self._host[1][:count]
            host2.copy_(selected2, non_blocking=True)
            host3.copy_(selected3, non_blocking=True)
            torch.cuda.current_stream(self.device).synchronize()
            points2, points3 = host2.numpy() + 0.5, host3.numpy()
        else:
            points2, points3 = selected2.cpu().numpy() + 0.5, selected3.cpu().numpy()
        K = camera_intrinsics(fov_x, fov_y, w, h)
        pose = solve_absolute_pose(
            points2,
            points3,
            K,
            reprojection_error_px=self.config["reprojection_error_px"],
            confidence=FROZEN.confidence,
            min_iterations=FROZEN.minimum_iterations,
            max_iterations=FROZEN.maximum_iterations,
            seed=self.seed,
        )
        return dict(
            pose=pose,
            matches=matches,
            keypoints=xy,
            descriptors=desc,
            intrinsic=K,
            runtime_ms=(time.perf_counter() - start) * 1000,
            success=bool(len(pose.inliers)),
            resolved_config=self.config,
        )
