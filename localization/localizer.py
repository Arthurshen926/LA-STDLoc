"""Minimal LaFGS deployment runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from localization.frontend import NativeSuperPointFrontend, SparseFeatures
from localization.matcher import Top1Matches, global_cosine_top1
from localization.pose_solver import PoseEstimate, camera_intrinsics, solve_absolute_pose
from map_learning.metric import SharedLowRankMetric


@dataclass(frozen=True)
class LocalizationResult:
    sparse_features: SparseFeatures
    matches: Top1Matches
    pose: PoseEstimate
    intrinsic: np.ndarray


def load_shared_metric(
    path: str | Path,
    *,
    anchor_ids: torch.Tensor,
    device: torch.device,
) -> SharedLowRankMetric:
    state = torch.load(path, map_location="cpu", weights_only=False)
    metric_ids = torch.as_tensor(state["landmark_indices"]).long().reshape(-1)
    if not torch.equal(metric_ids.cpu(), anchor_ids.cpu()):
        raise ValueError("metric state does not align with the compact anchor map")
    metric = SharedLowRankMetric(**state["metric_config"]).to(device)
    metric.load_state_dict(state["metric_state_dict"])
    return metric.eval()


class SparseLocalizer:
    def __init__(
        self,
        map_path: str | Path,
        metric_state_path: str | Path,
        *,
        device: torch.device | str = "cuda",
        keypoint_count: int = 2048,
        reprojection_error_px: float = 12.0,
        confidence: float = 0.99999,
        max_iterations: int = 100000,
        min_iterations: int = 1000,
        seed: int = 2026,
    ) -> None:
        self.device = torch.device(device)
        state = torch.load(map_path, map_location="cpu", weights_only=False)
        if state.get("schema") != "lafgs_materialized_anchor_map":
            raise ValueError("unsupported localization map schema")
        self.anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
        self.anchor_xyz = torch.as_tensor(
            state["anchor_xyz"], device=self.device
        ).float()
        self.anchor_features = F.normalize(
            torch.as_tensor(state["anchor_features"], device=self.device).float(),
            dim=1,
        )
        if not (
            self.anchor_ids.numel()
            == self.anchor_xyz.shape[0]
            == self.anchor_features.shape[0]
        ):
            raise ValueError("compact map rows do not align")
        metric = load_shared_metric(
            metric_state_path, anchor_ids=self.anchor_ids, device=self.device
        )
        self.frontend = NativeSuperPointFrontend(
            device=self.device, keypoint_count=keypoint_count, metric=metric
        )
        self.reprojection_error_px = float(reprojection_error_px)
        self.confidence = float(confidence)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.seed = int(seed)

    @torch.inference_mode()
    def localize(
        self,
        image: torch.Tensor,
        *,
        fov_x: float,
        fov_y: float,
        valid_mask: torch.Tensor | None = None,
    ) -> LocalizationResult:
        sparse = self.frontend(image, valid_mask=valid_mask)
        matches = global_cosine_top1(sparse.descriptors, self.anchor_features)
        points_2d = sparse.keypoints[matches.keypoint_indices].cpu().numpy()
        points_3d = self.anchor_xyz[matches.anchor_indices].cpu().numpy()
        height, width = sparse.image_hw
        intrinsic = camera_intrinsics(fov_x, fov_y, width, height)
        pose = solve_absolute_pose(
            points_2d + 0.5,
            points_3d,
            intrinsic,
            reprojection_error_px=self.reprojection_error_px,
            confidence=self.confidence,
            max_iterations=self.max_iterations,
            min_iterations=self.min_iterations,
            seed=self.seed,
        )
        return LocalizationResult(sparse, matches, pose, intrinsic)
