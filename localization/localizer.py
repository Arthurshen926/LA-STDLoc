"""Minimal LaFGS deployment runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from localization.frontend import NativeSuperPointFrontend, SparseFeatures
from localization.matcher import (
    Top1Matches,
    global_cosine_top1,
    global_cosine_top2,
    suppress_duplicate_anchor_matches,
)
from localization.pose_solver import (
    PoseEstimate,
    camera_intrinsics,
    solve_absolute_pose,
)
from map_learning.metric import SharedLowRankMetric


@dataclass(frozen=True)
class LocalizationResult:
    sparse_features: SparseFeatures
    matches: Top1Matches
    pose: PoseEstimate
    intrinsic: np.ndarray
    runtime_ms: dict[str, float]
    match_diagnostics: dict[str, int | float]


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
        suppress_duplicate_anchors: bool = False,
        guided_sampling: bool = False,
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
        self.suppress_duplicate_anchors = bool(suppress_duplicate_anchors)
        self.guided_sampling = bool(guided_sampling)
        self.anchor_matchability = torch.as_tensor(
            state.get("anchor_matchability", torch.ones_like(self.anchor_ids)),
            device=self.device,
        ).float()
        covariance = state.get("anchor_position_covariance")
        if covariance is None:
            self.anchor_uncertainty = torch.ones_like(
                self.anchor_matchability, device=self.device
            )
        else:
            trace = (
                torch.diagonal(
                    torch.as_tensor(covariance, device=self.device).float(),
                    dim1=1,
                    dim2=2,
                )
                .sum(dim=1)
                .clamp_min(1e-12)
            )
            median = trace.median().clamp_min(1e-12)
            self.anchor_uncertainty = trace / median

    @torch.inference_mode()
    def localize(
        self,
        image: torch.Tensor,
        *,
        fov_x: float,
        fov_y: float,
        valid_mask: torch.Tensor | None = None,
    ) -> LocalizationResult:
        def synchronize() -> None:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

        synchronize()
        total_started = time.perf_counter()
        frontend_started = total_started
        sparse = self.frontend(image, valid_mask=valid_mask)
        synchronize()
        frontend_ms = (time.perf_counter() - frontend_started) * 1000.0

        matching_started = time.perf_counter()
        guidance_quality = None
        if self.guided_sampling:
            top2 = global_cosine_top2(sparse.descriptors, self.anchor_features)
            raw_matches = Top1Matches(
                keypoint_indices=top2.keypoint_indices,
                anchor_indices=top2.anchor_indices[:, 0],
                scores=top2.scores[:, 0],
            )
            margin = (top2.scores[:, 0] - top2.scores[:, 1]).clamp_min(0)
            winner = raw_matches.anchor_indices
            reliability = self.anchor_matchability[winner].clamp(0.02, 1.0)
            certainty = (1.0 + self.anchor_uncertainty[winner]).reciprocal()
            guidance_quality = margin * reliability.sqrt() * certainty
        else:
            raw_matches = global_cosine_top1(sparse.descriptors, self.anchor_features)
        matches = (
            suppress_duplicate_anchor_matches(raw_matches)
            if self.suppress_duplicate_anchors
            else raw_matches
        )
        if self.guided_sampling:
            if self.suppress_duplicate_anchors:
                raise ValueError(
                    "guided sampling and duplicate suppression are separate ablations"
                )
            order = torch.argsort(guidance_quality, descending=True, stable=True)
            matches = Top1Matches(
                keypoint_indices=matches.keypoint_indices[order],
                anchor_indices=matches.anchor_indices[order],
                scores=matches.scores[order],
            )
        points_2d = sparse.keypoints[matches.keypoint_indices].cpu().numpy()
        points_3d = self.anchor_xyz[matches.anchor_indices].cpu().numpy()
        synchronize()
        matching_ms = (time.perf_counter() - matching_started) * 1000.0
        height, width = sparse.image_hw
        intrinsic = camera_intrinsics(fov_x, fov_y, width, height)

        ransac_started = time.perf_counter()
        pose = solve_absolute_pose(
            points_2d + 0.5,
            points_3d,
            intrinsic,
            reprojection_error_px=self.reprojection_error_px,
            confidence=self.confidence,
            max_iterations=self.max_iterations,
            min_iterations=self.min_iterations,
            seed=self.seed,
            progressive_sampling=self.guided_sampling,
        )
        ransac_ms = (time.perf_counter() - ransac_started) * 1000.0
        return LocalizationResult(
            sparse,
            matches,
            pose,
            intrinsic,
            {
                "frontend_ms": frontend_ms,
                "matching_ms": matching_ms,
                "ransac_ms": ransac_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000.0,
            },
            {
                "top1_match_count": int(raw_matches.scores.numel()),
                "retained_match_count": int(matches.scores.numel()),
                "duplicate_anchor_count": int(
                    raw_matches.scores.numel() - matches.scores.numel()
                ),
                "duplicate_anchor_fraction": float(
                    1.0
                    - matches.scores.numel() / max(int(raw_matches.scores.numel()), 1)
                ),
                "guided_sampling": int(self.guided_sampling),
            },
        )
