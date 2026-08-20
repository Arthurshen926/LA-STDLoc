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
    global_cosine_topk,
    maximum_weight_anchor_assignment,
    suppress_duplicate_anchor_matches,
)
from localization.pose_solver import (
    PoseEstimate,
    camera_intrinsics,
    poselib_camera,
    solve_absolute_pose,
    solve_group_diverse_absolute_pose,
)
from map_learning.context_metric import MapConsistentContextAdapter
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
    photometric_contract: dict | None = None,
) -> SharedLowRankMetric:
    state = torch.load(path, map_location="cpu", weights_only=False)
    metric_ids = torch.as_tensor(state["landmark_indices"]).long().reshape(-1)
    if not torch.equal(metric_ids.cpu(), anchor_ids.cpu()):
        raise ValueError("metric state does not align with the compact anchor map")
    metric_photometric = state.get("photometric_canonicalization_contract")
    if metric_photometric != photometric_contract:
        raise ValueError("map and metric photometric contracts do not align")
    metric = SharedLowRankMetric(**state["metric_config"]).to(device)
    metric.load_state_dict(state["metric_state_dict"])
    return metric.eval()


def load_context_descriptor_artifact(
    path: str | Path,
    *,
    base_anchor_ids: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, MapConsistentContextAdapter]:
    """Load a map-consistent context bank and strictly align it to a base map."""
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("schema") != "lafgs_map_consistent_context_descriptor":
        raise ValueError("unsupported context descriptor artifact schema")
    if bool(artifact.get("uses_test_queries", True)):
        raise ValueError("context descriptor artifact must be mapping-only")
    indices = torch.as_tensor(artifact["anchor_indices"]).long().reshape(-1)
    anchor_ids = torch.as_tensor(artifact["anchor_ids"]).long().reshape(-1)
    base_anchor_ids = torch.as_tensor(base_anchor_ids).long().reshape(-1)
    if indices.numel() and (
        int(indices.min()) < 0 or int(indices.max()) >= base_anchor_ids.numel()
    ):
        raise ValueError("context descriptor anchor index is outside the base map")
    if not torch.equal(base_anchor_ids[indices], anchor_ids):
        raise ValueError("context descriptor anchor IDs do not align with the base map")
    features = F.normalize(torch.as_tensor(artifact["anchor_features"]).float(), dim=1)
    if features.shape != (indices.numel(), 256):
        raise ValueError("context descriptor bank must have shape [N, 256]")
    exported = dict(artifact["adapter_config"])
    config = {
        key: exported[key]
        for key in (
            "descriptor_dim",
            "hidden_dim",
            "context_kernels",
            "maximum_residual_norm",
        )
    }
    config["context_mode"] = exported.get("context_mode", "multi_scale_global")
    # Artifacts produced before smooth radial trust regions used the original
    # hard clip. Preserve their exact runtime instead of silently changing it.
    config["residual_parameterization"] = exported.get(
        "residual_parameterization", "hard_clip_v1"
    )
    adapter = MapConsistentContextAdapter(**config).to(device)
    adapter.load_state_dict(artifact["adapter_state_dict"], strict=True)
    return indices, anchor_ids, features.to(device), adapter.eval()


class SparseLocalizer:
    def __init__(
        self,
        map_path: str | Path,
        metric_state_path: str | Path | None = None,
        *,
        context_state_path: str | Path | None = None,
        device: torch.device | str = "cuda",
        keypoint_count: int = 2048,
        nms_radius: int = 4,
        reprojection_error_px: float = 12.0,
        confidence: float = 0.99999,
        max_iterations: int = 100000,
        min_iterations: int = 1000,
        seed: int = 2026,
        suppress_duplicate_anchors: bool = False,
        guided_sampling: bool = False,
        group_aware_pose: bool = False,
        group_field: str = "parent_source_track_ids",
        group_hypothesis_samples: int = 32,
        assignment_topk: int = 0,
        assignment_dustbin_score: float = -1.0,
        profile_mode: bool = True,
        reuse_correspondence_buffers: bool = True,
    ) -> None:
        self.device = torch.device(device)
        state = torch.load(map_path, map_location="cpu", weights_only=False)
        if state.get("schema") != "lafgs_materialized_anchor_map":
            raise ValueError("unsupported localization map schema")
        if (metric_state_path is None) == (context_state_path is None):
            raise ValueError(
                "select exactly one descriptor protocol: shared metric or context"
            )
        base_anchor_ids = torch.as_tensor(state["anchor_ids"]).long()
        if context_state_path is not None:
            (
                context_indices,
                self.anchor_ids,
                self.anchor_features,
                context_adapter,
            ) = load_context_descriptor_artifact(
                context_state_path,
                base_anchor_ids=base_anchor_ids,
                device=self.device,
            )
            self.anchor_xyz = torch.as_tensor(
                state["anchor_xyz"], device=self.device
            ).float()[context_indices.to(self.device)]
            metric = None
        else:
            context_indices = torch.arange(base_anchor_ids.numel())
            self.anchor_ids = base_anchor_ids
            self.anchor_xyz = torch.as_tensor(
                state["anchor_xyz"], device=self.device
            ).float()
            self.anchor_features = F.normalize(
                torch.as_tensor(state["anchor_features"], device=self.device).float(),
                dim=1,
            )
            metric = load_shared_metric(
                metric_state_path,
                anchor_ids=self.anchor_ids,
                device=self.device,
                photometric_contract=state.get(
                    "photometric_canonicalization_contract"
                ),
            )
            context_adapter = None
        # This is the exact normalization that the historical matcher applied
        # to every map chunk for every query.  Materialize it once instead.
        self.anchor_features = F.normalize(self.anchor_features.float(), dim=1)
        if not (
            self.anchor_ids.numel()
            == self.anchor_xyz.shape[0]
            == self.anchor_features.shape[0]
        ):
            raise ValueError("compact map rows do not align")
        self.frontend = NativeSuperPointFrontend(
            device=self.device,
            keypoint_count=keypoint_count,
            nms_radius=nms_radius,
            metric=metric,
            context_adapter=context_adapter,
            photometric_contract=state.get("photometric_canonicalization_contract"),
        )
        self.photometric_canonicalization_contract = state.get(
            "photometric_canonicalization_contract"
        )
        self.reprojection_error_px = float(reprojection_error_px)
        self.confidence = float(confidence)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.seed = int(seed)
        self.suppress_duplicate_anchors = bool(suppress_duplicate_anchors)
        self.guided_sampling = bool(guided_sampling)
        self.assignment_topk = int(assignment_topk)
        self.assignment_dustbin_score = float(assignment_dustbin_score)
        self.profile_mode = bool(profile_mode)
        self.reuse_correspondence_buffers = bool(reuse_correspondence_buffers)
        self._camera_cache: dict[tuple[float, float, int, int], tuple[np.ndarray, dict]] = {}
        self._deployment_timing_events = None
        if self.device.type == "cuda" and not self.profile_mode:
            self._deployment_timing_events = tuple(
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            )
        self._points_2d_host = None
        self._points_3d_host = None
        if self.device.type == "cuda" and self.reuse_correspondence_buffers:
            # PoseLib consumes NumPy arrays immediately and localize is serial,
            # so these pinned buffers can safely be reused between queries.
            self._points_2d_host = torch.empty(
                (keypoint_count, 2), dtype=torch.float32, pin_memory=True
            )
            self._points_3d_host = torch.empty(
                (keypoint_count, 3), dtype=torch.float32, pin_memory=True
            )
        if self.assignment_topk < 0:
            raise ValueError("assignment top-K must be zero (disabled) or positive")
        if self.assignment_topk > int(self.anchor_features.shape[0]):
            raise ValueError("assignment top-K exceeds the Anchor count")
        if self.assignment_topk and (
            self.suppress_duplicate_anchors or self.guided_sampling
        ):
            raise ValueError(
                "capacity assignment, duplicate suppression, and guided sampling "
                "are separate deployment ablations"
            )
        self.group_aware_pose = bool(group_aware_pose)
        self.group_hypothesis_samples = int(group_hypothesis_samples)
        if self.group_aware_pose and (self.guided_sampling or self.assignment_topk):
            raise ValueError(
                "group-aware pose, guided sampling, and capacity assignment "
                "are separate ablations"
            )
        if self.group_aware_pose:
            if group_field not in state:
                raise ValueError(f"group-aware pose map misses {group_field}")
            groups = torch.as_tensor(state[group_field]).long().reshape(-1)
            if groups.shape != base_anchor_ids.shape:
                raise ValueError("pose correlation groups do not align with the map")
            unknown = groups < 0
            offset = int(groups[~unknown].max()) + 1 if bool((~unknown).any()) else 0
            groups = groups.clone()
            groups[unknown] = offset + torch.arange(groups.numel())[unknown]
            self.anchor_pose_groups = groups[context_indices].to(self.device)
        else:
            self.anchor_pose_groups = None
        self.anchor_matchability = torch.as_tensor(
            state.get("anchor_matchability", torch.ones_like(base_anchor_ids)),
            device=self.device,
        ).float()[context_indices.to(self.device)]
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
            self.anchor_uncertainty = self.anchor_uncertainty[
                context_indices.to(self.device)
            ]

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

        if self.profile_mode:
            synchronize()
        total_started = time.perf_counter()
        frontend_started = total_started
        frontend_start_event = frontend_end_event = matching_end_event = None
        if self._deployment_timing_events is not None:
            (
                frontend_start_event,
                frontend_end_event,
                matching_end_event,
            ) = self._deployment_timing_events
            frontend_start_event.record()
        sparse = self.frontend(image, valid_mask=valid_mask)
        if self.profile_mode:
            synchronize()
            frontend_ms = (time.perf_counter() - frontend_started) * 1000.0
        elif frontend_end_event is not None:
            frontend_end_event.record()
            frontend_ms = 0.0
        else:
            frontend_ms = (time.perf_counter() - frontend_started) * 1000.0

        matching_started = time.perf_counter()
        guidance_quality = None
        assignment = None
        if self.assignment_topk:
            topk = global_cosine_topk(
                sparse.descriptors,
                self.anchor_features,
                topk=self.assignment_topk,
                anchor_descriptors_normalized=True,
            )
            raw_matches = Top1Matches(
                keypoint_indices=topk.keypoint_indices,
                anchor_indices=topk.anchor_indices[:, 0],
                scores=topk.scores[:, 0],
            )
            assignment = maximum_weight_anchor_assignment(
                topk, dustbin_score=self.assignment_dustbin_score
            )
            matches = assignment.matches
        elif self.guided_sampling:
            top2 = global_cosine_top2(
                sparse.descriptors,
                self.anchor_features,
                anchor_descriptors_normalized=True,
            )
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
            raw_matches = global_cosine_top1(
                sparse.descriptors,
                self.anchor_features,
                anchor_descriptors_normalized=True,
            )
            matches = raw_matches
        if not self.assignment_topk:
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
        selected_2d = sparse.keypoints[matches.keypoint_indices]
        selected_3d = self.anchor_xyz[matches.anchor_indices]
        count = int(selected_2d.shape[0])
        if self._points_2d_host is not None and count <= self._points_2d_host.shape[0]:
            points_2d_tensor = self._points_2d_host[:count]
            points_3d_tensor = self._points_3d_host[:count]
            points_2d_tensor.copy_(selected_2d, non_blocking=True)
            points_3d_tensor.copy_(selected_3d, non_blocking=True)
            if matching_end_event is not None:
                matching_end_event.record()
            synchronize()
            points_2d = points_2d_tensor.numpy()
            points_3d = points_3d_tensor.numpy()
        else:
            points_2d = selected_2d.cpu().numpy()
            points_3d = selected_3d.cpu().numpy()
            if matching_end_event is not None:
                matching_end_event.record()
            synchronize()
        if frontend_start_event is not None:
            frontend_ms = float(frontend_start_event.elapsed_time(frontend_end_event))
            matching_ms = float(frontend_end_event.elapsed_time(matching_end_event))
        else:
            matching_ms = (time.perf_counter() - matching_started) * 1000.0
        height, width = sparse.image_hw
        camera_key = (float(fov_x), float(fov_y), int(width), int(height))
        cached_camera = self._camera_cache.get(camera_key)
        if cached_camera is None:
            intrinsic = camera_intrinsics(fov_x, fov_y, width, height)
            pose_camera = poselib_camera(intrinsic)
            self._camera_cache[camera_key] = (intrinsic, pose_camera)
        else:
            intrinsic, pose_camera = cached_camera

        ransac_started = time.perf_counter()
        solve_kwargs = {
            "reprojection_error_px": self.reprojection_error_px,
            "confidence": self.confidence,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
            "seed": self.seed,
        }
        if self.group_aware_pose:
            pose = solve_group_diverse_absolute_pose(
                points_2d + 0.5,
                points_3d,
                intrinsic,
                self.anchor_pose_groups[matches.anchor_indices].cpu().numpy(),
                group_hypothesis_samples=self.group_hypothesis_samples,
                **solve_kwargs,
            )
        else:
            pose = solve_absolute_pose(
                points_2d + 0.5,
                points_3d,
                intrinsic,
                progressive_sampling=self.guided_sampling,
                camera=pose_camera,
                **solve_kwargs,
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
                "group_aware_pose": int(self.group_aware_pose),
                "capacity_assignment": int(self.assignment_topk > 0),
                "assignment_topk": int(self.assignment_topk),
                "assignment_dustbin_score": float(self.assignment_dustbin_score),
                "assignment_candidate_edges": (
                    int(assignment.candidate_edge_count) if assignment else 0
                ),
                "assignment_eligible_edges": (
                    int(assignment.eligible_edge_count) if assignment else 0
                ),
                "assignment_unmatched_queries": (
                    int(assignment.unmatched_query_count) if assignment else 0
                ),
                "assignment_reassigned_queries": (
                    int(assignment.reassigned_query_count) if assignment else 0
                ),
                "assignment_top1_collisions": (
                    int(assignment.top1_collision_count) if assignment else 0
                ),
                "context_adapter": int(self.frontend.context_adapter is not None),
                "profile_mode": int(self.profile_mode),
                "reused_correspondence_buffers": int(
                    self._points_2d_host is not None
                ),
            },
        )
