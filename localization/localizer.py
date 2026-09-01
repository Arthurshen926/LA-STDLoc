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
    global_cosine_top2,
    global_cosine_topk,
    global_owner_prototype_top1,
    maximum_weight_anchor_assignment,
    suppress_duplicate_anchor_matches,
)
from localization.pose_solver import (
    PoseEstimate,
    camera_intrinsics,
    poselib_camera,
    refine_absolute_pose_from_initial,
    solve_absolute_pose,
    solve_group_diverse_absolute_pose,
)
from map_learning.context_metric import MapConsistentContextAdapter
from map_learning.metric import SharedLowRankMetric


def _reprojection_residuals(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    pose_w2c: np.ndarray,
    intrinsic: np.ndarray,
) -> np.ndarray:
    camera = (pose_w2c[:3, :3] @ points_3d.T).T + pose_w2c[:3, 3]
    homogeneous = (intrinsic @ camera.T).T
    depth = homogeneous[:, 2]
    projected = np.full_like(points_2d, np.inf, dtype=np.float64)
    valid = depth > 1e-12
    projected[valid] = homogeneous[valid, :2] / depth[valid, None]
    return np.linalg.norm(projected - points_2d, axis=1)


def _pose_update_magnitude(
    baseline_w2c: np.ndarray, candidate_w2c: np.ndarray
) -> tuple[float, float]:
    relative = candidate_w2c[:3, :3] @ baseline_w2c[:3, :3].T
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    rotation_deg = float(np.degrees(np.arccos(cosine)))
    baseline_center = np.linalg.inv(baseline_w2c)[:3, 3]
    candidate_center = np.linalg.inv(candidate_w2c)[:3, 3]
    translation_cm = float(np.linalg.norm(candidate_center - baseline_center) * 100.0)
    return translation_cm, rotation_deg


@dataclass(frozen=True)
class LocalizationResult:
    sparse_features: SparseFeatures
    matches: Top1Matches
    pose: PoseEstimate
    intrinsic: np.ndarray
    runtime_ms: dict[str, float]
    match_diagnostics: dict[str, int | float | str | list | None]


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
        topk_geometric_feedback: bool = False,
        sparse_lgcv_topk_feedback: bool = False,
        pose_conditioned_sparse_refinement: bool = False,
        refinement_pose_backend: str = "local",
        feedback_minimum_baseline_inliers: int = 128,
        feedback_maximum_baseline_inliers: int = 256,
        feedback_minimum_candidate_inlier_gain: int = 4,
        feedback_minimum_candidate_relative_inlier_gain: float = 0.0,
        feedback_maximum_candidate_ransac_iterations: int = 0,
        feedback_minimum_baseline_inlier_retention: float = 0.0,
        feedback_maximum_protected_median_residual_increase_px: float = -1.0,
        feedback_maximum_protected_p90_residual_increase_px: float = -1.0,
        feedback_maximum_pose_update_translation_cm: float = -1.0,
        feedback_maximum_pose_update_rotation_deg: float = -1.0,
        refinement_maximum_score_drop_from_top1: float = 0.03,
        refinement_view_direction_slack_deg: float = 15.0,
        refinement_maximum_changed_rows: int = 128,
        refinement_maximum_changed_to_baseline_inlier_ratio: float = 0.50,
        refinement_minimum_proposal_count: int = 60,
        refinement_minimum_proposal_relative_gain: float = 0.075,
        refinement_minimum_changed_inliers: int = 8,
        refinement_minimum_changed_inlier_fraction: float = 0.10,
        refinement_minimum_changed_inlier_spatial_cells: int = 3,
        refinement_maximum_changed_inlier_median_residual_px: float = 6.0,
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
        raw_view_support = state.get("anchor_view_support")
        self.anchor_view_support = None
        if raw_view_support is not None:
            if not isinstance(raw_view_support, dict):
                raise ValueError("Anchor view support must be a mapping")
            support_modes = torch.as_tensor(raw_view_support.get("direction_modes"))
            support_radii = torch.as_tensor(
                raw_view_support.get("direction_radius_deg")
            )
            support_count = torch.as_tensor(raw_view_support.get("mode_count"))
            support_minimum = torch.as_tensor(
                raw_view_support.get("minimum_distance_m")
            )
            support_maximum = torch.as_tensor(
                raw_view_support.get("maximum_distance_m")
            )
            base_count = base_anchor_ids.numel()
            if not (
                raw_view_support.get("schema")
                == "lafgs_v24_anchor_view_support"
                and raw_view_support.get("uses_test_queries") is False
                and support_modes.shape == (base_count, 2, 3)
                and support_radii.shape == (base_count, 2)
                and support_count.shape
                == support_minimum.shape
                == support_maximum.shape
                == (base_count,)
                and bool(((support_count == 1) | (support_count == 2)).all())
                and bool(torch.isfinite(support_modes).all())
                and bool(torch.isfinite(support_minimum).all())
                and bool(torch.isfinite(support_maximum).all())
                and bool((support_minimum > 0).all())
                and bool((support_maximum >= support_minimum).all())
            ):
                raise ValueError("Anchor view support does not align with the map")
            support_rows = context_indices.long()
            self.anchor_view_support = {
                "schema": raw_view_support["schema"],
                "uses_test_queries": False,
                "runtime_validated": True,
                "direction_modes": support_modes[support_rows]
                .float()
                .to(self.device),
                "direction_radius_deg": support_radii[support_rows]
                .float()
                .to(self.device),
                "mode_count": support_count[support_rows].long().to(self.device),
                "minimum_distance_m": support_minimum[support_rows]
                .float()
                .to(self.device),
                "maximum_distance_m": support_maximum[support_rows]
                .float()
                .to(self.device),
            }
        prototype_features = state.get("anchor_extra_prototype_features")
        prototype_owners = state.get("anchor_extra_prototype_owner_rows")
        if (prototype_features is None) != (prototype_owners is None):
            raise ValueError("sparse prototype features and owners must be paired")
        if prototype_features is None:
            self.anchor_extra_prototype_features = self.anchor_features.new_empty(
                (0, self.anchor_features.shape[1])
            )
            self.anchor_extra_prototype_owner_rows = torch.empty(
                0, dtype=torch.long, device=self.device
            )
        else:
            self.anchor_extra_prototype_features = F.normalize(
                torch.as_tensor(prototype_features, device=self.device).float(), dim=1
            )
            self.anchor_extra_prototype_owner_rows = torch.as_tensor(
                prototype_owners, device=self.device
            ).long().reshape(-1)
            if (
                self.anchor_extra_prototype_features.ndim != 2
                or self.anchor_extra_prototype_features.shape[0]
                != self.anchor_extra_prototype_owner_rows.numel()
                or self.anchor_extra_prototype_features.shape[1]
                != self.anchor_features.shape[1]
                or (
                    self.anchor_extra_prototype_owner_rows.numel()
                    and (
                        int(self.anchor_extra_prototype_owner_rows.min()) < 0
                        or int(self.anchor_extra_prototype_owner_rows.max())
                        >= self.anchor_features.shape[0]
                    )
                )
            ):
                raise ValueError("sparse prototype extension is invalid")
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
        self.topk_geometric_feedback = bool(topk_geometric_feedback)
        self.sparse_lgcv_topk_feedback = bool(sparse_lgcv_topk_feedback)
        self.pose_conditioned_sparse_refinement = bool(
            pose_conditioned_sparse_refinement
        )
        self.refinement_pose_backend = str(refinement_pose_backend)
        if self.refinement_pose_backend not in {"local", "robust"}:
            raise ValueError("refinement pose backend must be local or robust")
        if not self.pose_conditioned_sparse_refinement and (
            self.refinement_pose_backend != "local"
        ):
            raise ValueError(
                "robust refinement backend requires pose-conditioned refinement"
            )
        self.feedback_minimum_baseline_inliers = int(
            feedback_minimum_baseline_inliers
        )
        self.feedback_maximum_baseline_inliers = int(
            feedback_maximum_baseline_inliers
        )
        self.feedback_minimum_candidate_inlier_gain = int(
            feedback_minimum_candidate_inlier_gain
        )
        self.feedback_minimum_candidate_relative_inlier_gain = float(
            feedback_minimum_candidate_relative_inlier_gain
        )
        self.feedback_maximum_candidate_ransac_iterations = int(
            feedback_maximum_candidate_ransac_iterations
        )
        self.feedback_minimum_baseline_inlier_retention = float(
            feedback_minimum_baseline_inlier_retention
        )
        self.feedback_maximum_protected_median_residual_increase_px = float(
            feedback_maximum_protected_median_residual_increase_px
        )
        self.feedback_maximum_protected_p90_residual_increase_px = float(
            feedback_maximum_protected_p90_residual_increase_px
        )
        self.feedback_maximum_pose_update_translation_cm = float(
            feedback_maximum_pose_update_translation_cm
        )
        self.feedback_maximum_pose_update_rotation_deg = float(
            feedback_maximum_pose_update_rotation_deg
        )
        from map_learning.v24_pose_conditioned_sparse_refinement import (
            runtime_config as pose_conditioned_runtime_config,
        )

        self.pose_conditioned_selection_config = pose_conditioned_runtime_config(
            maximum_score_drop_from_top1=(
                refinement_maximum_score_drop_from_top1
            ),
            view_direction_slack_deg=refinement_view_direction_slack_deg,
            maximum_changed_rows=refinement_maximum_changed_rows,
            maximum_changed_to_baseline_inlier_ratio=(
                refinement_maximum_changed_to_baseline_inlier_ratio
            ),
        )
        self.refinement_minimum_changed_inliers = int(
            refinement_minimum_changed_inliers
        )
        self.refinement_minimum_proposal_count = int(
            refinement_minimum_proposal_count
        )
        self.refinement_minimum_proposal_relative_gain = float(
            refinement_minimum_proposal_relative_gain
        )
        if not (
            self.refinement_minimum_proposal_count >= 0
            and 0.0 <= self.refinement_minimum_proposal_relative_gain <= 1.0
        ):
            raise ValueError("V24 proposal pre-gate is invalid")
        self.refinement_minimum_changed_inlier_fraction = float(
            refinement_minimum_changed_inlier_fraction
        )
        self.refinement_minimum_changed_inlier_spatial_cells = int(
            refinement_minimum_changed_inlier_spatial_cells
        )
        self.refinement_maximum_changed_inlier_median_residual_px = float(
            refinement_maximum_changed_inlier_median_residual_px
        )
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
        if sum(
            (
                self.topk_geometric_feedback,
                self.sparse_lgcv_topk_feedback,
                self.pose_conditioned_sparse_refinement,
            )
        ) > 1:
            raise ValueError(
                "online sparse-refinement modes are separate ablations"
            )
        if (
            self.feedback_minimum_baseline_inliers < 4
            or self.feedback_maximum_baseline_inliers < 0
            or (
                self.feedback_maximum_baseline_inliers
                and self.feedback_maximum_baseline_inliers
                <= self.feedback_minimum_baseline_inliers
            )
        ):
            raise ValueError("feedback baseline inlier band is invalid")
        if not 0.0 <= self.feedback_minimum_baseline_inlier_retention <= 1.0:
            raise ValueError("feedback baseline inlier retention is invalid")
        if (
            self.feedback_minimum_candidate_inlier_gain < 0
            or self.feedback_minimum_candidate_relative_inlier_gain < 0.0
            or self.feedback_maximum_candidate_ransac_iterations < 0
        ):
            raise ValueError("feedback candidate acceptance gate is invalid")
        if (
            self.refinement_minimum_changed_inliers < 0
            or not 0.0 <= self.refinement_minimum_changed_inlier_fraction <= 1.0
            or self.refinement_minimum_changed_inlier_spatial_cells < 0
            or self.refinement_maximum_changed_inlier_median_residual_px < 0.0
        ):
            raise ValueError("pose-conditioned refinement support gate is invalid")
        if (
            self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or self.pose_conditioned_sparse_refinement
        ) and (
            self.assignment_topk
            or self.suppress_duplicate_anchors
            or self.guided_sampling
            or context_state_path is not None
        ):
            raise ValueError(
                "sparse LGCV Top-K feedback is a separate shared-metric deployment "
                "ablation"
            )
        if self.anchor_extra_prototype_owner_rows.numel() and (
            self.assignment_topk
            or self.guided_sampling
            or self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or self.pose_conditioned_sparse_refinement
            or context_state_path is not None
        ):
            raise ValueError(
                "sparse prototypes are currently authorized only for the minimal "
                "global-Top1 shared-metric deployment"
            )
        self.group_aware_pose = bool(group_aware_pose)
        self.group_hypothesis_samples = int(group_hypothesis_samples)
        if self.group_aware_pose and (
            self.guided_sampling
            or self.assignment_topk
            or self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or self.pose_conditioned_sparse_refinement
        ):
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
        self._sparse_feedback_anchor_xyz_cpu = (
            self.anchor_xyz.detach().float().cpu()
            if self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or self.pose_conditioned_sparse_refinement
            else None
        )
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
        if not (
            bool(torch.isfinite(self.anchor_matchability).all())
            and bool(torch.isfinite(self.anchor_uncertainty).all())
            and bool(torch.isfinite(self.anchor_xyz).all())
            and bool((self.anchor_uncertainty >= 0).all())
        ):
            raise ValueError("mapping reliability metadata is invalid")

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
        feedback_topk = None
        eager_feedback_topk = bool(
            self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or (
                self.pose_conditioned_sparse_refinement
                and self.refinement_pose_backend == "local"
            )
        )
        if eager_feedback_topk:
            feedback_topk = global_cosine_topk(
                sparse.descriptors,
                self.anchor_features,
                topk=64,
                anchor_descriptors_normalized=True,
            )
            raw_matches = Top1Matches(
                keypoint_indices=feedback_topk.keypoint_indices,
                anchor_indices=feedback_topk.anchor_indices[:, 0],
                scores=feedback_topk.scores[:, 0],
            )
            matches = raw_matches
        elif self.assignment_topk:
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
            raw_matches = global_owner_prototype_top1(
                sparse.descriptors,
                self.anchor_features,
                self.anchor_extra_prototype_features,
                self.anchor_extra_prototype_owner_rows,
                anchor_descriptors_normalized=True,
            )
            matches = raw_matches
        if not self.assignment_topk and not eager_feedback_topk and not (
            self.pose_conditioned_sparse_refinement
        ):
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
        baseline_pose_inlier_count = int(pose.inliers.size)
        feedback_geometry_ms = 0.0
        feedback_ransac_ms = 0.0
        feedback_eligible = False
        feedback_gate_passed = False
        feedback_accepted = False
        feedback_proposed_rows = 0
        feedback_supported_rows = 0
        feedback_candidate_inliers = 0
        feedback_candidate_inlier_gain = 0
        feedback_candidate_pose_w2c = None
        feedback_candidate_ransac_iterations = 0
        feedback_baseline_inlier_retention_fraction = 0.0
        feedback_protected_median_residual_increase_px = 0.0
        feedback_protected_p90_residual_increase_px = 0.0
        feedback_pose_update_translation_cm = 0.0
        feedback_pose_update_rotation_deg = 0.0
        feedback_changed_rows_entering_candidate_inliers = 0
        feedback_changed_inlier_fraction = 0.0
        feedback_changed_inlier_spatial_cells = 0
        feedback_changed_inlier_median_residual_px = 0.0
        feedback_support_passed = False
        feedback_eligible_edges = 0
        feedback_duplicate_owner_rejections = 0
        feedback_view_support_available = False
        feedback_view_support_rejected_edges = 0
        feedback_capacity_rejections = 0
        feedback_visible_anchor_count = 0
        feedback_candidate_pool_anchor_count = 0
        feedback_candidate_pool_global_fallback = False
        feedback_selected_rank_median = 0.0
        feedback_selected_rank_p90 = 0.0
        feedback_selected_score_drop_median = 0.0
        feedback_selected_score_drop_p90 = 0.0
        feedback_selected_joint_cost_median = 0.0
        feedback_selected_reprojection_median_px = 0.0
        feedback_rejected_by_protection = False
        if (
            self.topk_geometric_feedback
            or self.sparse_lgcv_topk_feedback
            or self.pose_conditioned_sparse_refinement
        ):
            if self._sparse_feedback_anchor_xyz_cpu is None:
                raise RuntimeError("online sparse-refinement state is missing")
            # Runtime import avoids a package cycle during localization module
            # initialization.  The selected functions contain no learned state
            # and never consume ground truth.
            from map_learning.v22_sparse_lgcv_feedback import (
                default_config as sparse_feedback_config,
                filter_provisional_assignment_with_sparse_lgcv,
            )
            from map_learning.v21_topk_geometric_feedback import (
                select_topk_geometry_rows,
            )

            feedback_cfg = sparse_feedback_config()
            feedback_eligible = bool(
                self.feedback_minimum_baseline_inliers <= int(pose.inliers.size)
                and (
                    not self.feedback_maximum_baseline_inliers
                    or int(pose.inliers.size)
                    < self.feedback_maximum_baseline_inliers
                )
            )
            if feedback_eligible:
                feedback_started = time.perf_counter()
                if feedback_topk is None:
                    # Robust V24 is staged: the normal Top-1/PnP path runs
                    # first, and exact Top-K is paid only by queries inside
                    # the predeclared refinement trigger band.
                    from map_learning.v24_pose_conditioned_sparse_refinement import (
                        build_pose_visible_topk,
                    )

                    visible_topk = build_pose_visible_topk(
                        query_descriptors=sparse.descriptors,
                        normalized_anchor_features=self.anchor_features,
                        baseline_anchor_rows=matches.anchor_indices,
                        baseline_scores=matches.scores,
                        anchor_xyz=self.anchor_xyz,
                        intrinsic=torch.as_tensor(intrinsic, device=self.device),
                        baseline_pose_w2c=torch.as_tensor(
                            pose.pose_w2c, device=self.device
                        ),
                        image_hw=sparse.image_hw,
                    )
                    feedback_topk = visible_topk["matches"]
                    feedback_visible_anchor_count = int(
                        visible_topk["visible_anchor_count"]
                    )
                    feedback_candidate_pool_anchor_count = int(
                        visible_topk["candidate_pool_anchor_count"]
                    )
                    feedback_candidate_pool_global_fallback = bool(
                        visible_topk["global_fallback"]
                    )
                baseline_rows_cpu = feedback_topk.anchor_indices[:, 0].cpu()
                if self.pose_conditioned_sparse_refinement:
                    from map_learning.v24_pose_conditioned_sparse_refinement import (
                        select_pose_conditioned_rows,
                    )

                    provisional_device = select_pose_conditioned_rows(
                        keypoints=sparse.keypoints.float() + 0.5,
                        topk_anchor_rows=feedback_topk.anchor_indices,
                        topk_scores=feedback_topk.scores,
                        baseline_inlier_rows=torch.as_tensor(
                            pose.inliers.copy(), device=self.device
                        ).long(),
                        anchor_xyz=self.anchor_xyz,
                        intrinsic=torch.as_tensor(
                            intrinsic, device=self.device
                        ).float(),
                        baseline_pose_w2c=torch.as_tensor(
                            pose.pose_w2c, device=self.device
                        ).float(),
                        anchor_view_support=self.anchor_view_support,
                        anchor_matchability=self.anchor_matchability,
                        anchor_uncertainty=self.anchor_uncertainty,
                        mapping_reliability_validated=True,
                        map_geometry_validated=True,
                        config=self.pose_conditioned_selection_config,
                    )
                    # This is the only V24 device-to-host transfer before the
                    # second solve: one selected row per query plus compact
                    # diagnostics, rather than the full [query, Top-K] bank.
                    synchronize()
                    provisional = {
                        key: value.detach().cpu()
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in provisional_device.items()
                    }
                    feedback_eligible_edges = int(
                        provisional["eligible_edge_count"]
                    )
                    feedback_duplicate_owner_rejections = int(
                        provisional["duplicate_candidate_owner_rejection_count"]
                    )
                    feedback_view_support_available = bool(
                        provisional["view_support_available"]
                    )
                    feedback_view_support_rejected_edges = int(
                        provisional["view_support_rejected_edge_count"]
                    )
                    feedback_capacity_rejections = int(
                        provisional["capacity_rejection_count"]
                    )
                else:
                    topk_cfg = dict(feedback_cfg["topk_geometry"])
                    topk_cfg["minimum_baseline_inlier_count_inclusive"] = (
                        self.feedback_minimum_baseline_inliers
                    )
                    topk_cfg["maximum_baseline_inlier_count_exclusive"] = (
                        self.feedback_maximum_baseline_inliers
                    )
                    provisional = select_topk_geometry_rows(
                        keypoints=sparse.keypoints.float().cpu() + 0.5,
                        topk_anchor_rows=feedback_topk.anchor_indices.cpu(),
                        topk_scores=feedback_topk.scores.cpu(),
                        baseline_anchor_rows=baseline_rows_cpu,
                        baseline_scores=feedback_topk.scores[:, 0].cpu(),
                        baseline_inlier_rows=torch.from_numpy(
                            pose.inliers.copy()
                        ).long(),
                        anchor_xyz=self._sparse_feedback_anchor_xyz_cpu,
                        intrinsic=torch.from_numpy(intrinsic),
                        baseline_pose_w2c=torch.from_numpy(pose.pose_w2c),
                        config=topk_cfg,
                        allow_runtime_inlier_band=True,
                    )
                feedback_proposed_rows = int(
                    provisional["changed_query_rows"].numel()
                )
                if self.pose_conditioned_sparse_refinement and feedback_proposed_rows:
                    selected_ranks = provisional["selected_candidate_ranks"].float()
                    selected_score_drop = provisional["selected_score_drop"].float()
                    selected_joint_cost = provisional["selected_joint_cost"].float()
                    selected_reprojection = provisional[
                        "selected_reprojection_residual_px"
                    ].float()
                    feedback_selected_rank_median = float(selected_ranks.median())
                    feedback_selected_rank_p90 = float(
                        torch.quantile(selected_ranks, 0.90)
                    )
                    feedback_selected_score_drop_median = float(
                        selected_score_drop.median()
                    )
                    feedback_selected_score_drop_p90 = float(
                        torch.quantile(selected_score_drop, 0.90)
                    )
                    feedback_selected_joint_cost_median = float(
                        selected_joint_cost.median()
                    )
                    feedback_selected_reprojection_median_px = float(
                        selected_reprojection.median()
                    )
                if self.sparse_lgcv_topk_feedback:
                    filtered = filter_provisional_assignment_with_sparse_lgcv(
                        keypoints=sparse.keypoints.float().cpu() + 0.5,
                        baseline_anchor_rows=baseline_rows_cpu,
                        provisional_anchor_rows=provisional["anchor_rows"],
                        provisional_changed_query_rows=provisional[
                            "changed_query_rows"
                        ],
                        baseline_inlier_rows=torch.from_numpy(
                            pose.inliers.copy()
                        ).long(),
                        anchor_xyz=self._sparse_feedback_anchor_xyz_cpu,
                        intrinsic=torch.from_numpy(intrinsic),
                        baseline_pose_w2c=torch.from_numpy(pose.pose_w2c),
                        config=feedback_cfg,
                    )
                    feedback_supported_rows = int(
                        filtered["supported_changed_query_rows"].numel()
                    )
                    supported_fraction = float(
                        feedback_supported_rows / max(feedback_proposed_rows, 1)
                    )
                    feedback_gate_passed = bool(
                        feedback_supported_rows
                        >= int(
                            feedback_cfg["minimum_query_supported_proposal_count"]
                        )
                        and supported_fraction
                        >= float(
                            feedback_cfg[
                                "minimum_query_supported_proposal_fraction"
                            ]
                        )
                    )
                else:
                    # The plain Top-K arm is the exact same provisional bundle
                    # without LGCV.  V24 additionally changes how that bundle is
                    # selected, but still runs at most one additional solve.
                    feedback_supported_rows = feedback_proposed_rows
                    feedback_gate_passed = bool(
                        feedback_proposed_rows
                        >= (
                            self.refinement_minimum_proposal_count
                            if self.pose_conditioned_sparse_refinement
                            else 1
                        )
                        and (
                            not self.pose_conditioned_sparse_refinement
                            or feedback_proposed_rows
                            / max(int(pose.inliers.size), 1)
                            >= self.refinement_minimum_proposal_relative_gain
                        )
                    )
                feedback_geometry_ms = (
                    time.perf_counter() - feedback_started
                ) * 1000.0
                if feedback_gate_passed:
                    candidate_rows_cpu = provisional["anchor_rows"]
                    candidate_points_3d = self._sparse_feedback_anchor_xyz_cpu[
                        candidate_rows_cpu
                    ].numpy()
                    baseline_inliers = np.asarray(pose.inliers, dtype=np.int64)
                    feedback_pose_started = time.perf_counter()
                    if (
                        self.pose_conditioned_sparse_refinement
                        and self.refinement_pose_backend == "local"
                    ):
                        optimization_rows = np.union1d(
                            baseline_inliers,
                            provisional["changed_query_rows"].numpy(),
                        )
                        candidate_pose = refine_absolute_pose_from_initial(
                            points_2d + 0.5,
                            candidate_points_3d,
                            intrinsic,
                            pose.pose_w2c,
                            optimization_rows,
                            reprojection_error_px=self.reprojection_error_px,
                            camera=pose_camera,
                        )
                    else:
                        candidate_pose = solve_absolute_pose(
                            points_2d + 0.5,
                            candidate_points_3d,
                            intrinsic,
                            camera=pose_camera,
                            **solve_kwargs,
                        )
                    feedback_ransac_ms = (
                        time.perf_counter() - feedback_pose_started
                    ) * 1000.0
                    feedback_candidate_inliers = int(candidate_pose.inliers.size)
                    feedback_candidate_pose_w2c = candidate_pose.pose_w2c.tolist()
                    feedback_candidate_ransac_iterations = int(
                        candidate_pose.diagnostics.get("iterations", 0)
                    )
                    feedback_candidate_inlier_gain = int(
                        feedback_candidate_inliers - int(pose.inliers.size)
                    )
                    feedback_candidate_relative_inlier_gain = float(
                        feedback_candidate_inlier_gain / max(int(pose.inliers.size), 1)
                    )
                    candidate_inliers = np.asarray(
                        candidate_pose.inliers, dtype=np.int64
                    )
                    retained = np.intersect1d(
                        baseline_inliers, candidate_inliers, assume_unique=True
                    )
                    feedback_baseline_inlier_retention_fraction = float(
                        retained.size / max(baseline_inliers.size, 1)
                    )
                    changed_numpy = provisional["changed_query_rows"].numpy()
                    feedback_changed_rows_entering_candidate_inliers = int(
                        np.intersect1d(
                            changed_numpy, candidate_inliers, assume_unique=True
                        ).size
                    )
                    feedback_changed_inlier_fraction = float(
                        feedback_changed_rows_entering_candidate_inliers
                        / max(feedback_proposed_rows, 1)
                    )
                    baseline_residual = _reprojection_residuals(
                        points_2d + 0.5,
                        points_3d,
                        pose.pose_w2c,
                        intrinsic,
                    )[baseline_inliers]
                    protected_candidate_residual = _reprojection_residuals(
                        points_2d + 0.5,
                        points_3d,
                        candidate_pose.pose_w2c,
                        intrinsic,
                    )[baseline_inliers]
                    candidate_assignment_residual = _reprojection_residuals(
                        points_2d + 0.5,
                        candidate_points_3d,
                        candidate_pose.pose_w2c,
                        intrinsic,
                    )
                    changed_candidate_inliers = np.intersect1d(
                        changed_numpy, candidate_inliers, assume_unique=True
                    )
                    if changed_candidate_inliers.size:
                        feedback_changed_inlier_median_residual_px = float(
                            np.median(
                                candidate_assignment_residual[
                                    changed_candidate_inliers
                                ]
                            )
                        )
                    if baseline_inliers.size:
                        feedback_protected_median_residual_increase_px = float(
                            np.median(protected_candidate_residual)
                            - np.median(baseline_residual)
                        )
                        feedback_protected_p90_residual_increase_px = float(
                            np.percentile(protected_candidate_residual, 90)
                            - np.percentile(baseline_residual, 90)
                        )
                    (
                        feedback_pose_update_translation_cm,
                        feedback_pose_update_rotation_deg,
                    ) = _pose_update_magnitude(
                        pose.pose_w2c, candidate_pose.pose_w2c
                    )
                    if self.pose_conditioned_sparse_refinement:
                        from map_learning.v24_pose_conditioned_sparse_refinement import (
                            changed_inlier_spatial_cell_count,
                        )

                        feedback_changed_inlier_spatial_cells = (
                            changed_inlier_spatial_cell_count(
                                keypoints=sparse.keypoints.float().cpu() + 0.5,
                                changed_query_rows=provisional[
                                    "changed_query_rows"
                                ],
                                candidate_inlier_rows=torch.from_numpy(
                                    candidate_inliers.copy()
                                ).long(),
                                image_hw=sparse.image_hw,
                            )
                        )
                        feedback_support_passed = bool(
                            feedback_changed_rows_entering_candidate_inliers
                            >= self.refinement_minimum_changed_inliers
                            and feedback_changed_inlier_fraction
                            >= self.refinement_minimum_changed_inlier_fraction
                            and feedback_changed_inlier_spatial_cells
                            >= self.refinement_minimum_changed_inlier_spatial_cells
                            and feedback_changed_inlier_median_residual_px
                            <= self.refinement_maximum_changed_inlier_median_residual_px
                        )
                    else:
                        feedback_support_passed = True
                    protection_passed = bool(
                        feedback_baseline_inlier_retention_fraction
                        >= self.feedback_minimum_baseline_inlier_retention
                        and (
                            self.feedback_maximum_protected_median_residual_increase_px
                            < 0
                            or feedback_protected_median_residual_increase_px
                            <= self.feedback_maximum_protected_median_residual_increase_px
                        )
                        and (
                            self.feedback_maximum_protected_p90_residual_increase_px
                            < 0
                            or feedback_protected_p90_residual_increase_px
                            <= self.feedback_maximum_protected_p90_residual_increase_px
                        )
                        and (
                            self.feedback_maximum_pose_update_translation_cm < 0
                            or feedback_pose_update_translation_cm
                            <= self.feedback_maximum_pose_update_translation_cm
                        )
                        and (
                            self.feedback_maximum_pose_update_rotation_deg < 0
                            or feedback_pose_update_rotation_deg
                            <= self.feedback_maximum_pose_update_rotation_deg
                        )
                        and (
                            not self.pose_conditioned_sparse_refinement
                            or (
                                feedback_baseline_inlier_retention_fraction >= 0.90
                                and feedback_protected_median_residual_increase_px
                                <= 0.50
                                and feedback_protected_p90_residual_increase_px <= 2.0
                                and feedback_pose_update_translation_cm <= 50.0
                                and feedback_pose_update_rotation_deg <= 2.0
                            )
                        )
                    )
                    feedback_rejected_by_protection = not protection_passed
                    feedback_accepted = bool(
                        feedback_candidate_inlier_gain
                        >= self.feedback_minimum_candidate_inlier_gain
                        and feedback_candidate_relative_inlier_gain
                        >= self.feedback_minimum_candidate_relative_inlier_gain
                        and (
                            not self.feedback_maximum_candidate_ransac_iterations
                            or int(candidate_pose.diagnostics.get("iterations", 0))
                            <= self.feedback_maximum_candidate_ransac_iterations
                        )
                        and protection_passed
                        and feedback_support_passed
                    )
                    if feedback_accepted:
                        changed = provisional["changed_query_rows"]
                        candidate_scores = feedback_topk.scores[:, 0].clone()
                        if changed.numel():
                            ranks = provisional["selected_candidate_ranks"].to(
                                feedback_topk.scores.device
                            )
                            changed_device = changed.to(feedback_topk.scores.device)
                            candidate_scores[changed_device] = feedback_topk.scores[
                                changed_device, ranks - 1
                            ]
                        matches = Top1Matches(
                            keypoint_indices=feedback_topk.keypoint_indices,
                            anchor_indices=candidate_rows_cpu.to(self.device),
                            scores=candidate_scores,
                        )
                        pose = candidate_pose
        return LocalizationResult(
            sparse,
            matches,
            pose,
            intrinsic,
            {
                "frontend_ms": frontend_ms,
                "matching_ms": matching_ms,
                "ransac_ms": ransac_ms,
                "feedback_geometry_ms": feedback_geometry_ms,
                "feedback_ransac_ms": feedback_ransac_ms,
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
                "sparse_lgcv_topk_feedback": int(
                    self.sparse_lgcv_topk_feedback
                ),
                "topk_geometric_feedback": int(self.topk_geometric_feedback),
                "pose_conditioned_sparse_refinement": int(
                    self.pose_conditioned_sparse_refinement
                ),
                "sparse_feedback_pose_backend": (
                    "poselib_local_nonlinear_refinement"
                    if self.pose_conditioned_sparse_refinement
                    and self.refinement_pose_backend == "local"
                    else "poselib_bounded_robust_reestimate"
                    if self.pose_conditioned_sparse_refinement
                    else "poselib_second_ransac"
                    if self.topk_geometric_feedback
                    or self.sparse_lgcv_topk_feedback
                    else "disabled"
                ),
                "sparse_feedback_minimum_baseline_inliers": int(
                    self.feedback_minimum_baseline_inliers
                ),
                "sparse_feedback_maximum_baseline_inliers": int(
                    self.feedback_maximum_baseline_inliers
                ),
                "sparse_feedback_eligible": int(feedback_eligible),
                "sparse_feedback_gate_passed": int(feedback_gate_passed),
                "sparse_feedback_accepted": int(feedback_accepted),
                "sparse_feedback_proposed_rows": feedback_proposed_rows,
                "sparse_feedback_proposal_relative_gain": float(
                    feedback_proposed_rows / max(baseline_pose_inlier_count, 1)
                ),
                "sparse_feedback_minimum_proposal_count": int(
                    self.refinement_minimum_proposal_count
                ),
                "sparse_feedback_minimum_proposal_relative_gain": float(
                    self.refinement_minimum_proposal_relative_gain
                ),
                "sparse_feedback_supported_rows": feedback_supported_rows,
                "sparse_feedback_candidate_inliers": feedback_candidate_inliers,
                "sparse_feedback_candidate_inlier_gain": (
                    feedback_candidate_inlier_gain
                ),
                "sparse_feedback_candidate_pose_w2c": (
                    feedback_candidate_pose_w2c
                ),
                "sparse_feedback_candidate_ransac_iterations": (
                    feedback_candidate_ransac_iterations
                ),
                "sparse_feedback_candidate_relative_inlier_gain": float(
                    feedback_candidate_inlier_gain
                    / max(feedback_candidate_inliers - feedback_candidate_inlier_gain, 1)
                ),
                "sparse_feedback_baseline_inlier_retention_fraction": (
                    feedback_baseline_inlier_retention_fraction
                ),
                "sparse_feedback_protected_median_residual_increase_px": (
                    feedback_protected_median_residual_increase_px
                ),
                "sparse_feedback_protected_p90_residual_increase_px": (
                    feedback_protected_p90_residual_increase_px
                ),
                "sparse_feedback_pose_update_translation_cm": (
                    feedback_pose_update_translation_cm
                ),
                "sparse_feedback_pose_update_rotation_deg": (
                    feedback_pose_update_rotation_deg
                ),
                "sparse_feedback_changed_rows_entering_candidate_inliers": (
                    feedback_changed_rows_entering_candidate_inliers
                ),
                "sparse_feedback_changed_inlier_fraction": (
                    feedback_changed_inlier_fraction
                ),
                "sparse_feedback_changed_inlier_spatial_cells": (
                    feedback_changed_inlier_spatial_cells
                ),
                "sparse_feedback_changed_inlier_median_residual_px": (
                    feedback_changed_inlier_median_residual_px
                ),
                "sparse_feedback_support_passed": int(feedback_support_passed),
                "sparse_feedback_eligible_candidate_edges": feedback_eligible_edges,
                "sparse_feedback_duplicate_owner_rejections": (
                    feedback_duplicate_owner_rejections
                ),
                "sparse_feedback_view_support_available": int(
                    feedback_view_support_available
                ),
                "sparse_feedback_view_support_rejected_edges": (
                    feedback_view_support_rejected_edges
                ),
                "sparse_feedback_capacity_rejections": feedback_capacity_rejections,
                "sparse_feedback_visible_anchor_count": (
                    feedback_visible_anchor_count
                ),
                "sparse_feedback_candidate_pool_anchor_count": (
                    feedback_candidate_pool_anchor_count
                ),
                "sparse_feedback_candidate_pool_global_fallback": int(
                    feedback_candidate_pool_global_fallback
                ),
                "sparse_feedback_selected_rank_median": (
                    feedback_selected_rank_median
                ),
                "sparse_feedback_selected_rank_p90": feedback_selected_rank_p90,
                "sparse_feedback_selected_score_drop_median": (
                    feedback_selected_score_drop_median
                ),
                "sparse_feedback_selected_score_drop_p90": (
                    feedback_selected_score_drop_p90
                ),
                "sparse_feedback_selected_joint_cost_median": (
                    feedback_selected_joint_cost_median
                ),
                "sparse_feedback_selected_reprojection_median_px": (
                    feedback_selected_reprojection_median_px
                ),
                "sparse_feedback_rejected_by_protection": int(
                    feedback_rejected_by_protection
                ),
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
