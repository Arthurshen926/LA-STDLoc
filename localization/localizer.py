"""Minimal LaFGS deployment runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.hashing import sha256_file
from localization.frontend import NativeSuperPointFrontend, SparseFeatures
from localization.matcher import (
    Top1Matches,
    global_cosine_top2,
    global_cosine_topk,
    global_owner_prototype_top1,
    maximum_weight_anchor_assignment,
    retain_diverse_confidence_matches,
    retain_high_score_matches,
    suppress_duplicate_anchor_matches,
)
from localization.pose_solver import (
    PoseEstimate,
    camera_intrinsics,
    poselib_camera,
    refine_absolute_pose_from_initial,
    solve_absolute_pose,
    solve_absolute_pose_from_hypothesis_core,
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


def _mapping_quality_polish_rows(
    strict_rows: np.ndarray,
    *,
    match_anchor_rows: torch.Tensor,
    anchor_quality: torch.Tensor,
    retention_fraction: float,
    minimum_count: int,
) -> np.ndarray:
    """Retain the most geometrically reliable strict inliers for local polish."""

    rows = np.asarray(strict_rows, dtype=np.int64).reshape(-1)
    owners = torch.as_tensor(match_anchor_rows).long().reshape(-1)
    quality = torch.as_tensor(anchor_quality, device=owners.device).float().reshape(-1)
    fraction = float(retention_fraction)
    minimum = int(minimum_count)
    if not (
        0.25 <= fraction <= 1.0
        and minimum >= 4
        and np.unique(rows).size == rows.size
        and (rows.size == 0 or (int(rows.min()) >= 0 and int(rows.max()) < owners.numel()))
        and (owners.numel() == 0 or (int(owners.min()) >= 0 and int(owners.max()) < quality.numel()))
        and bool(torch.isfinite(quality).all())
    ):
        raise ValueError("mapping-quality pose-polish rows are invalid")
    keep_count = min(
        rows.size,
        max(minimum, int(math.ceil(rows.size * fraction))),
    )
    if keep_count == rows.size:
        return rows.copy()
    row_tensor = torch.from_numpy(rows).to(owners.device)
    row_quality = quality[owners[row_tensor]]
    ordering = torch.argsort(row_quality, descending=True, stable=True)[:keep_count]
    return rows[ordering.detach().cpu().numpy()]


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


def _local_inliers_to_query_rows(
    matches: Top1Matches, inliers: np.ndarray
) -> np.ndarray:
    """Map solver-local inlier indices back to the full query registry."""

    rows = np.asarray(inliers, dtype=np.int64).reshape(-1)
    count = int(matches.keypoint_indices.numel())
    if rows.size and (
        int(rows.min()) < 0
        or int(rows.max()) >= count
        or np.unique(rows).size != rows.size
    ):
        raise ValueError("pose inliers do not index the active match registry")
    return (
        matches.keypoint_indices[
            torch.as_tensor(rows, device=matches.keypoint_indices.device).long()
        ]
        .detach()
        .long()
        .cpu()
        .numpy()
    )


def _query_level_feedback_count(
    detector_ranks: torch.Tensor | None,
    *,
    extracted_count: int,
    first_pass_query_cap: int,
    baseline_inlier_count: int,
    first_pass_match_count: int,
    expanded_reserve_maximum_inlier_fraction: float,
) -> tuple[int, bool]:
    """Choose the sparse Reserve size from T0 evidence only.

    A difficult query may use detector rows beyond the exact T0 prefix.  A
    stable query keeps the original prefix, preventing the larger extraction
    budget from changing its second-stage correspondence registry.
    """

    if not expanded_reserve_maximum_inlier_fraction:
        return int(extracted_count), False
    if first_pass_query_cap <= 0 or first_pass_match_count <= 0:
        raise ValueError("adaptive Reserve requires a non-empty first-pass cap")
    if detector_ranks is None:
        ranks = torch.arange(extracted_count)
    else:
        ranks = torch.as_tensor(detector_ranks).long().reshape(-1)
    if ranks.numel() != extracted_count:
        raise ValueError("detector ranks do not align with extracted keypoints")
    prefix = ranks < int(first_pass_query_cap)
    prefix_count = int(prefix.sum().item())
    if prefix_count <= 0:
        raise ValueError("adaptive Reserve has an empty first-pass registry")
    expected_prefix = torch.arange(prefix_count, device=prefix.device)
    if not torch.equal(
        torch.nonzero(prefix, as_tuple=False).reshape(-1), expected_prefix
    ):
        raise ValueError("first-pass detector rows must form an exact prefix")
    inlier_fraction = float(baseline_inlier_count) / float(first_pass_match_count)
    expanded = inlier_fraction <= float(expanded_reserve_maximum_inlier_fraction)
    return (int(extracted_count) if expanded else prefix_count), expanded


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
        view_conditioned_anchor_state_path: str | Path | None = None,
        view_conditioned_minimum_concentration: float = 0.0,
        view_conditioned_residual_scale: float = 1.0,
        view_conditioned_require_two_valid_modes: bool = False,
        view_conditioned_score_fusion: str = "replace",
        device: torch.device | str = "cuda",
        keypoint_count: int = 2048,
        nms_radius: int = 4,
        subpixel_keypoints: bool = False,
        subpixel_geometry_only: bool = False,
        subpixel_maximum_offset: float = 0.5,
        reprojection_error_px: float = 12.0,
        confidence: float = 0.99999,
        max_iterations: int = 100000,
        min_iterations: int = 1000,
        seed: int = 2026,
        ransac_hypothesis_core_size: int = 0,
        suppress_duplicate_anchors: bool = False,
        guided_sampling: bool = False,
        confidence_core_progressive_sampling: bool = False,
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
        refinement_projection_gate_px: float = 8.0,
        refinement_uncertainty_projection_gate_px: float = 0.0,
        refinement_uncertainty_maximum_baseline_inliers: int = 0,
        refinement_maximum_score_drop_from_top1: float = 0.03,
        refinement_reliability_adaptive_score_drop: bool = False,
        refinement_reliability_expanded_score_drop: float = 0.10,
        refinement_reliability_minimum_matchability_quantile: float = 0.50,
        refinement_reliability_maximum_uncertainty_quantile: float = 0.50,
        refinement_reliability_maximum_geometry_cost: float = 0.50,
        refinement_reliability_minimum_improvement_px: float = 4.0,
        refinement_view_direction_slack_deg: float = 15.0,
        refinement_maximum_changed_rows: int = 128,
        refinement_maximum_changed_to_baseline_inlier_ratio: float = 0.50,
        refinement_minimum_proposal_count: int = 60,
        refinement_minimum_proposal_relative_gain: float = 0.075,
        refinement_active_row_retrieval: bool = False,
        refinement_pre_topk_view_filter: bool = False,
        refinement_common_candidate_grid_gate: bool = False,
        refinement_minimum_common_grid_relative_energy_gain: float = 0.0,
        refinement_progressive_sampling: bool = False,
        refinement_allow_soft_inliers: bool = False,
        refinement_soft_inlier_minimum_residual_px: float = 6.0,
        refinement_soft_inlier_maximum_score_drop: float = 0.02,
        refinement_soft_inlier_minimum_improvement_px: float = 2.0,
        refinement_maximum_soft_inlier_changes: int = 16,
        refinement_pose_conditioned_mutual_matching: bool = False,
        refinement_set_level_reserve_selection: bool = False,
        refinement_projection_first_local_candidates: bool = False,
        refinement_projection_first_radius_px: float = 12.0,
        refinement_heldout_candidate_validation: bool = False,
        refinement_spatial_jackknife_diagnostic: bool = False,
        refinement_minimum_heldout_relative_energy_gain: float = 0.0,
        refinement_uncertainty_aware_projection: bool = False,
        refinement_maximum_uncertainty_projection_gate_px: float = 12.0,
        match_retention_fraction: float = 1.0,
        minimum_retained_match_count: int = 256,
        minimum_sufficient_confidence_core: bool = False,
        first_pass_query_cap: int = 0,
        refinement_expanded_reserve_maximum_inlier_fraction: float = 0.0,
        core_reserve_refinement: bool = False,
        core_reserve_reprojection_gate_px: float = 4.0,
        core_reserve_minimum_supported_rows: int = 16,
        final_pose_polish_reprojection_px: float = 0.0,
        final_pose_polish_minimum_inliers: int = 64,
        final_pose_polish_mapping_quality_fraction: float = 1.0,
        final_pose_polish_maximum_update_translation_cm: float = 10.0,
        final_pose_polish_maximum_update_rotation_deg: float = 0.10,
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
        if (
            view_conditioned_anchor_state_path is not None
            and context_state_path is not None
        ):
            raise ValueError(
                "V27 view-conditioned descriptors require the identity metric path"
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
                photometric_contract=state.get("photometric_canonicalization_contract"),
            )
            context_adapter = None
            if view_conditioned_anchor_state_path is not None and (
                metric.max_residual_norm != 0.0
                or any(
                    bool(torch.count_nonzero(value)) for value in metric.parameters()
                )
            ):
                raise ValueError(
                    "view-conditioned descriptors require an exact identity metric"
                )
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
                raw_view_support.get("schema") == "lafgs_v24_anchor_view_support"
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
                "direction_modes": support_modes[support_rows].float().to(self.device),
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
        self.view_conditioned_anchor_state = None
        self.view_conditioned_minimum_concentration = float(
            view_conditioned_minimum_concentration
        )
        self.view_conditioned_residual_scale = float(view_conditioned_residual_scale)
        self.view_conditioned_require_two_valid_modes = bool(
            view_conditioned_require_two_valid_modes
        )
        self.view_conditioned_score_fusion = str(view_conditioned_score_fusion)
        if not 0.0 <= self.view_conditioned_minimum_concentration <= 1.0:
            raise ValueError("V27 minimum descriptor concentration is invalid")
        if not 0.0 < self.view_conditioned_residual_scale <= 1.0:
            raise ValueError("V27 descriptor residual scale is invalid")
        if self.view_conditioned_score_fusion not in {"replace", "max_with_base"}:
            raise ValueError("V27 descriptor score fusion is invalid")
        if view_conditioned_anchor_state_path is not None:
            if self.anchor_view_support is None:
                raise ValueError(
                    "view-conditioned descriptors require F0 view support"
                )
            artifact = torch.load(
                view_conditioned_anchor_state_path,
                map_location="cpu",
                weights_only=False,
            )
            if artifact.get("schema") == (
                "lafgs_v27_mapping_view_conditioned_anchor_descriptors"
            ):
                from map_learning.v27_view_conditioned_anchor_descriptor import (
                    validate_artifact,
                )
            elif artifact.get("schema") == "anygsloc_v32_mapping_descriptor_mode_anchor":
                from map_learning.v32_descriptor_mode_anchor import validate_artifact
            else:
                raise ValueError("unsupported view-conditioned descriptor artifact")
            validate_artifact(artifact, map_state=state)
            stable_input = artifact["inputs"]["stable_map"]
            if stable_input.get("sha256") != sha256_file(map_path):
                raise ValueError("V27 descriptor state is bound to a different F0 map")
            self.view_conditioned_anchor_state = {
                "mode_features": torch.as_tensor(
                    artifact["mode_features"], device=self.device
                ).float(),
                "mode_valid": torch.as_tensor(
                    artifact["mode_valid"], device=self.device
                ).bool(),
                "mode_concentration": torch.as_tensor(
                    artifact["mode_concentration"], device=self.device
                ).float(),
                "mode_authorized": torch.as_tensor(
                    artifact.get("mode_authorized", artifact["mode_valid"]),
                    device=self.device,
                ).bool(),
            }
            if "mode_direction_vectors" in artifact:
                self.view_conditioned_anchor_state["direction_modes"] = (
                    torch.as_tensor(
                        artifact["mode_direction_vectors"], device=self.device
                    ).float()
                )
                self.view_conditioned_anchor_state["direction_radius_deg"] = (
                    torch.as_tensor(
                        artifact["mode_direction_radius_deg"], device=self.device
                    ).float()
                )
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
            self.anchor_extra_prototype_owner_rows = (
                torch.as_tensor(prototype_owners, device=self.device).long().reshape(-1)
            )
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
            subpixel_keypoints=subpixel_keypoints,
            subpixel_geometry_only=subpixel_geometry_only,
            subpixel_maximum_offset=subpixel_maximum_offset,
            metric=metric,
            context_adapter=context_adapter,
            photometric_contract=state.get("photometric_canonicalization_contract"),
        )
        self.photometric_canonicalization_contract = state.get(
            "photometric_canonicalization_contract"
        )
        self.subpixel_keypoints = bool(subpixel_keypoints)
        self.subpixel_geometry_only = bool(subpixel_geometry_only)
        self.subpixel_maximum_offset = float(subpixel_maximum_offset)
        self.reprojection_error_px = float(reprojection_error_px)
        self.confidence = float(confidence)
        self.max_iterations = int(max_iterations)
        self.min_iterations = int(min_iterations)
        self.seed = int(seed)
        self.ransac_hypothesis_core_size = int(ransac_hypothesis_core_size)
        if self.ransac_hypothesis_core_size and not (
            64 <= self.ransac_hypothesis_core_size <= keypoint_count
        ):
            raise ValueError("RANSAC hypothesis core size is invalid")
        self.suppress_duplicate_anchors = bool(suppress_duplicate_anchors)
        self.guided_sampling = bool(guided_sampling)
        self.confidence_core_progressive_sampling = bool(
            confidence_core_progressive_sampling
        )
        self.assignment_topk = int(assignment_topk)
        self.assignment_dustbin_score = float(assignment_dustbin_score)
        self.topk_geometric_feedback = bool(topk_geometric_feedback)
        self.sparse_lgcv_topk_feedback = bool(sparse_lgcv_topk_feedback)
        self.pose_conditioned_sparse_refinement = bool(
            pose_conditioned_sparse_refinement
        )
        self.refinement_spatial_jackknife_diagnostic = bool(
            refinement_spatial_jackknife_diagnostic
        )
        if (
            self.refinement_spatial_jackknife_diagnostic
            and not self.pose_conditioned_sparse_refinement
        ):
            raise ValueError(
                "spatial jackknife diagnostic requires pose-conditioned refinement"
            )
        self.refinement_pose_backend = str(refinement_pose_backend)
        self.refinement_projection_first_local_candidates = bool(
            refinement_projection_first_local_candidates
        )
        self.refinement_projection_first_radius_px = float(
            refinement_projection_first_radius_px
        )
        if self.refinement_pose_backend not in {"local", "robust"}:
            raise ValueError("refinement pose backend must be local or robust")
        if not self.pose_conditioned_sparse_refinement and (
            self.refinement_pose_backend != "local"
        ):
            raise ValueError(
                "robust refinement backend requires pose-conditioned refinement"
            )
        if self.refinement_projection_first_local_candidates and not (
            self.pose_conditioned_sparse_refinement
            and 4.0 <= self.refinement_projection_first_radius_px <= 32.0
        ):
            raise ValueError(
                "projection-first candidates require pose-conditioned refinement"
            )
        self.feedback_minimum_baseline_inliers = int(feedback_minimum_baseline_inliers)
        self.feedback_maximum_baseline_inliers = int(feedback_maximum_baseline_inliers)
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
            projection_gate_px=refinement_projection_gate_px,
            maximum_score_drop_from_top1=(refinement_maximum_score_drop_from_top1),
            reliability_adaptive_score_drop=(
                refinement_reliability_adaptive_score_drop
            ),
            reliability_expanded_score_drop=(
                refinement_reliability_expanded_score_drop
            ),
            reliability_minimum_matchability_quantile=(
                refinement_reliability_minimum_matchability_quantile
            ),
            reliability_maximum_uncertainty_quantile=(
                refinement_reliability_maximum_uncertainty_quantile
            ),
            reliability_maximum_geometry_cost=(
                refinement_reliability_maximum_geometry_cost
            ),
            reliability_minimum_reprojection_improvement_px=(
                refinement_reliability_minimum_improvement_px
            ),
            view_direction_slack_deg=refinement_view_direction_slack_deg,
            maximum_changed_rows=refinement_maximum_changed_rows,
            maximum_changed_to_baseline_inlier_ratio=(
                refinement_maximum_changed_to_baseline_inlier_ratio
            ),
            allow_soft_inliers=refinement_allow_soft_inliers,
            soft_inlier_minimum_baseline_residual_px=(
                refinement_soft_inlier_minimum_residual_px
            ),
            soft_inlier_maximum_score_drop=(refinement_soft_inlier_maximum_score_drop),
            soft_inlier_minimum_reprojection_improvement_px=(
                refinement_soft_inlier_minimum_improvement_px
            ),
            maximum_soft_inlier_changes=(refinement_maximum_soft_inlier_changes),
            pose_conditioned_mutual_matching=(
                refinement_pose_conditioned_mutual_matching
            ),
            set_level_reserve_selection=(refinement_set_level_reserve_selection),
            heldout_candidate_validation=(refinement_heldout_candidate_validation),
            uncertainty_aware_projection=(refinement_uncertainty_aware_projection),
            maximum_uncertainty_projection_gate_px=(
                refinement_maximum_uncertainty_projection_gate_px
            ),
        )
        self.refinement_uncertainty_projection_gate_px = float(
            refinement_uncertainty_projection_gate_px
        )
        self.refinement_uncertainty_maximum_baseline_inliers = int(
            refinement_uncertainty_maximum_baseline_inliers
        )
        uncertainty_gate_enabled = bool(
            self.refinement_uncertainty_projection_gate_px > 0.0
            or self.refinement_uncertainty_maximum_baseline_inliers > 0
        )
        if uncertainty_gate_enabled and bool(
            self.pose_conditioned_selection_config["uncertainty_aware_projection"]
        ):
            raise ValueError(
                "count-adaptive and covariance-adaptive projection are separate ablations"
            )
        if uncertainty_gate_enabled and not (
            self.refinement_uncertainty_projection_gate_px
            >= float(self.pose_conditioned_selection_config["projection_gate_px"])
            and self.refinement_uncertainty_projection_gate_px <= 24.0
            and self.refinement_uncertainty_maximum_baseline_inliers >= 4
        ):
            raise ValueError("V25 uncertainty-aware projection gate is invalid")
        self.refinement_minimum_changed_inliers = int(
            refinement_minimum_changed_inliers
        )
        self.refinement_minimum_proposal_count = int(refinement_minimum_proposal_count)
        self.refinement_minimum_proposal_relative_gain = float(
            refinement_minimum_proposal_relative_gain
        )
        self.refinement_active_row_retrieval = bool(refinement_active_row_retrieval)
        self.refinement_pre_topk_view_filter = bool(refinement_pre_topk_view_filter)
        self.refinement_common_candidate_grid_gate = bool(
            refinement_common_candidate_grid_gate
        )
        self.refinement_minimum_common_grid_relative_energy_gain = float(
            refinement_minimum_common_grid_relative_energy_gain
        )
        self.refinement_progressive_sampling = bool(refinement_progressive_sampling)
        self.refinement_minimum_heldout_relative_energy_gain = float(
            refinement_minimum_heldout_relative_energy_gain
        )
        if not -1.0 <= self.refinement_minimum_heldout_relative_energy_gain <= 1.0:
            raise ValueError("held-out relative energy gain is invalid")
        if not (
            self.refinement_minimum_proposal_count >= 0
            and 0.0 <= self.refinement_minimum_proposal_relative_gain <= 1.0
        ):
            raise ValueError("V24 proposal pre-gate is invalid")
        if (
            self.refinement_active_row_retrieval
            or self.refinement_pre_topk_view_filter
            or self.refinement_common_candidate_grid_gate
            or self.refinement_progressive_sampling
            or uncertainty_gate_enabled
            or bool(
                self.pose_conditioned_selection_config[
                    "pose_conditioned_mutual_matching"
                ]
            )
            or bool(
                self.pose_conditioned_selection_config["heldout_candidate_validation"]
            )
            or bool(
                self.pose_conditioned_selection_config["uncertainty_aware_projection"]
            )
            or bool(self.pose_conditioned_selection_config["allow_soft_inliers"])
        ) and not (
            self.pose_conditioned_sparse_refinement
            and self.refinement_pose_backend == "robust"
        ):
            raise ValueError(
                "enhanced retrieval requires robust pose-conditioned refinement"
            )
        if not -1.0 <= self.refinement_minimum_common_grid_relative_energy_gain <= 1.0:
            raise ValueError("V25 common candidate-grid gate is invalid")
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
        self.match_retention_fraction = float(match_retention_fraction)
        self.minimum_retained_match_count = int(minimum_retained_match_count)
        self.minimum_sufficient_confidence_core = bool(
            minimum_sufficient_confidence_core
        )
        self.first_pass_query_cap = int(first_pass_query_cap)
        self.refinement_expanded_reserve_maximum_inlier_fraction = float(
            refinement_expanded_reserve_maximum_inlier_fraction
        )
        self.core_reserve_refinement = bool(core_reserve_refinement)
        self.core_reserve_reprojection_gate_px = float(
            core_reserve_reprojection_gate_px
        )
        self.core_reserve_minimum_supported_rows = int(
            core_reserve_minimum_supported_rows
        )
        self.final_pose_polish_reprojection_px = float(
            final_pose_polish_reprojection_px
        )
        self.final_pose_polish_minimum_inliers = int(
            final_pose_polish_minimum_inliers
        )
        self.final_pose_polish_mapping_quality_fraction = float(
            final_pose_polish_mapping_quality_fraction
        )
        self.final_pose_polish_maximum_update_translation_cm = float(
            final_pose_polish_maximum_update_translation_cm
        )
        self.final_pose_polish_maximum_update_rotation_deg = float(
            final_pose_polish_maximum_update_rotation_deg
        )
        if not (
            0.25 <= self.match_retention_fraction <= 1.0
            and self.minimum_retained_match_count >= 4
            and self.first_pass_query_cap >= 0
            and 0.0 <= self.refinement_expanded_reserve_maximum_inlier_fraction <= 1.0
        ):
            raise ValueError("first-pass match retention configuration is invalid")
        if self.refinement_expanded_reserve_maximum_inlier_fraction and not (
            self.pose_conditioned_sparse_refinement
            and self.refinement_pose_backend == "robust"
            and self.first_pass_query_cap > 0
            and self.match_retention_fraction < 1.0
        ):
            raise ValueError(
                "adaptive expanded Reserve requires capped robust pose-conditioned refinement"
            )
        if self.match_retention_fraction < 1.0 and any(
            (
                self.assignment_topk,
                self.suppress_duplicate_anchors,
                self.guided_sampling,
                bool(group_aware_pose),
                self.topk_geometric_feedback,
                self.sparse_lgcv_topk_feedback,
            )
        ):
            raise ValueError(
                "first-pass score retention cannot be combined with this matcher mode"
            )
        if self.ransac_hypothesis_core_size and any(
            (
                self.guided_sampling,
                self.confidence_core_progressive_sampling,
                bool(group_aware_pose),
            )
        ):
            raise ValueError(
                "hypothesis-core RANSAC is a separate first-pose solver ablation"
            )
        if self.confidence_core_progressive_sampling and not (
            self.match_retention_fraction < 1.0
            and not self.guided_sampling
            and not bool(group_aware_pose)
        ):
            raise ValueError(
                "confidence-core progressive sampling requires a retained Core"
            )
        if self.core_reserve_refinement and self.pose_conditioned_sparse_refinement:
            raise ValueError(
                "core-reserve and pose-conditioned refinement are separate ablations"
            )
        if self.core_reserve_refinement and not (
            self.match_retention_fraction < 1.0
            and 1.0 <= self.core_reserve_reprojection_gate_px <= 8.0
            and self.core_reserve_minimum_supported_rows >= 4
        ):
            raise ValueError("core-reserve refinement configuration is invalid")
        if not 0.25 <= self.final_pose_polish_mapping_quality_fraction <= 1.0:
            raise ValueError("final pose-polish mapping-quality fraction is invalid")
        if not (
            self.final_pose_polish_maximum_update_translation_cm > 0.0
            and 0.0 < self.final_pose_polish_maximum_update_rotation_deg <= 1.0
        ):
            raise ValueError("final pose-polish update bound is invalid")
        if self.final_pose_polish_reprojection_px and not (
            1.0
            <= self.final_pose_polish_reprojection_px
            < self.reprojection_error_px
            and self.final_pose_polish_minimum_inliers >= 4
        ):
            raise ValueError("final sparse pose-polish configuration is invalid")
        self.reuse_correspondence_buffers = bool(reuse_correspondence_buffers)
        self._camera_cache: dict[
            tuple[float, float, int, int], tuple[np.ndarray, dict]
        ] = {}
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
        if (
            sum(
                (
                    self.topk_geometric_feedback,
                    self.sparse_lgcv_topk_feedback,
                    self.pose_conditioned_sparse_refinement,
                )
            )
            > 1
        ):
            raise ValueError("online sparse-refinement modes are separate ablations")
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
            self.anchor_position_covariance = None
            self.anchor_uncertainty = torch.ones_like(
                self.anchor_matchability, device=self.device
            )
        else:
            base_covariance = torch.as_tensor(covariance, device=self.device).float()
            trace = (
                torch.diagonal(
                    base_covariance,
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
            self.anchor_position_covariance = base_covariance[
                context_indices.to(self.device)
            ]
        if not (
            bool(torch.isfinite(self.anchor_matchability).all())
            and bool(torch.isfinite(self.anchor_uncertainty).all())
            and bool(torch.isfinite(self.anchor_xyz).all())
            and bool((self.anchor_uncertainty >= 0).all())
            and (
                self.anchor_position_covariance is None
                or (
                    self.anchor_position_covariance.shape
                    == (self.anchor_xyz.shape[0], 3, 3)
                    and bool(torch.isfinite(self.anchor_position_covariance).all())
                    and bool(
                        torch.allclose(
                            self.anchor_position_covariance,
                            self.anchor_position_covariance.transpose(1, 2),
                            atol=1e-5,
                            rtol=1e-5,
                        )
                    )
                )
            )
        ):
            raise ValueError("mapping reliability metadata is invalid")
        self.anchor_retrieval_quality = torch.sqrt(
            self.anchor_matchability.clamp(0.0, 1.0)
            * (1.0 + self.anchor_uncertainty).reciprocal()
        )

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
        core_second_best_scores = None
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
            core_second_best_scores = feedback_topk.scores[:, 1]
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
        elif self.minimum_sufficient_confidence_core:
            if self.anchor_extra_prototype_features.numel():
                raise ValueError("confidence Core v2 does not support extra prototypes")
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
            core_second_best_scores = top2.scores[:, 1]
            matches = raw_matches
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
        if (
            not self.assignment_topk
            and not eager_feedback_topk
            and not (self.pose_conditioned_sparse_refinement)
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
        if self.match_retention_fraction < 1.0:
            core_raw_matches = raw_matches
            core_second_scores = core_second_best_scores
            if self.first_pass_query_cap:
                detector_ranks = (
                    torch.arange(sparse.scores.numel(), device=self.device)
                    if sparse.detector_ranks is None
                    else sparse.detector_ranks
                )
                detector_mask = detector_ranks < self.first_pass_query_cap
                capped_count = int(detector_mask.sum().item())
                if capped_count < self.minimum_retained_match_count:
                    raise ValueError("first-pass query cap is below the minimum core")
                core_keep = detector_mask[raw_matches.keypoint_indices]
                core_raw_matches = Top1Matches(
                    keypoint_indices=raw_matches.keypoint_indices[core_keep],
                    anchor_indices=raw_matches.anchor_indices[core_keep],
                    scores=raw_matches.scores[core_keep],
                )
                if core_second_scores is not None:
                    core_second_scores = core_second_scores[core_keep]
            if self.minimum_sufficient_confidence_core:
                matches = retain_diverse_confidence_matches(
                    core_raw_matches,
                    keypoints=sparse.keypoints,
                    second_best_scores=core_second_scores,
                    anchor_matchability=self.anchor_matchability,
                    anchor_uncertainty=self.anchor_uncertainty,
                    anchor_xyz=self.anchor_xyz,
                    image_hw=sparse.image_hw,
                    retention_fraction=self.match_retention_fraction,
                    minimum_count=self.minimum_retained_match_count,
                )
            else:
                matches = retain_high_score_matches(
                    core_raw_matches,
                    retention_fraction=self.match_retention_fraction,
                    minimum_count=self.minimum_retained_match_count,
                )
        if self.confidence_core_progressive_sampling:
            order = torch.argsort(matches.scores, descending=True, stable=True)
            matches = Top1Matches(
                keypoint_indices=matches.keypoint_indices[order],
                anchor_indices=matches.anchor_indices[order],
                scores=matches.scores[order],
            )
        first_pass_match_count = int(matches.scores.numel())
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
        elif self.ransac_hypothesis_core_size:
            pose = solve_absolute_pose_from_hypothesis_core(
                points_2d + 0.5,
                points_3d,
                intrinsic,
                matches.scores.detach().float().cpu().numpy(),
                hypothesis_core_size=self.ransac_hypothesis_core_size,
                camera=pose_camera,
                **solve_kwargs,
            )
        else:
            pose = solve_absolute_pose(
                points_2d + 0.5,
                points_3d,
                intrinsic,
                progressive_sampling=(
                    self.guided_sampling
                    or self.confidence_core_progressive_sampling
                ),
                camera=pose_camera,
                **solve_kwargs,
            )
        ransac_ms = (time.perf_counter() - ransac_started) * 1000.0
        first_pass_ransac_iterations = int(pose.diagnostics.get("iterations", 0))
        first_pass_hypothesis_core_used = bool(
            pose.diagnostics.get("hypothesis_core_used", False)
        )
        first_pass_hypothesis_core_fallback = bool(
            pose.diagnostics.get("hypothesis_core_fallback", False)
        )
        baseline_inlier_query_rows = _local_inliers_to_query_rows(matches, pose.inliers)
        core_reserve_ms = 0.0
        core_reserve_supported_rows = 0
        core_reserve_optimization_rows = 0
        core_reserve_candidate_inliers = 0
        core_reserve_inlier_retention = 0.0
        core_reserve_median_residual_increase_px = 0.0
        core_reserve_pose_update_translation_cm = 0.0
        core_reserve_pose_update_rotation_deg = 0.0
        core_reserve_accepted = False
        if self.core_reserve_refinement:
            reserve_started = time.perf_counter()
            raw_query_rows = raw_matches.keypoint_indices.detach().long().cpu().numpy()
            full_points_2d = (
                sparse.keypoints[raw_matches.keypoint_indices]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            full_points_3d = (
                self.anchor_xyz[raw_matches.anchor_indices]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            core_query_rows = matches.keypoint_indices.detach().cpu().numpy()
            query_to_raw = np.full(int(sparse.keypoints.shape[0]), -1, dtype=np.int64)
            query_to_raw[raw_query_rows] = np.arange(
                raw_query_rows.size, dtype=np.int64
            )
            core_raw_rows = query_to_raw[core_query_rows]
            if bool((core_raw_rows < 0).any()):
                raise RuntimeError("retained match is absent from the raw registry")
            core_inlier_rows = core_raw_rows[np.asarray(pose.inliers, dtype=np.int64)]
            baseline_full_residual = _reprojection_residuals(
                full_points_2d + 0.5,
                full_points_3d,
                pose.pose_w2c,
                intrinsic,
            )
            reserve = np.ones(full_points_2d.shape[0], dtype=bool)
            reserve[core_raw_rows] = False
            supported_reserve_rows = np.flatnonzero(
                reserve
                & (baseline_full_residual <= self.core_reserve_reprojection_gate_px)
            )
            core_reserve_supported_rows = int(supported_reserve_rows.size)
            if supported_reserve_rows.size >= self.core_reserve_minimum_supported_rows:
                optimization_rows = np.union1d(core_inlier_rows, supported_reserve_rows)
                core_reserve_optimization_rows = int(optimization_rows.size)
                candidate_pose = refine_absolute_pose_from_initial(
                    full_points_2d + 0.5,
                    full_points_3d,
                    intrinsic,
                    pose.pose_w2c,
                    optimization_rows,
                    reprojection_error_px=self.reprojection_error_px,
                    camera=pose_camera,
                )
                core_reserve_candidate_inliers = int(candidate_pose.inliers.size)
                retained_core = np.intersect1d(
                    core_inlier_rows,
                    candidate_pose.inliers,
                    assume_unique=True,
                )
                core_reserve_inlier_retention = float(
                    retained_core.size / max(core_inlier_rows.size, 1)
                )
                candidate_full_residual = _reprojection_residuals(
                    full_points_2d + 0.5,
                    full_points_3d,
                    candidate_pose.pose_w2c,
                    intrinsic,
                )
                core_reserve_median_residual_increase_px = float(
                    np.median(candidate_full_residual[core_inlier_rows])
                    - np.median(baseline_full_residual[core_inlier_rows])
                )
                (
                    core_reserve_pose_update_translation_cm,
                    core_reserve_pose_update_rotation_deg,
                ) = _pose_update_magnitude(pose.pose_w2c, candidate_pose.pose_w2c)
                core_reserve_accepted = bool(
                    core_reserve_candidate_inliers >= int(pose.inliers.size)
                    and core_reserve_inlier_retention >= 0.98
                    and core_reserve_median_residual_increase_px <= 0.10
                    and core_reserve_pose_update_translation_cm <= 10.0
                    and core_reserve_pose_update_rotation_deg <= 0.10
                )
                if core_reserve_accepted:
                    pose = candidate_pose
                    matches = raw_matches
            core_reserve_ms = (time.perf_counter() - reserve_started) * 1000.0
        baseline_pose_inlier_count = int(pose.inliers.size)
        feedback_geometry_ms = 0.0
        feedback_ransac_ms = 0.0
        feedback_model_comparison_ms = 0.0
        feedback_eligible = False
        feedback_gate_passed = False
        feedback_accepted = False
        feedback_proposed_rows = 0
        feedback_supported_rows = 0
        feedback_candidate_inliers = 0
        feedback_baseline_comparison_inliers = int(pose.inliers.size)
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
        feedback_reliability_authorized_edges = 0
        feedback_reliability_expanded_budget_edges = 0
        feedback_reliability_expanded_selected_rows = 0
        feedback_reliability_fallback_query_rows = 0
        feedback_reliability_matchability_threshold = 0.0
        feedback_reliability_uncertainty_threshold = 0.0
        feedback_duplicate_owner_rejections = 0
        feedback_mutual_candidate_rejections = 0
        feedback_heldout_query_rows = 0
        feedback_heldout_candidate_edges = 0
        feedback_heldout_baseline_energy = 0.0
        feedback_heldout_candidate_energy = 0.0
        feedback_heldout_relative_energy_gain = 0.0
        feedback_heldout_baseline_assignments = 0
        feedback_heldout_candidate_assignments = 0
        feedback_heldout_passed = False
        feedback_pose_information_condition = 0.0
        feedback_expanded_projection_edges = 0
        feedback_projection_gate_p50_px = float(
            self.pose_conditioned_selection_config["projection_gate_px"]
        )
        feedback_projection_gate_p90_px = feedback_projection_gate_p50_px
        feedback_view_support_available = False
        feedback_view_support_rejected_edges = 0
        feedback_capacity_rejections = 0
        feedback_visible_anchor_count = 0
        feedback_candidate_pool_anchor_count = 0
        feedback_candidate_pool_global_fallback = False
        feedback_view_support_prefilter_fallback = False
        feedback_retrieval_query_count = 0
        feedback_projection_local_edge_count = 0
        feedback_projection_local_nonempty_query_count = 0
        feedback_view_supported_anchor_count = 0
        feedback_view_conditioned_selected_mode_anchor_count = 0
        feedback_view_conditioned_base_fallback_anchor_count = 0
        feedback_spatial_jackknife_available = False
        feedback_spatial_jackknife_baseline_task_p90 = 0.0
        feedback_spatial_jackknife_candidate_task_p90 = 0.0
        feedback_spatial_jackknife_relative_gain = 0.0
        feedback_spatial_jackknife_baseline_groups = 0
        feedback_spatial_jackknife_candidate_groups = 0
        feedback_soft_inlier_candidate_rows = 0
        feedback_soft_inlier_changed_rows = 0
        feedback_soft_inlier_capacity_rejections = 0
        feedback_hard_core_inlier_rows = 0
        feedback_selected_rank_median = 0.0
        feedback_selected_rank_p90 = 0.0
        feedback_selected_score_drop_median = 0.0
        feedback_selected_score_drop_p90 = 0.0
        feedback_selected_joint_cost_median = 0.0
        feedback_selected_reprojection_median_px = 0.0
        feedback_rejected_by_protection = False
        feedback_common_grid_baseline_energy = 0.0
        feedback_common_grid_candidate_energy = 0.0
        feedback_common_grid_relative_energy_gain = 0.0
        feedback_common_grid_baseline_duplicate_owners = 0
        feedback_common_grid_candidate_duplicate_owners = 0
        feedback_common_grid_passed = False
        feedback_progressive_sampling = False
        feedback_projection_gate_px = float(
            self.pose_conditioned_selection_config["projection_gate_px"]
        )
        feedback_query_count, feedback_reserve_expanded = _query_level_feedback_count(
            sparse.detector_ranks,
            extracted_count=int(sparse.descriptors.shape[0]),
            first_pass_query_cap=self.first_pass_query_cap,
            baseline_inlier_count=baseline_pose_inlier_count,
            first_pass_match_count=first_pass_match_count,
            expanded_reserve_maximum_inlier_fraction=(
                self.refinement_expanded_reserve_maximum_inlier_fraction
            ),
        )
        feedback_keypoints = sparse.keypoints[:feedback_query_count]
        feedback_descriptors = sparse.descriptors[:feedback_query_count]
        feedback_raw_anchor_rows = raw_matches.anchor_indices[:feedback_query_count]
        feedback_raw_scores = raw_matches.scores[:feedback_query_count]
        if baseline_inlier_query_rows.size and (
            int(baseline_inlier_query_rows.max()) >= feedback_query_count
        ):
            raise RuntimeError("T0 inliers escape the selected Reserve registry")
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
                    or int(pose.inliers.size) < self.feedback_maximum_baseline_inliers
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

                    retrieval_query_rows = None
                    if self.refinement_active_row_retrieval:
                        retrieval_mask = torch.ones(
                            feedback_descriptors.shape[0],
                            dtype=torch.bool,
                            device=self.device,
                        )
                        retrieval_mask[
                            torch.as_tensor(
                                baseline_inlier_query_rows.copy(), device=self.device
                            ).long()
                        ] = False
                        if bool(
                            self.pose_conditioned_selection_config["allow_soft_inliers"]
                        ):
                            baseline_residual_all = _reprojection_residuals(
                                points_2d + 0.5,
                                points_3d,
                                pose.pose_w2c,
                                intrinsic,
                            )
                            soft_local_rows = np.asarray(pose.inliers, dtype=np.int64)[
                                baseline_residual_all[
                                    np.asarray(pose.inliers, dtype=np.int64)
                                ]
                                >= float(
                                    self.pose_conditioned_selection_config[
                                        "soft_inlier_minimum_baseline_residual_px"
                                    ]
                                )
                            ]
                            soft_rows = (
                                matches.keypoint_indices[
                                    torch.as_tensor(
                                        soft_local_rows,
                                        device=matches.keypoint_indices.device,
                                    ).long()
                                ]
                                .detach()
                                .long()
                                .cpu()
                                .numpy()
                            )
                            if soft_rows.size:
                                retrieval_mask[
                                    torch.as_tensor(
                                        soft_rows, device=self.device
                                    ).long()
                                ] = True
                        retrieval_query_rows = torch.nonzero(
                            retrieval_mask, as_tuple=False
                        ).reshape(-1)

                    visible_topk = build_pose_visible_topk(
                        query_descriptors=feedback_descriptors,
                        query_keypoints=feedback_keypoints.float() + 0.5,
                        normalized_anchor_features=self.anchor_features,
                        baseline_anchor_rows=feedback_raw_anchor_rows,
                        baseline_scores=feedback_raw_scores,
                        anchor_xyz=self.anchor_xyz,
                        intrinsic=torch.as_tensor(intrinsic, device=self.device),
                        baseline_pose_w2c=torch.as_tensor(
                            pose.pose_w2c, device=self.device
                        ),
                        image_hw=sparse.image_hw,
                        retrieval_query_rows=retrieval_query_rows,
                        anchor_view_support=(
                            self.anchor_view_support
                            if self.refinement_pre_topk_view_filter
                            else None
                        ),
                        prefilter_mapping_view_support=(
                            self.refinement_pre_topk_view_filter
                        ),
                        view_direction_slack_deg=float(
                            self.pose_conditioned_selection_config[
                                "view_direction_slack_deg"
                            ]
                        ),
                        minimum_mapping_distance_ratio=float(
                            self.pose_conditioned_selection_config[
                                "minimum_mapping_distance_ratio"
                            ]
                        ),
                        maximum_mapping_distance_ratio=float(
                            self.pose_conditioned_selection_config[
                                "maximum_mapping_distance_ratio"
                            ]
                        ),
                        view_conditioned_descriptor_state=(
                            self.view_conditioned_anchor_state
                        ),
                        view_conditioned_minimum_concentration=(
                            self.view_conditioned_minimum_concentration
                        ),
                        view_conditioned_residual_scale=(
                            self.view_conditioned_residual_scale
                        ),
                        view_conditioned_require_two_valid_modes=(
                            self.view_conditioned_require_two_valid_modes
                        ),
                        view_conditioned_score_fusion=(
                            self.view_conditioned_score_fusion
                        ),
                        projection_first_local_candidates=(
                            self.refinement_projection_first_local_candidates
                        ),
                        projection_first_radius_px=(
                            self.refinement_projection_first_radius_px
                        ),
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
                    feedback_view_support_prefilter_fallback = bool(
                        visible_topk["view_support_prefilter_fallback"]
                    )
                    feedback_retrieval_query_count = int(
                        visible_topk["retrieval_query_count"]
                    )
                    feedback_projection_local_edge_count = int(
                        visible_topk["projection_local_edge_count"]
                    )
                    feedback_projection_local_nonempty_query_count = int(
                        visible_topk["projection_local_nonempty_query_count"]
                    )
                    feedback_view_supported_anchor_count = int(
                        visible_topk["view_supported_anchor_count"]
                    )
                    feedback_view_conditioned_selected_mode_anchor_count = int(
                        visible_topk["view_conditioned_selected_mode_anchor_count"]
                    )
                    feedback_view_conditioned_base_fallback_anchor_count = int(
                        visible_topk["view_conditioned_base_fallback_anchor_count"]
                    )
                baseline_rows_cpu = feedback_topk.anchor_indices[:, 0].cpu()
                if self.pose_conditioned_sparse_refinement:
                    from map_learning.v24_pose_conditioned_sparse_refinement import (
                        select_pose_conditioned_rows,
                    )

                    query_selection_config = dict(
                        self.pose_conditioned_selection_config
                    )
                    if (
                        self.refinement_uncertainty_projection_gate_px > 0.0
                        and int(pose.inliers.size)
                        <= self.refinement_uncertainty_maximum_baseline_inliers
                    ):
                        query_selection_config["projection_gate_px"] = float(
                            self.refinement_uncertainty_projection_gate_px
                        )
                    feedback_projection_gate_px = float(
                        query_selection_config["projection_gate_px"]
                    )
                    provisional_device = select_pose_conditioned_rows(
                        keypoints=feedback_keypoints.float() + 0.5,
                        topk_anchor_rows=feedback_topk.anchor_indices,
                        topk_scores=feedback_topk.scores,
                        baseline_inlier_rows=torch.as_tensor(
                            baseline_inlier_query_rows.copy(), device=self.device
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
                        anchor_position_covariance=(self.anchor_position_covariance),
                        mapping_reliability_validated=True,
                        map_geometry_validated=True,
                        image_hw=sparse.image_hw,
                        config=query_selection_config,
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
                    feedback_eligible_edges = int(provisional["eligible_edge_count"])
                    feedback_reliability_authorized_edges = int(
                        provisional["reliability_authorized_edge_count"]
                    )
                    feedback_reliability_expanded_budget_edges = int(
                        provisional["reliability_expanded_budget_edge_count"]
                    )
                    feedback_reliability_expanded_selected_rows = int(
                        provisional["reliability_expanded_selected_row_count"]
                    )
                    feedback_reliability_fallback_query_rows = int(
                        provisional["reliability_fallback_query_row_count"]
                    )
                    feedback_reliability_matchability_threshold = float(
                        provisional["reliability_matchability_threshold"]
                    )
                    feedback_reliability_uncertainty_threshold = float(
                        provisional["reliability_uncertainty_threshold"]
                    )
                    feedback_duplicate_owner_rejections = int(
                        provisional["duplicate_candidate_owner_rejection_count"]
                    )
                    feedback_mutual_candidate_rejections = int(
                        provisional["mutual_candidate_rejected_edge_count"]
                    )
                    feedback_heldout_query_rows = int(
                        provisional["heldout_validation_query_rows"].numel()
                    )
                    feedback_heldout_candidate_edges = int(
                        provisional["heldout_validation_edge_count"]
                    )
                    feedback_pose_information_condition = float(
                        provisional["pose_information_condition"]
                    )
                    feedback_expanded_projection_edges = int(
                        provisional["expanded_projection_edge_count"]
                    )
                    feedback_projection_gate_p50_px = float(
                        provisional["projection_gate_p50_px"]
                    )
                    feedback_projection_gate_p90_px = float(
                        provisional["projection_gate_p90_px"]
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
                    feedback_soft_inlier_candidate_rows = int(
                        provisional["soft_inlier_candidate_row_count"]
                    )
                    feedback_soft_inlier_changed_rows = int(
                        provisional["soft_inlier_changed_row_count"]
                    )
                    feedback_soft_inlier_capacity_rejections = int(
                        provisional["soft_inlier_capacity_rejection_count"]
                    )
                    feedback_hard_core_inlier_rows = int(
                        provisional["hard_core_inlier_row_count"]
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
                        keypoints=feedback_keypoints.float().cpu() + 0.5,
                        topk_anchor_rows=feedback_topk.anchor_indices.cpu(),
                        topk_scores=feedback_topk.scores.cpu(),
                        baseline_anchor_rows=baseline_rows_cpu,
                        baseline_scores=feedback_topk.scores[:, 0].cpu(),
                        baseline_inlier_rows=torch.from_numpy(
                            baseline_inlier_query_rows.copy()
                        ).long(),
                        anchor_xyz=self._sparse_feedback_anchor_xyz_cpu,
                        intrinsic=torch.from_numpy(intrinsic),
                        baseline_pose_w2c=torch.from_numpy(pose.pose_w2c),
                        config=topk_cfg,
                        allow_runtime_inlier_band=True,
                    )
                feedback_proposed_rows = int(provisional["changed_query_rows"].numel())
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
                        keypoints=feedback_keypoints.float().cpu() + 0.5,
                        baseline_anchor_rows=baseline_rows_cpu,
                        provisional_anchor_rows=provisional["anchor_rows"],
                        provisional_changed_query_rows=provisional[
                            "changed_query_rows"
                        ],
                        baseline_inlier_rows=torch.from_numpy(
                            baseline_inlier_query_rows.copy()
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
                        >= int(feedback_cfg["minimum_query_supported_proposal_count"])
                        and supported_fraction
                        >= float(
                            feedback_cfg["minimum_query_supported_proposal_fraction"]
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
                            or feedback_proposed_rows / max(int(pose.inliers.size), 1)
                            >= self.refinement_minimum_proposal_relative_gain
                        )
                        and (
                            not bool(
                                self.pose_conditioned_selection_config[
                                    "heldout_candidate_validation"
                                ]
                            )
                            or feedback_heldout_query_rows
                            >= int(
                                self.pose_conditioned_selection_config[
                                    "heldout_validation_minimum_rows"
                                ]
                            )
                        )
                    )
                feedback_geometry_ms = (time.perf_counter() - feedback_started) * 1000.0
                if feedback_gate_passed:
                    candidate_rows_cpu = provisional["anchor_rows"]
                    candidate_points_3d = self._sparse_feedback_anchor_xyz_cpu[
                        candidate_rows_cpu
                    ].numpy()
                    feedback_points_2d = (
                        feedback_keypoints[feedback_topk.keypoint_indices]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )
                    baseline_points_3d = self._sparse_feedback_anchor_xyz_cpu[
                        baseline_rows_cpu
                    ].numpy()
                    baseline_inliers = baseline_inlier_query_rows.copy()
                    feedback_baseline_comparison_inliers = int(
                        (
                            _reprojection_residuals(
                                feedback_points_2d + 0.5,
                                baseline_points_3d,
                                pose.pose_w2c,
                                intrinsic,
                            )
                            <= float(self.reprojection_error_px)
                        ).sum()
                    )
                    heldout_rows = (
                        provisional["heldout_validation_query_rows"].numpy()
                        if self.pose_conditioned_sparse_refinement
                        else np.empty(0, dtype=np.int64)
                    )
                    solver_rows = np.setdiff1d(
                        np.arange(feedback_points_2d.shape[0], dtype=np.int64),
                        heldout_rows,
                        assume_unique=True,
                    )
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
                            feedback_points_2d + 0.5,
                            candidate_points_3d,
                            intrinsic,
                            pose.pose_w2c,
                            optimization_rows,
                            reprojection_error_px=self.reprojection_error_px,
                            camera=pose_camera,
                        )
                    else:
                        candidate_order = None
                        if self.refinement_progressive_sampling:
                            baseline_order_residual = _reprojection_residuals(
                                feedback_points_2d + 0.5,
                                baseline_points_3d,
                                pose.pose_w2c,
                                intrinsic,
                            )
                            quality = np.full(
                                feedback_points_2d.shape[0],
                                -1.0,
                                dtype=np.float64,
                            )
                            quality[baseline_inliers] = 1.0 - baseline_order_residual[
                                baseline_inliers
                            ] / max(float(self.reprojection_error_px), 1e-12)
                            changed_order_rows = provisional[
                                "changed_query_rows"
                            ].numpy()
                            if changed_order_rows.size:
                                quality[changed_order_rows] = np.maximum(
                                    quality[changed_order_rows],
                                    1.0 - provisional["selected_joint_cost"].numpy(),
                                )
                            candidate_order = np.argsort(-quality, kind="stable")
                            if heldout_rows.size:
                                candidate_order = candidate_order[
                                    ~np.isin(
                                        candidate_order,
                                        heldout_rows,
                                        assume_unique=True,
                                    )
                                ]
                            ordered_pose = solve_absolute_pose(
                                feedback_points_2d[candidate_order] + 0.5,
                                candidate_points_3d[candidate_order],
                                intrinsic,
                                camera=pose_camera,
                                progressive_sampling=True,
                                **solve_kwargs,
                            )
                            candidate_pose = PoseEstimate(
                                ordered_pose.pose_w2c,
                                candidate_order[ordered_pose.inliers],
                                ordered_pose.diagnostics,
                            )
                            feedback_progressive_sampling = True
                        else:
                            solved_pose = solve_absolute_pose(
                                feedback_points_2d[solver_rows] + 0.5,
                                candidate_points_3d[solver_rows],
                                intrinsic,
                                camera=pose_camera,
                                **solve_kwargs,
                            )
                            candidate_pose = PoseEstimate(
                                solved_pose.pose_w2c,
                                solver_rows[solved_pose.inliers],
                                solved_pose.diagnostics,
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
                        feedback_candidate_inliers
                        - feedback_baseline_comparison_inliers
                    )
                    feedback_candidate_relative_inlier_gain = float(
                        feedback_candidate_inlier_gain
                        / max(feedback_baseline_comparison_inliers, 1)
                    )
                    if self.refinement_common_candidate_grid_gate:
                        from map_learning.v24_pose_conditioned_sparse_refinement import (
                            compare_poses_on_common_candidate_grid,
                        )

                        comparison_started = time.perf_counter()
                        comparison = compare_poses_on_common_candidate_grid(
                            keypoints=feedback_keypoints.float() + 0.5,
                            topk_anchor_rows=feedback_topk.anchor_indices,
                            topk_scores=feedback_topk.scores,
                            baseline_inlier_rows=torch.as_tensor(
                                baseline_inliers.copy(), device=self.device
                            ).long(),
                            anchor_xyz=self.anchor_xyz,
                            intrinsic=torch.as_tensor(
                                intrinsic, device=self.device
                            ).float(),
                            baseline_pose_w2c=torch.as_tensor(
                                pose.pose_w2c, device=self.device
                            ).float(),
                            candidate_pose_w2c=torch.as_tensor(
                                candidate_pose.pose_w2c, device=self.device
                            ).float(),
                            maximum_score_drop_from_top1=float(
                                self.pose_conditioned_selection_config[
                                    "maximum_score_drop_from_top1"
                                ]
                            ),
                            robust_scale_px=float(self.reprojection_error_px),
                        )
                        synchronize()
                        feedback_model_comparison_ms = (
                            time.perf_counter() - comparison_started
                        ) * 1000.0
                        feedback_common_grid_baseline_energy = float(
                            comparison["baseline_energy"].item()
                        )
                        feedback_common_grid_candidate_energy = float(
                            comparison["candidate_energy"].item()
                        )
                        feedback_common_grid_relative_energy_gain = float(
                            comparison["relative_energy_gain"].item()
                        )
                        feedback_common_grid_baseline_duplicate_owners = int(
                            comparison["baseline_duplicate_owner_count"]
                        )
                        feedback_common_grid_candidate_duplicate_owners = int(
                            comparison["candidate_duplicate_owner_count"]
                        )
                        feedback_common_grid_passed = bool(
                            feedback_common_grid_relative_energy_gain
                            >= self.refinement_minimum_common_grid_relative_energy_gain
                        )
                    if bool(
                        self.pose_conditioned_selection_config[
                            "heldout_candidate_validation"
                        ]
                    ):
                        from map_learning.v24_pose_conditioned_sparse_refinement import (
                            compare_poses_on_heldout_candidate_graph,
                        )

                        comparison_started = time.perf_counter()
                        heldout = compare_poses_on_heldout_candidate_graph(
                            keypoints=torch.from_numpy(
                                feedback_points_2d[heldout_rows] + 0.5
                            ),
                            candidate_anchor_rows=provisional[
                                "heldout_validation_anchor_rows"
                            ],
                            candidate_scores=provisional["heldout_validation_scores"],
                            candidate_edge_mask=provisional[
                                "heldout_validation_edge_mask"
                            ],
                            anchor_xyz=self._sparse_feedback_anchor_xyz_cpu,
                            intrinsic=torch.from_numpy(intrinsic),
                            baseline_pose_w2c=torch.from_numpy(pose.pose_w2c),
                            candidate_pose_w2c=torch.from_numpy(
                                candidate_pose.pose_w2c
                            ),
                            maximum_score_drop_from_top1=float(
                                self.pose_conditioned_selection_config[
                                    "maximum_score_drop_from_top1"
                                ]
                            ),
                            robust_scale_px=float(self.reprojection_error_px),
                        )
                        feedback_model_comparison_ms += (
                            time.perf_counter() - comparison_started
                        ) * 1000.0
                        feedback_heldout_baseline_energy = float(
                            heldout["baseline_energy"].item()
                        )
                        feedback_heldout_candidate_energy = float(
                            heldout["candidate_energy"].item()
                        )
                        feedback_heldout_relative_energy_gain = float(
                            heldout["relative_energy_gain"].item()
                        )
                        feedback_heldout_baseline_assignments = int(
                            heldout["baseline_assignment_count"]
                        )
                        feedback_heldout_candidate_assignments = int(
                            heldout["candidate_assignment_count"]
                        )
                        feedback_heldout_passed = bool(
                            feedback_heldout_relative_energy_gain
                            >= self.refinement_minimum_heldout_relative_energy_gain
                        )
                    candidate_inliers = np.asarray(
                        candidate_pose.inliers, dtype=np.int64
                    )
                    if self.refinement_spatial_jackknife_diagnostic:
                        from map_learning.v24_pose_conditioned_sparse_refinement import (
                            spatial_jackknife_pose_stability,
                        )

                        comparison_started = time.perf_counter()
                        try:
                            baseline_jackknife = spatial_jackknife_pose_stability(
                                keypoints=feedback_points_2d + 0.5,
                                points_3d=baseline_points_3d,
                                pose_w2c=pose.pose_w2c,
                                inlier_rows=baseline_inliers,
                                intrinsic=intrinsic,
                                image_hw=sparse.image_hw,
                                reprojection_error_px=self.reprojection_error_px,
                                camera=pose_camera,
                            )
                            candidate_jackknife = spatial_jackknife_pose_stability(
                                keypoints=feedback_points_2d + 0.5,
                                points_3d=candidate_points_3d,
                                pose_w2c=candidate_pose.pose_w2c,
                                inlier_rows=candidate_inliers,
                                intrinsic=intrinsic,
                                image_hw=sparse.image_hw,
                                reprojection_error_px=self.reprojection_error_px,
                                camera=pose_camera,
                            )
                            feedback_spatial_jackknife_available = True
                            feedback_spatial_jackknife_baseline_task_p90 = float(
                                baseline_jackknife["normalized_task_update_p90"]
                            )
                            feedback_spatial_jackknife_candidate_task_p90 = float(
                                candidate_jackknife["normalized_task_update_p90"]
                            )
                            feedback_spatial_jackknife_relative_gain = float(
                                (
                                    feedback_spatial_jackknife_baseline_task_p90
                                    - feedback_spatial_jackknife_candidate_task_p90
                                )
                                / max(
                                    feedback_spatial_jackknife_baseline_task_p90,
                                    1e-12,
                                )
                            )
                            feedback_spatial_jackknife_baseline_groups = int(
                                baseline_jackknife["valid_group_count"]
                            )
                            feedback_spatial_jackknife_candidate_groups = int(
                                candidate_jackknife["valid_group_count"]
                            )
                        except ValueError:
                            feedback_spatial_jackknife_available = False
                        feedback_model_comparison_ms += (
                            time.perf_counter() - comparison_started
                        ) * 1000.0
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
                        feedback_points_2d + 0.5,
                        baseline_points_3d,
                        pose.pose_w2c,
                        intrinsic,
                    )[baseline_inliers]
                    protected_candidate_residual = _reprojection_residuals(
                        feedback_points_2d + 0.5,
                        baseline_points_3d,
                        candidate_pose.pose_w2c,
                        intrinsic,
                    )[baseline_inliers]
                    candidate_assignment_residual = _reprojection_residuals(
                        feedback_points_2d + 0.5,
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
                                candidate_assignment_residual[changed_candidate_inliers]
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
                    ) = _pose_update_magnitude(pose.pose_w2c, candidate_pose.pose_w2c)
                    if self.pose_conditioned_sparse_refinement:
                        from map_learning.v24_pose_conditioned_sparse_refinement import (
                            changed_inlier_spatial_cell_count,
                        )

                        feedback_changed_inlier_spatial_cells = (
                            changed_inlier_spatial_cell_count(
                                keypoints=feedback_keypoints.float().cpu() + 0.5,
                                changed_query_rows=provisional["changed_query_rows"],
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
                            self.feedback_maximum_protected_p90_residual_increase_px < 0
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
                        and (
                            not self.refinement_common_candidate_grid_gate
                            or feedback_common_grid_passed
                        )
                        and (
                            not bool(
                                self.pose_conditioned_selection_config[
                                    "heldout_candidate_validation"
                                ]
                            )
                            or feedback_heldout_passed
                        )
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
        final_pose_polish_ms = 0.0
        final_pose_polish_rows = 0
        final_pose_polish_inlier_retention = 0.0
        final_pose_polish_update_translation_cm = 0.0
        final_pose_polish_update_rotation_deg = 0.0
        final_pose_polish_accepted = False
        if self.final_pose_polish_reprojection_px:
            polish_started = time.perf_counter()
            final_points_2d = (
                sparse.keypoints[matches.keypoint_indices]
                .detach()
                .float()
                .cpu()
                .numpy()
                + 0.5
            )
            final_points_3d = (
                self.anchor_xyz[matches.anchor_indices]
                .detach()
                .float()
                .cpu()
                .numpy()
            )
            original_inliers = np.asarray(pose.inliers, dtype=np.int64)
            original_residual = _reprojection_residuals(
                final_points_2d,
                final_points_3d,
                pose.pose_w2c,
                intrinsic,
            )
            strict_rows = original_inliers[
                original_residual[original_inliers]
                <= self.final_pose_polish_reprojection_px
            ]
            if strict_rows.size >= self.final_pose_polish_minimum_inliers:
                strict_rows = _mapping_quality_polish_rows(
                    strict_rows,
                    match_anchor_rows=matches.anchor_indices,
                    anchor_quality=self.anchor_retrieval_quality,
                    retention_fraction=(
                        self.final_pose_polish_mapping_quality_fraction
                    ),
                    minimum_count=self.final_pose_polish_minimum_inliers,
                )
            final_pose_polish_rows = int(strict_rows.size)
            if strict_rows.size >= self.final_pose_polish_minimum_inliers:
                polished = refine_absolute_pose_from_initial(
                    final_points_2d,
                    final_points_3d,
                    intrinsic,
                    pose.pose_w2c,
                    strict_rows,
                    reprojection_error_px=self.reprojection_error_px,
                    camera=pose_camera,
                )
                retained = np.intersect1d(
                    original_inliers, polished.inliers, assume_unique=True
                )
                final_pose_polish_inlier_retention = float(
                    retained.size / max(original_inliers.size, 1)
                )
                (
                    final_pose_polish_update_translation_cm,
                    final_pose_polish_update_rotation_deg,
                ) = _pose_update_magnitude(pose.pose_w2c, polished.pose_w2c)
                polished_residual = _reprojection_residuals(
                    final_points_2d,
                    final_points_3d,
                    polished.pose_w2c,
                    intrinsic,
                )
                final_pose_polish_accepted = bool(
                    final_pose_polish_inlier_retention >= 0.98
                    and np.median(polished_residual[strict_rows])
                    <= np.median(original_residual[strict_rows]) + 1e-6
                    and final_pose_polish_update_translation_cm
                    <= self.final_pose_polish_maximum_update_translation_cm
                    and final_pose_polish_update_rotation_deg
                    <= self.final_pose_polish_maximum_update_rotation_deg
                )
                if final_pose_polish_accepted:
                    diagnostics = dict(polished.diagnostics)
                    diagnostics["iterations"] = int(
                        pose.diagnostics.get("iterations", 0)
                    )
                    diagnostics["final_pose_polish"] = True
                    pose = PoseEstimate(
                        polished.pose_w2c, polished.inliers, diagnostics
                    )
            final_pose_polish_ms = (
                time.perf_counter() - polish_started
            ) * 1000.0
        return LocalizationResult(
            sparse,
            matches,
            pose,
            intrinsic,
            {
                "frontend_ms": frontend_ms,
                "matching_ms": matching_ms,
                "ransac_ms": ransac_ms,
                "core_reserve_ms": core_reserve_ms,
                "feedback_geometry_ms": feedback_geometry_ms,
                "feedback_ransac_ms": feedback_ransac_ms,
                "feedback_model_comparison_ms": feedback_model_comparison_ms,
                "final_pose_polish_ms": final_pose_polish_ms,
                "total_ms": (time.perf_counter() - total_started) * 1000.0,
            },
            {
                "top1_match_count": int(raw_matches.scores.numel()),
                "final_pose_polish_reprojection_px": float(
                    self.final_pose_polish_reprojection_px
                ),
                "final_pose_polish_mapping_quality_fraction": float(
                    self.final_pose_polish_mapping_quality_fraction
                ),
                "final_pose_polish_maximum_update_translation_cm": float(
                    self.final_pose_polish_maximum_update_translation_cm
                ),
                "final_pose_polish_maximum_update_rotation_deg": float(
                    self.final_pose_polish_maximum_update_rotation_deg
                ),
                "final_pose_polish_rows": int(final_pose_polish_rows),
                "final_pose_polish_inlier_retention_fraction": float(
                    final_pose_polish_inlier_retention
                ),
                "final_pose_polish_update_translation_cm": float(
                    final_pose_polish_update_translation_cm
                ),
                "final_pose_polish_update_rotation_deg": float(
                    final_pose_polish_update_rotation_deg
                ),
                "final_pose_polish_accepted": int(final_pose_polish_accepted),
                "superpoint_subpixel_keypoints": int(self.subpixel_keypoints),
                "superpoint_subpixel_geometry_only": int(
                    self.subpixel_geometry_only
                ),
                "superpoint_subpixel_maximum_offset": float(
                    self.subpixel_maximum_offset
                ),
                "retained_match_count": first_pass_match_count,
                "match_retention_fraction": float(self.match_retention_fraction),
                "minimum_sufficient_confidence_core": int(
                    self.minimum_sufficient_confidence_core
                ),
                "first_pass_query_cap": int(self.first_pass_query_cap),
                "ransac_hypothesis_core_size": int(
                    self.ransac_hypothesis_core_size
                ),
                "ransac_hypothesis_core_used": int(
                    first_pass_hypothesis_core_used
                ),
                "ransac_hypothesis_core_fallback": int(
                    first_pass_hypothesis_core_fallback
                ),
                "first_pass_ransac_iterations": first_pass_ransac_iterations,
                "refinement_expanded_reserve_maximum_inlier_fraction": float(
                    self.refinement_expanded_reserve_maximum_inlier_fraction
                ),
                "sparse_feedback_query_count": int(feedback_query_count),
                "sparse_feedback_reserve_expanded": int(feedback_reserve_expanded),
                "sparse_feedback_t0_inlier_fraction": float(
                    baseline_pose_inlier_count / max(first_pass_match_count, 1)
                ),
                "minimum_retained_match_count": int(self.minimum_retained_match_count),
                "core_reserve_refinement": int(self.core_reserve_refinement),
                "core_reserve_reprojection_gate_px": float(
                    self.core_reserve_reprojection_gate_px
                ),
                "core_reserve_supported_rows": int(core_reserve_supported_rows),
                "core_reserve_optimization_rows": int(core_reserve_optimization_rows),
                "core_reserve_candidate_inliers": int(core_reserve_candidate_inliers),
                "core_reserve_inlier_retention_fraction": float(
                    core_reserve_inlier_retention
                ),
                "core_reserve_median_residual_increase_px": float(
                    core_reserve_median_residual_increase_px
                ),
                "core_reserve_pose_update_translation_cm": float(
                    core_reserve_pose_update_translation_cm
                ),
                "core_reserve_pose_update_rotation_deg": float(
                    core_reserve_pose_update_rotation_deg
                ),
                "core_reserve_accepted": int(core_reserve_accepted),
                "score_filtered_match_count": int(
                    raw_matches.scores.numel() - first_pass_match_count
                    if self.match_retention_fraction < 1.0
                    else 0
                ),
                "duplicate_anchor_count": int(
                    0
                    if self.match_retention_fraction < 1.0
                    else raw_matches.scores.numel() - matches.scores.numel()
                ),
                "duplicate_anchor_fraction": float(
                    0.0
                    if self.match_retention_fraction < 1.0
                    else 1.0
                    - matches.scores.numel() / max(int(raw_matches.scores.numel()), 1)
                ),
                "guided_sampling": int(self.guided_sampling),
                "confidence_core_progressive_sampling": int(
                    self.confidence_core_progressive_sampling
                ),
                "group_aware_pose": int(self.group_aware_pose),
                "capacity_assignment": int(self.assignment_topk > 0),
                "sparse_lgcv_topk_feedback": int(self.sparse_lgcv_topk_feedback),
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
                    if self.topk_geometric_feedback or self.sparse_lgcv_topk_feedback
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
                "sparse_feedback_baseline_comparison_inliers": int(
                    feedback_baseline_comparison_inliers
                ),
                "sparse_feedback_candidate_inlier_gain": (
                    feedback_candidate_inlier_gain
                ),
                "sparse_feedback_candidate_pose_w2c": (feedback_candidate_pose_w2c),
                "sparse_feedback_candidate_ransac_iterations": (
                    feedback_candidate_ransac_iterations
                ),
                "sparse_feedback_candidate_relative_inlier_gain": float(
                    feedback_candidate_inlier_gain
                    / max(
                        feedback_candidate_inliers - feedback_candidate_inlier_gain, 1
                    )
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
                "sparse_feedback_reliability_adaptive_score_drop": int(
                    bool(
                        self.pose_conditioned_selection_config[
                            "reliability_adaptive_score_drop"
                        ]
                    )
                ),
                "sparse_feedback_reliability_authorized_edges": int(
                    feedback_reliability_authorized_edges
                ),
                "sparse_feedback_reliability_expanded_budget_edges": int(
                    feedback_reliability_expanded_budget_edges
                ),
                "sparse_feedback_reliability_expanded_selected_rows": int(
                    feedback_reliability_expanded_selected_rows
                ),
                "sparse_feedback_reliability_fallback_query_rows": int(
                    feedback_reliability_fallback_query_rows
                ),
                "sparse_feedback_reliability_matchability_threshold": float(
                    feedback_reliability_matchability_threshold
                ),
                "sparse_feedback_reliability_uncertainty_threshold": float(
                    feedback_reliability_uncertainty_threshold
                ),
                "sparse_feedback_duplicate_owner_rejections": (
                    feedback_duplicate_owner_rejections
                ),
                "sparse_feedback_pose_conditioned_mutual_matching": int(
                    bool(
                        self.pose_conditioned_selection_config[
                            "pose_conditioned_mutual_matching"
                        ]
                    )
                ),
                "sparse_feedback_mutual_candidate_rejections": int(
                    feedback_mutual_candidate_rejections
                ),
                "sparse_feedback_heldout_candidate_validation": int(
                    bool(
                        self.pose_conditioned_selection_config[
                            "heldout_candidate_validation"
                        ]
                    )
                ),
                "sparse_feedback_heldout_query_rows": int(feedback_heldout_query_rows),
                "sparse_feedback_heldout_candidate_edges": int(
                    feedback_heldout_candidate_edges
                ),
                "sparse_feedback_heldout_baseline_energy": float(
                    feedback_heldout_baseline_energy
                ),
                "sparse_feedback_heldout_candidate_energy": float(
                    feedback_heldout_candidate_energy
                ),
                "sparse_feedback_heldout_relative_energy_gain": float(
                    feedback_heldout_relative_energy_gain
                ),
                "sparse_feedback_heldout_minimum_relative_energy_gain": float(
                    self.refinement_minimum_heldout_relative_energy_gain
                ),
                "sparse_feedback_heldout_baseline_assignments": int(
                    feedback_heldout_baseline_assignments
                ),
                "sparse_feedback_heldout_candidate_assignments": int(
                    feedback_heldout_candidate_assignments
                ),
                "sparse_feedback_heldout_passed": int(feedback_heldout_passed),
                "sparse_feedback_uncertainty_aware_projection": int(
                    bool(
                        self.pose_conditioned_selection_config[
                            "uncertainty_aware_projection"
                        ]
                    )
                ),
                "sparse_feedback_pose_information_condition": float(
                    feedback_pose_information_condition
                ),
                "sparse_feedback_expanded_projection_edges": int(
                    feedback_expanded_projection_edges
                ),
                "sparse_feedback_projection_gate_p50_px": float(
                    feedback_projection_gate_p50_px
                ),
                "sparse_feedback_projection_gate_p90_px": float(
                    feedback_projection_gate_p90_px
                ),
                "sparse_feedback_view_support_available": int(
                    feedback_view_support_available
                ),
                "sparse_feedback_view_support_rejected_edges": (
                    feedback_view_support_rejected_edges
                ),
                "sparse_feedback_capacity_rejections": feedback_capacity_rejections,
                "sparse_feedback_visible_anchor_count": (feedback_visible_anchor_count),
                "sparse_feedback_candidate_pool_anchor_count": (
                    feedback_candidate_pool_anchor_count
                ),
                "sparse_feedback_candidate_pool_global_fallback": int(
                    feedback_candidate_pool_global_fallback
                ),
                "sparse_feedback_view_support_prefilter_fallback": int(
                    feedback_view_support_prefilter_fallback
                ),
                "sparse_feedback_retrieval_query_count": int(
                    feedback_retrieval_query_count
                ),
                "sparse_feedback_projection_first_local_candidates": int(
                    self.refinement_projection_first_local_candidates
                ),
                "sparse_feedback_projection_first_radius_px": float(
                    self.refinement_projection_first_radius_px
                ),
                "sparse_feedback_projection_local_edge_count": int(
                    feedback_projection_local_edge_count
                ),
                "sparse_feedback_projection_local_nonempty_query_count": int(
                    feedback_projection_local_nonempty_query_count
                ),
                "sparse_feedback_view_supported_anchor_count": int(
                    feedback_view_supported_anchor_count
                ),
                "sparse_feedback_view_conditioned_selected_mode_anchor_count": int(
                    feedback_view_conditioned_selected_mode_anchor_count
                ),
                "sparse_feedback_view_conditioned_base_fallback_anchor_count": int(
                    feedback_view_conditioned_base_fallback_anchor_count
                ),
                "sparse_feedback_spatial_jackknife_available": int(
                    feedback_spatial_jackknife_available
                ),
                "sparse_feedback_spatial_jackknife_baseline_task_p90": float(
                    feedback_spatial_jackknife_baseline_task_p90
                ),
                "sparse_feedback_spatial_jackknife_candidate_task_p90": float(
                    feedback_spatial_jackknife_candidate_task_p90
                ),
                "sparse_feedback_spatial_jackknife_relative_gain": float(
                    feedback_spatial_jackknife_relative_gain
                ),
                "sparse_feedback_spatial_jackknife_baseline_groups": int(
                    feedback_spatial_jackknife_baseline_groups
                ),
                "sparse_feedback_spatial_jackknife_candidate_groups": int(
                    feedback_spatial_jackknife_candidate_groups
                ),
                "sparse_feedback_active_row_retrieval": int(
                    self.refinement_active_row_retrieval
                ),
                "sparse_feedback_pre_topk_view_filter": int(
                    self.refinement_pre_topk_view_filter
                ),
                "sparse_feedback_common_candidate_grid_gate": int(
                    self.refinement_common_candidate_grid_gate
                ),
                "sparse_feedback_common_grid_baseline_energy": float(
                    feedback_common_grid_baseline_energy
                ),
                "sparse_feedback_common_grid_candidate_energy": float(
                    feedback_common_grid_candidate_energy
                ),
                "sparse_feedback_common_grid_relative_energy_gain": float(
                    feedback_common_grid_relative_energy_gain
                ),
                "sparse_feedback_common_grid_minimum_relative_energy_gain": float(
                    self.refinement_minimum_common_grid_relative_energy_gain
                ),
                "sparse_feedback_common_grid_baseline_duplicate_owners": int(
                    feedback_common_grid_baseline_duplicate_owners
                ),
                "sparse_feedback_common_grid_candidate_duplicate_owners": int(
                    feedback_common_grid_candidate_duplicate_owners
                ),
                "sparse_feedback_common_grid_passed": int(feedback_common_grid_passed),
                "sparse_feedback_progressive_sampling": int(
                    feedback_progressive_sampling
                ),
                "sparse_feedback_projection_gate_px": float(
                    feedback_projection_gate_px
                ),
                "sparse_feedback_soft_inliers_enabled": int(
                    bool(self.pose_conditioned_selection_config["allow_soft_inliers"])
                ),
                "sparse_feedback_soft_inlier_candidate_rows": int(
                    feedback_soft_inlier_candidate_rows
                ),
                "sparse_feedback_soft_inlier_changed_rows": int(
                    feedback_soft_inlier_changed_rows
                ),
                "sparse_feedback_soft_inlier_capacity_rejections": int(
                    feedback_soft_inlier_capacity_rejections
                ),
                "sparse_feedback_hard_core_inlier_rows": int(
                    feedback_hard_core_inlier_rows
                ),
                "sparse_feedback_selected_rank_median": (feedback_selected_rank_median),
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
                "reused_correspondence_buffers": int(self._points_2d_host is not None),
            },
        )
