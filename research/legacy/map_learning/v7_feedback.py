"""Fixed-plant V7 localization and strictly post-localization fault routing."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np
import poselib
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F

from features.extractor import FeatureExtractor


ROUTES = (
    "representation_deficit",
    "precision_deficit",
    "coverage_deficit",
    "unreliable_query",
    "nominal_success",
)


@dataclass(frozen=True)
class DiagnosticRegistry:
    anchor_ids: torch.Tensor
    anchor_xyz: torch.Tensor
    eligible: torch.Tensor


@dataclass(frozen=True)
class V7Top1Matches:
    keypoint_indices: torch.Tensor
    anchor_indices: torch.Tensor
    scores: torch.Tensor


@dataclass(frozen=True)
class V7PoseEstimate:
    pose_w2c: np.ndarray
    inliers: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class V7FixedPlantState:
    anchor_ids: torch.Tensor
    anchor_xyz: torch.Tensor
    anchor_features: torch.Tensor
    extractor: FeatureExtractor
    device: torch.device
    reprojection_error_px: float
    confidence: float
    maximum_iterations: int
    minimum_iterations: int
    seed: int
    diagnostic_registry: DiagnosticRegistry | None = None


@dataclass(frozen=True)
class V7LocalizationResult:
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor
    matches: V7Top1Matches
    pose: V7PoseEstimate
    intrinsic: np.ndarray
    runtime_ms: dict[str, float]
    active_anchor_ids: torch.Tensor
    active_anchor_xyz: torch.Tensor
    diagnostic_registry: DiagnosticRegistry | None
    solver_contract: Mapping[str, float | int] | None = None


@torch.inference_mode()
def _global_cosine_top1(
    query_descriptors: torch.Tensor,
    anchor_descriptors: torch.Tensor,
    *,
    chunk_size: int = 8192,
) -> V7Top1Matches:
    """The deployment Top-1 kernel, isolated from historical match variants."""

    query = F.normalize(torch.as_tensor(query_descriptors).float(), dim=1)
    anchors = torch.as_tensor(anchor_descriptors).float()
    if query.ndim != 2 or anchors.ndim != 2 or query.shape[1] != anchors.shape[1]:
        raise ValueError("V7 query and Anchor descriptor banks do not align")
    if anchors.shape[0] == 0:
        raise ValueError("V7 Anchor map is empty")
    best_scores = query.new_full((query.shape[0],), -torch.inf)
    best_indices = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
    for start in range(0, anchors.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), anchors.shape[0])
        scores = query @ anchors[start:stop].T
        values, rows = scores.max(dim=1)
        improve = values > best_scores
        best_scores[improve] = values[improve]
        best_indices[improve] = rows[improve] + start
    return V7Top1Matches(
        keypoint_indices=torch.arange(query.shape[0], device=query.device),
        anchor_indices=best_indices,
        scores=best_scores,
    )


def _poselib_camera(intrinsic: np.ndarray) -> dict[str, Any]:
    return {
        "model": "PINHOLE",
        "width": int(intrinsic[0, 2] * 2),
        "height": int(intrinsic[1, 2] * 2),
        "params": [
            intrinsic[0, 0],
            intrinsic[1, 1],
            intrinsic[0, 2],
            intrinsic[1, 2],
        ],
    }


def _solve_standard_poselib(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    *,
    reprojection_error_px: float,
    confidence: float,
    maximum_iterations: int,
    minimum_iterations: int,
    seed: int,
) -> V7PoseEstimate:
    if points_2d.shape[0] < 4:
        return V7PoseEstimate(
            np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int64), {}
        )
    pose, info = poselib.estimate_absolute_pose(
        points_2d,
        points_3d,
        _poselib_camera(intrinsic),
        {
            "max_iterations": int(maximum_iterations),
            "min_iterations": int(minimum_iterations),
            "max_reproj_error": float(reprojection_error_px),
            "success_prob": float(confidence),
            "progressive_sampling": False,
            "max_prosac_iterations": int(maximum_iterations),
            "seed": int(seed),
        },
        {"verbose": False},
    )
    if int(info["num_inliers"]) <= 0:
        return V7PoseEstimate(
            np.eye(4, dtype=np.float32), np.empty(0, dtype=np.int64), dict(info)
        )
    pose_w2c = np.concatenate(
        (pose.Rt, np.asarray([[0, 0, 0, 1]], dtype=np.float64)), axis=0
    ).astype(np.float32)
    return V7PoseEstimate(
        pose_w2c, np.flatnonzero(np.asarray(info["inliers"])), dict(info)
    )


def _validate_identity_metric(
    metric: Mapping[str, Any], anchor_ids: torch.Tensor
) -> None:
    if (
        metric.get("schema") != "lafgs_shared_metric_state"
        or metric.get("version") != 1
        or metric.get("step") != 0
        or metric.get("metric_config")
        != {"descriptor_dim": 256, "rank": 1, "max_residual_norm": 0.0}
        or not torch.equal(
            torch.as_tensor(metric.get("landmark_indices")).long(), anchor_ids
        )
    ):
        raise ValueError("V7 fixed plant requires an aligned identity metric")
    state = metric.get("metric_state_dict")
    if not isinstance(state, Mapping) or any(
        bool(torch.count_nonzero(torch.as_tensor(value))) for value in state.values()
    ):
        raise ValueError("V7 fixed plant forbids a learned descriptor transform")


def load_v7_fixed_plant(
    map_path: str | Path,
    metric_path: str | Path,
    *,
    device: str = "cuda",
    diagnostic_registry: DiagnosticRegistry | None = None,
    reprojection_error_px: float = 11.954343111400277,
    confidence: float = 0.99999,
    maximum_iterations: int = 100000,
    minimum_iterations: int = 1000,
    seed: int = 2026,
) -> V7FixedPlantState:
    """Load exactly one descriptor per Anchor and no optional online modules."""

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    state = torch.load(map_path, map_location="cpu", weights_only=False)
    metric = torch.load(metric_path, map_location="cpu", weights_only=False)
    if state.get("schema") != "lafgs_materialized_anchor_map":
        raise ValueError("unsupported V7 fixed-plant map")
    forbidden = {
        "anchor_extra_prototype_features",
        "anchor_extra_prototype_owner_rows",
        "v7_anchor_residual_parameter",
    }
    if forbidden & set(state):
        raise ValueError("V7 fixed plant contains a forbidden descriptor extension")
    anchor_ids = torch.as_tensor(state["anchor_ids"]).long().cpu()
    _validate_identity_metric(metric, anchor_ids)
    target = torch.device(device)
    if target.type != "cuda":
        raise ValueError("formal V7 localization requires CUDA")
    extractor = FeatureExtractor("sp", nms_radius=4).to(target).eval()
    extractor.requires_grad_(False)
    return V7FixedPlantState(
        anchor_ids=anchor_ids,
        anchor_xyz=torch.as_tensor(state["anchor_xyz"], device=target).float(),
        anchor_features=F.normalize(
            torch.as_tensor(state["anchor_features"], device=target).float(), dim=1
        ),
        extractor=extractor,
        device=target,
        reprojection_error_px=float(reprojection_error_px),
        confidence=float(confidence),
        maximum_iterations=int(maximum_iterations),
        minimum_iterations=int(minimum_iterations),
        seed=int(seed),
        diagnostic_registry=diagnostic_registry,
    )


@torch.inference_mode()
def localize_rgb_query(
    rgb: torch.Tensor,
    intrinsics: torch.Tensor,
    map_state: V7FixedPlantState,
) -> V7LocalizationResult:
    """Frozen RGB-only plant. GT, alpha, depth, and oracle inputs are impossible."""

    image = torch.as_tensor(rgb, device=map_state.device).float()
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3:
        raise ValueError("V7 query RGB must have shape [3,H,W] or [1,3,H,W]")
    intrinsic = torch.as_tensor(intrinsics).float().cpu()
    if intrinsic.shape != (3, 3) or not bool(torch.isfinite(intrinsic).all()):
        raise ValueError("V7 query intrinsics must be finite 3x3")
    torch.cuda.synchronize(map_state.device)
    started = time.perf_counter()
    torch.use_deterministic_algorithms(True)
    sparse = map_state.extractor.detectAndCompute(image, top_k=2048)[0]
    torch.use_deterministic_algorithms(False)
    keypoints = sparse["keypoints"]
    descriptors = F.normalize(sparse["descriptors"].float(), dim=1)
    scores = sparse["keypoint_scores"].float()
    torch.cuda.synchronize(map_state.device)
    frontend_ms = (time.perf_counter() - started) * 1000.0
    matching_started = time.perf_counter()
    matches = _global_cosine_top1(descriptors, map_state.anchor_features)
    points_2d = keypoints[matches.keypoint_indices].cpu().numpy() + 0.5
    points_3d = map_state.anchor_xyz[matches.anchor_indices].cpu().numpy()
    torch.cuda.synchronize(map_state.device)
    matching_ms = (time.perf_counter() - matching_started) * 1000.0
    ransac_started = time.perf_counter()
    intrinsic_numpy = intrinsic.numpy()
    pose = _solve_standard_poselib(
        points_2d,
        points_3d,
        intrinsic_numpy,
        reprojection_error_px=map_state.reprojection_error_px,
        confidence=map_state.confidence,
        maximum_iterations=map_state.maximum_iterations,
        minimum_iterations=map_state.minimum_iterations,
        seed=map_state.seed,
    )
    ransac_ms = (time.perf_counter() - ransac_started) * 1000.0
    return V7LocalizationResult(
        keypoints=keypoints.detach().cpu(),
        descriptors=descriptors.detach().cpu(),
        scores=scores.detach().cpu(),
        matches=V7Top1Matches(
            keypoint_indices=matches.keypoint_indices.detach().cpu(),
            anchor_indices=matches.anchor_indices.detach().cpu(),
            scores=matches.scores.detach().cpu(),
        ),
        pose=pose,
        intrinsic=intrinsic_numpy,
        runtime_ms={
            "frontend_ms": frontend_ms,
            "matching_ms": matching_ms,
            "ransac_ms": ransac_ms,
            "total_ms": (time.perf_counter() - started) * 1000.0,
        },
        active_anchor_ids=map_state.anchor_ids,
        active_anchor_xyz=map_state.anchor_xyz.detach().cpu(),
        diagnostic_registry=map_state.diagnostic_registry,
        solver_contract={
            "reprojection_error_px": map_state.reprojection_error_px,
            "confidence": map_state.confidence,
            "maximum_iterations": map_state.maximum_iterations,
            "minimum_iterations": map_state.minimum_iterations,
            "seed": map_state.seed,
        },
    )


def _pose_error(predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    relative = ground_truth[:3, :3] @ predicted[:3, :3].T
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    rotation = math.degrees(math.acos(float(cosine)))
    predicted_center = -predicted[:3, :3].T @ predicted[:3, 3]
    ground_truth_center = -ground_truth[:3, :3].T @ ground_truth[:3, 3]
    translation = float(np.linalg.norm(predicted_center - ground_truth_center) * 100.0)
    return rotation, translation


def pose_error(predicted: np.ndarray, ground_truth: np.ndarray) -> tuple[float, float]:
    """Public read-only pose metric used by non-controlling diagnostics."""

    return _pose_error(predicted, ground_truth)


def _project(
    xyz: torch.Tensor, pose_w2c: torch.Tensor, intrinsic: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = torch.as_tensor(xyz).double()
    pose = torch.as_tensor(pose_w2c).double()
    intrinsic = torch.as_tensor(intrinsic).double()
    camera = (pose[:3, :3] @ xyz.T + pose[:3, 3:4]).T
    valid = torch.isfinite(camera).all(1) & (camera[:, 2] > 1e-8)
    projected = torch.empty(camera.shape[0], 2, dtype=torch.float64)
    projected[:] = torch.nan
    homogeneous = (intrinsic @ camera[valid].T).T
    projected[valid] = homogeneous[:, :2] / homogeneous[:, 2:3]
    return projected.numpy(), valid.numpy(), camera[:, 2].numpy()


def _sample_query_raster(raster: torch.Tensor, keypoints: torch.Tensor) -> np.ndarray:
    value = torch.as_tensor(raster).float().squeeze()
    if value.ndim != 2:
        raise ValueError("feedback alpha/depth raster must reduce to [H,W]")
    xy = torch.floor(torch.as_tensor(keypoints).float()).long()
    x = xy[:, 0].clamp(0, value.shape[1] - 1)
    y = xy[:, 1].clamp(0, value.shape[0] - 1)
    return value[y, x].numpy()


def diagnose_feedback_query(
    localization_result: V7LocalizationResult,
    gt_pose: torch.Tensor,
    alpha: torch.Tensor,
    depth: torch.Tensor,
    quality_certificate: Mapping[str, Any],
    *,
    success_translation_cm: float = 5.0,
    success_rotation_deg: float = 5.0,
    oracle_reprojection_px: float = 4.0,
    minimum_oracle_correspondences: int = 16,
    precision_spatial_grid: tuple[int, int] = (4, 4),
    minimum_precision_spatial_cells: int = 6,
    minimum_precision_supporting_rows: int = 8,
    minimum_precision_translation_improvement_cm: float = 0.05,
    minimum_precision_translation_relative_improvement: float = 0.10,
    minimum_precision_rotation_improvement_deg: float = 0.005,
) -> dict[str, Any]:
    """Route a query only after the fixed plant has returned its pose."""

    alpha_tensor = torch.as_tensor(alpha).float()
    depth_tensor = torch.as_tensor(depth).float()
    raster_diagnostics = {
        "finite_alpha_fraction": float(torch.isfinite(alpha_tensor).float().mean()),
        "positive_depth_fraction": float(
            (torch.isfinite(depth_tensor) & (depth_tensor > 0)).float().mean()
        ),
    }
    decision = quality_certificate.get("decision")
    if decision not in {"ACCEPT", "UNCERTAIN", "REJECT"}:
        raise ValueError("feedback diagnosis requires a tri-state certificate")
    if decision != "ACCEPT":
        return {
            "category": "unreliable_query",
            "certificate_decision": decision,
            "can_drive_map_update": False,
            "included_in_feedback_statistics": decision == "UNCERTAIN",
            "raster_diagnostics": raster_diagnostics,
            "oracle_used_online": False,
        }
    if quality_certificate.get("can_drive_map_update") is not True:
        raise ValueError("ACCEPT certificate is not update-authorized")
    gt = torch.as_tensor(gt_pose).double()
    rotation, translation = _pose_error(localization_result.pose.pose_w2c, gt.numpy())
    pose_success = translation < float(success_translation_cm) and rotation < float(
        success_rotation_deg
    )
    keypoints = localization_result.keypoints + 0.5
    row_valid = torch.as_tensor(quality_certificate["row_valid"]).bool()
    if row_valid.shape != (keypoints.shape[0],):
        raise ValueError("certificate row mask differs from localized keypoints")

    active_projected, active_positive, active_depth = _project(
        localization_result.active_anchor_xyz, gt, localization_result.intrinsic
    )
    sampled_depth = _sample_query_raster(depth_tensor, keypoints)
    sampled_alpha = _sample_query_raster(alpha_tensor, keypoints)
    depth_tolerance = np.maximum(0.5, 0.10 * sampled_depth)
    winner = localization_result.matches.anchor_indices.numpy()
    winner_xy = active_projected[winner]
    winner_correct = (
        np.isfinite(winner_xy).all(1)
        & active_positive[winner]
        & np.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & (sampled_alpha >= 0.05)
        & (np.abs(active_depth[winner] - sampled_depth) <= depth_tolerance)
        & (
            np.linalg.norm(winner_xy - keypoints.numpy(), axis=1)
            <= float(oracle_reprojection_px)
        )
        & row_valid.numpy()
    )
    registry = localization_result.diagnostic_registry
    if registry is None:
        raise ValueError("P4 diagnosis requires an offline candidate registry")
    candidate_xy, candidate_positive, candidate_depth = _project(
        registry.anchor_xyz, gt, localization_result.intrinsic
    )
    candidate_valid = (
        candidate_positive & torch.as_tensor(registry.eligible).bool().numpy()
    )
    eligible_candidate_rows = np.flatnonzero(candidate_valid)
    tree = cKDTree(candidate_xy[eligible_candidate_rows])
    search_k = min(32, max(int(eligible_candidate_rows.size), 1))
    distances, nearest_local_rows = tree.query(keypoints[row_valid].numpy(), k=search_k)
    distances = np.asarray(distances).reshape(-1, search_k)
    nearest_local_rows = np.asarray(nearest_local_rows).reshape(-1, search_k)
    valid_query_rows = np.flatnonzero(row_valid.numpy())
    if eligible_candidate_rows.size:
        candidate_rows = eligible_candidate_rows[nearest_local_rows]
        query_depth = sampled_depth[valid_query_rows, None]
        query_alpha = sampled_alpha[valid_query_rows, None]
        oracle_pairs = (
            (distances <= float(oracle_reprojection_px))
            & np.isfinite(query_depth)
            & (query_depth > 0)
            & (query_alpha >= 0.05)
            & (
                np.abs(candidate_depth[candidate_rows] - query_depth)
                <= np.maximum(0.5, 0.10 * query_depth)
            )
        )
    else:
        candidate_rows = np.zeros_like(nearest_local_rows)
        oracle_pairs = np.zeros_like(distances, dtype=bool)
    oracle_supported = oracle_pairs.any(1)
    oracle_count = int(np.count_nonzero(oracle_supported))
    correct_top1_count = int(np.count_nonzero(winner_correct))
    solver_geometry = bool(
        not pose_success
        and oracle_count >= int(minimum_oracle_correspondences)
        and correct_top1_count >= int(minimum_oracle_correspondences)
    )
    # Precision evidence is deliberately restricted to the currently deployed
    # map.  It cannot add coverage: it asks whether a unique, spatially spread
    # set of already available Anchors would make the same standard PoseLib
    # solve materially more accurate.
    active_rows = np.flatnonzero(active_positive & np.isfinite(active_projected).all(1))
    active_search_k = min(32, max(int(active_rows.size), 1))
    alternative_candidates: list[tuple[float, int, int]] = []
    if active_rows.size:
        active_tree = cKDTree(active_projected[active_rows])
        active_distances, active_nearest = active_tree.query(
            keypoints[row_valid].numpy(), k=active_search_k
        )
        active_distances = np.asarray(active_distances).reshape(-1, active_search_k)
        active_nearest = np.asarray(active_nearest).reshape(-1, active_search_k)
        active_candidate_rows = active_rows[active_nearest]
        for local_query_row, query_row in enumerate(valid_query_rows):
            for neighbor in range(active_search_k):
                anchor_row = int(active_candidate_rows[local_query_row, neighbor])
                distance = float(active_distances[local_query_row, neighbor])
                if distance > float(oracle_reprojection_px):
                    break
                if (
                    np.isfinite(sampled_depth[query_row])
                    and sampled_depth[query_row] > 0
                    and sampled_alpha[query_row] >= 0.05
                    and abs(active_depth[anchor_row] - sampled_depth[query_row])
                    <= depth_tolerance[query_row]
                ):
                    alternative_candidates.append((distance, int(query_row), anchor_row))
    alternative_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used_query_rows: set[int] = set()
    used_anchor_rows: set[int] = set()
    selected_alternatives: list[tuple[int, int]] = []
    for _, query_row, anchor_row in alternative_candidates:
        if query_row in used_query_rows or anchor_row in used_anchor_rows:
            continue
        used_query_rows.add(query_row)
        used_anchor_rows.add(anchor_row)
        selected_alternatives.append((query_row, anchor_row))
    alternative_query_rows = np.asarray(
        [item[0] for item in selected_alternatives], dtype=np.int64
    )
    alternative_anchor_rows = np.asarray(
        [item[1] for item in selected_alternatives], dtype=np.int64
    )
    grid_width, grid_height = map(int, precision_spatial_grid)
    if grid_width <= 0 or grid_height <= 0:
        raise ValueError("precision spatial grid must be positive")
    image_height, image_width = map(int, depth_tensor.squeeze().shape)
    if alternative_query_rows.size:
        alternative_xy = keypoints[alternative_query_rows].numpy()
        cell_x = np.clip(
            (alternative_xy[:, 0] * grid_width / max(image_width, 1)).astype(int),
            0,
            grid_width - 1,
        )
        cell_y = np.clip(
            (alternative_xy[:, 1] * grid_height / max(image_height, 1)).astype(int),
            0,
            grid_height - 1,
        )
        alternative_spatial_cells = int(
            np.unique(cell_y * grid_width + cell_x).size
        )
    else:
        alternative_spatial_cells = 0
    changed_alternative = (
        alternative_anchor_rows != winner[alternative_query_rows]
        if alternative_query_rows.size
        else np.empty(0, dtype=bool)
    )
    precision_supporting_rows = int(np.count_nonzero(changed_alternative))
    alternative_pose = None
    alternative_rotation = math.nan
    alternative_translation = math.nan
    translation_improvement = math.nan
    rotation_improvement = math.nan
    precision_replay_eligible = bool(
        pose_success
        and alternative_query_rows.size >= int(minimum_oracle_correspondences)
        and alternative_spatial_cells >= int(minimum_precision_spatial_cells)
        and precision_supporting_rows >= int(minimum_precision_supporting_rows)
    )
    if precision_replay_eligible:
        solver_contract = localization_result.solver_contract or {
            "reprojection_error_px": 11.954343111400277,
            "confidence": 0.99999,
            "maximum_iterations": 100000,
            "minimum_iterations": 1000,
            "seed": 2026,
        }
        alternative_pose = _solve_standard_poselib(
            keypoints[alternative_query_rows].numpy(),
            localization_result.active_anchor_xyz[alternative_anchor_rows].numpy(),
            localization_result.intrinsic,
            reprojection_error_px=float(solver_contract["reprojection_error_px"]),
            confidence=float(solver_contract["confidence"]),
            maximum_iterations=int(solver_contract["maximum_iterations"]),
            minimum_iterations=int(solver_contract["minimum_iterations"]),
            seed=int(solver_contract["seed"]),
        )
        alternative_rotation, alternative_translation = _pose_error(
            alternative_pose.pose_w2c, gt.numpy()
        )
        translation_improvement = translation - alternative_translation
        rotation_improvement = rotation - alternative_rotation
    precision_deficit = bool(
        precision_replay_eligible
        and alternative_pose is not None
        and alternative_pose.inliers.size > 0
        and translation_improvement
        >= max(
            float(minimum_precision_translation_improvement_cm),
            float(minimum_precision_translation_relative_improvement) * translation,
        )
        and rotation_improvement
        >= float(minimum_precision_rotation_improvement_deg)
    )
    if precision_deficit:
        category = "precision_deficit"
    elif pose_success:
        category = "nominal_success"
    elif oracle_count < int(minimum_oracle_correspondences):
        category = "coverage_deficit"
    else:
        category = "representation_deficit"
    wrong_supported = oracle_supported & ~winner_correct[valid_query_rows]
    if precision_deficit:
        evidence_query_rows = alternative_query_rows[changed_alternative]
        positive_anchor_ids = localization_result.active_anchor_ids[
            alternative_anchor_rows[changed_alternative]
        ].clone().long()
        false_anchor_ids = localization_result.active_anchor_ids[
            winner[evidence_query_rows]
        ].clone().long()
    elif eligible_candidate_rows.size:
        evidence_query_rows = valid_query_rows[wrong_supported]
        first_supported = oracle_pairs.argmax(1)
        positive_rows = candidate_rows[
            np.arange(candidate_rows.shape[0]), first_supported
        ][wrong_supported]
        positive_anchor_ids = registry.anchor_ids[positive_rows].clone().long()
        false_anchor_ids = localization_result.active_anchor_ids[
            winner[evidence_query_rows]
        ].clone().long()
    else:
        evidence_query_rows = np.empty(0, dtype=np.int64)
        positive_anchor_ids = torch.empty(0, dtype=torch.long)
        false_anchor_ids = torch.empty(0, dtype=torch.long)
    return {
        "category": category,
        "certificate_decision": decision,
        "can_drive_map_update": (
            category == "precision_deficit"
            or (category == "representation_deficit" and not solver_geometry)
        ),
        "included_in_feedback_statistics": True,
        "translation_error_cm": translation,
        "rotation_error_deg": rotation,
        "pose_success": pose_success,
        "oracle_correspondence_count": oracle_count,
        "correct_top1_count": correct_top1_count,
        "valid_keypoint_count": int(row_valid.sum()),
        "solver_geometry_diagnostic": solver_geometry,
        "precision_diagnostic": {
            "replay_eligible": precision_replay_eligible,
            "deficit": precision_deficit,
            "alternative_correspondence_count": int(alternative_query_rows.size),
            "unique_anchor_count": int(np.unique(alternative_anchor_rows).size),
            "spatial_cell_count": alternative_spatial_cells,
            "supporting_row_count": precision_supporting_rows,
            "alternative_translation_error_cm": alternative_translation,
            "alternative_rotation_error_deg": alternative_rotation,
            "translation_improvement_cm": translation_improvement,
            "rotation_improvement_deg": rotation_improvement,
        },
        "oracle_used_online": False,
        "raster_diagnostics": raster_diagnostics,
        "descriptor_control_evidence": {
            "query_rows": torch.from_numpy(evidence_query_rows).long(),
            "query_descriptors": localization_result.descriptors[evidence_query_rows]
            .clone()
            .float(),
            "positive_anchor_ids": positive_anchor_ids,
            "false_attractor_anchor_ids": false_anchor_ids,
        },
    }
