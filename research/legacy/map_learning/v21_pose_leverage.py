"""Exact-PoseLib leverage audit for sparse, map-side descriptor repair.

The geometry in this module proposes *pose-valid candidate edges*.  It never
labels an Anchor negative and it never authorizes a map action.  A failed query
is descriptor-controllable only as an upper bound when replacing a finite set
of current winners with geometry-supported candidates makes the unchanged
Top-1/PoseLib plant pass R5.  Projective identity needs separate Track/
observation consensus before a controller may consume the result.

The pure functions deliberately accept a solver callback.  Production uses
``solve_absolute_pose``; tests can use a small deterministic discontinuous
plant without mocking module globals.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from localization.pose_solver import pose_error, solve_absolute_pose
from map_learning.v6_control_actions import (
    minimal_pose_correction_set,
    pose_priority_prefix_correction_set,
)
from map_learning.v8_feedback_controller import task_error
from map_learning.v21_correspondence_truth import (
    STATUS_EQUIVALENT,
    STATUS_UNIQUE,
    validate_query_truth_record,
)


SCHEMA = "lafgs_v21_pose_leverage_query"
VERSION = 1
R5_TRANSLATION_CM = 5.0
R5_ROTATION_DEG = 5.0

PROTECTION_ONLY = "protection_only"
DESCRIPTOR_CONTROLLABLE = "descriptor_controllable"
COVERAGE_LIMITED = "coverage_limited"
GEOMETRY_LIMITED = "geometry_limited"
SOLVER_LIMITED = "solver_limited"
ASSIGNMENT_INDETERMINATE = "candidate_assignment_indeterminate"
TRACK_DIAGNOSTIC_COVERAGE_LIMITED = "track_diagnostic_coverage_limited"

GAUSSIAN_GEOMETRY_SUPPORTED = "gaussian_geometry_supported_upper_bound"
REPROJECTION_UPPER_BOUND = "reprojection_upper_bound_only"
TRACK_CONSENSUS_DIAGNOSTIC = "track_consensus_diagnostic"
GAUSSIAN_GEOMETRY_SOURCE = "gaussian_geometry"


def _meets_pose_target(
    outcome: dict, *, translation_cm: float, rotation_deg: float
) -> bool:
    return bool(
        float(outcome["translation_error_cm"]) < float(translation_cm)
        and float(outcome["rotation_error_deg"]) < float(rotation_deg)
    )


def _tensor(value: Any, *, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype).detach().cpu()


def _validate_scene_geometry(
    *,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    xy = _tensor(keypoints, dtype=torch.float32)
    xyz = _tensor(anchor_xyz, dtype=torch.float32)
    calibration = _tensor(intrinsic, dtype=torch.float32)
    pose = _tensor(ground_truth_w2c, dtype=torch.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or xy.shape[0] == 0:
        raise ValueError("V21 keypoints must be a non-empty [rows,2] tensor")
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        raise ValueError("V21 Anchor XYZ must be a non-empty [anchors,3] tensor")
    if calibration.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("V21 intrinsics or ground-truth pose has the wrong shape")
    if not all(
        bool(torch.isfinite(value).all()) for value in (xy, xyz, calibration, pose)
    ):
        raise ValueError("V21 geometry contains a non-finite value")
    if float(calibration[0, 0]) <= 0.0 or float(calibration[1, 1]) <= 0.0:
        raise ValueError("V21 focal lengths must be positive")
    return xy, xyz, calibration, pose


def _equivalence_ids(
    equivalence_class_ids: torch.Tensor | None, anchor_count: int
) -> torch.Tensor:
    if equivalence_class_ids is None:
        return torch.arange(anchor_count, dtype=torch.long)
    equivalence = _tensor(equivalence_class_ids, dtype=torch.long).reshape(-1)
    if equivalence.shape != (anchor_count,) or bool((equivalence < 0).any()):
        raise ValueError("V21 equivalence registry does not align with the map")
    return equivalence


def project_anchor_geometry(
    *,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
) -> dict:
    """Project every frozen Anchor into one trusted feedback pose."""

    xyz = _tensor(anchor_xyz, dtype=torch.float32)
    calibration = _tensor(intrinsic, dtype=torch.float32)
    pose = _tensor(ground_truth_w2c, dtype=torch.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("V21 projection requires [anchors,3] XYZ")
    if calibration.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("V21 projection calibration or pose differs")
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ calibration.T
    depth = camera[:, 2]
    uv = homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8)
    valid = torch.isfinite(uv).all(1) & torch.isfinite(depth) & (depth > 1e-8)
    return {"uv": uv, "depth": depth, "valid": valid}


def _positive_parts(
    *,
    row_count: int,
    anchors: Sequence[list[torch.Tensor]],
    residuals: Sequence[list[torch.Tensor]],
    equivalence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor_parts: list[torch.Tensor] = []
    residual_parts: list[torch.Tensor] = []
    counts = []
    for row in range(row_count):
        if anchors[row]:
            local_anchors = torch.cat(anchors[row]).long()
            local_residuals = torch.cat(residuals[row]).float()
            best: dict[int, float] = {}
            for anchor, residual in zip(
                local_anchors.tolist(), local_residuals.tolist()
            ):
                best[int(anchor)] = min(best.get(int(anchor), math.inf), residual)
            ordered = sorted(best, key=lambda anchor: (best[anchor], anchor))
            local_anchors = torch.tensor(ordered, dtype=torch.long)
            local_residuals = torch.tensor(
                [best[anchor] for anchor in ordered], dtype=torch.float32
            )
        else:
            local_anchors = torch.empty(0, dtype=torch.long)
            local_residuals = torch.empty(0, dtype=torch.float32)
        anchor_parts.append(local_anchors)
        residual_parts.append(local_residuals)
        counts.append(local_anchors.numel())
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long),
            torch.tensor(counts, dtype=torch.long).cumsum(0),
        )
    )
    flat_anchors = (
        torch.cat(anchor_parts) if anchor_parts else torch.empty(0, dtype=torch.long)
    )
    flat_residuals = (
        torch.cat(residual_parts)
        if residual_parts
        else torch.empty(0, dtype=torch.float32)
    )
    return offsets, flat_anchors, equivalence[flat_anchors], flat_residuals


def _projection_grid_neighborhoods(
    *,
    keypoints: torch.Tensor,
    projected_uv: torch.Tensor,
    projected_valid: torch.Tensor,
    active_rows: torch.Tensor,
    threshold_px: float,
) -> dict[int, torch.Tensor]:
    """Return exact radius candidates without a rows-by-map distance matrix."""

    if active_rows.numel() == 0 or not bool(projected_valid.any()):
        return {}
    radius = float(threshold_px)
    active_xy = keypoints[active_rows]
    lower = active_xy.amin(0) - radius
    upper = active_xy.amax(0) + radius
    in_box = (
        projected_valid
        & (projected_uv[:, 0] >= lower[0])
        & (projected_uv[:, 0] <= upper[0])
        & (projected_uv[:, 1] >= lower[1])
        & (projected_uv[:, 1] <= upper[1])
    )
    visible_anchors = torch.nonzero(in_box, as_tuple=False).reshape(-1)
    if visible_anchors.numel() == 0:
        return {}
    anchor_cells = torch.floor(projected_uv[visible_anchors] / radius).long()
    query_cells = torch.floor(active_xy / radius).long()
    minimum_x = int(min(anchor_cells[:, 0].min(), query_cells[:, 0].min())) - 1
    maximum_x = int(max(anchor_cells[:, 0].max(), query_cells[:, 0].max())) + 1
    minimum_y = int(min(anchor_cells[:, 1].min(), query_cells[:, 1].min())) - 1
    width = maximum_x - minimum_x + 1
    if width <= 0:
        raise RuntimeError("V21 projection grid width is invalid")

    def keys(cells: torch.Tensor) -> torch.Tensor:
        return (cells[:, 1] - minimum_y) * width + (cells[:, 0] - minimum_x)

    anchor_keys = keys(anchor_cells)
    order = torch.argsort(anchor_keys, stable=True)
    sorted_keys = anchor_keys[order]
    sorted_anchors = visible_anchors[order]
    offsets = torch.tensor(
        [
            [-1, -1],
            [0, -1],
            [1, -1],
            [-1, 0],
            [0, 0],
            [1, 0],
            [-1, 1],
            [0, 1],
            [1, 1],
        ],
        dtype=torch.long,
    )
    neighbor_cells = query_cells[:, None, :] + offsets[None]
    neighbor_keys = keys(neighbor_cells.reshape(-1, 2)).reshape(
        active_rows.numel(), -1
    )
    left = torch.searchsorted(sorted_keys, neighbor_keys.reshape(-1), right=False)
    right = torch.searchsorted(sorted_keys, neighbor_keys.reshape(-1), right=True)
    left = left.reshape_as(neighbor_keys)
    right = right.reshape_as(neighbor_keys)
    output = {}
    for local, row in enumerate(active_rows.tolist()):
        parts = [
            sorted_anchors[int(start) : int(stop)]
            for start, stop in zip(left[local].tolist(), right[local].tolist())
            if int(stop) > int(start)
        ]
        if parts:
            output[int(row)] = torch.unique(torch.cat(parts), sorted=True)
    return output


def build_legal_positive_csr(
    *,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    equivalence_class_ids: torch.Tensor | None = None,
    row_valid_mask: torch.Tensor | None = None,
    gaussian_depth_at_keypoints: torch.Tensor | None = None,
    gaussian_alpha_at_keypoints: torch.Tensor | None = None,
    gaussian_valid_keypoint_mask: torch.Tensor | None = None,
    gaussian_relative_depth_spread_3x3: torch.Tensor | None = None,
    gaussian_local_valid_fraction_3x3: torch.Tensor | None = None,
    anchor_visibility_mask: torch.Tensor | None = None,
    reprojection_threshold_px: float = 2.0,
    minimum_alpha: float = 0.05,
    depth_absolute_m: float = 0.50,
    depth_relative: float = 0.10,
    maximum_relative_depth_spread: float = 0.25,
    minimum_local_valid_fraction: float = 0.50,
    chunk_size: int = 4096,
) -> dict:
    """Build a geometry-filtered, equivalence-aware candidate CSR.

    Gaussian depth/alpha can certify geometric compatibility, not Projective
    Track/Anchor identity.  Every result is therefore an upper bound until a
    separately validated identity-consensus artifact intersects these edges.
    No complement of this CSR is ever interpreted as a negative label.
    """

    xy, xyz, calibration, pose = _validate_scene_geometry(
        keypoints=keypoints,
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        ground_truth_w2c=ground_truth_w2c,
    )
    if (
        not math.isfinite(float(reprojection_threshold_px))
        or float(reprojection_threshold_px) <= 0.0
        or not math.isfinite(float(minimum_alpha))
        or not 0.0 <= float(minimum_alpha) <= 1.0
        or not math.isfinite(float(depth_absolute_m))
        or float(depth_absolute_m) < 0.0
        or not math.isfinite(float(depth_relative))
        or float(depth_relative) < 0.0
        or not math.isfinite(float(maximum_relative_depth_spread))
        or float(maximum_relative_depth_spread) < 0.0
        or not math.isfinite(float(minimum_local_valid_fraction))
        or not 0.0 <= float(minimum_local_valid_fraction) <= 1.0
        or int(chunk_size) < 1
    ):
        raise ValueError("V21 positive geometry thresholds are invalid")
    equivalence = _equivalence_ids(equivalence_class_ids, xyz.shape[0])
    row_valid = (
        torch.ones(xy.shape[0], dtype=torch.bool)
        if row_valid_mask is None
        else _tensor(row_valid_mask, dtype=torch.bool).reshape(-1)
    )
    if row_valid.shape != (xy.shape[0],):
        raise ValueError("V21 row-valid mask does not align with keypoints")
    visible = (
        torch.ones(xyz.shape[0], dtype=torch.bool)
        if anchor_visibility_mask is None
        else _tensor(anchor_visibility_mask, dtype=torch.bool).reshape(-1)
    )
    if visible.shape != (xyz.shape[0],):
        raise ValueError("V21 Anchor visibility mask does not align with the map")

    if (gaussian_depth_at_keypoints is None) != (
        gaussian_alpha_at_keypoints is None
    ):
        raise ValueError("V21 sampled Gaussian depth/alpha must be supplied together")
    geometry_supported = (
        gaussian_depth_at_keypoints is not None
        and gaussian_alpha_at_keypoints is not None
    )
    if not geometry_supported and any(
        value is not None
        for value in (
            gaussian_valid_keypoint_mask,
            gaussian_relative_depth_spread_3x3,
            gaussian_local_valid_fraction_3x3,
        )
    ):
        raise ValueError("V21 sampled Gaussian diagnostics require depth/alpha")
    evidence_mode = (
        GAUSSIAN_GEOMETRY_SUPPORTED
        if geometry_supported
        else REPROJECTION_UPPER_BOUND
    )
    observed_depth = None
    if gaussian_depth_at_keypoints is not None:
        observed_depth = _tensor(
            gaussian_depth_at_keypoints, dtype=torch.float32
        ).reshape(-1)
        if observed_depth.shape != (xy.shape[0],):
            raise ValueError("V21 sampled Gaussian depth does not align with rows")
        row_valid &= torch.isfinite(observed_depth) & (observed_depth > 0)
    if gaussian_alpha_at_keypoints is not None:
        observed_alpha = _tensor(
            gaussian_alpha_at_keypoints, dtype=torch.float32
        ).reshape(-1)
        if observed_alpha.shape != (xy.shape[0],):
            raise ValueError("V21 sampled Gaussian alpha does not align with rows")
        row_valid &= (
            torch.isfinite(observed_alpha)
            & (observed_alpha >= float(minimum_alpha))
            & (observed_alpha <= 1.0 + 1e-4)
        )
    gaussian_valid = torch.ones(xy.shape[0], dtype=torch.bool)
    if gaussian_valid_keypoint_mask is not None:
        gaussian_valid = _tensor(
            gaussian_valid_keypoint_mask, dtype=torch.bool
        ).reshape(-1)
        if gaussian_valid.shape != (xy.shape[0],):
            raise ValueError("V21 sampled Gaussian validity does not align with rows")
        row_valid &= gaussian_valid
    if gaussian_relative_depth_spread_3x3 is not None:
        depth_spread = _tensor(
            gaussian_relative_depth_spread_3x3, dtype=torch.float32
        ).reshape(-1)
        if (
            depth_spread.shape != (xy.shape[0],)
            or not bool(torch.isfinite(depth_spread[gaussian_valid]).all())
            or bool((depth_spread[gaussian_valid] < 0).any())
        ):
            raise ValueError("V21 sampled Gaussian depth spread is invalid")
        row_valid &= torch.isfinite(depth_spread) & (
            depth_spread <= float(maximum_relative_depth_spread)
        )
    if gaussian_local_valid_fraction_3x3 is not None:
        local_fraction = _tensor(
            gaussian_local_valid_fraction_3x3, dtype=torch.float32
        ).reshape(-1)
        if (
            local_fraction.shape != (xy.shape[0],)
            or not bool(torch.isfinite(local_fraction).all())
            or bool((local_fraction < 0).any())
            or bool((local_fraction > 1).any())
        ):
            raise ValueError("V21 sampled Gaussian valid fraction is invalid")
        row_valid &= local_fraction >= float(minimum_local_valid_fraction)

    projection = project_anchor_geometry(
        anchor_xyz=xyz,
        intrinsic=calibration,
        ground_truth_w2c=pose,
    )
    projected_uv = projection["uv"]
    projected_depth = projection["depth"]
    projected_valid = projection["valid"] & visible
    anchor_parts: list[list[torch.Tensor]] = [[] for _ in range(xy.shape[0])]
    residual_parts: list[list[torch.Tensor]] = [[] for _ in range(xy.shape[0])]
    active_rows = torch.nonzero(row_valid, as_tuple=False).reshape(-1)
    neighborhoods = _projection_grid_neighborhoods(
        keypoints=xy,
        projected_uv=projected_uv,
        projected_valid=projected_valid,
        active_rows=active_rows,
        threshold_px=reprojection_threshold_px,
    )
    for row, local_anchor in neighborhoods.items():
        for start in range(0, local_anchor.numel(), int(chunk_size)):
            anchors = local_anchor[start : start + int(chunk_size)]
            distance = torch.linalg.norm(projected_uv[anchors] - xy[row], dim=1)
            positive = torch.isfinite(distance)
            positive &= distance <= float(reprojection_threshold_px)
            positive &= projected_valid[anchors]
            if observed_depth is not None:
                tolerance = max(
                    float(depth_absolute_m),
                    abs(float(observed_depth[row])) * float(depth_relative),
                )
                positive &= (
                    projected_depth[anchors] - observed_depth[row]
                ).abs() <= tolerance
            if not bool(positive.any()):
                continue
            anchor_parts[row].append(anchors[positive])
            residual_parts[row].append(distance[positive])

    offsets, anchors, classes, residuals = _positive_parts(
        row_count=xy.shape[0],
        anchors=anchor_parts,
        residuals=residual_parts,
        equivalence=equivalence,
    )
    row_lengths = offsets[1:] - offsets[:-1]
    return {
        "schema": "lafgs_v21_legal_positive_csr",
        "version": VERSION,
        "row_count": int(xy.shape[0]),
        "anchor_count": int(xyz.shape[0]),
        "positive_offsets": offsets,
        "positive_anchor_rows": anchors,
        "positive_equivalence_class_ids": classes,
        "positive_reprojection_error_px": residuals,
        "row_has_legal_positive": row_lengths > 0,
        "row_valid_mask": row_valid,
        "legal_positive_edge_count": int(anchors.numel()),
        "legal_positive_row_count": int((row_lengths > 0).sum()),
        "legal_positive_equivalence_count": int(torch.unique(classes).numel()),
        "positive_evidence_mode": evidence_mode,
        "positive_source": GAUSSIAN_GEOMETRY_SOURCE,
        "geometry_supported_candidate": bool(geometry_supported),
        "candidate_edge_semantics": "pose_recovery_upper_bound_not_correspondence_truth",
        "deployable_positive_authorized": False,
        "positive_identity_authority": "none_requires_track_observation_consensus",
        "authorization_requires_exact_poselib": True,
        "topk_candidates_do_not_authorize": True,
        "negative_anchor_rows": torch.empty(0, dtype=torch.long),
        "unlabeled_rows_are_negative": False,
        "parameters": {
            "reprojection_threshold_px": float(reprojection_threshold_px),
            "minimum_alpha": float(minimum_alpha),
            "depth_absolute_m": float(depth_absolute_m),
            "depth_relative": float(depth_relative),
            "maximum_relative_depth_spread": float(
                maximum_relative_depth_spread
            ),
            "minimum_local_valid_fraction": float(minimum_local_valid_fraction),
            "chunk_size": int(chunk_size),
        },
    }


def build_track_consensus_diagnostic_csr(
    *,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    track_consensus_record: dict,
    equivalence_class_ids: torch.Tensor | None = None,
    row_valid_mask: torch.Tensor | None = None,
) -> dict:
    """Consume only UNIQUE/EQUIVALENT Track diagnostic edges.

    This is stronger identity evidence than Gaussian projection, but the
    diagnostic artifact explicitly withholds map/metric action authority.
    Its complement and its AMBIGUOUS/NO_TRUTH rows therefore remain unlabelled.
    """

    xy, xyz, calibration, pose = _validate_scene_geometry(
        keypoints=keypoints,
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        ground_truth_w2c=ground_truth_w2c,
    )
    validate_query_truth_record(track_consensus_record, anchor_count=xyz.shape[0])
    if int(track_consensus_record["keypoint_count"]) != int(xy.shape[0]):
        raise ValueError("V21 Track diagnostic rows do not align with keypoints")
    equivalence = _equivalence_ids(equivalence_class_ids, xyz.shape[0])
    status = _tensor(
        track_consensus_record["diagnostic_truth_status"], dtype=torch.int8
    ).reshape(-1)
    source_offsets = _tensor(
        track_consensus_record["diagnostic_positive_offsets"], dtype=torch.long
    ).reshape(-1)
    source_anchors = _tensor(
        track_consensus_record["diagnostic_positive_anchor_rows"], dtype=torch.long
    ).reshape(-1)
    if status.shape != (xy.shape[0],) or source_offsets.shape != (xy.shape[0] + 1,):
        raise ValueError("V21 Track diagnostic CSR does not align with keypoints")
    row_valid = (
        torch.ones(xy.shape[0], dtype=torch.bool)
        if row_valid_mask is None
        else _tensor(row_valid_mask, dtype=torch.bool).reshape(-1)
    )
    if row_valid.shape != (xy.shape[0],):
        raise ValueError("V21 Track diagnostic row-valid mask does not align")

    counts = source_offsets[1:] - source_offsets[:-1]
    decisive = (status == STATUS_UNIQUE) | (status == STATUS_EQUIVALENT)
    # The artifact validator already requires exactly these rows to own CSR
    # edges.  Recheck equivalence semantics before the oracle sees them.
    for row in torch.nonzero(decisive, as_tuple=False).reshape(-1).tolist():
        local = source_anchors[int(source_offsets[row]) : int(source_offsets[row + 1])]
        if torch.unique(local).numel() != local.numel():
            raise ValueError("V21 Track diagnostic row repeats an Anchor")
        if torch.unique(equivalence[local]).numel() != 1:
            raise ValueError("V21 EQUIVALENT diagnostic spans identity classes")
    keep_rows = decisive & row_valid
    edge_keep = torch.repeat_interleave(keep_rows, counts)
    anchors = source_anchors[edge_keep]
    kept_counts = counts * keep_rows.long()
    offsets = torch.cat(
        (torch.zeros(1, dtype=torch.long), kept_counts.cumsum(0))
    )
    edge_rows = torch.repeat_interleave(
        torch.arange(xy.shape[0], dtype=torch.long), kept_counts
    )
    projection = project_anchor_geometry(
        anchor_xyz=xyz,
        intrinsic=calibration,
        ground_truth_w2c=pose,
    )
    if anchors.numel() and not bool(projection["valid"][anchors].all()):
        raise ValueError("V21 Track diagnostic contains an unprojectable Anchor")
    residuals = (
        torch.linalg.norm(projection["uv"][anchors] - xy[edge_rows], dim=1)
        if anchors.numel()
        else torch.empty(0, dtype=torch.float32)
    )
    if not bool(torch.isfinite(residuals).all()):
        raise ValueError("V21 Track diagnostic reprojection residual is invalid")
    classes = equivalence[anchors]
    row_has_positive = kept_counts > 0
    return {
        "schema": "lafgs_v21_legal_positive_csr",
        "version": VERSION,
        "row_count": int(xy.shape[0]),
        "anchor_count": int(xyz.shape[0]),
        "positive_offsets": offsets,
        "positive_anchor_rows": anchors,
        "positive_equivalence_class_ids": classes,
        "positive_reprojection_error_px": residuals,
        "row_has_legal_positive": row_has_positive,
        "row_valid_mask": row_valid,
        "source_diagnostic_truth_status": status,
        "source_decisive_row_count": int(decisive.sum()),
        "source_ambiguous_or_no_truth_row_count": int((~decisive).sum()),
        "legal_positive_edge_count": int(anchors.numel()),
        "legal_positive_row_count": int(row_has_positive.sum()),
        "legal_positive_equivalence_count": int(torch.unique(classes).numel()),
        "positive_evidence_mode": TRACK_CONSENSUS_DIAGNOSTIC,
        "positive_source": TRACK_CONSENSUS_DIAGNOSTIC,
        "geometry_supported_candidate": True,
        "candidate_edge_semantics": (
            "mapping_track_consensus_diagnostic_not_controller_truth"
        ),
        "deployable_positive_authorized": False,
        "positive_identity_authority": (
            "diagnostic_only_teacher_did_not_authorize_map_or_metric_action"
        ),
        "track_diagnostic_is_action_authority": False,
        "authorization_requires_exact_poselib": True,
        "topk_candidates_do_not_authorize": True,
        "negative_anchor_rows": torch.empty(0, dtype=torch.long),
        "unlabeled_rows_are_negative": False,
        "parameters": {
            "tier_name": str(track_consensus_record["tier_name"]),
            "requested_action": str(track_consensus_record["requested_action"]),
            "source_action_authorized_ignored": bool(
                track_consensus_record["action_authorized"]
            ),
        },
    }


def _validate_positive_csr(
    legal_positive_csr: dict, *, row_count: int, anchor_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = _tensor(
        legal_positive_csr.get("positive_offsets", ()), dtype=torch.long
    ).reshape(-1)
    anchors = _tensor(
        legal_positive_csr.get("positive_anchor_rows", ()), dtype=torch.long
    ).reshape(-1)
    classes = _tensor(
        legal_positive_csr.get("positive_equivalence_class_ids", ()),
        dtype=torch.long,
    ).reshape(-1)
    residuals = _tensor(
        legal_positive_csr.get("positive_reprojection_error_px", ()),
        dtype=torch.float32,
    ).reshape(-1)
    if (
        offsets.shape != (row_count + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != anchors.numel()
        or bool(((offsets[1:] - offsets[:-1]) < 0).any())
        or not (anchors.shape == classes.shape == residuals.shape)
        or not bool(torch.isfinite(residuals).all())
    ):
        raise ValueError("V21 legal-positive CSR is invalid")
    if anchors.numel() and (
        int(anchors.min()) < 0 or int(anchors.max()) >= anchor_count
    ):
        raise ValueError("V21 legal-positive CSR references an invalid Anchor")
    return offsets, anchors, classes, residuals


def winner_legal_positive_mask(
    *,
    winner_anchor_rows: torch.Tensor,
    legal_positive_csr: dict,
    equivalence_class_ids: torch.Tensor,
) -> torch.Tensor:
    winners = _tensor(winner_anchor_rows, dtype=torch.long).reshape(-1)
    equivalence = _equivalence_ids(equivalence_class_ids, len(equivalence_class_ids))
    if winners.numel() and (
        int(winners.min()) < 0 or int(winners.max()) >= equivalence.numel()
    ):
        raise ValueError("V21 winner registry references an invalid Anchor")
    offsets, _, classes, _ = _validate_positive_csr(
        legal_positive_csr,
        row_count=winners.numel(),
        anchor_count=equivalence.numel(),
    )
    correct = torch.zeros(winners.numel(), dtype=torch.bool)
    for row in range(winners.numel()):
        local = classes[int(offsets[row]) : int(offsets[row + 1])]
        correct[row] = bool((local == equivalence[winners[row]]).any())
    return correct


def _descriptor_edge_scores(
    query_descriptors: torch.Tensor | None,
    anchor_features: torch.Tensor | None,
    *,
    row_count: int,
    anchor_count: int,
    edge_rows: torch.Tensor,
    edge_anchors: torch.Tensor,
) -> torch.Tensor | None:
    if query_descriptors is None and anchor_features is None:
        return None
    if query_descriptors is None or anchor_features is None:
        raise ValueError("V21 descriptor scoring requires both Query and map features")
    query = _tensor(query_descriptors, dtype=torch.float32)
    anchor = _tensor(anchor_features, dtype=torch.float32)
    if (
        query.ndim != 2
        or anchor.ndim != 2
        or query.shape[0] != row_count
        or anchor.shape[0] != anchor_count
        or query.shape[1] != anchor.shape[1]
        or not bool(torch.isfinite(query).all())
        or bool((torch.linalg.norm(query, dim=1) <= 1e-8).any())
    ):
        raise ValueError("V21 Query/map descriptor banks are invalid")
    rows = _tensor(edge_rows, dtype=torch.long).reshape(-1)
    anchors = _tensor(edge_anchors, dtype=torch.long).reshape(-1)
    if rows.shape != anchors.shape or (
        rows.numel()
        and (
            int(rows.min()) < 0
            or int(rows.max()) >= row_count
            or int(anchors.min()) < 0
            or int(anchors.max()) >= anchor_count
        )
    ):
        raise ValueError("V21 descriptor edge registry is invalid")
    if rows.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    selected = anchor[anchors]
    if not bool(torch.isfinite(selected).all()) or bool(
        (torch.linalg.norm(selected, dim=1) <= 1e-8).any()
    ):
        raise ValueError("V21 referenced map descriptors are invalid")
    # Only gather and normalize CSR-referenced anchors.  Normalizing the full
    # ~165k x D map for every Query dominates the actual PoseLib audit.
    return (F.normalize(query[rows], dim=1) * F.normalize(selected, dim=1)).sum(1)


def select_equivalence_unique_corrections(
    *,
    winner_anchor_rows: torch.Tensor,
    legal_positive_csr: dict,
    equivalence_class_ids: torch.Tensor,
    query_descriptors: torch.Tensor | None = None,
    anchor_features: torch.Tensor | None = None,
) -> dict:
    """Choose one conservative correction per row and positive equivalence.

    A current winner in any legal equivalence class is already correct.  Legal
    positives that would duplicate an already-correct equivalence vote are not
    introduced as oracle swaps.
    """

    winners = _tensor(winner_anchor_rows, dtype=torch.long).reshape(-1)
    equivalence = _equivalence_ids(equivalence_class_ids, len(equivalence_class_ids))
    if winners.numel() == 0 or int(winners.min()) < 0 or int(winners.max()) >= equivalence.numel():
        raise ValueError("V21 correction winners are empty or outside the map")
    offsets, anchors, classes, residuals = _validate_positive_csr(
        legal_positive_csr,
        row_count=winners.numel(),
        anchor_count=equivalence.numel(),
    )
    edge_rows = torch.repeat_interleave(
        torch.arange(winners.numel(), dtype=torch.long), offsets[1:] - offsets[:-1]
    )
    descriptor_scores = _descriptor_edge_scores(
        query_descriptors,
        anchor_features,
        row_count=winners.numel(),
        anchor_count=equivalence.numel(),
        edge_rows=edge_rows,
        edge_anchors=anchors,
    )
    correct = winner_legal_positive_mask(
        winner_anchor_rows=winners,
        legal_positive_csr=legal_positive_csr,
        equivalence_class_ids=equivalence,
    )
    occupied_classes = set(equivalence[winners[correct]].tolist())
    edges_by_row: dict[int, list[tuple]] = {}
    omitted_occupied_edge_count = 0
    threshold = float(
        legal_positive_csr.get("parameters", {}).get(
            "reprojection_threshold_px", 2.0
        )
    )
    for row in torch.nonzero(~correct, as_tuple=False).reshape(-1).tolist():
        start, stop = int(offsets[row]), int(offsets[row + 1])
        for index in range(start, stop):
            anchor = int(anchors[index])
            class_id = int(classes[index])
            if class_id in occupied_classes:
                omitted_occupied_edge_count += 1
                continue
            score = (
                float(descriptor_scores[index])
                if descriptor_scores is not None
                else -float(residuals[index])
            )
            priority = max(threshold - float(residuals[index]), 0.0)
            priority += 1e-3 * (score + 1.0) * 0.5
            edges_by_row.setdefault(row, []).append(
                (
                    -score,
                    float(residuals[index]),
                    row,
                    anchor,
                    class_id,
                    max(priority, 0.0),
                )
            )
    raw_edge_counts = {row: len(edges) for row, edges in edges_by_row.items()}
    raw_class_counts = Counter(
        int(edge[4]) for edges in edges_by_row.values() for edge in edges
    )
    for row in edges_by_row:
        # Only the best representative Anchor matters for a row/class edge.
        per_class: dict[int, tuple] = {}
        for edge in sorted(edges_by_row[row]):
            per_class.setdefault(int(edge[4]), edge)
        edges_by_row[row] = list(per_class.values())

    class_match: dict[int, tuple] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for edge in edges_by_row.get(row, []):
            class_id = int(edge[4])
            if class_id in occupied_classes or class_id in seen:
                continue
            seen.add(class_id)
            previous = class_match.get(class_id)
            if previous is None or augment(int(previous[2]), seen):
                class_match[class_id] = edge
                return True
        return False

    # Kuhn augmenting paths give maximum cardinality, unlike a residual-greedy
    # pass which can strand a row whose only class was taken by a flexible row.
    for row in sorted(edges_by_row, key=lambda value: (len(edges_by_row[value]), value)):
        augment(row, set())
    selected = sorted(
        (
            (int(edge[2]), int(edge[3]), int(edge[4]), float(edge[1]), float(edge[5]))
            for edge in class_match.values()
        ),
        key=lambda value: (value[0], value[1]),
    )
    return {
        "candidate_rows": torch.tensor(
            [value[0] for value in selected], dtype=torch.long
        ),
        "candidate_positive_anchor_rows": torch.tensor(
            [value[1] for value in selected], dtype=torch.long
        ),
        "candidate_positive_equivalence_class_ids": torch.tensor(
            [value[2] for value in selected], dtype=torch.long
        ),
        "candidate_reprojection_error_px": torch.tensor(
            [value[3] for value in selected], dtype=torch.float32
        ),
        "candidate_priority": torch.tensor(
            [value[4] for value in selected], dtype=torch.float32
        ),
        "established_positive_query_rows": torch.nonzero(
            correct, as_tuple=False
        ).reshape(-1),
        "established_positive_anchor_rows": winners[correct],
        "assignment_search_exhaustive": bool(
            omitted_occupied_edge_count == 0
            and all(count <= 1 for count in raw_edge_counts.values())
            and all(count <= 1 for count in raw_class_counts.values())
        ),
        "assignment_ambiguity_row_count": int(
            sum(count > 1 for count in raw_edge_counts.values())
        ),
        "omitted_occupied_equivalence_edge_count": int(
            omitted_occupied_edge_count
        ),
        "assignment_semantics": "one_deterministic_maximum_cardinality_assignment",
        "wrong_winner_anchor_rows_are_negative_labels": False,
    }


def select_equivalence_unique_legal_pairs(
    *, legal_positive_csr: dict, equivalence_class_ids: torch.Tensor
) -> dict:
    """Select a deterministic legal-only diagnostic correspondence set."""

    equivalence = _equivalence_ids(equivalence_class_ids, len(equivalence_class_ids))
    row_count = int(legal_positive_csr.get("row_count", -1))
    offsets, anchors, classes, residuals = _validate_positive_csr(
        legal_positive_csr,
        row_count=row_count,
        anchor_count=equivalence.numel(),
    )
    edges_by_row: dict[int, list[tuple[float, int, int, int]]] = {}
    for row in range(row_count):
        for index in range(int(offsets[row]), int(offsets[row + 1])):
            edges_by_row.setdefault(row, []).append(
                (
                    float(residuals[index]),
                    row,
                    int(anchors[index]),
                    int(classes[index]),
                )
            )
    raw_edge_counts = {row: len(edges) for row, edges in edges_by_row.items()}
    raw_class_counts = Counter(
        int(edge[3]) for edges in edges_by_row.values() for edge in edges
    )
    for row in edges_by_row:
        per_class: dict[int, tuple[float, int, int, int]] = {}
        for edge in sorted(edges_by_row[row]):
            per_class.setdefault(int(edge[3]), edge)
        edges_by_row[row] = list(per_class.values())
    class_match: dict[int, tuple[float, int, int, int]] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for edge in edges_by_row.get(row, []):
            class_id = int(edge[3])
            if class_id in seen:
                continue
            seen.add(class_id)
            previous = class_match.get(class_id)
            if previous is None or augment(int(previous[1]), seen):
                class_match[class_id] = edge
                return True
        return False

    for row in sorted(edges_by_row, key=lambda value: (len(edges_by_row[value]), value)):
        augment(row, set())
    selected = sorted(
        (
            (int(edge[1]), int(edge[2]), int(edge[3]), float(edge[0]))
            for edge in class_match.values()
        ),
        key=lambda value: (value[0], value[1]),
    )
    return {
        "query_rows": torch.tensor([value[0] for value in selected], dtype=torch.long),
        "anchor_rows": torch.tensor([value[1] for value in selected], dtype=torch.long),
        "equivalence_class_ids": torch.tensor(
            [value[2] for value in selected], dtype=torch.long
        ),
        "reprojection_error_px": torch.tensor(
            [value[3] for value in selected], dtype=torch.float32
        ),
        "assignment_search_exhaustive": bool(
            all(count <= 1 for count in raw_edge_counts.values())
            and all(count <= 1 for count in raw_class_counts.values())
        ),
        "assignment_ambiguity_row_count": int(
            sum(count > 1 for count in raw_edge_counts.values())
        ),
        "assignment_semantics": "one_deterministic_maximum_cardinality_assignment",
    }


def replay_pose_assignments(
    *,
    keypoints: torch.Tensor,
    anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    reprojection_error_px: float,
    seed: int = 2026,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Replay the unchanged one-shot pose plant and summarize exact R5."""

    xy = _tensor(keypoints, dtype=torch.float32)
    rows = _tensor(anchor_rows, dtype=torch.long).reshape(-1)
    xyz = _tensor(anchor_xyz, dtype=torch.float32)
    calibration = _tensor(intrinsic, dtype=torch.float32)
    truth = _tensor(ground_truth_w2c, dtype=torch.float32)
    if xy.shape != (rows.numel(), 2):
        raise ValueError("V21 pose replay keypoints and assignments do not align")
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= xyz.shape[0]):
        raise ValueError("V21 pose replay assignment is outside the map")
    if not math.isfinite(float(reprojection_error_px)) or float(reprojection_error_px) <= 0:
        raise ValueError("V21 PoseLib reprojection threshold must be positive")
    estimate = solver(
        xy.numpy(),
        xyz[rows].numpy(),
        calibration.numpy(),
        reprojection_error_px=float(reprojection_error_px),
        confidence=0.99999,
        max_iterations=100000,
        min_iterations=1000,
        seed=int(seed),
    )
    pose = np.asarray(estimate.pose_w2c)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("V21 exact pose replay returned an invalid pose")
    rotation_deg, translation_cm = pose_error(pose, truth.numpy())
    inliers = np.asarray(estimate.inliers, dtype=np.int64).reshape(-1)
    if inliers.size and (int(inliers.min()) < 0 or int(inliers.max()) >= rows.numel()):
        raise ValueError("V21 exact pose replay returned an invalid inlier row")
    task = task_error(translation_cm, rotation_deg)
    success = bool(
        translation_cm < R5_TRANSLATION_CM and rotation_deg < R5_ROTATION_DEG
    )
    return {
        "pose_w2c": torch.from_numpy(pose).float(),
        "translation_error_cm": float(translation_cm),
        "rotation_error_deg": float(rotation_deg),
        "task_error": float(task),
        "r5_success": success,
        "inlier_count": int(inliers.size),
        "inlier_query_rows": torch.from_numpy(inliers.copy()).long(),
    }


def _geometry_diagnostic(
    *,
    query_rows: torch.Tensor,
    anchor_rows: torch.Tensor,
    keypoints: torch.Tensor,
    anchor_xyz: torch.Tensor,
) -> dict:
    rows = _tensor(query_rows, dtype=torch.long).reshape(-1)
    anchors = _tensor(anchor_rows, dtype=torch.long).reshape(-1)
    xy = _tensor(keypoints, dtype=torch.float32)
    xyz = _tensor(anchor_xyz, dtype=torch.float32)
    if rows.shape != anchors.shape:
        raise ValueError("V21 legal-only geometry pairs do not align")
    if rows.numel() < 4:
        return {
            "degenerate": True,
            "reason": "fewer_than_four_unique_legal_correspondences",
            "correspondence_count": int(rows.numel()),
            "world_rank": 0,
            "image_rank": 0,
        }
    world_rank = int(torch.linalg.matrix_rank(xyz[anchors] - xyz[anchors].mean(0)))
    image_rank = int(torch.linalg.matrix_rank(xy[rows] - xy[rows].mean(0)))
    degenerate = world_rank < 2 or image_rank < 2
    return {
        "degenerate": bool(degenerate),
        "reason": (
            "collinear_or_collapsed_legal_geometry" if degenerate else "nondegenerate"
        ),
        "correspondence_count": int(rows.numel()),
        "world_rank": world_rank,
        "image_rank": image_rank,
    }


def _topk_diagnostic(
    *,
    candidate_anchor_rows: torch.Tensor | None,
    legal_positive_csr: dict,
    equivalence_class_ids: torch.Tensor,
) -> dict:
    if candidate_anchor_rows is None:
        return {
            "available": False,
            "candidate_count_per_row": 0,
            "legal_positive_recall_row_count": 0,
            "topk_is_authorization_source": False,
        }
    candidates = _tensor(candidate_anchor_rows, dtype=torch.long)
    row_count = int(legal_positive_csr["row_count"])
    equivalence = _equivalence_ids(equivalence_class_ids, len(equivalence_class_ids))
    if candidates.ndim != 2 or candidates.shape[0] != row_count:
        raise ValueError("V21 Top-K rows do not align with legal-positive CSR")
    if candidates.numel() and (
        int(candidates.min()) < 0 or int(candidates.max()) >= equivalence.numel()
    ):
        raise ValueError("V21 Top-K candidate is outside the frozen map")
    offsets, _, classes, _ = _validate_positive_csr(
        legal_positive_csr,
        row_count=row_count,
        anchor_count=equivalence.numel(),
    )
    recall = torch.zeros(row_count, dtype=torch.bool)
    for row in range(row_count):
        positive_classes = classes[int(offsets[row]) : int(offsets[row + 1])]
        recall[row] = bool(
            (equivalence[candidates[row]][:, None] == positive_classes[None]).any()
        ) if positive_classes.numel() else False
    return {
        "available": True,
        "candidate_count_per_row": int(candidates.shape[1]),
        "legal_positive_recall_row_count": int(recall.sum()),
        "legal_positive_recall_fraction": float(recall.float().mean()),
        "topk_is_authorization_source": False,
    }


def _patched_assignments(
    winners: torch.Tensor, rows: torch.Tensor, anchors: torch.Tensor
) -> torch.Tensor:
    output = winners.clone()
    if rows.numel():
        output[rows] = anchors
    return output


def _bundle_fixed_point_prune(
    *,
    rows: torch.Tensor,
    anchors: torch.Tensor,
    priorities: torch.Tensor,
    evaluate: Callable[[torch.Tensor, torch.Tensor], dict],
    is_success: Callable[[dict], bool],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    keep = torch.arange(rows.numel(), dtype=torch.long)
    changed = True
    while changed and keep.numel() > 1:
        changed = False
        order = sorted(
            range(keep.numel()),
            key=lambda local: (
                float(priorities[int(keep[local])]),
                int(rows[int(keep[local])]),
                int(anchors[int(keep[local])]),
            ),
        )
        for local in order:
            trial = torch.cat((keep[:local], keep[local + 1 :]))
            outcome = evaluate(rows[trial], anchors[trial])
            if is_success(outcome):
                keep = trial
                changed = True
                break
    outcome = evaluate(rows[keep], anchors[keep])
    return rows[keep], anchors[keep], priorities[keep], outcome


def analyze_pose_recovery_query(
    *,
    query_index: int,
    keypoints: torch.Tensor,
    winner_anchor_rows: torch.Tensor,
    anchor_xyz: torch.Tensor,
    intrinsic: torch.Tensor,
    ground_truth_w2c: torch.Tensor,
    equivalence_class_ids: torch.Tensor | None = None,
    query_descriptors: torch.Tensor | None = None,
    anchor_features: torch.Tensor | None = None,
    topk_candidate_anchor_rows: torch.Tensor | None = None,
    positive_source: str = GAUSSIAN_GEOMETRY_SOURCE,
    track_consensus_record: dict | None = None,
    row_valid_mask: torch.Tensor | None = None,
    gaussian_depth_at_keypoints: torch.Tensor | None = None,
    gaussian_alpha_at_keypoints: torch.Tensor | None = None,
    gaussian_valid_keypoint_mask: torch.Tensor | None = None,
    gaussian_relative_depth_spread_3x3: torch.Tensor | None = None,
    gaussian_local_valid_fraction_3x3: torch.Tensor | None = None,
    anchor_visibility_mask: torch.Tensor | None = None,
    positive_reprojection_px: float = 2.0,
    minimum_alpha: float = 0.05,
    depth_absolute_m: float = 0.50,
    depth_relative: float = 0.10,
    maximum_relative_depth_spread: float = 0.25,
    minimum_local_valid_fraction: float = 0.50,
    ransac_reprojection_px: float = 11.954343111400277,
    bundle_target_translation_cm: float = R5_TRANSLATION_CM,
    bundle_target_rotation_deg: float = R5_ROTATION_DEG,
    near_boundary_multiplier: float = 1.5,
    maximum_minimal_candidates: int = 24,
    maximum_minimal_set_size: int = 8,
    beam_width: int = 4,
    prefix_initial_set_size: int = 4,
    seed: int = 2026,
    solver: Callable = solve_absolute_pose,
) -> dict:
    """Audit one query using legal geometry and exact discrete PoseLib replay."""

    xy, xyz, calibration, truth = _validate_scene_geometry(
        keypoints=keypoints,
        anchor_xyz=anchor_xyz,
        intrinsic=intrinsic,
        ground_truth_w2c=ground_truth_w2c,
    )
    winners = _tensor(winner_anchor_rows, dtype=torch.long).reshape(-1)
    if winners.shape != (xy.shape[0],) or (
        winners.numel()
        and (int(winners.min()) < 0 or int(winners.max()) >= xyz.shape[0])
    ):
        raise ValueError("V21 baseline winners do not align with the Query/map")
    if (
        not math.isfinite(float(near_boundary_multiplier))
        or float(near_boundary_multiplier) < 1.0
        or int(maximum_minimal_candidates) < 1
        or int(maximum_minimal_set_size) < 1
        or int(beam_width) < 1
        or int(prefix_initial_set_size) < 1
        or not 0.0 < float(bundle_target_translation_cm) <= R5_TRANSLATION_CM
        or not 0.0 < float(bundle_target_rotation_deg) <= R5_ROTATION_DEG
    ):
        raise ValueError("V21 recovery search parameters are invalid")
    equivalence = _equivalence_ids(equivalence_class_ids, xyz.shape[0])
    gaussian_inputs_present = any(
        value is not None
        for value in (
            gaussian_depth_at_keypoints,
            gaussian_alpha_at_keypoints,
            gaussian_valid_keypoint_mask,
            gaussian_relative_depth_spread_3x3,
            gaussian_local_valid_fraction_3x3,
            anchor_visibility_mask,
        )
    )
    if positive_source == GAUSSIAN_GEOMETRY_SOURCE:
        if track_consensus_record is not None:
            raise ValueError("V21 positive sources are mutually exclusive")
        legal = build_legal_positive_csr(
            keypoints=xy,
            anchor_xyz=xyz,
            intrinsic=calibration,
            ground_truth_w2c=truth,
            equivalence_class_ids=equivalence,
            row_valid_mask=row_valid_mask,
            gaussian_depth_at_keypoints=gaussian_depth_at_keypoints,
            gaussian_alpha_at_keypoints=gaussian_alpha_at_keypoints,
            gaussian_valid_keypoint_mask=gaussian_valid_keypoint_mask,
            gaussian_relative_depth_spread_3x3=(
                gaussian_relative_depth_spread_3x3
            ),
            gaussian_local_valid_fraction_3x3=(
                gaussian_local_valid_fraction_3x3
            ),
            anchor_visibility_mask=anchor_visibility_mask,
            reprojection_threshold_px=positive_reprojection_px,
            minimum_alpha=minimum_alpha,
            depth_absolute_m=depth_absolute_m,
            depth_relative=depth_relative,
            maximum_relative_depth_spread=maximum_relative_depth_spread,
            minimum_local_valid_fraction=minimum_local_valid_fraction,
        )
    elif positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
        if track_consensus_record is None or gaussian_inputs_present:
            raise ValueError(
                "V21 Track diagnostic requires exactly one positive source"
            )
        if int(track_consensus_record.get("query_index", -1)) != int(query_index):
            raise ValueError("V21 Track diagnostic query identity differs")
        legal = build_track_consensus_diagnostic_csr(
            keypoints=xy,
            anchor_xyz=xyz,
            intrinsic=calibration,
            ground_truth_w2c=truth,
            track_consensus_record=track_consensus_record,
            equivalence_class_ids=equivalence,
            row_valid_mask=row_valid_mask,
        )
    else:
        raise ValueError("V21 positive source is unsupported")
    exact_replay_count = 0
    search_replay_count = 0
    replay_cache: dict[tuple[tuple[int, ...], tuple[int, ...]], dict] = {}

    def evaluate_bundle(rows: torch.Tensor, anchors: torch.Tensor) -> dict:
        nonlocal exact_replay_count
        rows = _tensor(rows, dtype=torch.long).reshape(-1)
        anchors = _tensor(anchors, dtype=torch.long).reshape(-1)
        key = (tuple(rows.tolist()), tuple(anchors.tolist()))
        if key not in replay_cache:
            replay_cache[key] = replay_pose_assignments(
                keypoints=xy,
                anchor_rows=_patched_assignments(winners, rows, anchors),
                anchor_xyz=xyz,
                intrinsic=calibration,
                ground_truth_w2c=truth,
                reprojection_error_px=ransac_reprojection_px,
                seed=seed,
                solver=solver,
            )
            exact_replay_count += 1
        return replay_cache[key]

    def counted_search_solver(*args, **kwargs):
        nonlocal search_replay_count
        search_replay_count += 1
        return solver(*args, **kwargs)

    baseline = evaluate_bundle(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long))
    baseline["bundle_target_success"] = _meets_pose_target(
        baseline,
        translation_cm=bundle_target_translation_cm,
        rotation_deg=bundle_target_rotation_deg,
    )
    baseline["near_boundary_failure"] = bool(
        not baseline["r5_success"]
        and max(
            baseline["translation_error_cm"] / R5_TRANSLATION_CM,
            baseline["rotation_error_deg"] / R5_ROTATION_DEG,
        )
        <= float(near_boundary_multiplier)
    )
    correct = winner_legal_positive_mask(
        winner_anchor_rows=winners,
        legal_positive_csr=legal,
        equivalence_class_ids=equivalence,
    )
    geometry_candidate_rows = torch.nonzero(correct, as_tuple=False).reshape(-1)
    baseline_inliers = baseline["inlier_query_rows"].clone()
    geometry_candidate_inliers = baseline_inliers[correct[baseline_inliers]]
    no_rows = torch.empty(0, dtype=torch.long)
    protection = {
        "query_rows": no_rows,
        "anchor_rows": no_rows,
        "pose_inlier_query_rows": no_rows,
        "pose_inlier_anchor_rows": no_rows,
        "geometry_candidate_query_rows": geometry_candidate_rows,
        "geometry_candidate_anchor_rows": winners[geometry_candidate_rows],
        "positive_source_candidate_query_rows": geometry_candidate_rows,
        "positive_source_candidate_anchor_rows": winners[geometry_candidate_rows],
        "pose_inlier_geometry_candidate_query_rows": geometry_candidate_inliers,
        "pose_inlier_geometry_candidate_anchor_rows": winners[
            geometry_candidate_inliers
        ],
        "baseline_pose_inlier_query_rows": baseline_inliers,
        "baseline_pose_inlier_anchor_rows": winners[baseline_inliers],
        "only_identity_certified_positive_inliers_are_protected": True,
        "identity_certified_positive_count": 0,
        "positive_source": positive_source,
        "positive_evidence_mode": legal["positive_evidence_mode"],
        "deployable_positive_authorized": bool(
            legal["deployable_positive_authorized"]
        ),
    }
    topk = _topk_diagnostic(
        candidate_anchor_rows=topk_candidate_anchor_rows,
        legal_positive_csr=legal,
        equivalence_class_ids=equivalence,
    )
    common = {
        "schema": SCHEMA,
        "version": VERSION,
        "query_index": int(query_index),
        "positive_source": positive_source,
        "uses_ground_truth_pose": True,
        "exact_poselib_authorizes_pose_recovery_claim_only": True,
        "controller_authorization_requires_identity_consensus": True,
        "route_is_pose_recovery_upper_bound": True,
        "bundle_target": {
            "translation_cm": float(bundle_target_translation_cm),
            "rotation_deg": float(bundle_target_rotation_deg),
            "standard_r5_reporting_translation_cm": R5_TRANSLATION_CM,
            "standard_r5_reporting_rotation_deg": R5_ROTATION_DEG,
            "changes_standard_r5_definition": False,
        },
        "topk": topk,
        "legal_positive_csr": legal,
        "negative_anchor_rows": torch.empty(0, dtype=torch.long),
        "unlabeled_rows_are_negative": False,
        "baseline": baseline,
        "protection": protection,
    }
    if baseline["r5_success"]:
        return {
            **common,
            "route": PROTECTION_ONLY,
            "controller_authorized": False,
            "authorization_reason": "baseline_success_protection_only",
            "one_assignment_lower_bound": None,
            "legal_only_diagnostic": None,
            "recovery_bundle": None,
            "exact_replay_count": exact_replay_count + search_replay_count,
        }

    corrections = select_equivalence_unique_corrections(
        winner_anchor_rows=winners,
        legal_positive_csr=legal,
        equivalence_class_ids=equivalence,
        query_descriptors=query_descriptors,
        anchor_features=anchor_features,
    )
    candidate_rows = corrections["candidate_rows"]
    candidate_anchors = corrections["candidate_positive_anchor_rows"]
    priorities = corrections["candidate_priority"]
    one_assignment = evaluate_bundle(candidate_rows, candidate_anchors)
    one_assignment["bundle_target_success"] = _meets_pose_target(
        one_assignment,
        translation_cm=bundle_target_translation_cm,
        rotation_deg=bundle_target_rotation_deg,
    )
    legal_pairs = select_equivalence_unique_legal_pairs(
        legal_positive_csr=legal, equivalence_class_ids=equivalence
    )
    geometry = _geometry_diagnostic(
        query_rows=legal_pairs["query_rows"],
        anchor_rows=legal_pairs["anchor_rows"],
        keypoints=xy,
        anchor_xyz=xyz,
    )
    if legal_pairs["query_rows"].numel() >= 4:
        legal_only = replay_pose_assignments(
            keypoints=xy[legal_pairs["query_rows"]],
            anchor_rows=legal_pairs["anchor_rows"],
            anchor_xyz=xyz,
            intrinsic=calibration,
            ground_truth_w2c=truth,
            reprojection_error_px=ransac_reprojection_px,
            seed=seed,
            solver=solver,
        )
        exact_replay_count += 1
    else:
        legal_only = None

    if one_assignment["bundle_target_success"] and candidate_rows.numel() > 0:
        small = minimal_pose_correction_set(
            keypoints=xy,
            xyz=xyz,
            winners=winners,
            candidate_rows=candidate_rows,
            candidate_positive_anchors=candidate_anchors,
            candidate_priority=priorities,
            intrinsics=calibration,
            ground_truth_pose_w2c=truth,
            reprojection_error_px=ransac_reprojection_px,
            maximum_candidates=min(
                int(maximum_minimal_candidates), int(candidate_rows.numel())
            ),
            maximum_set_size=maximum_minimal_set_size,
            beam_width=beam_width,
            seed=seed,
            solver=counted_search_solver,
        )
        selected_rows = None
        selected_anchors = None
        source = ""
        if small["correction_found"] and small["selected_rows"].numel() > 0:
            proposed_rows = _tensor(small["selected_rows"], dtype=torch.long)
            proposed_anchors = _tensor(
                small["selected_positive_anchors"], dtype=torch.long
            )
            if _meets_pose_target(
                evaluate_bundle(proposed_rows, proposed_anchors),
                translation_cm=bundle_target_translation_cm,
                rotation_deg=bundle_target_rotation_deg,
            ):
                selected_rows = proposed_rows
                selected_anchors = proposed_anchors
                source = "bounded_minimal_pose_correction_set_officially_replayed"
        if selected_rows is None:
            prefix = pose_priority_prefix_correction_set(
                keypoints=xy,
                xyz=xyz,
                winners=winners,
                candidate_rows=candidate_rows,
                candidate_positive_anchors=candidate_anchors,
                candidate_priority=priorities,
                intrinsics=calibration,
                ground_truth_pose_w2c=truth,
                reprojection_error_px=ransac_reprojection_px,
                maximum_candidates=int(candidate_rows.numel()),
                initial_set_size=prefix_initial_set_size,
                seed=seed,
                solver=counted_search_solver,
            )
            if prefix["correction_found"] and prefix["selected_rows"].numel() > 0:
                proposed_rows = _tensor(prefix["selected_rows"], dtype=torch.long)
                proposed_anchors = _tensor(
                    prefix["selected_positive_anchors"], dtype=torch.long
                )
                if _meets_pose_target(
                    evaluate_bundle(proposed_rows, proposed_anchors),
                    translation_cm=bundle_target_translation_cm,
                    rotation_deg=bundle_target_rotation_deg,
                ):
                    selected_rows = proposed_rows
                    selected_anchors = proposed_anchors
                    source = "pose_priority_prefix_officially_replayed"
        if selected_rows is None:
            order = torch.tensor(
                sorted(
                    range(candidate_rows.numel()),
                    key=lambda index: (
                        -float(priorities[index]),
                        int(candidate_rows[index]),
                        int(candidate_anchors[index]),
                    ),
                ),
                dtype=torch.long,
            )
            count = min(int(prefix_initial_set_size), int(order.numel()))
            while count > 0:
                proposed = order[:count]
                proposed_rows = candidate_rows[proposed]
                proposed_anchors = candidate_anchors[proposed]
                if _meets_pose_target(
                    evaluate_bundle(proposed_rows, proposed_anchors),
                    translation_cm=bundle_target_translation_cm,
                    rotation_deg=bundle_target_rotation_deg,
                ):
                    selected_rows = proposed_rows
                    selected_anchors = proposed_anchors
                    source = "official_bundle_target_priority_prefix"
                    break
                if count == int(order.numel()):
                    break
                count = min(count * 2, int(order.numel()))
        if selected_rows is None:
            # The exact one-assignment replay above is authoritative.  The V6
            # search helpers use a legacy Torch angle calculation, so a
            # boundary discrepancy must fall back to the officially replayed
            # full bundle instead of creating a false success or aborting.
            selected_rows = candidate_rows
            selected_anchors = candidate_anchors
            source = "official_exact_one_assignment_fallback"
        candidate_lookup = {
            (int(row), int(anchor)): index
            for index, (row, anchor) in enumerate(
                zip(candidate_rows.tolist(), candidate_anchors.tolist())
            )
        }
        selected_priority = torch.tensor(
            [
                priorities[candidate_lookup[(int(row), int(anchor))]]
                for row, anchor in zip(
                    selected_rows.tolist(), selected_anchors.tolist()
                )
            ],
            dtype=torch.float32,
        )
        bundle_rows, bundle_anchors, bundle_priority, bundle_pose = (
            _bundle_fixed_point_prune(
                rows=selected_rows,
                anchors=selected_anchors,
                priorities=selected_priority,
                evaluate=evaluate_bundle,
                is_success=lambda outcome: _meets_pose_target(
                    outcome,
                    translation_cm=bundle_target_translation_cm,
                    rotation_deg=bundle_target_rotation_deg,
                ),
            )
        )
        bundle_pose["bundle_target_success"] = _meets_pose_target(
            bundle_pose,
            translation_cm=bundle_target_translation_cm,
            rotation_deg=bundle_target_rotation_deg,
        )
        if not bundle_pose["bundle_target_success"]:
            raise RuntimeError("V21 recovery bundle failed its official pose target")
        necessity = []
        for local in range(bundle_rows.numel()):
            keep = torch.ones(bundle_rows.numel(), dtype=torch.bool)
            keep[local] = False
            without = evaluate_bundle(bundle_rows[keep], bundle_anchors[keep])
            without_target_success = _meets_pose_target(
                without,
                translation_cm=bundle_target_translation_cm,
                rotation_deg=bundle_target_rotation_deg,
            )
            necessity.append(
                {
                    "query_row": int(bundle_rows[local]),
                    "anchor_row": int(bundle_anchors[local]),
                    "equivalence_class_id": int(
                        equivalence[bundle_anchors[local]]
                    ),
                    "priority": float(bundle_priority[local]),
                    "removal_loses_recovery": bool(
                        bundle_pose["bundle_target_success"]
                        and not without_target_success
                    ),
                    "removal_loses_bundle_target": bool(
                        bundle_pose["bundle_target_success"]
                        and not without_target_success
                    ),
                    "without_row_bundle_target_success": bool(
                        without_target_success
                    ),
                    "without_row_r5_success": bool(without["r5_success"]),
                    "without_row_task_error": float(without["task_error"]),
                    "exact_task_loss_if_removed": float(
                        without["task_error"] - bundle_pose["task_error"]
                    ),
                    "exact_r5_loss_if_removed": int(bundle_pose["r5_success"])
                    - int(without["r5_success"]),
                    "exact_bundle_target_loss_if_removed": int(
                        bundle_pose["bundle_target_success"]
                    )
                    - int(without_target_success),
                }
            )
        recovery_bundle = {
            "search_source": source,
            "query_rows": bundle_rows,
            "anchor_rows": bundle_anchors,
            "equivalence_class_ids": equivalence[bundle_anchors],
            "priorities": bundle_priority,
            "row_necessity": necessity,
            "pose": bundle_pose,
            "bundle_target_translation_cm": float(
                bundle_target_translation_cm
            ),
            "bundle_target_rotation_deg": float(bundle_target_rotation_deg),
            "bundle_target_success": bool(bundle_pose["bundle_target_success"]),
            "standard_r5_success": bool(bundle_pose["r5_success"]),
            "exact_delta_task": float(
                baseline["task_error"] - bundle_pose["task_error"]
            ),
            "exact_delta_r5": int(bundle_pose["r5_success"])
            - int(baseline["r5_success"]),
            "inclusion_minimal": bool(
                all(value["removal_loses_recovery"] for value in necessity)
            ),
        }
        route = DESCRIPTOR_CONTROLLABLE
        authorized = bool(legal["deployable_positive_authorized"])
        if authorized:
            authorization_reason = "identity_consensus_and_exact_poselib_recovery"
        elif positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            authorization_reason = (
                "track_consensus_diagnostic_no_map_or_metric_action_authority"
            )
        else:
            authorization_reason = (
                f"{legal['positive_evidence_mode']}_no_identity_authority"
            )
    else:
        recovery_bundle = None
        if one_assignment["r5_success"] and candidate_rows.numel() > 0:
            route = DESCRIPTOR_CONTROLLABLE
        elif positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            route = TRACK_DIAGNOSTIC_COVERAGE_LIMITED
        elif legal_pairs["query_rows"].numel() < 4:
            route = COVERAGE_LIMITED
        elif not (
            corrections["assignment_search_exhaustive"]
            and legal_pairs["assignment_search_exhaustive"]
        ):
            route = ASSIGNMENT_INDETERMINATE
        elif geometry["degenerate"]:
            route = GEOMETRY_LIMITED
        elif legal_only is not None and legal_only["r5_success"]:
            # A clean legal subset works, but the unchanged full plant still
            # lacks enough labeled rows to survive its remaining outliers.
            route = COVERAGE_LIMITED
        else:
            route = SOLVER_LIMITED
        authorized = False
        if one_assignment["r5_success"] and candidate_rows.numel() > 0:
            authorization_reason = (
                "standard_r5_recovered_but_stricter_bundle_target_unavailable"
            )
        elif positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            authorization_reason = (
                "available_track_diagnostic_coverage_did_not_recover_"
                "no_global_solver_exclusion"
            )
        else:
            authorization_reason = f"{route}_has_no_exact_recovery_bundle"

    return {
        **common,
        "route": route,
        "controller_authorized": authorized,
        "authorization_reason": authorization_reason,
        "correction_candidates": corrections,
        "one_assignment_lower_bound": {
            **one_assignment,
            "assignment_search_exhaustive": bool(
                corrections["assignment_search_exhaustive"]
            ),
            "semantics": "one_deterministic_assignment_not_all_legal_assignments",
            "positive_source": positive_source,
        },
        "legal_only_diagnostic": {
            "pairs": legal_pairs,
            "geometry": geometry,
            "pose": legal_only,
        },
        "recovery_bundle": recovery_bundle,
        "exact_replay_count": exact_replay_count + search_replay_count,
    }


def summarize_pose_recovery(
    records: Sequence[dict], *, positive_source: str | None = None
) -> dict:
    """Summarize a shard without changing any per-query authorization."""

    allowed_sources = {GAUSSIAN_GEOMETRY_SOURCE, TRACK_CONSENSUS_DIAGNOSTIC}
    declared_sources = {
        str(record["positive_source"])
        for record in records
        if "positive_source" in record
    }
    missing_source_count = sum("positive_source" not in record for record in records)
    if positive_source is not None and positive_source not in allowed_sources:
        raise ValueError("V21 summary positive source is unsupported")
    if not declared_sources.issubset(allowed_sources):
        raise ValueError("V21 record positive source is unsupported")
    if missing_source_count:
        if declared_sources or positive_source == TRACK_CONSENSUS_DIAGNOSTIC:
            raise ValueError("V21 summary mixes or omits Track positive-source identity")
        if any(
            record.get("legal_positive_csr", {}).get("positive_evidence_mode")
            not in {GAUSSIAN_GEOMETRY_SUPPORTED, REPROJECTION_UPPER_BOUND}
            for record in records
        ):
            raise ValueError(
                "V21 non-geometry evidence cannot use legacy source inference"
            )
        legacy_geometry_records = True
    else:
        legacy_geometry_records = False
        if len(declared_sources) > 1 or (
            positive_source is not None and declared_sources not in ({positive_source}, set())
        ):
            raise ValueError("V21 summary positive-source registry differs")
    route_counts = Counter(str(record["route"]) for record in records)
    failures = [record for record in records if not record["baseline"]["r5_success"]]
    controllable = [
        record for record in failures if record["route"] == DESCRIPTOR_CONTROLLABLE
    ]
    authorized = [record for record in controllable if record["controller_authorized"]]
    recovered = []
    for record in controllable:
        bundle = record.get("recovery_bundle")
        if isinstance(bundle, dict) and bundle.get("exact_delta_r5") == 1:
            recovered.append(record)
    output = {
        "query_count": len(records),
        "baseline_r5_success_count": int(
            sum(bool(record["baseline"]["r5_success"]) for record in records)
        ),
        "baseline_failure_count": len(failures),
        "route_counts": dict(sorted(route_counts.items())),
        "descriptor_controllable_count": len(controllable),
        "exact_recovered_failure_count": len(recovered),
        "controller_authorized_query_count": len(authorized),
        "reprojection_upper_bound_query_count": int(
            sum(
                record["legal_positive_csr"]["positive_evidence_mode"]
                == REPROJECTION_UPPER_BOUND
                for record in records
            )
        ),
        "negative_anchor_label_count": 0,
    }
    if legacy_geometry_records or (not records and positive_source is None):
        # V1 geometry shards predate the explicit positive-source field.  This
        # fallback is accepted only when no record carries Track evidence; it
        # preserves their byte-level summary contract for strict aggregation.
        return {**output, "all_action_authority_is_exact_poselib": True}
    track_count = int(
        sum(
            record["positive_source"] == TRACK_CONSENSUS_DIAGNOSTIC
            for record in records
        )
    )
    return {
        **output,
        "track_consensus_diagnostic_query_count": track_count,
        "all_pose_recovery_claims_use_exact_poselib": True,
        "exact_poselib_is_controller_action_authority": False,
    }
