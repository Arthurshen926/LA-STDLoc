"""Pure metrics for a per-query Anchor localization failure chain."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


@dataclass(frozen=True)
class ProjectedAnchors:
    uv: np.ndarray
    depth: np.ndarray
    in_frame: np.ndarray
    visible: np.ndarray
    alpha_supported: np.ndarray
    depth_supported: np.ndarray


@dataclass(frozen=True)
class BipartiteCorrespondences:
    query_rows: np.ndarray
    anchor_rows: np.ndarray
    distances_px: np.ndarray
    edge_count: int

    @property
    def rank(self) -> int:
        return int(self.query_rows.size)


def project_gt_visible_anchors(
    anchor_xyz: np.ndarray,
    intrinsic: np.ndarray,
    pose_w2c: np.ndarray,
    *,
    image_size: tuple[int, int],
    rendered_alpha: np.ndarray | None = None,
    rendered_depth: np.ndarray | None = None,
    alpha_minimum: float = 0.05,
    depth_abs_tolerance_m: float = 0.05,
    depth_relative_tolerance: float = 0.02,
    depth_policy: str = "hard",
) -> ProjectedAnchors:
    """Project map geometry and apply the frozen nearest-pixel visibility test."""

    xyz = np.asarray(anchor_xyz, dtype=np.float64)
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    pose = np.asarray(pose_w2c, dtype=np.float64)
    width, height = (int(value) for value in image_size)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Anchor xyz must be a finite [N,3] matrix")
    if intrinsic.shape != (3, 3) or pose.shape != (4, 4):
        raise ValueError("intrinsic/pose shapes must be [3,3] and [4,4]")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    depth = camera[:, 2]
    homogeneous = camera @ intrinsic.T
    uv = homogeneous[:, :2] / np.maximum(depth[:, None], 1e-12)
    in_frame = (
        np.isfinite(uv).all(axis=1)
        & np.isfinite(depth)
        & (depth > 1e-5)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )
    alpha_supported = np.ones(xyz.shape[0], dtype=bool)
    depth_supported = np.ones(xyz.shape[0], dtype=bool)
    if (rendered_alpha is None) != (rendered_depth is None):
        raise ValueError("rendered alpha and depth must be supplied together")
    if depth_policy not in {"hard", "audit_only"}:
        raise ValueError("depth policy must be 'hard' or 'audit_only'")
    if rendered_alpha is not None:
        alpha = np.asarray(rendered_alpha, dtype=np.float64).squeeze()
        reference_depth = np.asarray(rendered_depth, dtype=np.float64).squeeze()
        if alpha.shape != (height, width) or reference_depth.shape != (height, width):
            raise ValueError("rendered alpha/depth dimensions differ from the query")
        x = np.clip(np.rint(uv[:, 0]), 0, width - 1).astype(np.int64)
        y = np.clip(np.rint(uv[:, 1]), 0, height - 1).astype(np.int64)
        sampled_alpha = alpha[y, x]
        sampled_depth = reference_depth[y, x]
        alpha_supported = np.isfinite(sampled_alpha) & (
            sampled_alpha >= float(alpha_minimum)
        )
        tolerance = float(depth_abs_tolerance_m) + float(
            depth_relative_tolerance
        ) * np.abs(sampled_depth)
        depth_supported = (
            np.isfinite(sampled_depth)
            & (sampled_depth > 1e-5)
            & (np.abs(depth - sampled_depth) <= tolerance)
        )
    visible = in_frame & alpha_supported
    if depth_policy == "hard":
        visible &= depth_supported
    return ProjectedAnchors(
        uv=uv,
        depth=depth,
        in_frame=in_frame,
        visible=visible,
        alpha_supported=alpha_supported,
        depth_supported=depth_supported,
    )


def nearby_anchor_edges(
    keypoints: np.ndarray,
    projected_uv: np.ndarray,
    eligible_anchors: np.ndarray,
    *,
    radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic query-row/Anchor edges inside a pixel radius."""

    keypoints = np.asarray(keypoints, dtype=np.float64)
    projected = np.asarray(projected_uv, dtype=np.float64)
    eligible = np.asarray(eligible_anchors, dtype=bool)
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("query keypoints must have shape [Q,2]")
    if projected.ndim != 2 or projected.shape[1] != 2:
        raise ValueError("projected Anchors must have shape [A,2]")
    if eligible.shape != (projected.shape[0],):
        raise ValueError("Anchor eligibility does not align with projections")
    if not math.isfinite(float(radius_px)) or float(radius_px) <= 0:
        raise ValueError("matching radius must be positive and finite")
    anchor_rows = np.flatnonzero(eligible)
    if anchor_rows.size == 0 or keypoints.shape[0] == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), np.empty(0, dtype=np.float64)
    tree = cKDTree(projected[anchor_rows])
    neighborhoods = tree.query_ball_point(keypoints, r=float(radius_px))
    query_result = []
    anchor_result = []
    distance_result = []
    for query_row, local_rows in enumerate(neighborhoods):
        if not local_rows:
            continue
        global_rows = anchor_rows[np.asarray(local_rows, dtype=np.int64)]
        distances = np.linalg.norm(projected[global_rows] - keypoints[query_row], axis=1)
        order = np.lexsort((global_rows, distances))
        query_result.extend([query_row] * len(order))
        anchor_result.extend(global_rows[order].tolist())
        distance_result.extend(distances[order].tolist())
    return (
        np.asarray(query_result, dtype=np.int64),
        np.asarray(anchor_result, dtype=np.int64),
        np.asarray(distance_result, dtype=np.float64),
    )


def maximum_cardinality_minimum_distance_matching(
    query_edge_rows: np.ndarray,
    anchor_edge_rows: np.ndarray,
    distances_px: np.ndarray,
    *,
    query_count: int,
) -> BipartiteCorrespondences:
    """Maximum-cardinality matching, then minimum total reprojection distance."""

    query_rows = np.asarray(query_edge_rows)
    anchor_rows = np.asarray(anchor_edge_rows)
    distances = np.asarray(distances_px, dtype=np.float64)
    if query_rows.dtype.kind not in "iu" or anchor_rows.dtype.kind not in "iu":
        raise ValueError("bipartite edge rows must be integer vectors")
    if query_rows.shape != anchor_rows.shape or distances.shape != query_rows.shape:
        raise ValueError("bipartite edges and distances do not align")
    if int(query_count) < 0:
        raise ValueError("query count cannot be negative")
    if query_rows.size and (
        np.any(query_rows < 0) or np.any(query_rows >= int(query_count))
    ):
        raise ValueError("query edge row is outside the registry")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("edge distances must be finite and non-negative")
    if int(query_count) == 0 or query_rows.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return BipartiteCorrespondences(empty, empty.copy(), np.empty(0), int(query_rows.size))
    unique_anchors, compact = np.unique(anchor_rows, return_inverse=True)
    dummy_penalty = max(float(distances.max(initial=0.0)) + 1.0, 1.0) * (
        int(query_count) + 1
    )
    all_rows = np.concatenate((query_rows.astype(np.int64), np.arange(query_count)))
    all_columns = np.concatenate(
        (compact.astype(np.int64), unique_anchors.size + np.arange(query_count))
    )
    # Sparse matching removes exact zero weights, so add a tiny deterministic
    # positive term without changing the distance ordering.
    real_weights = distances + 1e-9 * (compact + 1)
    all_weights = np.concatenate((real_weights, np.full(query_count, dummy_penalty)))
    graph = coo_matrix(
        (all_weights, (all_rows, all_columns)),
        shape=(query_count, unique_anchors.size + query_count),
    ).tocsr()
    matched_rows, matched_columns = min_weight_full_bipartite_matching(graph)
    real = matched_columns < unique_anchors.size
    selected_queries = matched_rows[real].astype(np.int64)
    selected_anchors = unique_anchors[matched_columns[real]].astype(np.int64)
    distance_lookup = {
        (int(q), int(a)): float(d)
        for q, a, d in zip(query_rows, anchor_rows, distances)
    }
    selected_distances = np.asarray(
        [distance_lookup[(int(q), int(a))] for q, a in zip(selected_queries, selected_anchors)]
    )
    order = np.argsort(selected_queries, kind="stable")
    return BipartiteCorrespondences(
        selected_queries[order], selected_anchors[order], selected_distances[order],
        int(query_rows.size),
    )


def grid_coverage(points_2d: np.ndarray, image_size: tuple[int, int], *, bins: int = 4) -> int:
    points = np.asarray(points_2d, dtype=np.float64)
    width, height = (int(value) for value in image_size)
    if points.size == 0:
        return 0
    x = np.clip(np.floor(points[:, 0] * bins / width), 0, bins - 1).astype(int)
    y = np.clip(np.floor(points[:, 1] * bins / height), 0, bins - 1).astype(int)
    return int(np.unique(y * int(bins) + x).size)


def geometry_diagnostics(
    points_2d: np.ndarray,
    points_3d: np.ndarray,
    intrinsic: np.ndarray,
    pose_w2c: np.ndarray,
    *,
    image_size: tuple[int, int],
    seed: int,
    maximum_minimal_sets: int = 64,
) -> dict:
    """Report 2D/3D distribution, D-opt information, and PnP-set viability."""

    points_2d = np.asarray(points_2d, dtype=np.float64)
    points_3d = np.asarray(points_3d, dtype=np.float64)
    count = int(points_3d.shape[0])
    if points_2d.shape != (count, 2):
        raise ValueError("2D/3D correspondence rows do not align")
    centered = points_3d - points_3d.mean(axis=0) if count else points_3d
    singular = np.linalg.svd(centered, compute_uv=False) if count else np.zeros(3)
    normalized_singular = singular / max(float(singular[0]), 1e-12)
    information_logdet = None
    information_rank = 0
    if count:
        points_tensor = torch.as_tensor(points_3d).double()
        K_tensor = torch.as_tensor(intrinsic).double()
        pose_tensor = torch.as_tensor(pose_w2c).double()
        jacobian = task_scaled_pose_jacobian(
            pose_jacobian_analytic(points_tensor, K_tensor, pose_tensor),
            translation_scale=0.05,
            rotation_scale=math.radians(5.0),
        )
        information = fisher_contributions(jacobian).sum(dim=0)
        eigenvalues = torch.linalg.eigvalsh(information).clamp_min(0)
        information_rank = int((eigenvalues > 1e-8).sum())
        information_logdet = float(torch.logdet(information + 1e-6 * torch.eye(6)))
    tested = nondegenerate = 0
    if count >= 4:
        generator = np.random.default_rng(int(seed))
        seen = set()
        attempts = max(int(maximum_minimal_sets) * 20, 100)
        for _ in range(attempts):
            sample = tuple(sorted(generator.choice(count, size=4, replace=False).tolist()))
            if sample in seen:
                continue
            seen.add(sample)
            xyz = points_3d[np.asarray(sample)]
            uv = points_2d[np.asarray(sample)]
            xyz_s = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
            uv_s = np.linalg.svd(uv - uv.mean(axis=0), compute_uv=False)
            tested += 1
            if (
                xyz_s[1] / max(float(xyz_s[0]), 1e-12) > 1e-3
                and uv_s[1] / max(float(uv_s[0]), 1e-12) > 1e-3
            ):
                nondegenerate += 1
            if tested == int(maximum_minimal_sets):
                break
    return {
        "correspondence_count": count,
        "grid_coverage_4x4": grid_coverage(points_2d, image_size),
        "xyz_extent_m": (
            (points_3d.max(axis=0) - points_3d.min(axis=0)).tolist()
            if count else [0.0, 0.0, 0.0]
        ),
        "xyz_axis_ratios": normalized_singular.tolist(),
        "information_rank": information_rank,
        "d_opt_logdet_regularized": information_logdet,
        "minimal_sets_tested": tested,
        "nondegenerate_minimal_set_count": nondegenerate,
        "nondegenerate_minimal_set_rate": (
            float(nondegenerate / tested) if tested else 0.0
        ),
        "pnp_geometry_solvable": bool(
            count >= 4 and information_rank >= 6 and nondegenerate > 0
        ),
    }


def descriptor_recall_diagnostics(
    scores: np.ndarray,
    top_anchor_rows: np.ndarray,
    positive_query_rows: np.ndarray,
    positive_anchor_rows: np.ndarray,
    anchor_type: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
) -> dict:
    """Compute geometric-positive recall and margins, split by Anchor type."""

    scores = np.asarray(scores, dtype=np.float64)
    top = np.asarray(top_anchor_rows)
    q_edges = np.asarray(positive_query_rows)
    a_edges = np.asarray(positive_anchor_rows)
    types = np.asarray(anchor_type)
    if scores.ndim != 2 or top.ndim != 2 or scores.shape[0] != top.shape[0]:
        raise ValueError("score matrix and Top-K rows do not align")
    if scores.shape[1] != types.size:
        raise ValueError("score columns and Anchor types do not align")
    positives_by_query = [[] for _ in range(scores.shape[0])]
    for query_row, anchor_row in zip(q_edges.tolist(), a_edges.tolist()):
        positives_by_query[int(query_row)].append(int(anchor_row))

    def summarize(kind: str, type_value: int | None) -> dict:
        eligible_rows = []
        recalls = {k: [] for k in ks}
        margins = []
        for query_row, positive_rows in enumerate(positives_by_query):
            if type_value is not None:
                positive_rows = [row for row in positive_rows if int(types[row]) == type_value]
            if not positive_rows:
                continue
            eligible_rows.append(query_row)
            positive_set = set(positive_rows)
            for k in ks:
                recalls[k].append(any(int(row) in positive_set for row in top[query_row, :k]))
            best_positive = float(scores[query_row, positive_rows].max())
            wrong = scores[query_row].copy()
            wrong[np.asarray(positives_by_query[query_row], dtype=np.int64)] = -np.inf
            margins.append(best_positive - float(wrong.max()))
        return {
            "kind": kind,
            "eligible_query_row_count": len(eligible_rows),
            "recall_percent": {
                f"r_at_{k}": (100.0 * float(np.mean(recalls[k])) if recalls[k] else 0.0)
                for k in ks
            },
            "best_positive_minus_best_wrong_margin": {
                "mean": float(np.mean(margins)) if margins else None,
                "median": float(np.median(margins)) if margins else None,
                "positive_fraction": float(np.mean(np.asarray(margins) > 0)) if margins else None,
            },
        }

    return {
        "all": summarize("all", None),
        "track": summarize("track", 1),
        "surface_completion": summarize("surface_completion", 0),
    }


def classify_failure_chain(
    *,
    visible_geometry_solvable: bool,
    detector_matching_rank: int,
    oracle_matching_rank: int,
    top32_positive_matching_rank: int,
    oracle_pose_correct: bool,
    deployed_pose_correct: bool,
) -> str:
    if not visible_geometry_solvable:
        return "L1_MAP_COVERAGE_OR_GEOMETRY"
    if detector_matching_rank < 4 or oracle_matching_rank < 4:
        return "L2_DETECTOR_ACCESS"
    if top32_positive_matching_rank < 4:
        return "L3_DESCRIPTOR_RECALL"
    if oracle_pose_correct and not deployed_pose_correct:
        return "L4_CANDIDATE_SELECTION_OR_STRUCTURE"
    if not oracle_pose_correct:
        return "L5_SOLVER_GAP_ON_ORACLE_CORRESPONDENCES"
    return "PASS_OR_UNEXPLAINED_NONFAILURE"
