"""Formal fixed-plant mapping feedback and LOO audits for the V6 closed loop."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from common.v6_contracts import (
    DESCRIPTOR_CLEAN_LABEL_SEMANTICS,
    DESCRIPTOR_POSE_WEIGHT_SEMANTICS,
    exact_identity_positive_contract,
    require_mapping_only,
)
from evidence.observation_provider import ObservationProvider
from evidence.projective_loo import LeaveOneQueryOutProjectiveMap
from features.raster_sampling import sample_raster_at_grid_uv
from localization.matcher import global_cosine_topk
from localization.pose_solver import solve_absolute_pose
from map_learning.self_localization_feedback import build_self_localization_feedback
from topology.layered_sufficiency import (
    DEFAULT_VISIBILITY_GRID,
    visibility_image_cells,
)
from topology.pose_information import (
    fisher_contributions,
    pose_jacobian_analytic,
    task_scaled_pose_jacobian,
)


_POSE_INFORMATION_REGULARIZATION = 1e-9
_POSE_TRANSLATION_SCALE_M = 0.05
_POSE_ROTATION_SCALE_DEG = 5.0


def _maximum_matching(edges: list[list[int]]) -> tuple[int, list[tuple[int, int]]]:
    row_for_anchor: dict[int, int] = {}

    def augment(row: int, seen: set[int]) -> bool:
        for anchor in edges[row]:
            if anchor in seen:
                continue
            seen.add(anchor)
            previous = row_for_anchor.get(anchor)
            if previous is None or augment(previous, seen):
                row_for_anchor[anchor] = row
                return True
        return False

    for row in range(len(edges)):
        augment(row, set())
    pairs = [(row, anchor) for anchor, row in sorted(row_for_anchor.items())]
    return len(pairs), pairs


def _project(
    xyz: torch.Tensor, K: torch.Tensor, pose: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = xyz @ pose[:3, :3].T + pose[:3, 3]
    homogeneous = camera @ K.T
    return homogeneous[:, :2] / homogeneous[:, 2:].clamp_min(1e-8), camera[:, 2]


def _layer_edges(
    keypoints: torch.Tensor,
    projected: torch.Tensor,
    visible_rows: torch.Tensor,
    radius_px: float,
) -> list[list[int]]:
    result = [[] for _ in range(int(keypoints.shape[0]))]
    if keypoints.numel() == 0 or visible_rows.numel() == 0:
        return result
    if not float(radius_px) > 0:
        raise ValueError("positive radius must be positive")
    visible_xy = projected[visible_rows].float().numpy()
    tree = cKDTree(visible_xy)
    neighbors = tree.query_ball_point(
        keypoints.float().numpy(), r=float(radius_px), return_sorted=True
    )
    for row, local_rows in enumerate(neighbors):
        if len(local_rows):
            result[row] = visible_rows[
                torch.as_tensor(local_rows, dtype=torch.long)
            ].tolist()
    return result


def _depth_certified_pose_valid_edges(
    geometry_edges: list[list[int]],
    *,
    anchor_depth: torch.Tensor,
    keypoint_depth: torch.Tensor | None,
    absolute_tolerance_m: float = 0.25,
    relative_tolerance: float = 0.05,
) -> tuple[list[list[int]], bool]:
    """Certify non-identity alternatives against a rendered surface depth.

    A 2D reprojection neighborhood alone is never promoted to a pose-valid
    positive.  When the observation cache has no aligned keypoint depth, the
    certified set is intentionally empty and the broad edges remain diagnostic
    ambiguity only.
    """

    certified = [[] for _ in geometry_edges]
    if keypoint_depth is None:
        return certified, False
    reference = torch.as_tensor(keypoint_depth).float().reshape(-1)
    if reference.numel() != len(geometry_edges):
        raise ValueError("keypoint depth and geometry rows differ")
    for row, anchors in enumerate(geometry_edges):
        surface = float(reference[row])
        if not math.isfinite(surface) or surface <= 0.0 or not anchors:
            continue
        candidates = torch.as_tensor(anchors, dtype=torch.long)
        candidate_depth = anchor_depth[candidates].float()
        tolerance = float(absolute_tolerance_m) + float(relative_tolerance) * surface
        valid = (
            torch.isfinite(candidate_depth)
            & (candidate_depth > 0.0)
            & ((candidate_depth - surface).abs() <= tolerance)
        )
        certified[row] = candidates[valid].tolist()
    return certified, True


def _aligned_keypoint_surface_depth(
    view,
    *,
    alpha_minimum: float,
) -> tuple[torch.Tensor | None, str]:
    """Resolve Gaussian surface depth on the exact sparse detector rows.

    Older render caches store the dense native depth/alpha rasters but omit the
    redundant ``*_at_keypoints`` columns.  Treating those caches as having no
    depth silently disabled every certified non-identity alternative.  The
    nearest-cell sampling below is the same frozen raster convention used by
    projective association/completion; invalid or transparent samples remain
    unavailable rather than being promoted.
    """

    if view.keypoint_depth is not None:
        depth = torch.as_tensor(view.keypoint_depth).float().clone()
        source = "native_depth_at_keypoints"
    elif view.depth is not None:
        depth = sample_raster_at_grid_uv(view.depth, view.keypoints).float()
        source = "sampled_native_depth_raster_at_sparse_keypoints"
    else:
        return None, "unavailable"
    valid = torch.isfinite(depth) & (depth > 0.0)
    if view.keypoint_validity is not None:
        valid &= torch.as_tensor(view.keypoint_validity).bool()
    elif view.valid_mask is not None:
        valid &= sample_raster_at_grid_uv(view.valid_mask, view.keypoints).bool()
    if view.keypoint_alpha is not None:
        alpha = torch.as_tensor(view.keypoint_alpha).float()
        valid &= torch.isfinite(alpha) & (alpha >= float(alpha_minimum))
    elif view.alpha is not None:
        alpha = sample_raster_at_grid_uv(view.alpha, view.keypoints).float()
        valid &= torch.isfinite(alpha) & (alpha >= float(alpha_minimum))
    depth[~valid] = torch.nan
    return depth, source


def _exact_identity_anchor_by_query(
    state: dict,
    observations: ObservationProvider,
) -> list[torch.Tensor]:
    """Resolve each mapping keypoint to its one certified Anchor identity.

    The projective observation CSR is association lineage, unlike a reprojection
    radius.  A query/keypoint occurring under multiple Anchors is therefore an
    invalid identity artifact rather than a multi-positive training example.
    """

    names = list(state.get("v6_mapping_query_names", ()))
    if names != list(observations.names):
        raise ValueError("V6 map and identity query registries differ")
    csr = state.get("projective_anchor_observations")
    if not isinstance(csr, dict):
        raise ValueError(
            "exact identity positives require projective Anchor observations"
        )
    if (
        csr.get("schema") != "lafgs_projective_anchor_observations"
        or int(csr.get("version", -1)) != 1
    ):
        raise ValueError("unsupported exact identity observation schema")
    required_fields = (
        "observation_offsets",
        "query_indices",
        "keypoint_indices",
    )
    if any(field not in csr for field in required_fields):
        raise ValueError("exact identity observation CSR fields are incomplete")
    offsets = torch.as_tensor(csr.get("observation_offsets"))
    query = torch.as_tensor(csr.get("query_indices"))
    keypoint = torch.as_tensor(csr.get("keypoint_indices"))
    anchor_count = int(torch.as_tensor(state["anchor_ids"]).numel())
    if offsets.dtype != torch.long or offsets.shape != (anchor_count + 1,):
        raise ValueError("exact identity observation offsets must be int64 [N+1]")
    if (
        query.dtype != torch.long
        or keypoint.dtype != torch.long
        or query.ndim != 1
        or keypoint.shape != query.shape
    ):
        raise ValueError("exact identity observation columns must be aligned int64")
    if (
        int(offsets[0]) != 0
        or bool((offsets[1:] <= offsets[:-1]).any())
        or int(offsets[-1]) != int(query.numel())
    ):
        raise ValueError(
            "every Anchor must have a non-empty exact identity observation row"
        )
    if query.numel() and (
        int(query.min()) < 0 or int(query.max()) >= len(observations)
    ):
        raise ValueError("exact identity query index is out of range")

    anchor = torch.repeat_interleave(
        torch.arange(anchor_count, dtype=torch.long), offsets[1:] - offsets[:-1]
    )
    order = torch.argsort(query, stable=True)
    counts = torch.bincount(query, minlength=len(observations))
    query_offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    result = []
    for query_index in range(len(observations)):
        view = observations.build_view(query_index)
        row_count = int(view.descriptors.shape[0])
        identity = torch.full((row_count,), -1, dtype=torch.long)
        start = int(query_offsets[query_index])
        stop = int(query_offsets[query_index + 1])
        positions = order[start:stop]
        rows = keypoint[positions]
        if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= row_count):
            raise ValueError(
                f"exact identity keypoint index is invalid for {view.image_name}"
            )
        if rows.numel() != torch.unique(rows).numel():
            raise ValueError(
                f"query/keypoint identity is assigned to multiple Anchors: "
                f"{view.image_name}"
            )
        identity[rows] = anchor[positions]
        result.append(identity)
    return result


def _partition_identity_edges(
    identity_anchor: torch.Tensor,
    geometry_edges: list[list[int]],
    active: torch.Tensor,
) -> dict[str, list[list[int]]]:
    """Partition candidates into exact positives, ignores, and negatives."""

    if identity_anchor.shape != (len(geometry_edges),):
        raise ValueError("identity and geometry query rows differ")
    exact = [[] for _ in geometry_edges]
    ambiguous = [[] for _ in geometry_edges]
    incompatible = [[] for _ in geometry_edges]
    inactive = [[] for _ in geometry_edges]
    active_identity = [[] for _ in geometry_edges]
    ignored = [[] for _ in geometry_edges]
    lineage = [[] for _ in geometry_edges]
    for row, geometry in enumerate(geometry_edges):
        identity = int(identity_anchor[row])
        geometry_set = set(int(anchor) for anchor in geometry)
        if identity >= 0:
            lineage[row] = [identity]
            if not bool(active[identity]):
                inactive[row] = [identity]
            else:
                active_identity[row] = [identity]
                if identity in geometry_set:
                    exact[row] = [identity]
                else:
                    incompatible[row] = [identity]
        ambiguous[row] = sorted(anchor for anchor in geometry_set if anchor != identity)
        ignored[row] = sorted(set(ambiguous[row] + incompatible[row]))
    return {
        "lineage": lineage,
        "exact": exact,
        "ambiguous": ambiguous,
        "incompatible": incompatible,
        "inactive": inactive,
        "active_identity": active_identity,
        "ignored": ignored,
    }


def _edge_pairs(edges: list[list[int]]) -> torch.Tensor:
    return torch.tensor(
        [(row, anchor) for row, anchors in enumerate(edges) for anchor in anchors],
        dtype=torch.long,
    ).reshape(-1, 2)


def _anchor_unique_pose_rows(
    query_rows: torch.Tensor,
    anchor_ids: torch.Tensor,
    *,
    keypoints: torch.Tensor,
    projected: torch.Tensor,
    winner_scores: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one independent pose constraint per Anchor.

    The lowest known-pose reprojection residual wins.  Descriptor score and
    query-row index deterministically break exact residual ties.
    """

    query_rows = torch.as_tensor(query_rows, dtype=torch.long).reshape(-1)
    anchor_ids = torch.as_tensor(anchor_ids, dtype=torch.long).reshape(-1)
    if query_rows.shape != anchor_ids.shape:
        raise ValueError("pose query rows and Anchor IDs differ")
    selected_rows = []
    selected_residuals = []
    for anchor in torch.unique(anchor_ids, sorted=True).tolist():
        candidates = query_rows[anchor_ids == int(anchor)]
        residual = torch.linalg.norm(
            keypoints[candidates] - projected[int(anchor)], dim=1
        )
        minimum = residual.min()
        tied = candidates[residual == minimum]
        if tied.numel() > 1:
            scores = winner_scores[tied]
            tied = tied[scores == scores.max()]
        chosen = int(tied.min())
        selected_rows.append(chosen)
        selected_residuals.append(
            float(torch.linalg.norm(keypoints[chosen] - projected[int(anchor)]))
        )
    return (
        torch.tensor(selected_rows, dtype=torch.long),
        torch.tensor(selected_residuals, dtype=torch.float64),
    )


def _visible_spatial_rank(
    projected: torch.Tensor,
    visible_rows: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    grid_rows: int = DEFAULT_VISIBILITY_GRID[0],
    grid_cols: int = DEFAULT_VISIBILITY_GRID[1],
) -> int:
    """Count independently occupied image cells, not raw visible Anchors."""

    if visible_rows.numel() == 0:
        return 0
    cells = visibility_image_cells(
        projected[visible_rows],
        image_hw=image_hw,
        grid_shape=(grid_rows, grid_cols),
    )
    return int(torch.unique(cells).numel())


def _pose_neighborhoods(
    observations: ObservationProvider,
    neighbor_count: int,
) -> list[torch.Tensor]:
    """Return deterministic query-local pose neighborhoods.

    Translation and rotation are normalized by their mapping-trajectory
    nearest-neighbor scales.  This avoids filename/sequence heuristics while
    preventing adjacent trajectory frames from leaking into LOO maps.
    """

    count = len(observations)
    neighbor_count = min(max(int(neighbor_count), 1), count)
    if neighbor_count == 1:
        return [torch.tensor([index], dtype=torch.long) for index in range(count)]
    poses = torch.stack(
        [observations.build_view(index).pose_w2c.float() for index in range(count)]
    )
    rotations = poses[:, :3, :3]
    centers = -(rotations.transpose(1, 2) @ poses[:, :3, 3, None]).squeeze(-1)
    translation = torch.cdist(centers, centers)
    trace = torch.einsum("aij,bij->ab", rotations, rotations)
    rotation = torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0))
    diagonal = torch.eye(count, dtype=torch.bool)

    def nearest_scale(distance: torch.Tensor, fallback: float) -> torch.Tensor:
        masked = distance.masked_fill(diagonal | (distance <= 1e-8), torch.inf)
        nearest = masked.min(1).values
        finite = nearest[torch.isfinite(nearest)]
        if finite.numel() == 0:
            return torch.tensor(float(fallback), dtype=distance.dtype)
        return finite.median().clamp_min(1e-6)

    translation_scale = nearest_scale(translation, 1.0)
    rotation_scale = nearest_scale(rotation, math.radians(15.0))
    distance = translation / translation_scale + rotation / rotation_scale
    # Stable sorting makes repeated/same-pose cameras deterministic.
    order = torch.argsort(distance, dim=1, stable=True)
    return [torch.sort(order[index, :neighbor_count]).values for index in range(count)]


def _positive_score_statistics(
    dense_scores: torch.Tensor,
    positives_by_row: list[list[int]],
    *,
    ignored_by_row: list[list[int]] | None = None,
    chunk_size: int = 64,
) -> dict[int, tuple[float, float, int, int, int]]:
    """Return exact stable-rank statistics with one bank scan per query row.

    Stable descending argsort breaks equal-score ties by the original Anchor
    row. Counting larger scores plus equal scores at smaller rows is exactly
    equivalent. Rows are processed in bounded batches so rank and best-wrong
    share one vectorized scan rather than separately scanning the full bank
    for every positive keypoint.
    """

    if dense_scores.ndim != 2 or len(positives_by_row) != dense_scores.shape[0]:
        raise ValueError("positive edges and dense score rows differ")
    if ignored_by_row is None:
        ignored_by_row = [[] for _ in positives_by_row]
    if len(ignored_by_row) != len(positives_by_row):
        raise ValueError("ignored edges and dense score rows differ")
    if any(
        set(positives) & set(ignored)
        for positives, ignored in zip(positives_by_row, ignored_by_row)
    ):
        raise ValueError("positive and ignored Anchor identities overlap")
    if int(chunk_size) < 1:
        raise ValueError("positive statistics chunk size must be positive")
    positive_query_rows = [
        row for row, positives in enumerate(positives_by_row) if positives
    ]
    result: dict[int, tuple[float, float, int, int, int]] = {}
    anchor_count = int(dense_scores.shape[1])
    anchor_rows = torch.arange(anchor_count, device=dense_scores.device)
    for start in range(0, len(positive_query_rows), int(chunk_size)):
        rows_list = positive_query_rows[start : start + int(chunk_size)]
        rows = torch.tensor(rows_list, dtype=torch.long, device=dense_scores.device)
        maximum_degree = max(len(positives_by_row[row]) for row in rows_list)
        padded_cpu = torch.full(
            (len(rows_list), maximum_degree),
            -1,
            dtype=torch.long,
        )
        lengths = torch.tensor(
            [len(positives_by_row[row]) for row in rows_list], dtype=torch.long
        )
        flat = torch.tensor(
            [anchor for row in rows_list for anchor in positives_by_row[row]],
            dtype=torch.long,
        )
        local = torch.repeat_interleave(torch.arange(len(rows_list)), lengths)
        offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
        within = torch.arange(flat.numel()) - torch.repeat_interleave(
            offsets[:-1], lengths
        )
        padded_cpu[local, within] = flat
        padded = padded_cpu.to(dense_scores.device)
        valid = padded >= 0
        gathered = dense_scores[rows[:, None], padded.clamp_min(0)]
        gathered = gathered.masked_fill(~valid, -torch.inf)
        best_positive = gathered.max(1).values
        best_anchor = (
            torch.where(
                valid & (gathered == best_positive[:, None]),
                padded,
                anchor_count,
            )
            .min(1)
            .values
        )
        scores = dense_scores[rows]
        ranking_scores = scores.clone()
        wrong = scores.clone()
        ignored_lengths = torch.tensor(
            [len(ignored_by_row[row]) for row in rows_list], dtype=torch.long
        )
        if int(ignored_lengths.sum()):
            ignored_anchor = torch.tensor(
                [anchor for row in rows_list for anchor in ignored_by_row[row]],
                dtype=torch.long,
                device=dense_scores.device,
            )
            ignored_local = torch.repeat_interleave(
                torch.arange(len(rows_list), device=dense_scores.device),
                ignored_lengths.to(dense_scores.device),
            )
            ranking_scores[ignored_local, ignored_anchor] = -torch.inf
            wrong[ignored_local, ignored_anchor] = -torch.inf
        ranks = 1 + (
            (ranking_scores > best_positive[:, None])
            | (
                (ranking_scores == best_positive[:, None])
                & (anchor_rows[None] < best_anchor[:, None])
            )
        ).sum(1)
        local_rows = torch.arange(len(rows_list), device=dense_scores.device)
        local_rows = local_rows[:, None].expand_as(padded)
        wrong[local_rows[valid], padded[valid]] = -torch.inf
        best_wrong = wrong.max(1).values
        best_wrong_anchor = torch.where(
            torch.isfinite(best_wrong),
            torch.argmax(wrong, dim=1),
            torch.full_like(torch.argmax(wrong, dim=1), -1),
        )
        packed = torch.stack(
            [best_positive, best_wrong, ranks, best_anchor, best_wrong_anchor], dim=1
        ).cpu()
        for row, values in zip(rows_list, packed.tolist()):
            result[row] = (
                float(values[0]),
                float(values[1]),
                int(values[2]),
                int(values[3]),
                int(values[4]),
            )
    return result


def _legal_descriptor_pair_is_clean(
    positive_score: float, legal_negative_score: float
) -> bool:
    """Classify a feedback pair without consulting ignored global winners."""

    return bool(
        math.isfinite(positive_score)
        and math.isfinite(legal_negative_score)
        and positive_score >= legal_negative_score
    )


def _fixed_hypothesis_counterfactual_pose_weights(
    *,
    triplets: torch.Tensor,
    winners: torch.Tensor,
    keypoints: torch.Tensor,
    xyz: torch.Tensor,
    intrinsics: torch.Tensor,
    current_pose_w2c: torch.Tensor,
    ground_truth_pose_w2c: torch.Tensor,
    reprojection_error_px: float,
) -> torch.Tensor:
    """Score an actual winner flip against fixed current/ground-truth hypotheses.

    The online solver is unchanged.  Offline, each legal negative-to-exact-positive
    replacement receives the normalized improvement in the standard RANSAC
    consensus margin ``support(GT) - support(current)``.  A row that does not
    replace the deployed winner cannot change the current hypothesis and gets
    zero weight.
    """

    values = torch.as_tensor(triplets, dtype=torch.long).reshape(-1, 4)
    if values.numel() == 0:
        return torch.empty(0, dtype=torch.float32)
    rows, positive, negative, _ = values.T
    if bool((rows < 0).any()) or bool((rows >= keypoints.shape[0]).any()):
        raise ValueError("descriptor counterfactual query row is invalid")
    if bool((positive < 0).any()) or bool((positive >= xyz.shape[0]).any()):
        raise ValueError("descriptor counterfactual positive Anchor is invalid")
    if bool((negative < 0).any()) or bool((negative >= xyz.shape[0]).any()):
        raise ValueError("descriptor counterfactual negative Anchor is invalid")

    def residual(anchor_rows: torch.Tensor, pose_w2c: torch.Tensor) -> torch.Tensor:
        projected, depth = _project(
            xyz[anchor_rows].float(), intrinsics.float(), pose_w2c.float()
        )
        error = torch.linalg.norm(projected - keypoints[rows].float(), dim=1)
        valid = torch.isfinite(error) & torch.isfinite(depth) & (depth > 0)
        return torch.where(valid, error, torch.full_like(error, float("inf")))

    threshold = float(reprojection_error_px)
    if not threshold > 0.0:
        raise ValueError("counterfactual reprojection threshold must be positive")
    positive_gt = residual(positive, ground_truth_pose_w2c) <= threshold
    positive_current = residual(positive, current_pose_w2c) <= threshold
    negative_gt = residual(negative, ground_truth_pose_w2c) <= threshold
    negative_current = residual(negative, current_pose_w2c) <= threshold
    before_margin = negative_gt.float() - negative_current.float()
    after_margin = positive_gt.float() - positive_current.float()
    # Each binary hypothesis-support margin lies in [-1, 1], hence a winner
    # replacement can improve it by at most two.
    gain = ((after_margin - before_margin) / 2.0).clamp(0.0, 1.0)
    deployed_winner_replaced = winners[rows].long() == negative
    return gain * deployed_winner_replaced.float()


def _summary(rows: list[dict]) -> dict:
    te = np.asarray([row["te_cm"] for row in rows], dtype=np.float64)
    ae = np.asarray([row["ae_deg"] for row in rows], dtype=np.float64)
    tail = max(int(math.ceil(0.05 * len(rows))), 1)
    correspondence_count = sum(int(row["correspondences"]) for row in rows)
    correct_count = sum(int(row["correct_winners"]) for row in rows)
    inlier_count = sum(int(row["inliers"]) for row in rows)
    clean_inlier_count = sum(int(row["clean_inliers"]) for row in rows)
    positive_rows = sum(int(row["positive_rows"]) for row in rows)
    identity_lineage_rows = sum(int(row["identity_lineage_rows"]) for row in rows)
    ambiguous_rows = sum(int(row["geometry_ambiguous_rows"]) for row in rows)
    exact_correct_rows = sum(
        int(row["top1_exact_identity_correct_rows"]) for row in rows
    )
    return {
        "query_count": len(rows),
        "median_te_cm": float(np.median(te)),
        "mean_te_cm": float(np.mean(te)),
        "p90_te_cm": float(np.percentile(te, 90)),
        "cvar95_te_cm": float(np.sort(te)[-tail:].mean()),
        "median_ae_deg": float(np.median(ae)),
        "recall_5cm_5deg_percent": float(np.mean((te < 5.0) & (ae < 5.0)) * 100.0),
        "catastrophic_100cm_count": int((te >= 100.0).sum()),
        "raw_gt_precision_percent": 100.0
        * correct_count
        / max(correspondence_count, 1),
        "raw_exact_identity_precision_percent": (
            100.0 * exact_correct_rows / max(correspondence_count, 1)
        ),
        "exact_identity_lineage_rows": identity_lineage_rows,
        "exact_identity_positive_rows": positive_rows,
        "geometry_compatible_ambiguous_rows": ambiguous_rows,
        "certified_pose_valid_alternative_rows": sum(
            int(row.get("certified_pose_valid_alternative_rows", 0))
            for row in rows
        ),
        "top1_exact_identity_correct_rows": exact_correct_rows,
        "top1_geometry_compatible_ambiguous_rows": sum(
            int(row["top1_geometry_ambiguous_rows"]) for row in rows
        ),
        "top1_identity_projective_incompatible_rows": sum(
            int(row["top1_identity_incompatible_rows"]) for row in rows
        ),
        "top1_negative_rows": sum(int(row["top1_negative_rows"]) for row in rows),
        "exact_identity_top1_recall_percent": (
            100.0 * exact_correct_rows / max(positive_rows, 1)
        ),
        "pose_information_duplicate_rows_removed": sum(
            int(row["pose_information_duplicate_rows_removed"]) for row in rows
        ),
        "inlier_gt_precision_percent": 100.0
        * clean_inlier_count
        / max(inlier_count, 1),
        "correct_anchor_recall_at_1_percent": 100.0
        * sum(int(row["correct_anchor_rank_le_1"]) for row in rows)
        / max(positive_rows, 1),
        "correct_anchor_recall_at_16_percent": 100.0
        * sum(int(row["correct_anchor_rank_le_16"]) for row in rows)
        / max(positive_rows, 1),
        "mean_poselib_iterations": float(
            np.mean([row["poselib_iterations"] for row in rows])
        ),
        "online_latency_ms": float(np.mean([row["online_latency_ms"] for row in rows])),
        "oracle_feedback_localization_latency_ms": float(
            np.mean([row["online_latency_ms"] for row in rows])
        ),
        "loo_feedback_latency_ms": float(
            np.mean([row["loo_feedback_latency_ms"] for row in rows])
        ),
    }


def _descriptor_training_query_masks(state: dict, query_count: int) -> dict:
    """Return declared training-split and direct-gradient query masks.

    Older residual checkpoints did not serialize this registry.  When a
    non-zero residual exists without a registry, fail closed and treat every
    mapping query as training-reused rather than claiming query LOO.
    """

    training_split = torch.zeros(int(query_count), dtype=torch.bool)
    gradient_reused = torch.zeros(int(query_count), dtype=torch.bool)
    explicit = False
    report = state.get("v6_descriptor_distillation")
    if isinstance(report, dict):
        training_value = report.get("training_query_indices")
        selected_value = report.get("selected_query_indices", training_value)
        if training_value is not None:
            rows = torch.as_tensor(training_value, dtype=torch.long).reshape(-1)
            if rows.numel() and (
                int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
            ):
                raise ValueError("descriptor training query registry is invalid")
            training_split[rows] = True
        if selected_value is not None:
            rows = torch.as_tensor(selected_value, dtype=torch.long).reshape(-1)
            if rows.numel() and (
                int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
            ):
                raise ValueError("descriptor gradient query registry is invalid")
            gradient_reused[rows] = True
        explicit = bool(report.get("training_query_registry_explicit", False))
        if training_value is not None or selected_value is not None:
            return {
                "training_split": training_split,
                "gradient_reused": gradient_reused,
                "training_registry_explicit": explicit,
                "descriptor_dependency_present": True,
            }
        training_split[:] = True
        gradient_reused[:] = True
        return {
            "training_split": training_split,
            "gradient_reused": gradient_reused,
            "training_registry_explicit": False,
            "descriptor_dependency_present": True,
        }
    residual = state.get("anchor_descriptor_residual")
    if residual is not None and bool(torch.as_tensor(residual).abs().max() > 0):
        training_split[:] = True
        gradient_reused[:] = True
        dependency_present = True
    else:
        dependency_present = False
    return {
        "training_split": training_split,
        "gradient_reused": gradient_reused,
        "training_registry_explicit": False,
        "descriptor_dependency_present": dependency_present,
    }


def _descriptor_training_query_mask(state: dict, query_count: int) -> torch.Tensor:
    """Backward-compatible direct-gradient mask used by focused tests/tools."""

    return _descriptor_training_query_masks(state, query_count)["gradient_reused"]


def _reconstruction_target_query_mask(state: dict, query_count: int) -> torch.Tensor:
    """Queries whose rendered depth selected a reconstruction seed region."""

    mask = torch.zeros(int(query_count), dtype=torch.bool)
    report = state.get("v6_reconstruction_distillation")
    if not isinstance(report, dict):
        # Initial mapping completion Anchors share the same candidate kind as
        # later feedback-targeted reconstruction, so kind alone is not a
        # dependency signal.  Legacy targeted rounds did persist this exact
        # feedback lineage even before the target registry was added.
        legacy_reconstruction = (
            state.get("provenance", {}).get("v6_reconstruction_feedback_sha256")
            is not None
        )
        if legacy_reconstruction:
            mask[:] = True
        return mask
    rows = torch.as_tensor(
        report.get("target_query_indices", ()), dtype=torch.long
    ).reshape(-1)
    if rows.numel() and (int(rows.min()) < 0 or int(rows.max()) >= int(query_count)):
        raise ValueError("reconstruction target query registry is invalid")
    mask[rows] = True
    return mask


def _reconstruction_training_query_mask(
    state: dict, query_count: int
) -> torch.Tensor:
    """Queries used anywhere by feedback-driven reconstruction.

    Target replay is narrower: target-query depth selects seed regions, while
    every training query may supply eligible reconstruction observations.  An
    older report without an explicit training/support registry is therefore
    not evidence of independence and is conservatively treated as depending
    on every mapping query.
    """

    target = _reconstruction_target_query_mask(state, query_count)
    report = state.get("v6_reconstruction_distillation")
    if not isinstance(report, dict):
        return target
    dependency = torch.zeros(int(query_count), dtype=torch.bool)
    registry_present = False
    for key in ("training_query_indices", "eligible_support_query_indices"):
        if key not in report:
            continue
        registry_present = True
        rows = torch.as_tensor(report[key], dtype=torch.long).reshape(-1)
        if rows.numel() and (
            int(rows.min()) < 0 or int(rows.max()) >= int(query_count)
        ):
            raise ValueError("reconstruction training/support registry is invalid")
        dependency[rows] = True
    if not registry_present or not bool(dependency.any()):
        dependency[:] = True
    dependency |= target
    return dependency


def _selection_training_query_mask(state: dict, query_count: int) -> torch.Tensor:
    """Queries whose feedback directly determined Anchor selection."""

    mask = torch.zeros(int(query_count), dtype=torch.bool)
    report = state.get("v6_selection_distillation")
    if not isinstance(report, dict):
        return mask
    rows = torch.as_tensor(
        report.get("training_query_indices", ()), dtype=torch.long
    ).reshape(-1)
    # Older selection maps did not serialize dependencies.  Fail closed.
    if rows.numel() == 0:
        mask[:] = True
        return mask
    if int(rows.min()) < 0 or int(rows.max()) >= int(query_count):
        raise ValueError("selection training query registry is invalid")
    mask[rows] = True
    return mask


@torch.inference_mode()
def evaluate_query_local_feedback(
    *,
    state: dict,
    observations: ObservationProvider,
    source_map_sha256: str,
    query_cache_sha256: str,
    scene_calibration_sha256: str,
    feedback_calibration_binding_sha256: str,
    device: torch.device,
    positive_radius_px: float,
    alpha_minimum: float,
    required_rank: int,
    ransac_reprojection_px: float,
    seed: int,
    loo_pose_neighbors: int = 1,
    required_visibility_rank: int = 4,
    required_detectable_rank: int | None = None,
    loo_affected_anchor_policy: str = "rebuild",
    feedback_observer_mode: str = "fixed_map",
    pose_logdet_target: float = 0.0,
    pose_min_eigenvalue_target: float = 0.0,
) -> dict:
    """One global Top-1 and one standard PoseLib solve per mapping query.

    ``fixed_map`` (F0) measures the actual deployment plant.  The optional
    ``descriptor_leave_self_out`` mode (F1) holds geometry and topology fixed
    while removing only the current query's direct descriptor contribution.
    ``full_loo`` (F2) retains the historical geometry-rebuild stress test.
    """

    evaluation_started = time.perf_counter()
    if not math.isfinite(float(pose_logdet_target)):
        raise ValueError("pose logdet target must be finite")
    if (
        not math.isfinite(float(pose_min_eigenvalue_target))
        or float(pose_min_eigenvalue_target) < 0.0
    ):
        raise ValueError(
            "pose minimum-eigenvalue target must be finite and non-negative"
        )
    require_mapping_only(state.get("provenance", {}), label="V6 feedback map")
    if feedback_observer_mode not in {
        "fixed_map",
        "descriptor_leave_self_out",
        "full_loo",
    }:
        raise ValueError("unknown V6 feedback observer mode")
    replay = None
    if feedback_observer_mode != "fixed_map":
        # F2 may rebuild or conservatively purge affected Anchors.  F1 only
        # needs access to observation descriptors and never changes geometry.
        replay = LeaveOneQueryOutProjectiveMap(
            state,
            observations,
            affected_anchor_policy=(
                "rebuild"
                if feedback_observer_mode == "descriptor_leave_self_out"
                else loo_affected_anchor_policy
            ),
        )
    identity_anchor_by_query = _exact_identity_anchor_by_query(state, observations)
    positive_identity_contract = exact_identity_positive_contract()
    descriptor_identity_supervision_available = not (
        feedback_observer_mode == "full_loo"
        and loo_affected_anchor_policy == "purge"
    )
    base_xyz = torch.as_tensor(state["anchor_xyz"]).float()
    base_bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    query_rows = []
    feedback_records = []
    descriptor_masks = _descriptor_training_query_masks(state, len(observations))
    descriptor_training_split = descriptor_masks["training_split"]
    descriptor_gradient_reused = descriptor_masks["gradient_reused"]
    reconstruction_target_reused = _reconstruction_target_query_mask(
        state, len(observations)
    )
    reconstruction_training_reused = _reconstruction_training_query_mask(
        state, len(observations)
    )
    selection_training_reused = _selection_training_query_mask(state, len(observations))
    pose_neighborhoods = _pose_neighborhoods(observations, loo_pose_neighbors)
    print(
        f"[v6-feedback] prepared {feedback_observer_mode} observer "
        f"({loo_affected_anchor_policy}) "
        f"for {len(observations)} queries and {base_xyz.shape[0]} Anchors "
        f"in {time.perf_counter() - evaluation_started:.1f}s",
        flush=True,
    )
    for query_index, name in enumerate(observations.names):
        if query_index % 25 == 0:
            print(
                f"[v6-feedback] query {query_index + 1}/{len(observations)}: {name}",
                flush=True,
            )
        view = observations.build_view(query_index)
        loo_started = time.perf_counter()
        if feedback_observer_mode == "fixed_map":
            excluded_queries = torch.empty(0, dtype=torch.long)
            update = {
                "anchor_rows": torch.empty(0, dtype=torch.long),
                "valid": torch.empty(0, dtype=torch.bool),
                "contract": {
                    "query_descriptor_loo": False,
                    "query_geometry_loo": False,
                    "affected_anchor_policy": "fixed_map",
                    "geometry_held_fixed": True,
                },
            }
        elif feedback_observer_mode == "descriptor_leave_self_out":
            excluded_queries = torch.tensor([query_index], dtype=torch.long)
            update = replay.descriptor_only_update(query_index)
        else:
            excluded_queries = pose_neighborhoods[query_index]
            update = replay.query_update(
                query_index, excluded_queries=excluded_queries
            )
        loo_latency_ms = (time.perf_counter() - loo_started) * 1000.0
        online_started = time.perf_counter()
        xyz = base_xyz
        bank = base_bank
        active = torch.ones(xyz.shape[0], dtype=torch.bool)
        affected = update["anchor_rows"]
        if affected.numel():
            if bool(update["valid"].any()):
                bank = base_bank.clone()
                valid_affected = affected[update["valid"]]
                bank[valid_affected] = F.normalize(
                    update["anchor_features"][update["valid"]], dim=1
                )
                if feedback_observer_mode == "full_loo":
                    xyz = base_xyz.clone()
                    xyz[valid_affected] = update["anchor_xyz"][update["valid"]]
            if feedback_observer_mode == "full_loo":
                active[affected] = update["valid"]
        projected, depth = _project(xyz, view.intrinsics.float(), view.pose_w2c.float())
        height, width = view.image_hw
        visible = (
            active
            & torch.isfinite(projected).all(1)
            & torch.isfinite(depth)
            & (depth > 0)
            & (projected[:, 0] >= 0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0)
            & (projected[:, 1] < height)
        )
        if view.alpha is not None:
            x = projected[:, 0].round().long().clamp(0, width - 1)
            y = projected[:, 1].round().long().clamp(0, height - 1)
            visible &= torch.isfinite(view.alpha[y, x]) & (
                view.alpha[y, x] >= float(alpha_minimum)
            )
        visible_rows = torch.nonzero(visible, as_tuple=False).reshape(-1)
        visible_image_cells = visibility_image_cells(
            projected[visible_rows],
            image_hw=(height, width),
        )
        visible_rank = int(torch.unique(visible_image_cells).numel())
        keypoints = view.physical_keypoints.float()
        geometry_edges = _layer_edges(
            keypoints, projected, visible_rows, float(positive_radius_px)
        )
        identity_partition = _partition_identity_edges(
            identity_anchor_by_query[query_index], geometry_edges, active
        )
        positive_edges = identity_partition["exact"]
        ambiguous_edges = identity_partition["ambiguous"]
        keypoint_surface_depth, pose_valid_depth_source = (
            _aligned_keypoint_surface_depth(
                view,
                alpha_minimum=float(alpha_minimum),
            )
        )
        certified_edges, pose_valid_depth_available = (
            _depth_certified_pose_valid_edges(
                geometry_edges,
                anchor_depth=depth,
                keypoint_depth=keypoint_surface_depth,
            )
        )
        certified_alternatives = []
        for row, anchors in enumerate(certified_edges):
            identity = int(identity_anchor_by_query[query_index][row])
            certified_alternatives.append(
                sorted(int(anchor) for anchor in anchors if int(anchor) != identity)
            )
        ignored_edges = identity_partition["ignored"]
        detectable_rank, detectable_pairs = _maximum_matching(geometry_edges)
        query_descriptor = F.normalize(view.descriptors.float(), dim=1).to(device)
        active_rows = torch.nonzero(active, as_tuple=False).reshape(-1)
        if active_rows.numel() < 4:
            raise ValueError(f"query {name} has fewer than four LOO-valid Anchors")
        matches = global_cosine_topk(
            query_descriptor,
            bank[active_rows].to(device),
            topk=1,
            anchor_descriptors_normalized=True,
        )
        winners = active_rows[matches.anchor_indices[:, 0].cpu()]
        correct = torch.tensor(
            [int(winner) in positive_edges[row] for row, winner in enumerate(winners)],
            dtype=torch.bool,
        )
        ambiguous = torch.tensor(
            [int(winner) in ambiguous_edges[row] for row, winner in enumerate(winners)],
            dtype=torch.bool,
        )
        identity_incompatible = torch.tensor(
            [
                int(winner) in identity_partition["incompatible"][row]
                for row, winner in enumerate(winners)
            ],
            dtype=torch.bool,
        )
        negative = ~(correct | ambiguous | identity_incompatible)
        matching_rank, matching_pairs = _maximum_matching(
            [
                [int(winners[row])] if bool(correct[row]) else []
                for row in range(len(winners))
            ]
        )
        estimate = solve_absolute_pose(
            keypoints.numpy(),
            xyz[winners].numpy(),
            view.intrinsics.float().numpy(),
            reprojection_error_px=float(ransac_reprojection_px),
            confidence=0.99999,
            max_iterations=100000,
            min_iterations=1000,
            seed=int(seed),
        )
        inliers = torch.as_tensor(estimate.inliers).long().reshape(-1)
        pose = torch.as_tensor(estimate.pose_w2c).float()
        rotation = pose[:3, :3] @ view.pose_w2c[:3, :3].float().T
        cosine = ((torch.trace(rotation) - 1.0) / 2.0).clamp(-1, 1)
        ae_deg = float(torch.rad2deg(torch.acos(cosine)))
        center_est = -(pose[:3, :3].T @ pose[:3, 3])
        gt = view.pose_w2c.float()
        center_gt = -(gt[:3, :3].T @ gt[:3, 3])
        te_cm = float(torch.linalg.norm(center_est - center_gt) * 100.0)
        online_latency_ms = (time.perf_counter() - online_started) * 1000.0
        clean_rows = (
            inliers[correct[inliers]]
            if inliers.numel()
            else torch.empty(0, dtype=torch.long)
        )
        clean_ids = winners[clean_rows]
        harmful_rows = (
            inliers[negative[inliers]]
            if inliers.numel()
            else torch.empty(0, dtype=torch.long)
        )
        harmful_ids = winners[harmful_rows]
        ambiguous_inlier_rows = (
            inliers[(ambiguous | identity_incompatible)[inliers]]
            if inliers.numel()
            else torch.empty(0, dtype=torch.long)
        )
        winner_scores = matches.scores[:, 0].cpu()
        pose_clean_rows, pose_reprojection_errors = _anchor_unique_pose_rows(
            clean_rows,
            clean_ids,
            keypoints=keypoints,
            projected=projected,
            winner_scores=winner_scores,
        )
        pose_clean_ids = winners[pose_clean_rows]
        dense_scores = query_descriptor @ bank.to(device).T
        dense_scores[:, ~active.to(device)] = -torch.inf
        best_positive = []
        best_wrong = []
        correct_anchor_ranks = []
        confusion_pairs = []
        descriptor_triplets = []
        descriptor_triplet_harmful_inlier = []
        descriptor_triplet_legal_pair_clean = []
        positive_statistics = _positive_score_statistics(
            dense_scores, positive_edges, ignored_by_row=ignored_edges
        )
        for row, positives in enumerate(positive_edges):
            if positives:
                (
                    positive_score,
                    wrong_score,
                    rank,
                    best,
                    best_wrong_anchor,
                ) = positive_statistics[row]
                best_positive.append(positive_score)
                best_wrong.append(wrong_score)
                correct_anchor_ranks.append(rank)
                if bool(negative[row]):
                    confusion_pairs.append((row, int(winners[row]), best))
                if (
                    descriptor_identity_supervision_available
                    and best_wrong_anchor >= 0
                    and math.isfinite(wrong_score)
                ):
                    legal_pair_clean = _legal_descriptor_pair_is_clean(
                        positive_score, wrong_score
                    )
                    descriptor_triplets.append(
                        (row, best, best_wrong_anchor, int(legal_pair_clean))
                    )
                    descriptor_triplet_harmful_inlier.append(
                        bool((harmful_rows == row).any())
                    )
                    descriptor_triplet_legal_pair_clean.append(legal_pair_clean)
        if pose_clean_ids.numel():
            clean_jacobian = pose_jacobian_analytic(
                xyz[pose_clean_ids].double(),
                view.intrinsics.double(),
                view.pose_w2c.double(),
            )
            clean_jacobian = task_scaled_pose_jacobian(
                clean_jacobian,
                translation_scale=_POSE_TRANSLATION_SCALE_M,
                rotation_scale=math.radians(_POSE_ROTATION_SCALE_DEG),
            )
            clean_information = fisher_contributions(clean_jacobian)
            total_information = clean_information.sum(0)
        else:
            clean_information = torch.empty((0, 6, 6), dtype=torch.float64)
            total_information = torch.zeros((6, 6), dtype=torch.float64)
        total_information = (total_information + total_information.T) * 0.5
        information_rank = int(torch.linalg.matrix_rank(total_information))
        regularized_information = total_information + (
            torch.eye(6, dtype=torch.float64) * _POSE_INFORMATION_REGULARIZATION
        )
        information_logdet = float(torch.linalg.slogdet(regularized_information)[1])
        information_min_eigenvalue = float(
            torch.linalg.eigvalsh(regularized_information)[0].clamp_min(0.0)
        )
        information_sufficient = bool(
            information_rank >= 6
            and information_logdet >= float(pose_logdet_target)
            and information_min_eigenvalue >= float(pose_min_eigenvalue_target)
        )
        exact_identity_pairs = _edge_pairs(identity_partition["lineage"])
        active_identity_pairs = _edge_pairs(identity_partition["active_identity"])
        exact_identity_positive_pairs = _edge_pairs(positive_edges)
        triplet_harmful_mask = torch.tensor(
            descriptor_triplet_harmful_inlier, dtype=torch.bool
        )
        triplet_legal_pair_clean_mask = torch.tensor(
            descriptor_triplet_legal_pair_clean, dtype=torch.bool
        )
        descriptor_triplet_tensor = torch.tensor(
            descriptor_triplets, dtype=torch.long
        ).reshape(-1, 4)
        triplet_pose_weights = _fixed_hypothesis_counterfactual_pose_weights(
            triplets=descriptor_triplet_tensor,
            winners=winners,
            keypoints=keypoints,
            xyz=xyz,
            intrinsics=view.intrinsics,
            current_pose_w2c=pose,
            ground_truth_pose_w2c=view.pose_w2c,
            reprojection_error_px=float(ransac_reprojection_px),
        )
        feedback_records.append(
            {
                "image_name": name,
                "visible_rank": int(visible_rank),
                "visible_anchor_count": int(visible_rows.numel()),
                "detectable_rank": int(detectable_rank),
                "correct_anchor_rank": min(correct_anchor_ranks, default=0),
                "matching_rank": int(matching_rank),
                "winner_anchor": int(winners[0]) if winners.numel() else -1,
                "best_positive_score": max(best_positive, default=-1.0),
                "best_wrong_score": max(best_wrong, default=-1.0),
                "clean_inlier_anchor_ids": torch.unique(clean_ids),
                "harmful_inlier_anchor_ids": torch.unique(harmful_ids),
                "ambiguous_inlier_anchor_ids": torch.unique(
                    winners[ambiguous_inlier_rows]
                ),
                "query_rows": torch.arange(keypoints.shape[0], dtype=torch.long),
                "winner_anchor_ids": winners,
                "winner_scores": winner_scores,
                "winner_identity_correct_mask": correct,
                "top1_exact_identity_correct_mask": correct,
                "top1_geometry_compatible_ambiguous_mask": ambiguous,
                "top1_identity_projective_incompatible_mask": (identity_incompatible),
                "top1_negative_mask": negative,
                "inlier_query_rows": inliers,
                "inlier_clean_mask": correct[inliers]
                if inliers.numel()
                else torch.empty(0, dtype=torch.bool),
                "visible_anchor_ids": visible_rows,
                "visible_anchor_image_cells": visible_image_cells,
                "exact_identity_pairs": exact_identity_pairs,
                "exact_identity_lineage_pairs": exact_identity_pairs,
                "active_identity_pairs": active_identity_pairs,
                "exact_identity_positive_pairs": exact_identity_positive_pairs,
                "identity_inactive_pairs": _edge_pairs(identity_partition["inactive"]),
                "identity_projective_incompatible_pairs": _edge_pairs(
                    identity_partition["incompatible"]
                ),
                "projective_compatible_ambiguous_pairs": _edge_pairs(ambiguous_edges),
                "certified_pose_valid_alternative_pairs": _edge_pairs(
                    certified_alternatives
                ),
                "pose_valid_depth_supervision_available": bool(
                    pose_valid_depth_available
                ),
                "pose_valid_depth_source": pose_valid_depth_source,
                "pose_valid_depth_absolute_tolerance_m": 0.25,
                "pose_valid_depth_relative_tolerance": 0.05,
                "identity_positive_count": sum(
                    int(bool(edges)) for edges in positive_edges
                ),
                "identity_active_count": sum(
                    int(bool(edges)) for edges in identity_partition["active_identity"]
                ),
                "identity_lineage_count": sum(
                    int(bool(edges)) for edges in identity_partition["lineage"]
                ),
                "identity_inactive_count": sum(
                    int(bool(edges)) for edges in identity_partition["inactive"]
                ),
                "identity_projective_incompatible_count": sum(
                    int(bool(edges)) for edges in identity_partition["incompatible"]
                ),
                "geometry_ambiguous_count": sum(
                    len(edges) for edges in ambiguous_edges
                ),
                "certified_pose_valid_alternative_count": sum(
                    len(edges) for edges in certified_alternatives
                ),
                "detectable_pairs": torch.tensor(
                    detectable_pairs, dtype=torch.long
                ).reshape(-1, 2),
                "matching_pairs": torch.tensor(
                    matching_pairs, dtype=torch.long
                ).reshape(-1, 2),
                "confusion_pairs": torch.tensor(
                    confusion_pairs, dtype=torch.long
                ).reshape(-1, 3),
                "descriptor_triplets": descriptor_triplet_tensor,
                "descriptor_triplet_harmful_inlier_mask": triplet_harmful_mask,
                "descriptor_triplet_pose_weights": triplet_pose_weights,
                "descriptor_triplet_legal_pair_clean_mask": (
                    triplet_legal_pair_clean_mask
                ),
                "descriptor_identity_supervision_available": (
                    descriptor_identity_supervision_available
                ),
                "excluded_query_indices": excluded_queries,
                "dependency_group_ids": torch.unique(
                    torch.as_tensor(state["dependency_group_ids"])[winners]
                ),
                "clean_inlier_pose_anchor_ids": pose_clean_ids,
                "clean_inlier_pose_query_rows": pose_clean_rows,
                "clean_inlier_pose_reprojection_errors_px": (pose_reprojection_errors),
                "pose_information_duplicate_rows_removed": int(
                    clean_rows.numel() - pose_clean_rows.numel()
                ),
                "pose_information_anchor_unique": True,
                "clean_inlier_pose_information": clean_information,
                "pose_information_rank": information_rank,
                "pose_information_logdet": information_logdet,
                "pose_information_min_eigenvalue": information_min_eigenvalue,
                "pose_information_contribution": information_logdet,
                "pose_information_sufficient": information_sufficient,
                "pose_logdet_target": float(pose_logdet_target),
                "pose_min_eigenvalue_target": float(
                    pose_min_eigenvalue_target
                ),
                "pose_information_regularization": (
                    _POSE_INFORMATION_REGULARIZATION
                ),
                "pose_success": bool(te_cm < 5.0 and ae_deg < 5.0),
                "estimated_pose_w2c": pose,
                "te_cm": te_cm,
                "ae_deg": ae_deg,
                "query_descriptor_loo": bool(
                    feedback_observer_mode != "fixed_map"
                    and not descriptor_gradient_reused[query_index]
                ),
                "observer_descriptor_leave_self_out": bool(
                    feedback_observer_mode
                    in {"descriptor_leave_self_out", "full_loo"}
                ),
                "descriptor_training_query_reused": bool(
                    descriptor_gradient_reused[query_index]
                ),
                "descriptor_training_split_member": bool(
                    descriptor_training_split[query_index]
                ),
                "query_geometry_loo": bool(
                    feedback_observer_mode == "full_loo"
                    and not reconstruction_target_reused[query_index]
                ),
                "query_raw_geometry_observation_loo": (
                    feedback_observer_mode == "full_loo"
                ),
                "query_candidate_topology_loo": not bool(
                    reconstruction_training_reused[query_index]
                    or selection_training_reused[query_index]
                ),
                "reconstruction_target_query_reused": bool(
                    reconstruction_target_reused[query_index]
                ),
                "reconstruction_training_query_reused": bool(
                    reconstruction_training_reused[query_index]
                ),
                "selection_training_query_reused": bool(
                    selection_training_reused[query_index]
                ),
                "independent_mapping_validation_query": bool(
                    (
                        not descriptor_masks["descriptor_dependency_present"]
                        or (
                            descriptor_masks["training_registry_explicit"]
                            and not descriptor_training_split[query_index]
                        )
                    )
                    and not reconstruction_training_reused[query_index]
                    and not selection_training_reused[query_index]
                ),
                "pose_neighborhood_loo": (
                    feedback_observer_mode == "full_loo"
                    and int(excluded_queries.numel()) > 1
                ),
                "feedback_observer_mode": feedback_observer_mode,
                "affected_anchor_policy": update["contract"]["affected_anchor_policy"],
            }
        )
        query_rows.append(
            {
                "query_index": query_index,
                "image_name": name,
                "te_cm": te_cm,
                "ae_deg": ae_deg,
                "inliers": int(inliers.numel()),
                "clean_inliers": int(correct[inliers].sum()) if inliers.numel() else 0,
                "correct_winners": int(correct.sum()),
                "positive_rows": int(len(correct_anchor_ranks)),
                "identity_lineage_rows": sum(
                    int(bool(edges)) for edges in identity_partition["lineage"]
                ),
                "geometry_ambiguous_rows": sum(
                    int(bool(edges)) for edges in ambiguous_edges
                ),
                "certified_pose_valid_alternative_rows": sum(
                    int(bool(edges)) for edges in certified_alternatives
                ),
                "top1_exact_identity_correct_rows": int(correct.sum()),
                "top1_geometry_ambiguous_rows": int(ambiguous.sum()),
                "top1_identity_incompatible_rows": int(identity_incompatible.sum()),
                "top1_negative_rows": int(negative.sum()),
                "pose_information_duplicate_rows_removed": int(
                    clean_rows.numel() - pose_clean_rows.numel()
                ),
                "correct_anchor_rank_le_1": int(
                    sum(rank <= 1 for rank in correct_anchor_ranks)
                ),
                "correct_anchor_rank_le_16": int(
                    sum(rank <= 16 for rank in correct_anchor_ranks)
                ),
                "correspondences": int(winners.numel()),
                "pose_solves": 1,
                "poselib_iterations": int(estimate.diagnostics.get("iterations", 0)),
                "online_latency_ms": online_latency_ms,
                "loo_feedback_latency_ms": loo_latency_ms,
                "descriptor_training_query_reused": bool(
                    descriptor_gradient_reused[query_index]
                ),
                "descriptor_training_split_member": bool(
                    descriptor_training_split[query_index]
                ),
                "reconstruction_target_query_reused": bool(
                    reconstruction_target_reused[query_index]
                ),
                "reconstruction_training_query_reused": bool(
                    reconstruction_training_reused[query_index]
                ),
                "selection_training_query_reused": bool(
                    selection_training_reused[query_index]
                ),
                "independent_mapping_validation_query": bool(
                    (
                        not descriptor_masks["descriptor_dependency_present"]
                        or (
                            descriptor_masks["training_registry_explicit"]
                            and not descriptor_training_split[query_index]
                        )
                    )
                    and not reconstruction_training_reused[query_index]
                    and not selection_training_reused[query_index]
                ),
                "pose_information_rank": information_rank,
                "pose_information_logdet": information_logdet,
                "pose_information_min_eigenvalue": information_min_eigenvalue,
                "pose_information_sufficient": information_sufficient,
                "detectable_matching_pairs": detectable_pairs,
                "top1_correct_pairs": matching_pairs,
            }
        )
    feedback = build_self_localization_feedback(
        query_names=list(observations.names),
        records=feedback_records,
        required_rank=int(required_rank),
        required_visibility_rank=int(required_visibility_rank),
        required_detectable_rank=(
            int(required_rank)
            if required_detectable_rank is None
            else int(required_detectable_rank)
        ),
        source_map_sha256=source_map_sha256,
        query_cache_sha256=query_cache_sha256,
        scene_calibration_sha256=scene_calibration_sha256,
        feedback_calibration_binding_sha256=(
            feedback_calibration_binding_sha256
        ),
        positive_identity_contract=positive_identity_contract,
        pose_logdet_target=float(pose_logdet_target),
        pose_min_eigenvalue_target=float(pose_min_eigenvalue_target),
    )
    summary = _summary(query_rows)
    summary["anchor_count"] = int(base_xyz.shape[0])
    validation_rows = []
    if bool(descriptor_masks["training_registry_explicit"]):
        validation_rows = [
            row
            for row in query_rows
            if not bool(row["descriptor_training_split_member"])
        ]
    training_replay_rows = [
        row for row in query_rows if bool(row["descriptor_training_split_member"])
    ]
    gradient_reuse_rows = [
        row for row in query_rows if bool(row["descriptor_training_query_reused"])
    ]
    validation_summary = None if not validation_rows else _summary(validation_rows)
    independent_validation_rows = [
        row for row in query_rows if bool(row["independent_mapping_validation_query"])
    ]
    independent_validation_summary = (
        None
        if not independent_validation_rows
        else _summary(independent_validation_rows)
    )
    training_replay_summary = (
        None if not training_replay_rows else _summary(training_replay_rows)
    )
    gradient_reuse_summary = (
        None if not gradient_reuse_rows else _summary(gradient_reuse_rows)
    )
    reconstruction_replay_rows = [
        row for row in query_rows if bool(row["reconstruction_target_query_reused"])
    ]
    reconstruction_replay_summary = (
        None if not reconstruction_replay_rows else _summary(reconstruction_replay_rows)
    )
    selection_replay_rows = [
        row for row in query_rows if bool(row["selection_training_query_reused"])
    ]
    selection_replay_summary = (
        None if not selection_replay_rows else _summary(selection_replay_rows)
    )
    descriptor_query_loo = bool(
        feedback_observer_mode != "fixed_map"
        and not descriptor_gradient_reused.any()
    )
    return {
        "schema": "lafgs_v6_query_local_feedback_evaluation",
        "version": 6,
        "uses_source_mapping_rgb": False,
        "uses_test_queries": False,
        "queries": query_rows,
        "summary": summary,
        "descriptor_validation_summary": validation_summary,
        "independent_mapping_validation_summary": independent_validation_summary,
        "descriptor_training_replay_summary": training_replay_summary,
        "descriptor_gradient_reuse_summary": gradient_reuse_summary,
        "reconstruction_target_replay_summary": reconstruction_replay_summary,
        "selection_training_replay_summary": selection_replay_summary,
        "feedback": feedback,
        "contract": {
            "query_descriptor_loo": descriptor_query_loo,
            "descriptor_training_split_query_count": int(
                descriptor_training_split.sum()
            ),
            "descriptor_gradient_reuse_query_count": int(
                descriptor_gradient_reused.sum()
            ),
            "descriptor_validation_query_count": len(validation_rows),
            "independent_mapping_validation_query_count": len(
                independent_validation_rows
            ),
            "descriptor_training_registry_explicit": bool(
                descriptor_masks["training_registry_explicit"]
            ),
            "descriptor_dependency_present": bool(
                descriptor_masks["descriptor_dependency_present"]
            ),
            "independent_mapping_validation_available": bool(
                independent_validation_rows
            ),
            "query_geometry_loo": bool(
                feedback_observer_mode == "full_loo"
                and not reconstruction_target_reused.any()
            ),
            "query_raw_geometry_observation_loo": (
                feedback_observer_mode == "full_loo"
            ),
            "query_candidate_topology_loo": not bool(
                reconstruction_training_reused.any()
                or selection_training_reused.any()
            ),
            "reconstruction_target_reuse_query_count": int(
                reconstruction_target_reused.sum()
            ),
            "reconstruction_training_reuse_query_count": int(
                reconstruction_training_reused.sum()
            ),
            "selection_training_reuse_query_count": int(
                selection_training_reused.sum()
            ),
            "feedback_observer_mode": feedback_observer_mode,
            "deployment_plant_geometry_held_fixed": (
                feedback_observer_mode != "full_loo"
            ),
            "pose_neighborhood_loo": (
                feedback_observer_mode == "full_loo"
                and int(loo_pose_neighbors) > 1
            ),
            "loo_pose_neighbors": int(loo_pose_neighbors),
            "affected_anchor_policy": (
                "fixed_map"
                if feedback_observer_mode == "fixed_map"
                else "descriptor_only"
                if feedback_observer_mode == "descriptor_leave_self_out"
                else loo_affected_anchor_policy
            ),
            "positive_radius_px": float(positive_radius_px),
            "positive_identity": positive_identity_contract,
            "identity_supervision_unavailable_query_count": int(
                feedback["identity_supervision_unavailable_query_count"]
            ),
            "positive_radius_role": "projective_compatibility_and_ambiguity_ignore",
            "descriptor_strong_positives_are_exact_identity_only": True,
            "geometry_compatible_nonidentity_is_ignored": True,
            "certified_pose_valid_alternatives_require_depth": True,
            "uncertified_2d_neighbors_are_diagnostic_only": True,
            "descriptor_triplet_pose_weight_semantics": (
                DESCRIPTOR_POSE_WEIGHT_SEMANTICS
            ),
            "descriptor_triplet_pose_weight_formula": (
                "max(0,((I[pos@gt]-I[pos@current])-"
                "(I[neg@gt]-I[neg@current]))/2) when neg_is_deployed_winner"
            ),
            "descriptor_triplet_clean_semantics": (
                DESCRIPTOR_CLEAN_LABEL_SEMANTICS
            ),
            "descriptor_identity_supervision_available": (
                descriptor_identity_supervision_available
            ),
            "diagnostic_purge_suppresses_descriptor_triplets": (
                feedback_observer_mode == "full_loo"
                and loo_affected_anchor_policy == "purge"
            ),
            "estimated_pose_w2c_role": (
                "paired_winning_pose_diagnostic_only_not_training"
            ),
            "pose_information_anchor_unique": True,
            "pose_information_unique_row_policy": (
                "lowest_gt_reprojection_residual_then_highest_descriptor_score"
            ),
            "pose_information_task_translation_scale_m": (
                _POSE_TRANSLATION_SCALE_M
            ),
            "pose_information_task_rotation_scale_deg": (
                _POSE_ROTATION_SCALE_DEG
            ),
            "pose_information_regularization": _POSE_INFORMATION_REGULARIZATION,
            "pose_logdet_target": float(pose_logdet_target),
            "pose_min_eigenvalue_target": float(pose_min_eigenvalue_target),
            "pose_information_sufficiency": (
                "unregularized_rank_ge_6_and_task_scaled_regularized_"
                "logdet_and_min_eigenvalue_meet_targets"
            ),
            "alpha_minimum": float(alpha_minimum),
            "required_matching_rank": int(required_rank),
            "required_visibility_rank": int(required_visibility_rank),
            "required_detectable_rank": int(
                required_rank
                if required_detectable_rank is None
                else required_detectable_rank
            ),
            "ransac_reprojection_px": float(ransac_reprojection_px),
            "ransac_seed": int(seed),
            "evaluation_device": str(device),
            "affected_anchor_holdout_is_exact_rebuild": (
                feedback_observer_mode == "full_loo"
                and loo_affected_anchor_policy == "rebuild"
            ),
            "purged_holdout_is_exact_rebuild": (
                False
                if feedback_observer_mode == "full_loo"
                and loo_affected_anchor_policy == "purge"
                else None
            ),
            "reported_online_latency_is_oracle_feedback_localization": True,
            "global_top1": True,
            "pose_solves_per_query": 1,
            "retrieval": False,
            "refinement": False,
        },
    }
