"""View-conditioned failure atlas and active synthetic-view planning."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F

from localization_training.appearance_family import trajectory_id
from localization_training.candidate_basin_teacher import family_topk
from localization_training.shared_metric import SharedLowRankMetric


@dataclass(frozen=True)
class FailureAtlasConfig:
    topk: int = 16
    clean_threshold_px: float = 2.0
    coarse_clean_threshold_px: float = 4.0
    harmful_threshold_px: float = 12.0
    minimum_matchable_rate: float = 0.08
    minimum_render_topk_recall: float = 0.25
    assignment_gap: float = 0.08
    risk_tail_quantile: float = 0.75
    maximum_planned_views: int = 64
    maximum_views_per_source: int = 2
    maximum_views_per_trajectory: int = 16
    maximum_views_per_component: int = 8
    interpolation_alphas: tuple[float, ...] = (0.35, 0.5, 0.65)
    planner_mode: str = "viewpoint_completion"
    partner_candidates: int = 4
    minimum_normalized_view_gap: float = 0.75
    minimum_global_pose_novelty: float = 0.75
    maximum_normalized_pair_distance: float = 6.0
    maximum_pair_rotation_degrees: float = 55.0
    view_gap_weight: float = 0.75
    anchor_coverage_weight: float = 0.5
    artifact_risk_weight: float = 0.75


def _record_positive_sets(record: dict) -> dict[int, set[int]]:
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    positives = torch.as_tensor(record["positive_indices"]).long()
    if offsets.numel() != rows.numel() + 1:
        raise ValueError("complete-positive CSR offsets do not align")
    return {
        int(row): set(
            int(value)
            for value in positives[offsets[index] : offsets[index + 1]]
        )
        for index, row in enumerate(rows.tolist())
    }


def _entropy(values: torch.Tensor) -> float:
    values = torch.as_tensor(values).long()
    values = values[values >= 0]
    if not values.numel():
        return 0.0
    counts = torch.bincount(values).float()
    probability = counts[counts > 0] / counts.sum()
    return float(-(probability * probability.log()).sum())


def _dominant(values: torch.Tensor, default: int = -1) -> int:
    values = torch.as_tensor(values).long()
    values = values[values >= 0]
    if not values.numel():
        return int(default)
    return int(torch.bincount(values).argmax())


def _camera_depths(
    xyz: torch.Tensor,
    anchors: torch.Tensor,
    pose_w2c: torch.Tensor,
) -> torch.Tensor:
    points = xyz[anchors]
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    return camera[:, 2]


def _safe_rate(mask: torch.Tensor) -> float:
    return float(torch.as_tensor(mask).float().mean()) if mask.numel() else 0.0


def _basin_query_stats(record: dict | None) -> dict:
    if record is None:
        return {
            "basin_evidence_available": False,
            "strict_set_count": 0,
            "precision_set_count": 0,
            "harmful_set_count": 0,
            "repair_set_count": 0,
        }
    levels = torch.as_tensor(record["basin_level"]).long()
    types = torch.as_tensor(record["set_types"]).long()
    repair = torch.as_tensor(record["repair_order"]).long()
    return {
        "basin_evidence_available": True,
        "strict_set_count": int((levels == 3).sum()),
        "precision_set_count": int((levels >= 2).sum()),
        "harmful_set_count": int((types == 1).sum()),
        "repair_set_count": int((repair > 0).sum()),
    }


def _failure_class(record: dict, config: FailureAtlasConfig) -> str:
    if record["matchable_rate"] < float(config.minimum_matchable_rate):
        return "geometry_or_coverage_deficiency"
    if (
        record["positive_topk_recall"] - record["legal_top1_recall"]
        >= float(config.assignment_gap)
    ):
        return "repeated_assignment_deficiency"
    if (
        record["basin_evidence_available"]
        and
        record["strict_set_count"] == 0
        and record["positive_topk_recall"]
        >= float(config.minimum_render_topk_recall)
    ):
        return "basin_composition_deficiency"
    return "view_appearance_deficiency"


def _risk_scores(records: list[dict]) -> None:
    if not records:
        return
    hypotheses = np.asarray(
        [math.log1p(max(float(value["hypotheses"]), 0.0)) for value in records]
    )
    hypothesis_scale = max(float(np.median(hypotheses)), 1e-6)
    harmful = np.asarray(
        [float(value["harmful_set_count"]) for value in records]
    )
    positive_harmful = harmful[harmful > 0]
    harmful_scale = (
        max(float(np.median(positive_harmful)), 1.0)
        if positive_harmful.size
        else 1.0
    )
    for record, log_hypotheses in zip(records, hypotheses.tolist()):
        record["risk_terms"] = {
            "translation": min(float(record["te_cm"]) / 50.0, 4.0),
            "hypotheses": min(log_hypotheses / hypothesis_scale, 3.0),
            "raw_impurity": 1.0 - float(
                record["raw_gt_precision_coarse"]
            ),
            "harmful_basin": min(
                float(record["harmful_set_count"]) / harmful_scale, 3.0
            ),
            "strict_deficit": float(
                record["basin_evidence_available"]
                and record["strict_set_count"] == 0
            ),
        }
        terms = record["risk_terms"]
        record["risk"] = float(
            terms["translation"]
            + 0.35 * terms["hypotheses"]
            + terms["raw_impurity"]
            + 0.5 * terms["harmful_basin"]
            + 0.5 * terms["strict_deficit"]
        )


def _depth_bins(records: list[dict]) -> None:
    depths = np.asarray(
        [float(record["median_anchor_depth"]) for record in records],
        dtype=np.float64,
    )
    finite = depths[np.isfinite(depths) & (depths > 0)]
    thresholds = (
        np.quantile(finite, [0.25, 0.5, 0.75])
        if finite.size
        else np.asarray([1.0, 2.0, 4.0])
    )
    for record in records:
        depth = float(record["median_anchor_depth"])
        record["scale_bin"] = int(
            np.searchsorted(thresholds, depth, side="right")
            if np.isfinite(depth)
            else 4
        )


def _aggregate_cells(records: list[dict]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for record in records:
        key = (
            int(record["source_component"]),
            int(record["view_bin"]),
            int(record["scale_bin"]),
            str(record["appearance_group"]),
        )
        grouped[key].append(record)
    cells = []
    for key, values in grouped.items():
        failure_counts = Counter(
            str(value["failure_class"]) for value in values
        )
        cells.append(
            {
                "cell": {
                    "source_component": key[0],
                    "view_bin": key[1],
                    "scale_bin": key[2],
                    "appearance_group": key[3],
                },
                "query_count": len(values),
                "query_names": [value["query_name"] for value in values],
                "risk_mean": float(
                    np.mean([value["risk"] for value in values])
                ),
                "risk_max": float(
                    np.max([value["risk"] for value in values])
                ),
                "te_cm_mean": float(
                    np.mean([value["te_cm"] for value in values])
                ),
                "raw_gt_precision_coarse_mean": float(
                    np.mean(
                        [
                            value["raw_gt_precision_coarse"]
                            for value in values
                        ]
                    )
                ),
                "positive_topk_recall_mean": float(
                    np.mean(
                        [value["positive_topk_recall"] for value in values]
                    )
                ),
                "positive_topk_rate_mean": float(
                    np.mean(
                        [value["positive_topk_rate"] for value in values]
                    )
                ),
                "strict_query_fraction": float(
                    np.mean(
                        [value["strict_set_count"] > 0 for value in values]
                    )
                ),
                "render_eligible_fraction": float(
                    np.mean([value["render_eligible"] for value in values])
                ),
                "failure_counts": dict(sorted(failure_counts.items())),
            }
        )
    cells.sort(
        key=lambda value: (
            -value["risk_mean"],
            -value["query_count"],
            value["cell"]["source_component"],
        )
    )
    return cells


def build_failure_atlas(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict | None,
    dynamic: dict,
    positives: dict,
    cache: dict,
    query_bins: dict[str, int],
    basin_teacher: dict | None,
    config: FailureAtlasConfig,
    device: torch.device,
    progress=None,
) -> dict:
    """Audit real self-localization outcomes under the deployed matcher."""
    names = list(dynamic["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("failure atlas query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(dynamic["anchor_count"]) != anchor_count:
        raise ValueError("dynamic outcomes do not align with atlas map")
    if int(positives["anchor_count"]) != anchor_count:
        raise ValueError("positive teacher does not align with atlas map")
    teacher_by_name = (
        {
            str(record["query_name"]): record
            for record in basin_teacher["records"]
        }
        if basin_teacher is not None
        else {}
    )
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    xyz = torch.as_tensor(state["anchor_xyz"]).float()
    component_ids = torch.as_tensor(
        state.get(
            "coarse_dependency_group_ids",
            state.get(
                "dependency_group_ids",
                state["source_primitive_ids"],
            ),
        )
    ).long()
    metric = metric.to(device).eval()
    records = []
    with torch.no_grad():
        for query_index, name in enumerate(names):
            cached = cache[name]
            outcome = dynamic["records"][query_index]
            rows = torch.as_tensor(outcome["query_rows"]).long()
            raw = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float()[rows],
                dim=1,
            ).to(device)
            query, _ = metric(raw)
            if family is None:
                _, top_anchors = torch.topk(
                    query @ bank.T,
                    k=min(int(config.topk), int(bank.shape[0])),
                    dim=1,
                )
                top_modes = torch.full_like(top_anchors, -1)
            else:
                _, top_anchors, top_modes, _ = family_topk(
                    query, bank, family, config.topk
                )
            top_anchors = top_anchors.cpu()
            top_modes = top_modes.cpu()
            positives_by_row = _record_positive_sets(
                positives["records"][query_index]
            )
            legal_top1 = torch.as_tensor(
                [
                    int(top_anchors[index, 0])
                    in positives_by_row.get(int(row), set())
                    for index, row in enumerate(rows.tolist())
                ],
                dtype=torch.bool,
            )
            legal_topk = torch.as_tensor(
                [
                    bool(
                        set(int(value) for value in top_anchors[index].tolist())
                        & positives_by_row.get(int(row), set())
                    )
                    for index, row in enumerate(rows.tolist())
                ],
                dtype=torch.bool,
            )
            matchable = torch.as_tensor(
                [
                    bool(positives_by_row.get(int(row), set()))
                    for row in rows.tolist()
                ],
                dtype=torch.bool,
            )
            gt_errors = torch.as_tensor(
                outcome["gt_reprojection_errors_px"]
            ).float()
            inlier = torch.as_tensor(outcome["ransac_inlier_mask"]).bool()
            top1 = top_anchors[:, 0]
            harmful = gt_errors > float(config.harmful_threshold_px)
            components = component_ids[top1]
            source_component = _dominant(
                components[harmful]
                if bool(harmful.any())
                else components
            )
            depths = _camera_depths(
                xyz,
                top1,
                torch.as_tensor(cached["pose_w2c"]).float(),
            )
            basin_stats = _basin_query_stats(teacher_by_name.get(name))
            record = {
                "query_index": query_index,
                "query_name": str(name),
                "appearance_group": trajectory_id(name),
                "view_bin": int(query_bins[name]),
                "source_component": source_component,
                "te_cm": float(outcome["te_cm"]),
                "re_deg": float(outcome["re_deg"]),
                "hypotheses": int(outcome.get("hypotheses", 0)),
                "query_row_count": int(rows.numel()),
                "matchable_rate": _safe_rate(matchable),
                "legal_top1_rate": _safe_rate(legal_top1),
                "legal_top1_recall": _safe_rate(legal_top1[matchable]),
                "positive_topk_rate": _safe_rate(legal_topk),
                "positive_topk_recall": _safe_rate(legal_topk[matchable]),
                "raw_gt_precision": _safe_rate(
                    gt_errors <= float(config.clean_threshold_px)
                ),
                "raw_gt_precision_coarse": _safe_rate(
                    gt_errors <= float(config.coarse_clean_threshold_px)
                ),
                "inlier_gt_precision": _safe_rate(
                    gt_errors[inlier] <= float(config.clean_threshold_px)
                ),
                "inlier_gt_precision_coarse": _safe_rate(
                    gt_errors[inlier]
                    <= float(config.coarse_clean_threshold_px)
                ),
                "harmful_match_rate": _safe_rate(harmful),
                "family_activation_rate": _safe_rate(top_modes[:, 0] >= 0),
                "family_mode_entropy": _entropy(top_modes[:, 0]),
                "median_anchor_depth": float(
                    depths[depths > 0].median()
                    if bool((depths > 0).any())
                    else float("nan")
                ),
                **basin_stats,
            }
            record["failure_class"] = _failure_class(record, config)
            record["render_eligible"] = bool(
                record["failure_class"]
                != "geometry_or_coverage_deficiency"
                and source_component >= 0
            )
            records.append(record)
            if progress is not None:
                progress(query_index + 1, len(names))
    _depth_bins(records)
    _risk_scores(records)
    risk_threshold = float(
        np.quantile(
            [record["risk"] for record in records],
            float(config.risk_tail_quantile),
        )
    )
    for record in records:
        record["high_risk"] = bool(record["risk"] >= risk_threshold)
        record["render_eligible"] = bool(
            record["render_eligible"] and record["high_risk"]
        )
    cells = _aggregate_cells(records)
    failures = Counter(record["failure_class"] for record in records)
    return {
        "schema": "lafgs_view_conditioned_failure_atlas",
        "version": 2,
        "matcher": "base_metric" if family is None else "appearance_family",
        "query_names": names,
        "anchor_count": anchor_count,
        "records": records,
        "cells": cells,
        "summary": {
            "query_count": len(records),
            "cell_count": len(cells),
            "risk_threshold": risk_threshold,
            "render_eligible_query_count": sum(
                record["render_eligible"] for record in records
            ),
            "failure_counts": dict(sorted(failures.items())),
            "te_cm_median": float(
                np.median([record["te_cm"] for record in records])
            ),
            "te_cm_mean": float(
                np.mean([record["te_cm"] for record in records])
            ),
            "raw_gt_precision_coarse_mean": float(
                np.mean(
                    [
                        record["raw_gt_precision_coarse"]
                        for record in records
                    ]
                )
            ),
            "positive_topk_recall_mean": float(
                np.mean(
                    [record["positive_topk_recall"] for record in records]
                )
            ),
            "positive_topk_rate_mean": float(
                np.mean([record["positive_topk_rate"] for record in records])
            ),
            "strict_query_fraction": float(
                np.mean(
                    [record["strict_set_count"] > 0 for record in records]
                )
            ),
        },
        "config": asdict(config),
    }


def _frame_number(name: str) -> int:
    match = re.search(r"(\d+)(?!.*\d)", str(name))
    return int(match.group(1)) if match else -1


def _orthogonalized_rotation(value: torch.Tensor) -> torch.Tensor:
    u, _, vh = torch.linalg.svd(value)
    rotation = u @ vh
    if float(torch.det(rotation)) < 0:
        u = u.clone()
        u[:, -1] *= -1
        rotation = u @ vh
    return rotation


def interpolate_pose_w2c(
    first: torch.Tensor, second: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Interpolate camera centers and project rotation back to SO(3)."""
    first_c2w = torch.linalg.inv(torch.as_tensor(first).double())
    second_c2w = torch.linalg.inv(torch.as_tensor(second).double())
    alpha = float(alpha)
    result = torch.eye(4, dtype=torch.float64)
    result[:3, :3] = _orthogonalized_rotation(
        (1.0 - alpha) * first_c2w[:3, :3]
        + alpha * second_c2w[:3, :3]
    )
    result[:3, 3] = (
        (1.0 - alpha) * first_c2w[:3, 3]
        + alpha * second_c2w[:3, 3]
    )
    return torch.linalg.inv(result).float()


def _camera_center_and_forward(pose_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = torch.linalg.inv(torch.as_tensor(pose_w2c).double())
    return c2w[:3, 3], F.normalize(c2w[:3, 2].float(), dim=0).double()


def _rotation_gap_degrees(first: torch.Tensor, second: torch.Tensor) -> float:
    cosine = float(torch.dot(first, second).clamp(-1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _mapping_view_scale(names: list[str], cache: dict) -> float:
    """Estimate a scene-independent local camera-spacing scale."""
    by_trajectory: dict[str, list[str]] = defaultdict(list)
    for name in names:
        by_trajectory[trajectory_id(name)].append(name)
    distances = []
    for sequence in by_trajectory.values():
        sequence.sort(key=lambda value: (_frame_number(value), value))
        for left, right in zip(sequence, sequence[1:]):
            left_center, _ = _camera_center_and_forward(cache[left]["pose_w2c"])
            right_center, _ = _camera_center_and_forward(cache[right]["pose_w2c"])
            distance = float(torch.linalg.norm(left_center - right_center))
            if math.isfinite(distance) and distance > 1e-6:
                distances.append(distance)
    if distances:
        return max(float(np.median(distances)), 1e-4)
    centers = torch.stack(
        [_camera_center_and_forward(cache[name]["pose_w2c"])[0] for name in names]
    )
    if centers.shape[0] < 2:
        return 1.0
    pairwise = torch.cdist(centers.float(), centers.float())
    pairwise.fill_diagonal_(torch.inf)
    return max(float(pairwise.amin(dim=1).median()), 1e-4)


def plan_failure_conditioned_views(
    *,
    atlas: dict,
    cache: dict,
    config: FailureAtlasConfig,
) -> list[dict]:
    """Plan in-envelope views that complete missing localization viewpoints."""
    records = {
        str(record["query_name"]): record for record in atlas["records"]
    }
    by_trajectory: dict[str, list[str]] = defaultdict(list)
    for name in atlas["query_names"]:
        by_trajectory[trajectory_id(name)].append(str(name))
    for names in by_trajectory.values():
        names.sort(key=lambda value: (_frame_number(value), value))
    all_names = [str(name) for name in atlas["query_names"]]
    view_scale = _mapping_view_scale(all_names, cache)
    centers = {}
    forwards = {}
    for name in all_names:
        centers[name], forwards[name] = _camera_center_and_forward(
            cache[name]["pose_w2c"]
        )
    candidates = []
    for name, failure in records.items():
        if not bool(failure["render_eligible"]):
            continue
        source_trajectory = trajectory_id(name)
        if str(config.planner_mode) == "adjacent":
            sequence = by_trajectory[source_trajectory]
            position = sequence.index(name)
            neighbors = []
            if position > 0:
                neighbors.append(sequence[position - 1])
            if position + 1 < len(sequence):
                neighbors.append(sequence[position + 1])
        elif str(config.planner_mode) == "viewpoint_completion":
            partner_scores = []
            for partner in all_names:
                if partner == name:
                    continue
                partner_failure = records[partner]
                different_trajectory = trajectory_id(partner) != source_trajectory
                different_bin = int(partner_failure["view_bin"]) != int(
                    failure["view_bin"]
                )
                if not different_trajectory and not different_bin:
                    continue
                normalized_distance = float(
                    torch.linalg.norm(centers[name] - centers[partner])
                ) / view_scale
                rotation_gap = _rotation_gap_degrees(
                    forwards[name], forwards[partner]
                )
                if normalized_distance > float(
                    config.maximum_normalized_pair_distance
                ):
                    continue
                if rotation_gap > float(config.maximum_pair_rotation_degrees):
                    continue
                view_gap = (
                    min(normalized_distance / 3.0, 2.0)
                    + min(
                        rotation_gap
                        / max(float(config.maximum_pair_rotation_degrees), 1e-6),
                        1.0,
                    )
                    + 0.5 * float(different_trajectory)
                    + 0.25 * float(different_bin)
                )
                if view_gap < float(config.minimum_normalized_view_gap):
                    continue
                artifact_risk = (
                    max(normalized_distance - 3.0, 0.0) / 3.0
                    + rotation_gap
                    / max(float(config.maximum_pair_rotation_degrees), 1e-6)
                )
                partner_scores.append(
                    (view_gap - artifact_risk, view_gap, artifact_risk, partner)
                )
            partner_scores.sort(key=lambda value: (-value[0], value[3]))
            neighbors = [
                value[3]
                for value in partner_scores[: max(int(config.partner_candidates), 1)]
            ]
        else:
            raise ValueError(f"unsupported failure-view planner mode: {config.planner_mode}")
        for neighbor in neighbors:
            normalized_distance = float(
                torch.linalg.norm(centers[name] - centers[neighbor])
            ) / view_scale
            rotation_gap = _rotation_gap_degrees(
                forwards[name], forwards[neighbor]
            )
            cross_trajectory = trajectory_id(neighbor) != source_trajectory
            for alpha in config.interpolation_alphas:
                left, right = name, neighbor
                pose = interpolate_pose_w2c(
                    cache[left]["pose_w2c"],
                    cache[right]["pose_w2c"],
                    alpha,
                )
                synthetic_center, synthetic_forward = _camera_center_and_forward(pose)
                global_pose_gaps = [
                    float(
                        torch.linalg.norm(synthetic_center - centers[real_name])
                    )
                    / view_scale
                    + _rotation_gap_degrees(
                        synthetic_forward, forwards[real_name]
                    )
                    / max(float(config.maximum_pair_rotation_degrees), 1e-6)
                    for real_name in all_names
                ]
                nearest_real_pose_gap = min(global_pose_gaps)
                if nearest_real_pose_gap < float(
                    config.minimum_global_pose_novelty
                ):
                    continue
                height, width = cache[left]["native_input_hw"]
                K = torch.as_tensor(cache[left]["native_K"]).float()
                interpolation_novelty = min(float(alpha), 1.0 - float(alpha))
                view_gap = (
                    interpolation_novelty * normalized_distance
                    + interpolation_novelty
                    * rotation_gap
                    / max(float(config.maximum_pair_rotation_degrees), 1e-6)
                    + 0.5 * float(cross_trajectory)
                )
                coverage_need = 1.0 - float(failure["positive_topk_recall"])
                artifact_risk = (
                    max(normalized_distance - 3.0, 0.0) / 3.0
                    + rotation_gap
                    / max(float(config.maximum_pair_rotation_degrees), 1e-6)
                    + 0.25 * abs(float(alpha) - 0.5)
                )
                assignment_need = max(
                    failure["positive_topk_recall"]
                    - failure["legal_top1_recall"],
                    0.0,
                )
                acquisition = float(failure["risk"]) * (
                    1.0
                    + assignment_need
                )
                acquisition += float(config.view_gap_weight) * view_gap
                acquisition += float(config.anchor_coverage_weight) * coverage_need
                acquisition -= float(config.artifact_risk_weight) * artifact_risk
                candidates.append(
                    {
                        "query_id": (
                            f"failure_render:{left}:{neighbor}:"
                            f"{float(alpha):.2f}"
                        ),
                        "source_query": left,
                        "neighbor_query": right,
                        "synthetic_alpha": float(alpha),
                        "pose_w2c": pose.tolist(),
                        "width": int(width),
                        "height": int(height),
                        "fovx": float(
                            2.0
                            * math.atan(
                                float(width) / (2.0 * float(K[0, 0]))
                            )
                        ),
                        "fovy": float(
                            2.0
                            * math.atan(
                                float(height) / (2.0 * float(K[1, 1]))
                            )
                        ),
                        "K": K,
                        "view_bin": int(failure["view_bin"]),
                        "source_component": int(
                            failure["source_component"]
                        ),
                        "failure_class": failure["failure_class"],
                        "risk": float(failure["risk"]),
                        "view_gap": float(view_gap),
                        "anchor_coverage_need": float(coverage_need),
                        "artifact_risk": float(artifact_risk),
                        "normalized_pair_distance": float(normalized_distance),
                        "pair_rotation_degrees": float(rotation_gap),
                        "cross_trajectory": bool(cross_trajectory),
                        "nearest_real_pose_gap": float(nearest_real_pose_gap),
                        "planner_mode": str(config.planner_mode),
                        "acquisition": acquisition,
                    }
                )
    candidates.sort(
        key=lambda value: (
            -value["acquisition"],
            value["query_id"],
        )
    )
    selected = []
    used_pairs = set()
    source_counts = Counter()
    trajectory_counts = Counter()
    component_counts = Counter()
    for candidate in candidates:
        pair = (
            candidate["source_query"],
            candidate["neighbor_query"],
            candidate["synthetic_alpha"],
        )
        if pair in used_pairs:
            continue
        source = str(candidate["source_query"])
        trajectory = trajectory_id(source)
        component = int(candidate["source_component"])
        if source_counts[source] >= int(config.maximum_views_per_source):
            continue
        if trajectory_counts[trajectory] >= int(
            config.maximum_views_per_trajectory
        ):
            continue
        if component_counts[component] >= int(
            config.maximum_views_per_component
        ):
            continue
        used_pairs.add(pair)
        selected.append(candidate)
        source_counts[source] += 1
        trajectory_counts[trajectory] += 1
        component_counts[component] += 1
        if len(selected) >= int(config.maximum_planned_views):
            break
    return selected
