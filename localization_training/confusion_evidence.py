"""Confusion-conditioned evidence for active localization reconstruction."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.appearance_family import trajectory_id
from localization_training.candidate_basin_teacher import project_errors
from localization_training.failure_atlas import interpolate_pose_w2c
from localization_training.shared_metric import SharedLowRankMetric
from localization_training.synthetic_evidence import (
    SyntheticEvidenceConfig,
    project_existing_anchors,
    render_visible_anchor_mask,
)


@dataclass(frozen=True)
class ConfusionGraphConfig:
    minimum_occurrences: int = 2
    minimum_trajectories: int = 1
    harmful_threshold_px: float = 12.0
    clean_threshold_px: float = 4.0
    maximum_events_per_edge: int = 64


@dataclass(frozen=True)
class ContrastiveEvidenceConfig:
    strong_radius_px: float = 2.0
    ambiguous_radius_px: float = 6.0
    maximum_positives_per_keypoint: int = 4
    maximum_negatives_per_keypoint: int = 4
    maximum_negative_score_gap: float = 0.12
    minimum_hard_negative_pairs: int = 8
    minimum_edge_occurrences: int = 2
    require_negative_visibility: bool = False
    require_distinct_source_primitive: bool = True
    require_distinct_dependency_group: bool = True
    restrict_strong_to_active_target: bool = False


@dataclass(frozen=True)
class ConfusionViewPlanningConfig:
    maximum_planned_views: int = 64
    maximum_edges: int = 256
    maximum_events_per_edge: int = 3
    maximum_pose_neighbors: int = 4
    maximum_views_per_edge: int = 2
    maximum_views_per_source: int = 2
    maximum_views_per_trajectory: int = 16
    minimum_edge_occurrences: int = 5
    minimum_edge_trajectories: int = 1
    maximum_neighbor_scale: float = 4.0
    maximum_view_angle_deg: float = 40.0
    image_margin_px: float = 16.0
    interpolation_alphas: tuple[float, ...] = (0.35, 0.5, 0.65)
    cross_trajectory_bonus: float = 0.35
    different_view_bin_bonus: float = 0.15
    confusing_family_visible_bonus: float = 0.10


def _positive_lookup(record: dict) -> dict[int, list[int]]:
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    if offsets.numel() != rows.numel() + 1:
        raise ValueError("positive teacher CSR is malformed")
    return {
        int(row): [
            int(value)
            for value in indices[offsets[index] : offsets[index + 1]]
        ]
        for index, row in enumerate(rows.tolist())
    }


def family_pair_scores(
    query: torch.Tensor,
    anchors: torch.Tensor,
    bank: torch.Tensor,
    family: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score specified query-anchor pairs under deployed family retrieval."""
    query = torch.as_tensor(query)
    anchors = torch.as_tensor(anchors, device=query.device).long()
    bank = torch.as_tensor(bank, device=query.device)
    if query.ndim != 2 or anchors.ndim != 1 or len(query) != len(anchors):
        raise ValueError("family pair scoring expects aligned query-anchor rows")
    scores = (query * bank[anchors]).sum(dim=1)
    modes = torch.full_like(anchors, -1)
    prototypes = F.normalize(
        torch.as_tensor(family["prototype_features"]).float(), dim=1
    ).to(query.device)
    parents = torch.as_tensor(
        family["prototype_anchor_indices"], device=query.device
    ).long()
    bias = torch.as_tensor(
        family.get("prototype_bias", torch.zeros(len(prototypes))),
        device=query.device,
    ).float()
    temperature = torch.as_tensor(
        family.get("prototype_temperature", torch.ones(len(prototypes))),
        device=query.device,
    ).float()
    for parent in anchors.unique().tolist():
        rows = torch.nonzero(anchors == int(parent), as_tuple=False).reshape(-1)
        family_modes = torch.nonzero(
            parents == int(parent), as_tuple=False
        ).reshape(-1)
        if not family_modes.numel():
            continue
        values = query[rows] @ prototypes[family_modes].T
        values = values / temperature[family_modes][None] + bias[
            family_modes
        ][None]
        best, local = values.max(dim=1)
        improve = best > scores[rows]
        if bool(improve.any()):
            selected = rows[improve]
            scores[selected] = best[improve]
            modes[selected] = family_modes[local[improve]]
    return scores, modes


def _best_legal_assignments(
    *,
    query: torch.Tensor,
    rows: torch.Tensor,
    positive_lookup: dict[int, list[int]],
    bank: torch.Tensor,
    family: dict,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pair_rows = []
    pair_anchors = []
    pair_slots = []
    for slot, row in enumerate(rows.tolist()):
        for anchor in positive_lookup.get(int(row), []):
            pair_rows.append(slot)
            pair_anchors.append(anchor)
            pair_slots.append(slot)
    output_anchor = torch.full((len(rows),), -1, dtype=torch.long)
    output_score = torch.full((len(rows),), -torch.inf)
    output_mode = torch.full((len(rows),), -1, dtype=torch.long)
    if not pair_rows:
        return output_anchor, output_score, output_mode
    pair_rows_tensor = torch.as_tensor(pair_rows, device=query.device).long()
    pair_anchor_tensor = torch.as_tensor(
        pair_anchors, device=query.device
    ).long()
    scores, modes = family_pair_scores(
        query[pair_rows_tensor], pair_anchor_tensor, bank, family
    )
    for pair_index, slot in enumerate(pair_slots):
        score = float(scores[pair_index])
        if score > float(output_score[slot]):
            output_score[slot] = score
            output_anchor[slot] = int(pair_anchors[pair_index])
            output_mode[slot] = int(modes[pair_index])
    return output_anchor, output_score, output_mode


def _image_cell(
    point: torch.Tensor, *, width: int, height: int, columns: int = 8, rows: int = 6
) -> int:
    x = min(max(int(float(point[0]) / max(width, 1) * columns), 0), columns - 1)
    y = min(max(int(float(point[1]) / max(height, 1) * rows), 0), rows - 1)
    return y * columns + x


def _camera_center_and_forward(pose_w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    c2w = torch.linalg.inv(torch.as_tensor(pose_w2c).double())
    center = c2w[:3, 3]
    forward = c2w[:3, 2]
    return center, forward / torch.linalg.vector_norm(forward).clamp_min(1e-8)


def _view_angle_deg(first: torch.Tensor, second: torch.Tensor) -> float:
    cosine = torch.dot(first, second).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cosine)))


def _project_pair(
    *,
    xyz: torch.Tensor,
    anchors: tuple[int, int],
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    width: int,
    height: int,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = xyz[torch.as_tensor(anchors).long()]
    camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
    depth = camera[:, 2]
    projected = camera[:, :2] / depth[:, None].clamp_min(1e-8)
    projected = projected @ K[:2, :2].T + K[:2, 2]
    visible = (
        (depth > 1e-4)
        & (projected[:, 0] >= float(margin))
        & (projected[:, 0] < float(width) - float(margin))
        & (projected[:, 1] >= float(margin))
        & (projected[:, 1] < float(height) - float(margin))
    )
    return projected, visible


def plan_confusion_conditioned_views(
    *,
    confusion_graph: dict,
    state: dict,
    cache: dict,
    query_bins: dict[str, int],
    config: ConfusionViewPlanningConfig,
) -> list[dict]:
    """Plan safe pose-space views for directed family-confusion cells."""
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(confusion_graph["anchor_count"]) != anchor_count:
        raise ValueError("confusion view planner map does not align with graph")
    names = [
        str(name)
        for name in confusion_graph["query_names"]
        if str(name) in cache
    ]
    if not names:
        raise ValueError("confusion view planner has no cached cameras")
    missing_bins = [name for name in names if name not in query_bins]
    if missing_bins:
        raise ValueError(
            f"confusion view planner misses view bins for {missing_bins[:3]}"
        )
    centers = []
    forwards = []
    for name in names:
        center, forward = _camera_center_and_forward(cache[name]["pose_w2c"])
        centers.append(center)
        forwards.append(forward)
    centers = torch.stack(centers)
    forwards = torch.stack(forwards)
    distances = torch.cdist(centers.float(), centers.float()).double()
    distances.fill_diagonal_(float("inf"))
    finite_neighbors = torch.topk(
        distances,
        k=min(8, max(len(names) - 1, 1)),
        dim=1,
        largest=False,
    ).values
    local_scales = torch.where(
        torch.isfinite(finite_neighbors),
        finite_neighbors,
        torch.nan,
    ).nanmedian(dim=1).values.clamp_min(1e-3)
    name_to_index = {name: index for index, name in enumerate(names)}
    events_by_edge: dict[int, list[dict]] = defaultdict(list)
    for event in confusion_graph["events"]:
        if str(event["query_name"]) in name_to_index:
            events_by_edge[int(event["edge_index"])].append(event)
    edges = [
        edge
        for edge in confusion_graph["edges"]
        if int(edge["occurrences"]) >= int(config.minimum_edge_occurrences)
        and int(edge["trajectory_count"])
        >= int(config.minimum_edge_trajectories)
    ][: int(config.maximum_edges)]
    xyz = torch.as_tensor(state["anchor_xyz"]).double()
    candidates = []
    for edge in edges:
        edge_index = int(edge["edge_index"])
        events = sorted(
            events_by_edge.get(edge_index, []),
            key=lambda value: (
                -float(value["pose_blame"]),
                -float(value["score_margin"]),
                str(value["query_name"]),
                int(value["query_row"]),
            ),
        )[: int(config.maximum_events_per_edge)]
        for event in events:
            source = str(event["query_name"])
            source_index = name_to_index[source]
            source_scale = float(local_scales[source_index])
            neighbor_order = torch.argsort(distances[source_index]).tolist()
            neighbors = []
            for neighbor_index in neighbor_order:
                distance = float(distances[source_index, neighbor_index])
                if not math.isfinite(distance):
                    continue
                if distance > float(config.maximum_neighbor_scale) * source_scale:
                    break
                angle = _view_angle_deg(
                    forwards[source_index], forwards[neighbor_index]
                )
                if angle > float(config.maximum_view_angle_deg):
                    continue
                neighbors.append((neighbor_index, distance, angle))
                if len(neighbors) >= int(config.maximum_pose_neighbors):
                    break
            cached = cache[source]
            height, width = (
                int(value) for value in cached["native_input_hw"]
            )
            K = torch.as_tensor(cached["native_K"]).double()
            anchors = (
                int(edge["correct_anchor"]),
                int(edge["confusing_anchor"]),
            )
            for neighbor_index, distance, angle in neighbors:
                neighbor = names[neighbor_index]
                cross_trajectory = (
                    trajectory_id(source) != trajectory_id(neighbor)
                )
                different_view_bin = (
                    int(query_bins[source]) != int(query_bins[neighbor])
                )
                for alpha in config.interpolation_alphas:
                    pose = interpolate_pose_w2c(
                        cached["pose_w2c"],
                        cache[neighbor]["pose_w2c"],
                        float(alpha),
                    ).double()
                    projected, visible = _project_pair(
                        xyz=xyz,
                        anchors=anchors,
                        pose_w2c=pose,
                        K=K,
                        width=width,
                        height=height,
                        margin=float(config.image_margin_px),
                    )
                    if not bool(visible[0]):
                        continue
                    baseline_ratio = (
                        float(alpha) * distance / max(source_scale, 1e-6)
                    )
                    pair_separation = float(
                        torch.linalg.vector_norm(
                            projected[0] - projected[1]
                        )
                        / max(math.hypot(width, height), 1.0)
                    )
                    utility = (
                        math.log1p(max(float(edge["weight"]), 0.0))
                        * (1.0 + min(baseline_ratio, 2.0))
                        * (
                            1.0
                            + float(config.cross_trajectory_bonus)
                            * float(cross_trajectory)
                            + float(config.different_view_bin_bonus)
                            * float(different_view_bin)
                            + float(config.confusing_family_visible_bonus)
                            * float(visible[1])
                        )
                        * (1.0 + min(pair_separation, 0.5))
                    )
                    candidates.append(
                        {
                            "query_id": (
                                f"confusion_render:e{edge_index}:"
                                f"{source}:{neighbor}:{float(alpha):.2f}"
                            ),
                            "source_query": source,
                            "neighbor_query": neighbor,
                            "synthetic_alpha": float(alpha),
                            "pose_w2c": pose.float().tolist(),
                            "width": width,
                            "height": height,
                            "fovx": float(
                                2.0
                                * math.atan(
                                    float(width)
                                    / (2.0 * float(K[0, 0]))
                                )
                            ),
                            "fovy": float(
                                2.0
                                * math.atan(
                                    float(height)
                                    / (2.0 * float(K[1, 1]))
                                )
                            ),
                            "K": K.float(),
                            "edge_index": edge_index,
                            "correct_anchor": anchors[0],
                            "confusing_anchor": anchors[1],
                            "image_cell": int(event["image_cell"]),
                            "view_bin": int(query_bins[source]),
                            "source_view_bin": int(query_bins[source]),
                            "neighbor_view_bin": int(query_bins[neighbor]),
                            "cross_trajectory": bool(cross_trajectory),
                            "source_neighbor_distance": distance,
                            "source_local_scale": source_scale,
                            "view_angle_deg": angle,
                            "correct_family_visible": bool(visible[0]),
                            "confusing_family_visible": bool(visible[1]),
                            "projected_pair_separation": pair_separation,
                            "edge_weight": float(edge["weight"]),
                            "edge_occurrences": int(edge["occurrences"]),
                            "edge_trajectory_count": int(
                                edge["trajectory_count"]
                            ),
                            "pose_blame": float(event["pose_blame"]),
                            "acquisition": float(utility),
                            "planning_policy": (
                                "confusion_cell_pose_space_neighbor"
                            ),
                        }
                    )
    candidates.sort(
        key=lambda value: (-value["acquisition"], value["query_id"])
    )
    selected = []
    edge_counts = Counter()
    source_counts = Counter()
    trajectory_counts = Counter()
    seen_poses = set()
    for candidate in candidates:
        edge = int(candidate["edge_index"])
        source = str(candidate["source_query"])
        trajectory = trajectory_id(source)
        pose_key = (
            source,
            str(candidate["neighbor_query"]),
            float(candidate["synthetic_alpha"]),
        )
        if pose_key in seen_poses:
            continue
        if edge_counts[edge] >= int(config.maximum_views_per_edge):
            continue
        if source_counts[source] >= int(config.maximum_views_per_source):
            continue
        if trajectory_counts[trajectory] >= int(
            config.maximum_views_per_trajectory
        ):
            continue
        selected.append(candidate)
        seen_poses.add(pose_key)
        edge_counts[edge] += 1
        source_counts[source] += 1
        trajectory_counts[trajectory] += 1
        if len(selected) >= int(config.maximum_planned_views):
            break
    return selected


def build_anchor_family_confusion_graph(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict,
    dynamic: dict,
    positives: dict,
    cache: dict,
    query_bins: dict[str, int],
    config: ConfusionGraphConfig,
    device: torch.device,
    progress=None,
) -> dict:
    """Build directed correct-family -> harmful-family assignment edges."""
    names = list(dynamic["query_names"])
    if names != list(positives["query_names"]):
        raise ValueError("confusion graph query registries differ")
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if int(dynamic["anchor_count"]) != anchor_count:
        raise ValueError("dynamic outcomes do not align with confusion map")
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    xyz = torch.as_tensor(state["anchor_xyz"]).double().numpy()
    source_ids = torch.as_tensor(state["source_primitive_ids"]).long()
    dependency_ids = torch.as_tensor(state["dependency_group_ids"]).long()
    metric = metric.to(device).eval()
    edge_events: dict[tuple[int, int], list[dict]] = defaultdict(list)
    with torch.no_grad():
        for query_index, name in enumerate(names):
            outcome = dynamic["records"][query_index]
            cached = cache[name]
            rows = torch.as_tensor(outcome["query_rows"]).long()
            raw = F.normalize(
                torch.as_tensor(cached["native_descriptors"]).float()[rows],
                dim=1,
            ).to(device)
            query, _ = metric(raw)
            lookup = _positive_lookup(positives["records"][query_index])
            correct, correct_scores, correct_modes = _best_legal_assignments(
                query=query,
                rows=rows,
                positive_lookup=lookup,
                bank=bank,
                family=family,
            )
            chosen = torch.as_tensor(
                outcome["top1_anchor_indices"]
            ).long()
            chosen_scores = torch.as_tensor(outcome["top1_scores"]).float()
            chosen_modes = torch.as_tensor(
                outcome.get(
                    "top1_mode_indices", torch.full_like(chosen, -1)
                )
            ).long()
            gt_errors = torch.as_tensor(
                outcome["gt_reprojection_errors_px"]
            ).float()
            inliers = torch.as_tensor(outcome["ransac_inlier_mask"]).bool()
            points = torch.as_tensor(cached["native_keypoints"]).float()[rows]
            height, width = cached["native_input_hw"]
            for slot in range(len(rows)):
                positive = int(correct[slot])
                negative = int(chosen[slot])
                if positive < 0 or negative in lookup.get(int(rows[slot]), []):
                    continue
                error = float(gt_errors[slot])
                if error <= float(config.clean_threshold_px):
                    continue
                margin = float(chosen_scores[slot] - correct_scores[slot])
                event = {
                    "query_index": int(query_index),
                    "query_name": str(name),
                    "query_row": int(rows[slot]),
                    "correct_anchor": positive,
                    "confusing_anchor": negative,
                    "correct_mode": int(correct_modes[slot]),
                    "confusing_mode": int(chosen_modes[slot]),
                    "score_margin": margin,
                    "gt_reprojection_error_px": error,
                    "harmful": bool(
                        error > float(config.harmful_threshold_px)
                    ),
                    "ransac_inlier": bool(inliers[slot]),
                    "pose_blame": float(
                        min(error / max(config.harmful_threshold_px, 1e-6), 4.0)
                        * (1.0 + float(inliers[slot]))
                        * (1.0 + min(float(outcome["te_cm"]) / 50.0, 3.0))
                    ),
                    "trajectory": trajectory_id(str(name)),
                    "view_bin": int(query_bins[name]),
                    "image_cell": _image_cell(
                        points[slot], width=int(width), height=int(height)
                    ),
                    "source_primitive_pair": [
                        int(source_ids[positive]),
                        int(source_ids[negative]),
                    ],
                    "dependency_group_pair": [
                        int(dependency_ids[positive]),
                        int(dependency_ids[negative]),
                    ],
                }
                edge_events[(positive, negative)].append(event)
            if progress is not None:
                progress(query_index + 1, len(names))
    edges = []
    retained_events = []
    for (positive, negative), events in edge_events.items():
        trajectories = sorted({value["trajectory"] for value in events})
        if len(events) < int(config.minimum_occurrences):
            continue
        if len(trajectories) < int(config.minimum_trajectories):
            continue
        harmful = sum(bool(value["harmful"]) for value in events)
        survivors = sum(
            bool(value["harmful"]) and bool(value["ransac_inlier"])
            for value in events
        )
        mean_margin = sum(value["score_margin"] for value in events) / len(events)
        pose_blame = sum(value["pose_blame"] for value in events)
        edge = {
            "edge_index": len(edges),
            "correct_anchor": int(positive),
            "confusing_anchor": int(negative),
            "occurrences": len(events),
            "harmful_occurrences": harmful,
            "harmful_ransac_survivors": survivors,
            "trajectory_count": len(trajectories),
            "trajectories": trajectories,
            "query_count": len({value["query_name"] for value in events}),
            "mean_score_margin": float(mean_margin),
            "maximum_score_margin": float(
                max(value["score_margin"] for value in events)
            ),
            "pose_blame": float(pose_blame),
            "weight": float(
                len(events)
                + 2.0 * harmful
                + 3.0 * survivors
                + pose_blame
                + len(trajectories)
            ),
            "source_primitive_pair": events[0]["source_primitive_pair"],
            "dependency_group_pair": events[0]["dependency_group_pair"],
        }
        edges.append(edge)
        for event in sorted(
            events,
            key=lambda value: (
                -value["pose_blame"],
                -value["score_margin"],
                value["query_name"],
                value["query_row"],
            ),
        )[: int(config.maximum_events_per_edge)]:
            retained_events.append({**event, "edge_index": edge["edge_index"]})
    edges.sort(key=lambda value: (-value["weight"], value["correct_anchor"], value["confusing_anchor"]))
    remap = {
        int(edge["edge_index"]): index for index, edge in enumerate(edges)
    }
    for index, edge in enumerate(edges):
        edge["edge_index"] = index
    retained_events = [
        {**event, "edge_index": remap[int(event["edge_index"])]}
        for event in retained_events
    ]
    retained_events.sort(
        key=lambda value: (
            value["edge_index"],
            -value["pose_blame"],
            value["query_name"],
            value["query_row"],
        )
    )
    confusion_by_query = Counter(
        value["query_name"] for value in retained_events
    )
    return {
        "schema": "lafgs_anchor_family_confusion_graph",
        "version": 1,
        "anchor_count": anchor_count,
        "query_names": names,
        "edges": edges,
        "events": retained_events,
        "summary": {
            "raw_directed_edge_count": len(edge_events),
            "retained_directed_edge_count": len(edges),
            "retained_event_count": len(retained_events),
            "affected_query_count": len(confusion_by_query),
            "cross_trajectory_edge_count": sum(
                int(edge["trajectory_count"] > 1) for edge in edges
            ),
            "harmful_occurrence_count": sum(
                edge["harmful_occurrences"] for edge in edges
            ),
            "harmful_ransac_survivor_count": sum(
                edge["harmful_ransac_survivors"] for edge in edges
            ),
        },
        "config": asdict(config),
    }


def _csr_from_lists(values: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    flattened = []
    for row in values:
        flattened.extend(int(value) for value in row)
        offsets.append(len(flattened))
    return torch.as_tensor(offsets, dtype=torch.long), torch.as_tensor(
        flattened, dtype=torch.long
    )


def build_contrastive_synthetic_record(
    *,
    record: dict,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict,
    confusion_graph: dict,
    rendered_depth: torch.Tensor,
    alpha: torch.Tensor,
    visibility_config: SyntheticEvidenceConfig,
    config: ContrastiveEvidenceConfig,
    device: torch.device,
) -> dict:
    """Relabel one accepted render with strong, ambiguous and hard negatives."""
    if not bool(record.get("accepted", True)):
        raise ValueError("contrastive relabeling requires accepted evidence")
    if not bool(record.get("config", {}).get("require_support_mask", False)):
        raise ValueError("contrastive evidence requires support-masked keypoints")
    keypoints = torch.as_tensor(record["native_keypoints"]).float().to(device)
    raw = F.normalize(
        torch.as_tensor(record["native_descriptors"]).float(), dim=1
    ).to(device)
    xyz = torch.as_tensor(state["anchor_xyz"]).float().to(device)
    projected, depth, in_front = project_existing_anchors(
        xyz,
        torch.as_tensor(record["pose_w2c"]).float().to(device),
        torch.as_tensor(record["native_K"]).float().to(device),
    )
    height, width = record["native_input_hw"]
    visible = render_visible_anchor_mask(
        projected_xy=projected,
        anchor_depth=depth,
        rendered_depth=torch.as_tensor(rendered_depth).to(device),
        alpha=torch.as_tensor(alpha).to(device),
        width=int(width),
        height=int(height),
        config=visibility_config,
    )
    visible &= in_front
    visible_indices = torch.nonzero(visible, as_tuple=False).reshape(-1)
    if not len(keypoints) or not len(visible_indices):
        raise ValueError("accepted synthetic evidence has no visible support")
    distances = torch.cdist(keypoints, projected[visible_indices])
    count = min(
        max(int(config.maximum_positives_per_keypoint), 1),
        len(visible_indices),
    )
    nearest_distance, nearest_local = torch.topk(
        distances, k=count, largest=False, dim=1
    )
    nearest_anchor = visible_indices[nearest_local]
    active_target = record.get("active_evidence_target")
    strong_lists = []
    ambiguous_lists = []
    for row in range(len(keypoints)):
        strong_lists.append(
            [
                int(anchor)
                for anchor, distance in zip(
                    nearest_anchor[row].tolist(),
                    nearest_distance[row].tolist(),
                )
                if distance <= float(config.strong_radius_px)
            ]
        )
        ambiguous_lists.append(
            [
                int(anchor)
                for anchor, distance in zip(
                    nearest_anchor[row].tolist(),
                    nearest_distance[row].tolist(),
                )
                if float(config.strong_radius_px)
                < distance
                <= float(config.ambiguous_radius_px)
            ]
        )
    if (
        active_target is not None
        and config.restrict_strong_to_active_target
    ):
        target = int(active_target["correct_anchor"])
        strong_lists = [
            [anchor for anchor in values if int(anchor) == target]
            for values in strong_lists
        ]
    adjacency: dict[int, list[dict]] = defaultdict(list)
    target_edge_index = (
        int(active_target["edge_index"])
        if active_target is not None
        else None
    )
    for edge in confusion_graph["edges"]:
        if (
            target_edge_index is not None
            and int(edge["edge_index"]) != target_edge_index
        ):
            continue
        if int(edge["occurrences"]) < int(config.minimum_edge_occurrences):
            continue
        if active_target is not None and (
            int(edge["correct_anchor"])
            != int(active_target["correct_anchor"])
            or int(edge["confusing_anchor"])
            != int(active_target["confusing_anchor"])
        ):
            raise ValueError(
                "active render target does not align with confusion edge"
            )
        adjacency[int(edge["correct_anchor"])].append(edge)
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = metric.to(device).eval()
    with torch.no_grad():
        query, _ = metric(raw)
    source_ids = torch.as_tensor(
        state["source_primitive_ids"], device=device
    ).long()
    dependency_ids = torch.as_tensor(
        state["dependency_group_ids"], device=device
    ).long()
    negative_lists: list[list[int]] = [[] for _ in range(len(keypoints))]
    negative_positive_lists: list[list[int]] = [
        [] for _ in range(len(keypoints))
    ]
    negative_weight_lists: list[list[float]] = [
        [] for _ in range(len(keypoints))
    ]
    negative_margin_lists: list[list[float]] = [
        [] for _ in range(len(keypoints))
    ]
    for row, positives in enumerate(strong_lists):
        candidates = []
        for positive in positives:
            for edge in adjacency.get(int(positive), []):
                negative = int(edge["confusing_anchor"])
                if (
                    config.require_negative_visibility
                    and not bool(visible[negative])
                ):
                    continue
                if bool(visible[negative]) and float(
                    torch.linalg.vector_norm(
                        projected[negative] - keypoints[row]
                    )
                ) <= float(config.ambiguous_radius_px):
                    continue
                if (
                    config.require_distinct_source_primitive
                    and int(source_ids[positive]) == int(source_ids[negative])
                ):
                    continue
                if (
                    config.require_distinct_dependency_group
                    and int(dependency_ids[positive])
                    == int(dependency_ids[negative])
                ):
                    continue
                candidates.append((positive, negative, edge))
        if not candidates:
            continue
        positive_anchors = torch.as_tensor(
            [value[0] for value in candidates], device=device
        ).long()
        negative_anchors = torch.as_tensor(
            [value[1] for value in candidates], device=device
        ).long()
        repeated_query = query[row : row + 1].expand(len(candidates), -1)
        positive_scores, _ = family_pair_scores(
            repeated_query, positive_anchors, bank, family
        )
        negative_scores, _ = family_pair_scores(
            repeated_query, negative_anchors, bank, family
        )
        score_margin = negative_scores - positive_scores
        ranked = sorted(
            range(len(candidates)),
            key=lambda index: (
                -float(candidates[index][2]["weight"]),
                -float(score_margin[index]),
                candidates[index][1],
            ),
        )
        seen = set()
        for index in ranked:
            positive, negative, edge = candidates[index]
            if negative in seen:
                continue
            if float(score_margin[index]) < -float(
                config.maximum_negative_score_gap
            ):
                continue
            seen.add(negative)
            negative_lists[row].append(negative)
            negative_positive_lists[row].append(positive)
            negative_weight_lists[row].append(float(edge["weight"]))
            negative_margin_lists[row].append(float(score_margin[index]))
            if len(negative_lists[row]) >= int(
                config.maximum_negatives_per_keypoint
            ):
                break
    strong_offsets, strong_indices = _csr_from_lists(strong_lists)
    ambiguous_offsets, ambiguous_indices = _csr_from_lists(ambiguous_lists)
    negative_offsets, negative_indices = _csr_from_lists(negative_lists)
    _, negative_positive_indices = _csr_from_lists(negative_positive_lists)
    flat_weights = torch.as_tensor(
        [value for row in negative_weight_lists for value in row],
        dtype=torch.float32,
    )
    flat_margins = torch.as_tensor(
        [value for row in negative_margin_lists for value in row],
        dtype=torch.float32,
    )
    output = {
        **record,
        "positive_offsets": strong_offsets,
        "positive_indices": strong_indices,
        "ambiguous_offsets": ambiguous_offsets,
        "ambiguous_indices": ambiguous_indices,
        "hard_negative_offsets": negative_offsets,
        "hard_negative_indices": negative_indices,
        "hard_negative_positive_indices": negative_positive_indices,
        "hard_negative_weights": flat_weights,
        "hard_negative_score_margins": flat_margins,
        "positive_pair_count": int(strong_indices.numel()),
        "strong_positive_pair_count": int(strong_indices.numel()),
        "ambiguous_pair_count": int(ambiguous_indices.numel()),
        "hard_negative_pair_count": int(negative_indices.numel()),
        "contrastive_accepted": bool(
            negative_indices.numel()
            >= int(config.minimum_hard_negative_pairs)
        ),
        "contrastive_config": asdict(config),
        "label_policy": (
            "depth-visible 2px strong and 2-6px ambiguous anchors; global "
            "real confusion-graph negatives outside map identity groups"
        ),
        "raster_provenance_policy": (
            "unavailable_in_renderer_v1; map source/dependency identity and "
            "rendered depth/alpha are an explicit conservative fallback"
        ),
        "active_evidence_target": active_target,
    }
    return output


def pack_contrastive_synthetic_evidence(
    records: list[dict],
    *,
    source: dict,
    confusion_graph: dict,
    include_positive_only_records: bool = False,
) -> dict:
    contrastive = [
        record
        for record in records
        if bool(record.get("accepted", True))
        and bool(record["contrastive_accepted"])
    ]
    accepted = (
        [
            record
            for record in records
            if bool(record.get("accepted", True))
        ]
        if include_positive_only_records
        else contrastive
    )
    return {
        "schema": "lafgs_confusion_contrastive_synthetic_evidence",
        "version": 1,
        "anchor_count": int(confusion_graph["anchor_count"]),
        "query_names": [record["query_name"] for record in accepted],
        "records": accepted,
        "summary": {
            "candidate_view_count": len(records),
            "positive_view_count": len(accepted),
            "contrastive_view_count": len(contrastive),
            "strong_positive_pair_count": sum(
                record["strong_positive_pair_count"] for record in accepted
            ),
            "ambiguous_pair_count": sum(
                record["ambiguous_pair_count"] for record in accepted
            ),
            "hard_negative_pair_count": sum(
                record["hard_negative_pair_count"] for record in accepted
            ),
            "targeted_confusion_edge_count": len(
                {
                    (int(positive), int(negative))
                    for record in accepted
                    for positive, negative in zip(
                        torch.as_tensor(
                            record["hard_negative_positive_indices"]
                        ).tolist(),
                        torch.as_tensor(
                            record["hard_negative_indices"]
                        ).tolist(),
                    )
                }
            ),
        },
        "source": dict(source),
        "provenance": {
            **dict(source),
            "label_policy": (
                "real confusion graph plus support-masked synthetic RGB"
            ),
        },
        "confusion_graph_summary": dict(confusion_graph["summary"]),
    }


def synthetic_separability_oracle(
    *,
    state: dict,
    metric: SharedLowRankMetric,
    family: dict,
    evidence: dict,
    device: torch.device,
) -> dict:
    """Measure the best rendered descriptor margin for targeted family pairs."""
    if evidence.get("schema") != (
        "lafgs_confusion_contrastive_synthetic_evidence"
    ):
        raise ValueError("separability oracle requires contrastive evidence")
    bank = F.normalize(
        torch.as_tensor(state["anchor_features"]).float(), dim=1
    ).to(device)
    metric = metric.to(device).eval()
    observations: dict[tuple[int, int], list[float]] = defaultdict(list)
    with torch.no_grad():
        for record in evidence["records"]:
            pair_count = int(record.get("hard_negative_pair_count", 0))
            if pair_count <= 0:
                continue
            offsets = torch.as_tensor(
                record["hard_negative_offsets"]
            ).long()
            pair_rows = torch.repeat_interleave(
                torch.arange(len(offsets) - 1),
                offsets[1:] - offsets[:-1],
            )
            positive = torch.as_tensor(
                record["hard_negative_positive_indices"]
            ).long()
            negative = torch.as_tensor(
                record["hard_negative_indices"]
            ).long()
            raw = F.normalize(
                torch.as_tensor(record["native_descriptors"]).float()[
                    pair_rows
                ],
                dim=1,
            ).to(device)
            query, _ = metric(raw)
            positive_scores, _ = family_pair_scores(
                query, positive.to(device), bank, family
            )
            negative_scores, _ = family_pair_scores(
                query, negative.to(device), bank, family
            )
            for pos, neg, margin in zip(
                positive.tolist(),
                negative.tolist(),
                (positive_scores - negative_scores).tolist(),
            ):
                observations[(int(pos), int(neg))].append(float(margin))
    edges = []
    for (positive, negative), margins in sorted(observations.items()):
        values = torch.as_tensor(margins).float()
        edges.append(
            {
                "correct_anchor": positive,
                "confusing_anchor": negative,
                "observation_count": len(margins),
                "maximum_margin": float(values.max()),
                "median_margin": float(values.median()),
                "separable": bool((values > 0).any()),
                "separable_002": bool((values > 0.02).any()),
                "separable_005": bool((values > 0.05).any()),
            }
        )
    maxima = torch.as_tensor(
        [edge["maximum_margin"] for edge in edges]
    ).float()
    return {
        "schema": "lafgs_rendered_pair_separability_oracle",
        "version": 1,
        "edge_count": len(edges),
        "observation_count": sum(len(values) for values in observations.values()),
        "separable_edge_fraction": (
            float((maxima > 0).float().mean()) if maxima.numel() else 0.0
        ),
        "separable_002_edge_fraction": (
            float((maxima > 0.02).float().mean())
            if maxima.numel()
            else 0.0
        ),
        "separable_005_edge_fraction": (
            float((maxima > 0.05).float().mean())
            if maxima.numel()
            else 0.0
        ),
        "maximum_margin_median": (
            float(maxima.median()) if maxima.numel() else float("nan")
        ),
        "maximum_margin_mean": (
            float(maxima.mean()) if maxima.numel() else float("nan")
        ),
        "edges": edges,
    }
