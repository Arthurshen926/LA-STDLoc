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
    arc_yaw_degrees: tuple[float, ...] = (-10.0, -5.0, 5.0, 10.0)
    arc_vertical_fractions: tuple[float, ...] = (0.0,)
    minimum_pose_novelty: float = 0.15
    maximum_safe_envelope_scale: float = 3.0
    context_neighbor_count: int = 16
    context_separation_weight: float = 8.0
    pose_novelty_weight: float = 0.5
    diversity_weight: float = 0.5


def filter_confusion_graph_by_context_oracle(
    confusion_graph: dict,
    oracle: dict,
    *,
    method: str = "O1_cross_trajectory_2d",
    minimum_positive_fraction: float = 0.5,
    minimum_records: int = 2,
) -> tuple[dict, dict]:
    """Keep only confusion edges that a real-query context oracle can separate."""

    if oracle.get("schema") != "lafgs_real_2d3d_context_oracle":
        raise ValueError("unsupported context oracle")
    edge_outcomes = defaultdict(list)
    for record in oracle["records"]:
        margins = record.get("margins", {})
        if method not in margins:
            raise ValueError(f"context oracle does not contain {method}")
        if margins[method] is None:
            continue
        edge_outcomes[int(record["edge_index"])].append(
            float(margins[method]) > 0.0
        )
    eligible = {
        edge
        for edge, outcomes in edge_outcomes.items()
        if len(outcomes) >= int(minimum_records)
        and sum(outcomes) / len(outcomes)
        >= float(minimum_positive_fraction)
    }
    filtered = {
        **confusion_graph,
        "edges": [
            edge
            for edge in confusion_graph["edges"]
            if int(edge["edge_index"]) in eligible
        ],
        "events": [
            event
            for event in confusion_graph["events"]
            if int(event["edge_index"]) in eligible
        ],
    }
    return filtered, {
        "method": str(method),
        "minimum_positive_fraction": float(minimum_positive_fraction),
        "minimum_records": int(minimum_records),
        "input_edge_count": len(confusion_graph["edges"]),
        "oracle_edge_count": len(edge_outcomes),
        "eligible_edge_count": len(eligible),
        "retained_graph_edge_count": len(filtered["edges"]),
    }


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


def _look_at_pose_w2c(center, target, reference_down):
    center = torch.as_tensor(center).double()
    target = torch.as_tensor(target).double()
    forward = F.normalize(target - center, dim=0)
    down = F.normalize(torch.as_tensor(reference_down).double(), dim=0)
    right = torch.linalg.cross(down, forward)
    if float(torch.linalg.vector_norm(right)) < 1e-6:
        right = torch.linalg.cross(
            center.new_tensor([0.0, 1.0, 0.0]), forward
        )
    right = F.normalize(right, dim=0)
    down = F.normalize(torch.linalg.cross(forward, right), dim=0)
    rotation_c2w = torch.stack((right, down, forward), dim=1)
    pose = torch.eye(4, dtype=torch.double)
    pose[:3, :3] = rotation_c2w.T
    pose[:3, 3] = -rotation_c2w.T @ center
    return pose


def _rotate_about_axis(vector, axis, angle_degrees):
    vector = torch.as_tensor(vector).double()
    axis = F.normalize(torch.as_tensor(axis).double(), dim=0)
    angle = math.radians(float(angle_degrees))
    return (
        vector * math.cos(angle)
        + torch.linalg.cross(axis, vector) * math.sin(angle)
        + axis * torch.dot(axis, vector) * (1.0 - math.cos(angle))
    )


def _pose_novelty(pose_w2c, centers, forwards, local_scale):
    center, forward = _camera_center_and_forward(pose_w2c)
    translation = torch.linalg.vector_norm(centers - center, dim=1)
    angles = torch.rad2deg(
        torch.acos((forwards @ forward).clamp(-1.0, 1.0))
    )
    normalized_translation = translation / max(float(local_scale), 1e-6)
    combined = torch.sqrt(
        normalized_translation.square() + (angles / 20.0).square()
    )
    nearest = int(torch.argmin(combined))
    return (
        float(combined[nearest]),
        float(normalized_translation[nearest]),
        float(angles[nearest]),
    )


def _projected_context_separation(
    xyz, anchors, pose_w2c, K, image_diagonal, neighbor_indices
):
    layouts = []
    for anchor in anchors:
        indices = torch.cat(
            (
                torch.as_tensor([anchor]),
                neighbor_indices[int(anchor)].cpu(),
            )
        ).long()
        points = xyz[indices]
        camera = points @ pose_w2c[:3, :3].T + pose_w2c[:3, 3]
        valid = camera[:, 2] > 1e-4
        if not bool(valid[0]) or int(valid[1:].sum()) < 2:
            return 0.0
        projected = camera[:, :2] / camera[:, 2:3].clamp_min(1e-8)
        projected = projected @ K[:2, :2].T + K[:2, 2]
        layouts.append(
            (projected[1:][valid[1:]] - projected[0])
            / max(float(image_diagonal), 1.0)
        )
    distance = torch.cdist(layouts[0].float(), layouts[1].float())
    return float(
        0.5
        * (
            distance.min(dim=1).values.mean()
            + distance.min(dim=0).values.mean()
        )
    )


def _select_diverse_confusion_views(candidates, config):
    if not candidates:
        return []
    selected = []
    remaining = torch.ones(len(candidates), dtype=torch.bool)
    centers = torch.as_tensor(
        [candidate["camera_center"] for candidate in candidates]
    ).double()
    forwards = F.normalize(
        torch.as_tensor(
            [candidate["camera_forward"] for candidate in candidates]
        ).double(),
        dim=1,
    )
    scales = torch.as_tensor(
        [candidate["source_local_scale"] for candidate in candidates]
    ).double().clamp_min(1e-6)
    acquisitions = torch.as_tensor(
        [candidate["acquisition"] for candidate in candidates]
    ).double()
    minimum_diversity = torch.full(
        (len(candidates),), float("inf"), dtype=torch.double
    )
    edge_counts = Counter()
    source_counts = Counter()
    trajectory_counts = Counter()
    while bool(remaining.any()) and len(selected) < int(
        config.maximum_planned_views
    ):
        eligible = remaining.clone()
        for index, candidate in enumerate(candidates):
            if not bool(eligible[index]):
                continue
            edge = int(candidate["edge_index"])
            source = str(candidate["source_query"])
            trajectory = trajectory_id(source)
            if edge_counts[edge] >= int(config.maximum_views_per_edge):
                eligible[index] = False
                continue
            if source_counts[source] >= int(config.maximum_views_per_source):
                eligible[index] = False
                continue
            if trajectory_counts[trajectory] >= int(
                config.maximum_views_per_trajectory
            ):
                eligible[index] = False
        if not bool(eligible.any()):
            break
        if selected:
            diversity = minimum_diversity.clamp(0.0, 2.0)
        else:
            diversity = torch.zeros_like(minimum_diversity)
        scores = acquisitions * (
            1.0 + float(config.diversity_weight) * diversity
        )
        scores[~eligible] = -torch.inf
        best_index = int(torch.argmax(scores))
        best_score = float(scores[best_index])
        candidate = candidates[best_index]
        candidate["diverse_acquisition"] = float(best_score)
        selected.append(candidate)
        remaining[best_index] = False
        edge_counts[int(candidate["edge_index"])] += 1
        source = str(candidate["source_query"])
        source_counts[source] += 1
        trajectory_counts[trajectory_id(source)] += 1
        translation = torch.linalg.vector_norm(
            centers - centers[best_index], dim=1
        ) / scales
        angles = torch.rad2deg(
            torch.acos(
                (forwards @ forwards[best_index]).clamp(-1.0, 1.0)
            )
        ) / 20.0
        minimum_diversity = torch.minimum(
            minimum_diversity,
            torch.sqrt(translation.square() + angles.square()),
        )
    return selected


def plan_reference_guided_confusion_views(
    *,
    confusion_graph: dict,
    state: dict,
    cache: dict,
    query_bins: dict[str, int],
    config: ConfusionViewPlanningConfig,
) -> list[dict]:
    """Plan local, confusion-centered arcs between reliable real views."""

    xyz = torch.as_tensor(state["anchor_xyz"]).double()
    if int(confusion_graph["anchor_count"]) != len(xyz):
        raise ValueError("reference-guided planner map does not align")
    names = [
        str(name)
        for name in confusion_graph["query_names"]
        if str(name) in cache
    ]
    if not names or any(name not in query_bins for name in names):
        raise ValueError("reference-guided planner camera registry is invalid")
    centers, forwards, downs = [], [], []
    for name in names:
        pose = torch.as_tensor(cache[name]["pose_w2c"]).double()
        c2w = torch.linalg.inv(pose)
        center, forward = _camera_center_and_forward(pose)
        centers.append(center)
        forwards.append(forward)
        downs.append(F.normalize(c2w[:3, 1], dim=0))
    centers = torch.stack(centers)
    forwards = torch.stack(forwards)
    downs = torch.stack(downs)
    pose_distances = torch.cdist(centers.float(), centers.float()).double()
    pose_distances.fill_diagonal_(float("inf"))
    nearest = torch.topk(
        pose_distances,
        k=min(8, max(len(names) - 1, 1)),
        dim=1,
        largest=False,
    ).values
    local_scales = torch.where(
        torch.isfinite(nearest), nearest, torch.nan
    ).nanmedian(dim=1).values.clamp_min(1e-3)
    name_to_index = {name: index for index, name in enumerate(names)}
    events_by_edge = defaultdict(list)
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
    candidates = []
    for edge in edges:
        edge_index = int(edge["edge_index"])
        anchors = (
            int(edge["correct_anchor"]),
            int(edge["confusing_anchor"]),
        )
        component_center = xyz[
            torch.as_tensor(anchors).long()
        ].mean(dim=0)
        neighbor_indices = {}
        maximum_context_neighbors = min(
            max(int(config.context_neighbor_count), 1),
            max(len(xyz) - 1, 1),
        )
        for anchor in anchors:
            anchor_distance = torch.linalg.vector_norm(
                xyz - xyz[int(anchor)], dim=1
            )
            anchor_distance[int(anchor)] = torch.inf
            neighbor_indices[int(anchor)] = torch.topk(
                anchor_distance,
                k=maximum_context_neighbors,
                largest=False,
            ).indices
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
            references = []
            for neighbor_index in torch.argsort(
                pose_distances[source_index]
            ).tolist():
                distance = float(
                    pose_distances[source_index, neighbor_index]
                )
                if not math.isfinite(distance):
                    continue
                if distance > float(config.maximum_neighbor_scale) * source_scale:
                    break
                angle = _view_angle_deg(
                    forwards[source_index], forwards[neighbor_index]
                )
                if angle <= float(config.maximum_view_angle_deg):
                    references.append((neighbor_index, distance, angle))
                if len(references) >= int(config.maximum_pose_neighbors):
                    break
            cached = cache[source]
            height, width = map(int, cached["native_input_hw"])
            K = torch.as_tensor(cached["native_K"]).double()
            image_diagonal = math.hypot(width, height)
            for neighbor_index, reference_distance, reference_angle in references:
                neighbor = names[neighbor_index]
                source_radial = centers[source_index] - component_center
                neighbor_radial = centers[neighbor_index] - component_center
                source_radius = float(torch.linalg.vector_norm(source_radial))
                neighbor_radius = float(torch.linalg.vector_norm(neighbor_radial))
                if min(source_radius, neighbor_radius) < 1e-4:
                    continue
                source_radial = F.normalize(source_radial, dim=0)
                neighbor_radial = F.normalize(neighbor_radial, dim=0)
                for alpha in config.interpolation_alphas:
                    radial = F.normalize(
                        (1.0 - float(alpha)) * source_radial
                        + float(alpha) * neighbor_radial,
                        dim=0,
                    )
                    radius = (
                        (1.0 - float(alpha)) * source_radius
                        + float(alpha) * neighbor_radius
                    )
                    reference_down = F.normalize(
                        (1.0 - float(alpha)) * downs[source_index]
                        + float(alpha) * downs[neighbor_index],
                        dim=0,
                    )
                    for yaw in config.arc_yaw_degrees:
                        yaw_radial = F.normalize(
                            _rotate_about_axis(
                                radial, -reference_down, float(yaw)
                            ),
                            dim=0,
                        )
                        for vertical in config.arc_vertical_fractions:
                            center = (
                                component_center
                                + radius * yaw_radial
                                - float(vertical) * radius * reference_down
                            )
                            pose = _look_at_pose_w2c(
                                center, component_center, reference_down
                            )
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
                            novelty, translation_novelty, angle_novelty = (
                                _pose_novelty(
                                    pose, centers, forwards, source_scale
                                )
                            )
                            nearest_train = float(
                                torch.linalg.vector_norm(
                                    centers - center, dim=1
                                ).min()
                            )
                            if novelty < float(config.minimum_pose_novelty):
                                continue
                            if nearest_train > (
                                float(config.maximum_safe_envelope_scale)
                                * source_scale
                            ):
                                continue
                            context_separation = (
                                _projected_context_separation(
                                    xyz,
                                    anchors,
                                    pose,
                                    K,
                                    image_diagonal,
                                    neighbor_indices,
                                )
                            )
                            pair_separation = float(
                                torch.linalg.vector_norm(
                                    projected[0] - projected[1]
                                )
                                / max(image_diagonal, 1.0)
                            )
                            cross_trajectory = (
                                trajectory_id(source)
                                != trajectory_id(neighbor)
                            )
                            different_bin = (
                                int(query_bins[source])
                                != int(query_bins[neighbor])
                            )
                            utility = math.log1p(
                                max(float(edge["weight"]), 0.0)
                            )
                            utility *= (
                                1.0
                                + float(config.context_separation_weight)
                                * context_separation
                            )
                            utility *= (
                                1.0
                                + float(config.pose_novelty_weight)
                                * min(novelty, 2.0)
                            )
                            utility *= (
                                1.0
                                + float(config.cross_trajectory_bonus)
                                * float(cross_trajectory)
                                + float(config.different_view_bin_bonus)
                                * float(different_bin)
                                + float(
                                    config.confusing_family_visible_bonus
                                )
                                * float(visible[1])
                            )
                            _, camera_forward = _camera_center_and_forward(
                                pose
                            )
                            candidates.append(
                                {
                                    "query_id": (
                                        f"confusion_arc:e{edge_index}:{source}:"
                                        f"{neighbor}:{float(alpha):.2f}:"
                                        f"{float(yaw):+.1f}:{float(vertical):+.3f}"
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
                                            width / (2.0 * float(K[0, 0]))
                                        )
                                    ),
                                    "fovy": float(
                                        2.0
                                        * math.atan(
                                            height / (2.0 * float(K[1, 1]))
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
                                    "source_neighbor_distance": reference_distance,
                                    "source_local_scale": source_scale,
                                    "view_angle_deg": reference_angle,
                                    "correct_family_visible": bool(visible[0]),
                                    "confusing_family_visible": bool(visible[1]),
                                    "projected_pair_separation": pair_separation,
                                    "projected_context_separation": context_separation,
                                    "pose_novelty": novelty,
                                    "pose_novelty_translation": translation_novelty,
                                    "pose_novelty_angle_deg": angle_novelty,
                                    "nearest_train_distance": nearest_train,
                                    "arc_yaw_degrees": float(yaw),
                                    "arc_vertical_fraction": float(vertical),
                                    "camera_center": center.float().tolist(),
                                    "camera_forward": camera_forward.float().tolist(),
                                    "edge_weight": float(edge["weight"]),
                                    "edge_occurrences": int(edge["occurrences"]),
                                    "edge_trajectory_count": int(
                                        edge["trajectory_count"]
                                    ),
                                    "pose_blame": float(event["pose_blame"]),
                                    "acquisition": float(utility),
                                    "planning_policy": (
                                        "confusion_component_reference_arc"
                                    ),
                                }
                            )
    unique = {}
    for candidate in candidates:
        previous = unique.get(candidate["query_id"])
        if (
            previous is None
            or candidate["acquisition"] > previous["acquisition"]
        ):
            unique[candidate["query_id"]] = candidate
    return _select_diverse_confusion_views(list(unique.values()), config)


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
