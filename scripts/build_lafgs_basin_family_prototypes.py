#!/usr/bin/env python3
"""Build basin-conditioned descriptor families without duplicating geometry."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


def trajectory_id(query_name: str) -> str:
    return str(query_name).split("/", 1)[0]


def bitwise_union(values: torch.Tensor) -> int:
    output = 0
    for value in torch.as_tensor(values).reshape(-1).tolist():
        output |= int(value)
    return output


def robust_family_prototype(
    descriptors: torch.Tensor,
    weights: torch.Tensor,
    *,
    trim_fraction: float,
) -> torch.Tensor:
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    weights = torch.as_tensor(weights).float().clamp_min(1e-6)
    agreement = descriptors @ descriptors.T
    medoid = int((agreement * weights[None]).sum(dim=1).argmax())
    keep_count = max(
        1, int(round(descriptors.shape[0] * (1.0 - float(trim_fraction))))
    )
    keep = torch.topk(agreement[medoid], keep_count).indices
    return F.normalize(
        (descriptors[keep] * weights[keep, None]).sum(dim=0), dim=0
    )


def discover_spherical_modes(
    descriptors: torch.Tensor,
    primary: torch.Tensor,
    *,
    maximum_modes: int,
    minimum_cluster_size: int,
    minimum_separation: float,
    trim_fraction: float,
    iterations: int = 8,
) -> list[dict]:
    """Discover coherent secondary modes while keeping the primary fixed."""
    descriptors = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    primary = F.normalize(torch.as_tensor(primary).float(), dim=0)
    if descriptors.shape[0] < int(minimum_cluster_size):
        return []
    modes = [primary]
    assignments = torch.zeros(descriptors.shape[0], dtype=torch.long)
    maximum_modes = max(int(maximum_modes), 1)
    while len(modes) < maximum_modes:
        stacked = torch.stack(modes)
        similarities = descriptors @ stacked.T
        best_similarity = similarities.max(dim=1).values
        seed_order = torch.argsort(best_similarity)
        accepted = None
        for seed_index in seed_order.tolist():
            if 1.0 - float(best_similarity[seed_index]) < float(
                minimum_separation
            ):
                break
            trial_modes = modes + [descriptors[seed_index]]
            for _ in range(max(int(iterations), 1)):
                trial_stack = torch.stack(trial_modes)
                trial_assignments = (descriptors @ trial_stack.T).argmax(dim=1)
                selected = torch.nonzero(
                    trial_assignments == len(trial_modes) - 1,
                    as_tuple=False,
                ).reshape(-1)
                if selected.numel() < int(minimum_cluster_size):
                    break
                updated = robust_family_prototype(
                    descriptors[selected],
                    torch.ones(selected.numel()),
                    trim_fraction=trim_fraction,
                )
                if float(updated @ trial_modes[-1]) > 1.0 - 1e-6:
                    trial_modes[-1] = updated
                    break
                trial_modes[-1] = updated
            else:
                selected = torch.nonzero(
                    trial_assignments == len(trial_modes) - 1,
                    as_tuple=False,
                ).reshape(-1)
            if selected.numel() < int(minimum_cluster_size):
                continue
            accepted = (trial_modes[-1], trial_assignments)
            break
        if accepted is None:
            break
        modes.append(accepted[0])
        assignments = accepted[1]
    if len(modes) == 1:
        return []
    similarities = descriptors @ torch.stack(modes).T
    assignments = similarities.argmax(dim=1)
    output = []
    for mode_index in range(1, len(modes)):
        selected = torch.nonzero(
            assignments == mode_index, as_tuple=False
        ).reshape(-1)
        if selected.numel() < int(minimum_cluster_size):
            continue
        prototype = robust_family_prototype(
            descriptors[selected],
            torch.ones(selected.numel()),
            trim_fraction=trim_fraction,
        )
        cluster_similarity = descriptors[selected] @ prototype
        keep_count = max(
            int(minimum_cluster_size),
            int(round(selected.numel() * (1.0 - float(trim_fraction)))),
        )
        if keep_count < selected.numel():
            selected = selected[
                torch.topk(cluster_similarity, keep_count).indices
            ]
            prototype = F.normalize(descriptors[selected].mean(dim=0), dim=0)
            cluster_similarity = descriptors[selected] @ prototype
        output.append(
            {
                "prototype": prototype,
                "observation_indices": selected,
                "observation_count": int(selected.numel()),
                "dispersion": float(1.0 - cluster_similarity.mean()),
                "primary_similarity": float(prototype @ primary),
                "activation_gain_mean": float(
                    (
                        descriptors[selected] @ prototype
                        - descriptors[selected] @ primary
                    ).mean()
                ),
            }
        )
    return output


def _metric_descriptors(
    metric: SharedLowRankMetric,
    raw: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        transformed, _ = metric(
            F.normalize(torch.as_tensor(raw).float().to(device), dim=1)
        )
    return transformed.half().cpu()


def build_appearance_mode_pool(
    state: dict,
    positives: dict,
    cache: dict,
    query_bins: dict[str, int],
    metric: SharedLowRankMetric,
    *,
    dynamic: dict | None,
    basin_teacher: dict | None,
    maximum_modes_per_anchor: int,
    minimum_observations: int,
    minimum_trajectories: int,
    minimum_view_bins: int,
    minimum_separation: float,
    maximum_primary_similarity: float,
    trim_fraction: float,
    initial_bias: float,
    device: torch.device,
) -> tuple[list[dict], dict]:
    """Generate high-recall modes from all legal observations before Basin gating."""
    names = list(positives["query_names"])
    if dynamic is not None and names != list(dynamic["query_names"]):
        raise ValueError("dynamic outcomes and positive teacher registries differ")
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    descriptor_blocks = []
    anchor_blocks = []
    query_blocks = []
    row_blocks = []
    bin_blocks = []
    provenance_blocks = []
    basin_by_name = (
        {str(record["query_name"]): record for record in basin_teacher["records"]}
        if basin_teacher is not None
        else {}
    )
    for query_index, (name, record) in enumerate(
        zip(names, positives["records"])
    ):
        rows = torch.as_tensor(record["query_rows"]).long()
        offsets = torch.as_tensor(record["positive_offsets"]).long()
        anchors = torch.as_tensor(record["positive_indices"]).long()
        counts = offsets[1:] - offsets[:-1]
        pair_rows = torch.repeat_interleave(rows, counts)
        extra: dict[tuple[int, int], int] = {}
        if dynamic is not None:
            outcome = dynamic["records"][query_index]
            clean = torch.as_tensor(outcome["gt_reprojection_errors_px"]).float() <= 4.0
            for row, anchor in zip(
                torch.as_tensor(outcome["query_rows"]).long()[clean].tolist(),
                torch.as_tensor(outcome["top1_anchor_indices"]).long()[clean].tolist(),
            ):
                extra[(int(row), int(anchor))] = 2
        blame = basin_by_name.get(name)
        if blame is not None:
            for row, anchor in zip(
                torch.as_tensor(blame["blame_rows"]).long().tolist(),
                torch.as_tensor(blame["blame_positive_anchors"]).long().tolist(),
            ):
                extra[(int(row), int(anchor))] = (
                    extra.get((int(row), int(anchor)), 0) | 4
                )
        pair_positions = {
            (int(row), int(anchor)): index
            for index, (row, anchor) in enumerate(
                zip(pair_rows.tolist(), anchors.tolist())
            )
        }
        provenance = torch.ones(len(anchors), dtype=torch.uint8)
        for key, source in extra.items():
            position = pair_positions.get(key)
            if position is not None:
                provenance[position] = int(provenance[position]) | int(source)
        extra_items = [
            (row, anchor, source)
            for (row, anchor), source in extra.items()
            if (row, anchor) not in pair_positions
        ]
        all_rows = pair_rows
        all_anchors = anchors
        if extra_items:
            all_rows = torch.cat(
                (all_rows, torch.as_tensor([item[0] for item in extra_items]))
            )
            all_anchors = torch.cat(
                (
                    all_anchors,
                    torch.as_tensor([item[1] for item in extra_items]),
                )
            )
            provenance = torch.cat(
                (
                    provenance,
                    torch.as_tensor(
                        [item[2] for item in extra_items], dtype=torch.uint8
                    ),
                )
            )
        unique_rows, inverse = torch.unique(all_rows, sorted=True, return_inverse=True)
        transformed = _metric_descriptors(
            metric,
            torch.as_tensor(cache[name]["native_descriptors"])[unique_rows],
            device,
        )
        descriptor_blocks.append(transformed[inverse])
        anchor_blocks.append(all_anchors)
        query_blocks.append(
            torch.full((len(all_anchors),), query_index, dtype=torch.int32)
        )
        row_blocks.append(all_rows.to(torch.int32))
        bin_blocks.append(
            torch.full(
                (len(all_anchors),),
                int(query_bins[name]),
                dtype=torch.int16,
            )
        )
        provenance_blocks.append(provenance)
    descriptors = torch.cat(descriptor_blocks)
    anchors = torch.cat(anchor_blocks).long()
    query_indices = torch.cat(query_blocks)
    rows = torch.cat(row_blocks)
    bins = torch.cat(bin_blocks)
    provenance = torch.cat(provenance_blocks)
    order = torch.argsort(anchors)
    anchors = anchors[order]
    descriptors = descriptors[order]
    query_indices = query_indices[order]
    rows = rows[order]
    bins = bins[order]
    provenance = provenance[order]
    counts = torch.bincount(anchors, minlength=bank.shape[0])
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    candidates = []
    observation_query_indices = []
    observation_rows = []
    observation_provenance = []
    mode_offsets = [0]
    for anchor in torch.nonzero(
        counts >= int(minimum_observations), as_tuple=False
    ).reshape(-1).tolist():
        start, stop = int(offsets[anchor]), int(offsets[anchor + 1])
        anchor_descriptors = descriptors[start:stop].float()
        modes = discover_spherical_modes(
            anchor_descriptors,
            bank[anchor],
            maximum_modes=maximum_modes_per_anchor,
            minimum_cluster_size=minimum_observations,
            minimum_separation=minimum_separation,
            trim_fraction=trim_fraction,
        )
        for mode_id, mode in enumerate(modes, start=1):
            local = mode.pop("observation_indices")
            selected_queries = query_indices[start:stop][local]
            selected_bins = bins[start:stop][local]
            trajectories = {
                trajectory_id(names[int(index)])
                for index in selected_queries.tolist()
            }
            view_bins = set(selected_bins.tolist())
            if (
                len(trajectories) < int(minimum_trajectories)
                or len(view_bins) < int(minimum_view_bins)
                or mode["primary_similarity"]
                > float(maximum_primary_similarity)
            ):
                continue
            mode.update(
                {
                    "source_anchor": int(anchor),
                    "mode_id": int(mode_id),
                    "trajectory_count": len(trajectories),
                    "view_bin_count": len(view_bins),
                    "provenance_mask": bitwise_union(
                        provenance[start:stop][local]
                    ),
                    "utility": float(
                        mode["activation_gain_mean"]
                        * mode["observation_count"]
                    ),
                    "_observation_slot": len(observation_query_indices),
                }
            )
            candidates.append(mode)
            observation_query_indices.append(selected_queries)
            observation_rows.append(rows[start:stop][local])
            observation_provenance.append(provenance[start:stop][local])
            mode_offsets.append(
                mode_offsets[-1] + int(mode["observation_count"])
            )
    candidates.sort(
        key=lambda value: (
            -value["utility"],
            -value["trajectory_count"],
            -value["view_bin_count"],
            value["source_anchor"],
            value["mode_id"],
        )
    )
    # Sorting candidates requires applying the same permutation to CSR observations.
    ordered_observations = [
        (
            observation_query_indices[int(value["_observation_slot"])],
            observation_rows[int(value["_observation_slot"])],
            observation_provenance[int(value["_observation_slot"])],
        )
        for value in candidates
    ]
    for value in candidates:
        value.pop("_observation_slot")
    mode_offsets = [0]
    for value in candidates:
        mode_offsets.append(mode_offsets[-1] + int(value["observation_count"]))
    observation_payload = {
        "offsets": torch.as_tensor(mode_offsets, dtype=torch.long),
        "query_indices": torch.cat([value[0] for value in ordered_observations])
        if ordered_observations
        else torch.empty(0, dtype=torch.int32),
        "query_rows": torch.cat([value[1] for value in ordered_observations])
        if ordered_observations
        else torch.empty(0, dtype=torch.int32),
        "provenance": torch.cat([value[2] for value in ordered_observations])
        if ordered_observations
        else torch.empty(0, dtype=torch.uint8),
    }
    for value in candidates:
        value["initial_bias"] = min(float(initial_bias), 0.0)
    return candidates, observation_payload


def collapse_duplicate_conflicts(base: dict, conflict: dict) -> list[dict]:
    base_count = int(torch.as_tensor(base["anchor_xyz"]).shape[0])
    conflict_count = int(torch.as_tensor(conflict["anchor_xyz"]).shape[0])
    metadata = conflict.get("basin_conflict_anchors")
    if not isinstance(metadata, dict):
        raise ValueError("conflict map has no basin conflict metadata")
    parents = torch.as_tensor(metadata["source_anchor_rows"]).long()
    if conflict_count != base_count + parents.numel():
        raise ValueError("conflict rows are not an appended suffix")
    for key in ("anchor_xyz", "source_primitive_ids"):
        if not torch.equal(
            torch.as_tensor(base[key]),
            torch.as_tensor(conflict[key])[:base_count],
        ):
            raise ValueError(f"conflict map does not preserve base {key}")
    features = F.normalize(
        torch.as_tensor(conflict["anchor_features"]).float()[base_count:], dim=1
    )
    groups = torch.as_tensor(metadata.get("query_groups", -torch.ones_like(parents)))
    counts = torch.as_tensor(metadata.get("observation_counts", torch.ones_like(parents)))
    return [
        {
            "source_anchor": int(parent),
            "prototype": feature,
            "query_group": int(group),
            "observation_count": int(count),
            "trajectory_count": 0,
            "view_bin_count": 1,
            "oof_gain": 0.0,
            "oof_positive_fraction": 0.0,
            "utility": float(count),
        }
        for parent, feature, group, count in zip(
            parents.tolist(), features, groups.tolist(), counts.tolist()
        )
    ]


def _collect_stable_observations(
    teacher: dict,
    cache: dict,
    query_bins: dict[str, int],
    metric: SharedLowRankMetric,
) -> dict[int, list[dict]]:
    observations: dict[tuple[int, str, int], dict] = {}
    for record in teacher["records"]:
        name = str(record["query_name"])
        types = torch.as_tensor(record["set_types"]).long()
        correct = torch.as_tensor(record["correct_basin"]).bool()
        te_cm = torch.as_tensor(record["te_cm"]).float()
        re_deg = torch.as_tensor(record["re_deg"]).float()
        rows = torch.as_tensor(record["set_query_rows"]).long()
        anchors = torch.as_tensor(record["set_anchor_indices"]).long()
        valid_sets = torch.nonzero(
            correct & ((types == 0) | (types == 2)), as_tuple=False
        ).reshape(-1)
        blame_rows = torch.as_tensor(record["blame_rows"]).long()
        referenced_rows = torch.unique(
            torch.cat((rows[valid_sets].reshape(-1), blame_rows))
        )
        raw_native = torch.as_tensor(
            cache[name]["native_descriptors"]
        ).float()
        selected_native = F.normalize(raw_native[referenced_rows], dim=1)
        with torch.no_grad():
            selected_native, _ = metric(selected_native)
        native = {
            int(row): descriptor
            for row, descriptor in zip(
                referenced_rows.tolist(), selected_native
            )
        }
        for set_index in valid_sets.tolist():
            reward = float(
                torch.exp(
                    -te_cm[set_index].clamp_min(0) / 15.0
                    - re_deg[set_index].clamp_min(0) / 2.0
                )
            )
            for row, anchor in zip(
                rows[set_index].tolist(), anchors[set_index].tolist()
            ):
                key = (int(anchor), name, int(row))
                previous = observations.get(key)
                if previous is None or reward > previous["weight"]:
                    observations[key] = {
                        "anchor": int(anchor),
                        "query_name": name,
                        "row": int(row),
                        "descriptor": native[int(row)],
                        "weight": max(reward, 1e-3),
                        "view_bin": int(query_bins[name]),
                        "counterfactual": False,
                    }
        for row, anchor, weight in zip(
            blame_rows.tolist(),
            torch.as_tensor(record["blame_positive_anchors"]).long().tolist(),
            torch.as_tensor(record["blame_weights"]).float().tolist(),
        ):
            key = (int(anchor), name, int(row))
            counterfactual_weight = max(float(weight), 1e-3)
            previous = observations.get(key)
            if previous is None or counterfactual_weight > previous["weight"]:
                observations[key] = {
                    "anchor": int(anchor),
                    "query_name": name,
                    "row": int(row),
                    "descriptor": native[int(row)],
                    "weight": counterfactual_weight,
                    "view_bin": int(query_bins[name]),
                    "counterfactual": True,
                }
            else:
                previous["counterfactual"] = True
    grouped: dict[int, list[dict]] = defaultdict(list)
    for value in observations.values():
        grouped[int(value["anchor"])].append(value)
    return grouped


def _leave_trajectory_out_gain(
    values: list[dict],
    primary: torch.Tensor,
    *,
    trim_fraction: float,
) -> tuple[float, float]:
    gains = []
    trajectories = sorted({trajectory_id(value["query_name"]) for value in values})
    for held_out in trajectories:
        train = [
            value
            for value in values
            if trajectory_id(value["query_name"]) != held_out
        ]
        held = [
            value
            for value in values
            if trajectory_id(value["query_name"]) == held_out
        ]
        if not train or not held:
            continue
        prototype = robust_family_prototype(
            torch.stack([value["descriptor"] for value in train]),
            torch.as_tensor([value["weight"] for value in train]),
            trim_fraction=trim_fraction,
        )
        query = torch.stack([value["descriptor"] for value in held])
        gains.extend(((query @ prototype) - (query @ primary)).tolist())
    if not gains:
        return float("-inf"), 0.0
    tensor = torch.as_tensor(gains)
    return float(tensor.mean()), float((tensor > 0).float().mean())


def build_cross_view_families(
    state: dict,
    teacher: dict,
    cache: dict,
    query_bins: dict[str, int],
    metric: SharedLowRankMetric,
    *,
    minimum_observations: int,
    minimum_trajectories: int,
    minimum_view_bins: int,
    minimum_counterfactual_observations: int,
    minimum_oof_gain: float,
    minimum_oof_positive_fraction: float,
    trim_fraction: float,
    maximum_primary_similarity: float,
) -> list[dict]:
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    grouped = _collect_stable_observations(
        teacher, cache, query_bins, metric
    )
    candidates = []
    for anchor, all_values in grouped.items():
        primary = bank[anchor]
        descriptors = torch.stack([value["descriptor"] for value in all_values])
        similarity = descriptors @ primary
        seed = descriptors[int(similarity.argmin())]
        secondary_assignment = (descriptors @ seed) > similarity
        values = [
            value
            for value, selected in zip(all_values, secondary_assignment.tolist())
            if selected
        ]
        if len(values) < int(minimum_observations):
            continue
        trajectories = {trajectory_id(value["query_name"]) for value in values}
        view_bins = {int(value["view_bin"]) for value in values}
        counterfactual_count = sum(value["counterfactual"] for value in values)
        if (
            len(trajectories) < int(minimum_trajectories)
            or len(view_bins) < int(minimum_view_bins)
            or counterfactual_count < int(minimum_counterfactual_observations)
        ):
            continue
        oof_gain, positive_fraction = _leave_trajectory_out_gain(
            values, primary, trim_fraction=trim_fraction
        )
        if (
            oof_gain < float(minimum_oof_gain)
            or positive_fraction < float(minimum_oof_positive_fraction)
        ):
            continue
        weights = torch.as_tensor([value["weight"] for value in values])
        prototype = robust_family_prototype(
            torch.stack([value["descriptor"] for value in values]),
            weights,
            trim_fraction=trim_fraction,
        )
        primary_similarity = float(prototype @ primary)
        if primary_similarity > float(maximum_primary_similarity):
            continue
        candidates.append(
            {
                "source_anchor": int(anchor),
                "prototype": prototype,
                "observation_count": len(values),
                "trajectory_count": len(trajectories),
                "view_bin_count": len(view_bins),
                "counterfactual_observation_count": int(counterfactual_count),
                "primary_similarity": primary_similarity,
                "oof_gain": oof_gain,
                "oof_positive_fraction": positive_fraction,
                "utility": oof_gain
                * positive_fraction
                * float(weights.sum()),
            }
        )
    candidates.sort(
        key=lambda value: (
            -value["utility"],
            -value["trajectory_count"],
            -value["view_bin_count"],
            value["source_anchor"],
        )
    )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--mode",
        choices=("collapse_duplicate", "cross_view_stable", "appearance_pool"),
        required=True,
    )
    parser.add_argument("--conflict-map", default="")
    parser.add_argument("--basin-teacher", default="")
    parser.add_argument("--query-cache", default="")
    parser.add_argument("--track-payload", default="")
    parser.add_argument("--metric-state", default="")
    parser.add_argument("--complete-positive-teacher", default="")
    parser.add_argument("--dynamic-outcomes", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--maximum-modes-per-anchor", type=int, default=3)
    parser.add_argument("--minimum-separation", type=float, default=0.08)
    parser.add_argument("--initial-bias", type=float, default=-0.01)
    parser.add_argument("--maximum-prototypes", type=int, default=0)
    parser.add_argument("--minimum-observations", type=int, default=4)
    parser.add_argument("--minimum-trajectories", type=int, default=2)
    parser.add_argument("--minimum-view-bins", type=int, default=3)
    parser.add_argument("--minimum-counterfactual-observations", type=int, default=1)
    parser.add_argument("--minimum-oof-gain", type=float, default=0.01)
    parser.add_argument("--minimum-oof-positive-fraction", type=float, default=0.6)
    parser.add_argument("--trim-fraction", type=float, default=0.2)
    parser.add_argument("--maximum-primary-similarity", type=float, default=0.98)
    args = parser.parse_args()

    state = torch.load(args.map, map_location="cpu", weights_only=False)
    observation_payload = None
    if args.mode == "collapse_duplicate":
        if not args.conflict_map:
            raise ValueError("collapse_duplicate requires --conflict-map")
        conflict = torch.load(
            args.conflict_map, map_location="cpu", weights_only=False
        )
        candidates = collapse_duplicate_conflicts(state, conflict)
    elif args.mode == "cross_view_stable":
        required = (
            args.basin_teacher,
            args.query_cache,
            args.track_payload,
            args.metric_state,
        )
        if not all(required):
            raise ValueError("cross_view_stable inputs are incomplete")
        teacher = torch.load(
            args.basin_teacher, map_location="cpu", weights_only=False
        )
        cache_payload = torch.load(
            args.query_cache, map_location="cpu", weights_only=False
        )
        cache = cache_payload.get("queries", cache_payload)
        payload = torch.load(
            args.track_payload, map_location="cpu", weights_only=False
        )
        query_bins = {
            name: int(group)
            for name, group in zip(
                payload["query_names"], payload["query_bins"].tolist()
            )
        }
        metric_payload = torch.load(
            args.metric_state, map_location="cpu", weights_only=False
        )
        metric = SharedLowRankMetric(**metric_payload["metric_config"]).eval()
        metric.load_state_dict(metric_payload["metric_state_dict"])
        candidates = build_cross_view_families(
            state,
            teacher,
            cache,
            query_bins,
            metric,
            minimum_observations=args.minimum_observations,
            minimum_trajectories=args.minimum_trajectories,
            minimum_view_bins=args.minimum_view_bins,
            minimum_counterfactual_observations=(
                args.minimum_counterfactual_observations
            ),
            minimum_oof_gain=args.minimum_oof_gain,
            minimum_oof_positive_fraction=args.minimum_oof_positive_fraction,
            trim_fraction=args.trim_fraction,
            maximum_primary_similarity=args.maximum_primary_similarity,
        )
    else:
        required = (
            args.complete_positive_teacher,
            args.query_cache,
            args.track_payload,
            args.metric_state,
        )
        if not all(required):
            raise ValueError("appearance_pool inputs are incomplete")
        positives = torch.load(
            args.complete_positive_teacher,
            map_location="cpu",
            weights_only=False,
        )
        cache_payload = torch.load(
            args.query_cache, map_location="cpu", weights_only=False
        )
        cache = cache_payload.get("queries", cache_payload)
        payload = torch.load(
            args.track_payload, map_location="cpu", weights_only=False
        )
        query_bins = {
            name: int(group)
            for name, group in zip(
                payload["query_names"], payload["query_bins"].tolist()
            )
        }
        metric_payload = torch.load(
            args.metric_state, map_location="cpu", weights_only=False
        )
        device = torch.device(args.device)
        metric = SharedLowRankMetric(**metric_payload["metric_config"]).to(device)
        metric.load_state_dict(metric_payload["metric_state_dict"])
        metric.eval()
        dynamic = (
            torch.load(
                args.dynamic_outcomes, map_location="cpu", weights_only=False
            )
            if args.dynamic_outcomes
            else None
        )
        basin_teacher = (
            torch.load(
                args.basin_teacher, map_location="cpu", weights_only=False
            )
            if args.basin_teacher
            else None
        )
        candidates, observation_payload = build_appearance_mode_pool(
            state,
            positives,
            cache,
            query_bins,
            metric,
            dynamic=dynamic,
            basin_teacher=basin_teacher,
            maximum_modes_per_anchor=args.maximum_modes_per_anchor,
            minimum_observations=args.minimum_observations,
            minimum_trajectories=args.minimum_trajectories,
            minimum_view_bins=args.minimum_view_bins,
            minimum_separation=args.minimum_separation,
            maximum_primary_similarity=args.maximum_primary_similarity,
            trim_fraction=args.trim_fraction,
            initial_bias=args.initial_bias,
            device=device,
        )
    if int(args.maximum_prototypes) > 0:
        candidates = candidates[: int(args.maximum_prototypes)]
        if observation_payload is not None:
            offsets = observation_payload["offsets"]
            observation_stop = int(offsets[len(candidates)])
            observation_payload = {
                "offsets": offsets[: len(candidates) + 1].clone(),
                "query_indices": observation_payload["query_indices"][
                    :observation_stop
                ].clone(),
                "query_rows": observation_payload["query_rows"][
                    :observation_stop
                ].clone(),
                "provenance": observation_payload["provenance"][
                    :observation_stop
                ].clone(),
            }
    prototype_features = (
        torch.stack([value["prototype"] for value in candidates])
        if candidates
        else torch.empty(
            0, torch.as_tensor(state["anchor_features"]).reshape(
                len(state["anchor_features"]), -1
            ).shape[1]
        )
    )
    prototype_features = prototype_features.detach()
    output = {
        "schema": "lafgs_basin_family_prototypes",
        "version": 1,
        "landmark_indices": torch.arange(
            torch.as_tensor(state["anchor_xyz"]).shape[0], dtype=torch.long
        ),
        "prototype_features": prototype_features,
        "prototype_anchor_indices": torch.as_tensor(
            [value["source_anchor"] for value in candidates], dtype=torch.long
        ),
        "prototype_bias": torch.as_tensor(
            [value.get("initial_bias", 0.0) for value in candidates],
            dtype=torch.float32,
        ),
        "prototype_temperature": torch.ones(len(candidates)),
        "config": vars(args),
        "families": [
            {key: value for key, value in candidate.items() if key != "prototype"}
            for candidate in candidates
        ],
    }
    if observation_payload is not None:
        output["mode_observations"] = observation_payload
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    report = {
        "schema": output["schema"],
        "prototype_count": len(candidates),
        "geometry_anchor_count": int(
            torch.as_tensor(state["anchor_xyz"]).shape[0]
        ),
        "unique_parent_count": len(
            {value["source_anchor"] for value in candidates}
        ),
        "config": vars(args),
        "families": output["families"],
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
