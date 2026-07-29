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
        "--mode", choices=("collapse_duplicate", "cross_view_stable"), required=True
    )
    parser.add_argument("--conflict-map", default="")
    parser.add_argument("--basin-teacher", default="")
    parser.add_argument("--query-cache", default="")
    parser.add_argument("--track-payload", default="")
    parser.add_argument("--metric-state", default="")
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
    if args.mode == "collapse_duplicate":
        if not args.conflict_map:
            raise ValueError("collapse_duplicate requires --conflict-map")
        conflict = torch.load(
            args.conflict_map, map_location="cpu", weights_only=False
        )
        candidates = collapse_duplicate_conflicts(state, conflict)
    else:
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
    if int(args.maximum_prototypes) > 0:
        candidates = candidates[: int(args.maximum_prototypes)]
    prototype_features = (
        torch.stack([value["prototype"] for value in candidates])
        if candidates
        else torch.empty(
            0, torch.as_tensor(state["anchor_features"]).reshape(
                len(state["anchor_features"]), -1
            ).shape[1]
        )
    )
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
        "config": vars(args),
        "families": [
            {key: value for key, value in candidate.items() if key != "prototype"}
            for candidate in candidates
        ],
    }
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
