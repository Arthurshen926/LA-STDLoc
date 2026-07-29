#!/usr/bin/env python3
"""Add query-group descriptor anchors for repeatable counterfactual conflicts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from localization_training.shared_metric import SharedLowRankMetric


def verify_teacher_anchor_prefix(state: dict, teacher: dict) -> None:
    teacher_count = int(teacher["anchor_count"])
    state_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    if teacher_count > state_count:
        raise ValueError("basin teacher has more anchors than the target map")
    source_record = teacher.get("artifacts", {}).get("map")
    if not source_record or not Path(source_record["path"]).is_file():
        raise ValueError("basin teacher does not expose its source map")
    source = torch.load(
        source_record["path"], map_location="cpu", weights_only=False
    )
    for key in ("anchor_ids", "anchor_xyz", "source_primitive_ids"):
        expected = torch.as_tensor(source[key])[:teacher_count]
        actual = torch.as_tensor(state[key])[:teacher_count]
        if expected.shape != actual.shape or not torch.equal(expected, actual):
            raise ValueError(f"target map does not preserve basin-teacher {key} prefix")


def build_conflict_groups(
    state: dict,
    teacher: dict,
    cache: dict,
    query_groups: torch.Tensor,
    *,
    minimum_observations: int,
    minimum_margin_gain: float,
    prototype_weight: float,
    query_metric: SharedLowRankMetric | None = None,
) -> list[dict]:
    bank = F.normalize(torch.as_tensor(state["anchor_features"]).float(), dim=1)
    observations: dict[tuple[int, int], list[tuple[torch.Tensor, int, float]]] = (
        defaultdict(list)
    )
    for record in teacher["records"]:
        query_index = int(record["query_index"])
        rows = torch.as_tensor(record["blame_rows"]).long()
        if not rows.numel():
            continue
        descriptors = F.normalize(
            torch.as_tensor(
                cache[record["query_name"]]["native_descriptors"]
            ).float()[rows],
            dim=1,
        )
        if query_metric is not None:
            descriptors, _ = query_metric(descriptors)
        positive = torch.as_tensor(record["blame_positive_anchors"]).long()
        harmful = torch.as_tensor(record["blame_harmful_anchors"]).long()
        weights = torch.as_tensor(record["blame_weights"]).float()
        group = int(query_groups[query_index])
        for descriptor, pos, neg, weight in zip(
            descriptors, positive.tolist(), harmful.tolist(), weights.tolist()
        ):
            observations[(int(pos), group)].append(
                (descriptor, int(neg), float(weight))
            )
    candidates = []
    for (anchor, group), values in observations.items():
        query_count = len(values)
        if query_count < int(minimum_observations):
            continue
        descriptors = torch.stack([value[0] for value in values])
        weights = torch.as_tensor([value[2] for value in values]).clamp_min(1e-6)
        prototype = F.normalize(
            (descriptors * weights[:, None]).sum(dim=0), dim=0
        )
        prototype = F.normalize(
            float(prototype_weight) * prototype
            + (1.0 - float(prototype_weight)) * bank[anchor],
            dim=0,
        )
        harmful = torch.as_tensor([value[1] for value in values]).long()
        original_margin = (
            descriptors @ bank[anchor]
            - torch.einsum("bd,bd->b", descriptors, bank[harmful])
        )
        prototype_margin = (
            descriptors @ prototype
            - torch.einsum("bd,bd->b", descriptors, bank[harmful])
        )
        gain = float(
            ((prototype_margin - original_margin) * weights).sum()
            / weights.sum().clamp_min(1e-8)
        )
        if gain < float(minimum_margin_gain):
            continue
        candidates.append(
            {
                "source_anchor": anchor,
                "query_group": group,
                "observation_count": query_count,
                "weighted_margin_gain": gain,
                "prototype": prototype,
                "utility": gain * float(weights.sum()),
            }
        )
    candidates.sort(
        key=lambda value: (
            -value["utility"],
            -value["observation_count"],
            value["source_anchor"],
            value["query_group"],
        )
    )
    return candidates


def append_conflict_anchors(
    state: dict, candidates: list[dict], maximum_additions: int
) -> dict:
    output = dict(state)
    anchor_count = int(torch.as_tensor(state["anchor_xyz"]).shape[0])
    selected = candidates[: int(maximum_additions)]
    source_rows = torch.as_tensor(
        [value["source_anchor"] for value in selected], dtype=torch.long
    )
    for key, value in state.items():
        if not (
            torch.is_tensor(value)
            and value.ndim
            and int(value.shape[0]) == anchor_count
        ):
            continue
        additions = value[source_rows].clone()
        if key == "anchor_features":
            additions = torch.stack([item["prototype"] for item in selected]).to(
                value.dtype
            )
        elif key in {"anchor_ids", "fine_identity_ids"}:
            start = int(value.max()) + 1 if value.numel() else 0
            additions = torch.arange(
                start, start + len(selected), dtype=value.dtype
            )
        output[key] = torch.cat((value, additions), dim=0)
    output["canonical_anchor_count"] = anchor_count + len(selected)
    output["basin_conflict_anchors"] = {
        "schema": "lafgs_basin_conflict_anchors",
        "source_anchor_rows": source_rows,
        "query_groups": torch.as_tensor(
            [value["query_group"] for value in selected], dtype=torch.long
        ),
        "observation_counts": torch.as_tensor(
            [value["observation_count"] for value in selected], dtype=torch.long
        ),
        "weighted_margin_gains": torch.as_tensor(
            [value["weighted_margin_gain"] for value in selected]
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--basin-teacher", required=True)
    parser.add_argument("--query-cache", required=True)
    parser.add_argument("--track-payload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--maximum-additions", type=int, default=256)
    parser.add_argument("--minimum-observations", type=int, default=2)
    parser.add_argument("--minimum-margin-gain", type=float, default=0.05)
    parser.add_argument("--prototype-weight", type=float, default=0.75)
    parser.add_argument("--metric-state", default="")
    args = parser.parse_args()
    state = torch.load(args.map, map_location="cpu", weights_only=False)
    teacher = torch.load(
        args.basin_teacher, map_location="cpu", weights_only=False
    )
    verify_teacher_anchor_prefix(state, teacher)
    cache_payload = torch.load(
        args.query_cache, map_location="cpu", weights_only=False
    )
    cache = cache_payload.get("queries", cache_payload)
    payload = torch.load(
        args.track_payload, map_location="cpu", weights_only=False
    )
    query_metric = None
    if args.metric_state:
        metric_payload = torch.load(
            args.metric_state, map_location="cpu", weights_only=False
        )
        query_metric = SharedLowRankMetric(
            **metric_payload["metric_config"]
        ).eval()
        query_metric.load_state_dict(metric_payload["metric_state_dict"])
    teacher_index = {
        name: index for index, name in enumerate(teacher["query_names"])
    }
    query_groups = torch.empty(len(teacher["query_names"]), dtype=torch.long)
    assigned = torch.zeros(len(teacher["query_names"]), dtype=torch.bool)
    for name, group in zip(payload["query_names"], payload["query_bins"].tolist()):
        if name in teacher_index:
            query_groups[teacher_index[name]] = int(group)
            assigned[teacher_index[name]] = True
    if not bool(assigned.all()):
        raise ValueError("track payload does not cover every basin-teacher query")
    candidates = build_conflict_groups(
        state,
        teacher,
        cache,
        query_groups,
        minimum_observations=args.minimum_observations,
        minimum_margin_gain=args.minimum_margin_gain,
        prototype_weight=args.prototype_weight,
        query_metric=query_metric,
    )
    output = append_conflict_anchors(
        state, candidates, maximum_additions=args.maximum_additions
    )
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, path)
    report = {
        "schema": "lafgs_basin_conflict_anchor_build",
        "source_anchor_count": int(torch.as_tensor(state["anchor_xyz"]).shape[0]),
        "eligible_conflict_family_count": len(candidates),
        "added_anchor_count": min(len(candidates), int(args.maximum_additions)),
        "output_anchor_count": int(
            torch.as_tensor(output["anchor_xyz"]).shape[0]
        ),
        "config": vars(args),
        "selected": [
            {key: value for key, value in item.items() if key != "prototype"}
            for item in candidates[: int(args.maximum_additions)]
        ],
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
