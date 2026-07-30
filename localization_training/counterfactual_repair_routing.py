"""Route counterfactual repairs using the support of the repair event itself."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


ROUTE_PRIMARY = 0
ROUTE_FAMILY = 1
ROUTE_REJECT = 2


@dataclass(frozen=True)
class RepairRoutingConfig:
    minimum_primary_repair_trajectories: int = 2
    minimum_family_repair_observations: int = 2
    minimum_family_repair_trajectories: int = 2
    minimum_descriptor_cosine: float = 0.6
    family_prototype_bias: float = -0.05


def assign_repair_routes(
    occurrences: list[tuple[int, str]],
    config: RepairRoutingConfig,
) -> list[int]:
    """Assign primary/family/reject from repeated repair evidence."""

    grouped: dict[int, list[tuple[int, str]]] = defaultdict(list)
    for index, (target, trajectory) in enumerate(occurrences):
        if int(target) >= 0:
            grouped[int(target)].append((index, str(trajectory)))
    routes = [ROUTE_REJECT] * len(occurrences)
    for values in grouped.values():
        trajectories = {trajectory for _, trajectory in values}
        if len(trajectories) >= int(
            config.minimum_primary_repair_trajectories
        ):
            route = ROUTE_PRIMARY
        elif len(values) >= int(config.minimum_family_repair_observations):
            route = ROUTE_FAMILY
        else:
            route = ROUTE_REJECT
        for index, _ in values:
            routes[index] = route
    return routes


def assign_descriptor_consistent_repair_routes(
    occurrences: list[tuple[int, str]],
    descriptors: torch.Tensor,
    config: RepairRoutingConfig,
) -> tuple[list[int], list[int]]:
    """Route only descriptor-coherent, cross-trajectory repair modes."""

    descriptors = F.normalize(
        torch.as_tensor(descriptors).float().reshape(len(occurrences), -1),
        dim=1,
    )
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, (target, _) in enumerate(occurrences):
        if int(target) >= 0:
            grouped[int(target)].append(index)
    routes = [ROUTE_REJECT] * len(occurrences)
    family_clusters = [-1] * len(occurrences)
    next_cluster = 0
    threshold = float(config.minimum_descriptor_cosine)

    for indices in grouped.values():
        remaining = list(indices)
        modes: list[list[int]] = []
        while remaining:
            values = descriptors[torch.as_tensor(remaining).long()]
            similarity = values @ values.T
            best: tuple[tuple[int, int, float, int], list[int]] | None = None
            for local, seed in enumerate(remaining):
                # Complete-link growth prevents two mutually inconsistent
                # appearances from entering one mode merely because both are
                # close to a permissive seed.
                support_local = [local]
                candidates = [
                    value
                    for value in range(len(remaining))
                    if value != local
                    and float(similarity[local, value]) >= threshold
                ]
                candidates.sort(
                    key=lambda value: (
                        -float(similarity[local, value]),
                        remaining[value],
                    )
                )
                for value in candidates:
                    if bool(
                        (
                            similarity[
                                value,
                                torch.as_tensor(support_local).long(),
                            ]
                            >= threshold
                        ).all()
                    ):
                        support_local.append(value)
                support = [remaining[value] for value in support_local]
                trajectories = {
                    str(occurrences[value][1]) for value in support
                }
                if (
                    len(support)
                    < int(config.minimum_family_repair_observations)
                    or len(trajectories)
                    < int(config.minimum_family_repair_trajectories)
                ):
                    continue
                score = (
                    len(trajectories),
                    len(support),
                    float(
                        similarity[
                            torch.as_tensor(support_local).long()
                        ][:, torch.as_tensor(support_local).long()].min()
                    ),
                    -int(seed),
                )
                if best is None or score > best[0]:
                    best = (score, support)
            if best is None:
                break
            support = best[1]
            modes.append(support)
            selected = set(support)
            remaining = [value for value in remaining if value not in selected]

        covered = sum(len(mode) for mode in modes)
        if (
            len(modes) == 1
            and covered == len(indices)
            and len(
                {
                    str(occurrences[value][1])
                    for value in modes[0]
                }
            )
            >= int(config.minimum_primary_repair_trajectories)
        ):
            for index in modes[0]:
                routes[index] = ROUTE_PRIMARY
            continue
        for mode in modes:
            for index in mode:
                routes[index] = ROUTE_FAMILY
                family_clusters[index] = next_cluster
            next_cluster += 1
    return routes, family_clusters


def _repair_targets(record: dict) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.as_tensor(record["query_rows"]).long()
    if "counterfactual_pose_best" in record:
        targets = torch.as_tensor(
            record["counterfactual_pose_best"]
        ).long()
    elif "target_anchor_indices" in record:
        targets = torch.as_tensor(
            record["target_anchor_indices"]
        ).long()
        if "accepted" in record:
            accepted = torch.as_tensor(record["accepted"]).bool()
            targets = torch.where(
                accepted, targets, torch.full_like(targets, -1)
            )
    else:
        raise ValueError("counterfactual audit has no repair target")
    if len(rows) != len(targets):
        raise ValueError("counterfactual repair targets do not align")
    return rows, targets


def _pack(values: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    counts = torch.as_tensor([len(value) for value in values], dtype=torch.long)
    offsets = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    if int(offsets[-1]) == 0:
        return offsets, torch.empty(0, dtype=torch.long)
    return offsets, torch.cat(values).long()


def route_counterfactual_repairs(
    *,
    audit: dict,
    positive_teacher: dict,
    family: dict,
    query_cache: dict,
    metric,
    anchor_count: int,
    config: RepairRoutingConfig,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    """Reroute strict targets and materialize robust family prototypes."""

    names = list(audit["query_names"])
    if names != list(positive_teacher["query_names"]):
        raise ValueError("repair routing query registries differ")
    cache = query_cache.get("queries", query_cache)
    occurrences = []
    occurrence_location = []
    raw_descriptors = []
    for query_index, (name, record) in enumerate(
        zip(names, audit["records"])
    ):
        rows, targets = _repair_targets(record)
        for local, (row, target) in enumerate(
            zip(rows.tolist(), targets.tolist())
        ):
            occurrences.append((int(target), name.split("/", 1)[0]))
            occurrence_location.append((query_index, local))
            raw_descriptors.append(
                torch.as_tensor(cache[name]["native_descriptors"])[int(row)]
            )
    raw = F.normalize(torch.stack(raw_descriptors).float().to(device), dim=1)
    with torch.no_grad():
        descriptors, _ = metric(raw)
        descriptors = F.normalize(descriptors, dim=1).cpu()
    routes, family_cluster_ids = assign_descriptor_consistent_repair_routes(
        occurrences,
        descriptors,
        config,
    )

    family_features = [
        value.clone()
        for value in torch.as_tensor(family["prototype_features"]).float()
    ]
    family_parents = torch.as_tensor(
        family["prototype_anchor_indices"]
    ).long().tolist()
    family_bias = torch.as_tensor(
        family.get("prototype_bias", torch.zeros(len(family_features)))
    ).float().tolist()
    family_temperature = torch.as_tensor(
        family.get(
            "prototype_temperature", torch.ones(len(family_features))
        )
    ).float().tolist()
    family_occurrences: dict[int, list[int]] = defaultdict(list)
    for index, route in enumerate(routes):
        if route == ROUTE_FAMILY:
            family_occurrences[int(family_cluster_ids[index])].append(index)
    family_representation = {}
    for cluster, indices in sorted(family_occurrences.items()):
        values = descriptors[torch.as_tensor(indices).long()]
        similarity = values @ values.T
        medoid = values[int(torch.argmax(similarity.mean(dim=1)))]
        family_representation[cluster] = anchor_count + len(family_features)
        family_features.append(medoid)
        family_parents.append(int(occurrences[indices[0]][0]))
        family_bias.append(float(config.family_prototype_bias))
        family_temperature.append(1.0)

    routed_records = [dict(record) for record in audit["records"]]
    target_representation_by_record = [
        torch.full(
            (len(torch.as_tensor(record["query_rows"])),),
            -1,
            dtype=torch.long,
        )
        for record in routed_records
    ]
    route_by_record = [
        torch.full_like(value, ROUTE_REJECT, dtype=torch.int8)
        for value in target_representation_by_record
    ]
    for index, ((target, _), route) in enumerate(zip(occurrences, routes)):
        query_index, local = occurrence_location[index]
        route_by_record[query_index][local] = int(route)
        if route == ROUTE_PRIMARY:
            target_representation_by_record[query_index][local] = int(target)
        elif route == ROUTE_FAMILY:
            target_representation_by_record[query_index][local] = int(
                family_representation[int(family_cluster_ids[index])]
            )
    for index, record in enumerate(routed_records):
        record["route"] = route_by_record[index]
        record["target_representation"] = (
            target_representation_by_record[index]
        )

    routed_teacher_records = []
    route_counts = {"primary": 0, "family": 0, "reject": 0}
    target_rows = 0
    target_pairs = 0
    for audit_record, teacher_record in zip(
        routed_records, positive_teacher["records"]
    ):
        audit_rows, audit_targets = _repair_targets(audit_record)
        audit_routes = torch.as_tensor(audit_record["route"]).long()
        routed = {
            int(row): (
                int(target) if int(route) != ROUTE_REJECT else -1
            )
            for row, target, route in zip(
                audit_rows.tolist(),
                audit_targets.tolist(),
                audit_routes.tolist(),
            )
        }
        offsets = torch.as_tensor(teacher_record["positive_offsets"]).long()
        indices = torch.as_tensor(teacher_record["positive_indices"]).long()
        rows = torch.as_tensor(teacher_record["query_rows"]).long()
        values = []
        for row_index, row in enumerate(rows.tolist()):
            if int(row) in routed:
                target = routed[int(row)]
                value = (
                    torch.tensor([target], dtype=torch.long)
                    if target >= 0
                    else torch.empty(0, dtype=torch.long)
                )
            else:
                value = indices[
                    int(offsets[row_index]) : int(offsets[row_index + 1])
                ]
            values.append(value)
            target_rows += int(len(value) > 0)
            target_pairs += len(value)
        positive_offsets, positive_indices = _pack(values)
        routed_teacher_records.append(
            {
                **teacher_record,
                "positive_offsets": positive_offsets,
                "positive_indices": positive_indices,
            }
        )
        for route in audit_routes.tolist():
            route_counts[("primary", "family", "reject")[int(route)]] += 1

    routed_audit = {
        **audit,
        "version": int(audit.get("version", 1)) + 1,
        "records": routed_records,
        "summary": {
            **dict(audit.get("summary", {})),
            "event_support_routing": {
                "route_counts": route_counts,
                "family_prototype_count": len(family_occurrences),
                "descriptor_coherence_enabled": True,
            },
        },
        "repair_routing_config": asdict(config),
    }
    routed_teacher = {
        **positive_teacher,
        "version": int(positive_teacher.get("version", 1)) + 1,
        "records": routed_teacher_records,
        "diagnostics": {
            **dict(positive_teacher.get("diagnostics", {})),
            "positive_rows": int(target_rows),
            "strong_pair_count": int(target_pairs),
            "event_support_routing": routed_audit["summary"][
                "event_support_routing"
            ],
        },
    }
    routed_family = {
        **family,
        "prototype_features": torch.stack(family_features),
        "prototype_anchor_indices": torch.as_tensor(
            family_parents, dtype=torch.long
        ),
        "prototype_bias": torch.as_tensor(family_bias).float(),
        "prototype_temperature": torch.as_tensor(
            family_temperature
        ).float(),
        "counterfactual_repair_routing": {
            "config": asdict(config),
            "added_family_prototype_count": len(family_occurrences),
        },
    }
    return routed_audit, routed_teacher, routed_family
