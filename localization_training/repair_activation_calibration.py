"""Cross-trajectory safety calibration for counterfactual repair routes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from localization_training.counterfactual_repair_routing import (
    ROUTE_FAMILY,
    ROUTE_PRIMARY,
    ROUTE_REJECT,
)


@dataclass(frozen=True)
class RepairActivationCalibrationConfig:
    minimum_global_precision: float = 0.8
    minimum_trajectory_precision: float = 0.5
    minimum_supported_trajectories: int = 2
    minimum_true_activations_per_trajectory: int = 1
    activation_margin: float = 0.0
    descriptor_batch_size: int = 8192


def _descriptor_medoid(descriptors: torch.Tensor) -> torch.Tensor:
    values = F.normalize(torch.as_tensor(descriptors).float(), dim=1)
    if not len(values):
        raise ValueError("a repair mode needs at least one descriptor")
    similarity = values @ values.T
    return values[int(torch.argmax(similarity.mean(dim=1)))]


def _positive_rows_by_target(
    record: dict,
    targets: set[int],
) -> dict[int, set[int]]:
    rows = torch.as_tensor(record["query_rows"]).long()
    offsets = torch.as_tensor(record["positive_offsets"]).long()
    indices = torch.as_tensor(record["positive_indices"]).long()
    output: dict[int, set[int]] = defaultdict(set)
    for local, row in enumerate(rows.tolist()):
        start = int(offsets[local])
        stop = int(offsets[local + 1])
        for target in indices[start:stop].tolist():
            target = int(target)
            if target in targets:
                output[target].add(int(row))
    return output


def _transform_descriptors(
    descriptors: torch.Tensor,
    metric,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    values = torch.as_tensor(descriptors).float()
    transformed = []
    with torch.no_grad():
        for start in range(0, len(values), max(int(batch_size), 1)):
            block, _ = metric(
                F.normalize(
                    values[start : start + max(int(batch_size), 1)].to(device),
                    dim=1,
                )
            )
            transformed.append(F.normalize(block, dim=1).cpu())
    return torch.cat(transformed) if transformed else values


def _mode_occurrences(
    routed_audit: dict,
    query_cache: dict,
) -> tuple[dict[tuple[int, int], list[dict]], dict[tuple[int, int], int]]:
    cache = query_cache.get("queries", query_cache)
    occurrences: dict[tuple[int, int], list[dict]] = defaultdict(list)
    parents: dict[tuple[int, int], int] = {}
    for record in routed_audit["records"]:
        name = str(record["query_name"])
        trajectory = name.split("/", 1)[0]
        rows = torch.as_tensor(record["query_rows"]).long()
        targets = torch.as_tensor(record["target_anchor_indices"]).long()
        routes = torch.as_tensor(record["route"]).long()
        representations = torch.as_tensor(
            record["target_representation"]
        ).long()
        for row, target, route, representation in zip(
            rows.tolist(),
            targets.tolist(),
            routes.tolist(),
            representations.tolist(),
        ):
            if int(route) not in (ROUTE_PRIMARY, ROUTE_FAMILY):
                continue
            key = (int(route), int(representation))
            if int(representation) < 0 or int(target) < 0:
                raise ValueError("accepted repair route has no target")
            previous = parents.setdefault(key, int(target))
            if previous != int(target):
                raise ValueError("one repair mode maps to multiple anchors")
            occurrences[key].append(
                {
                    "query_name": name,
                    "trajectory": trajectory,
                    "query_row": int(row),
                    "descriptor": torch.as_tensor(
                        cache[name]["native_descriptors"]
                    )[int(row)],
                }
            )
    return occurrences, parents


def calibrate_repair_route_activations(
    *,
    routed_audit: dict,
    routed_teacher: dict,
    routed_family: dict,
    base_family_count: int,
    positive_teacher: dict,
    query_cache: dict,
    dynamic_outcomes: dict,
    metric,
    anchor_count: int,
    config: RepairActivationCalibrationConfig,
    device: torch.device,
) -> tuple[dict, dict, dict, dict]:
    """Apply a leave-one-trajectory-out false-attractor gate.

    A candidate mode is reconstructed without one supporting trajectory and
    scored against every deployment row in that trajectory. It survives only
    when its activations are sufficiently precise and recover a verified
    positive in every required held-out trajectory.
    """

    names = list(routed_audit["query_names"])
    if not (
        names
        == list(routed_teacher["query_names"])
        == list(positive_teacher["query_names"])
        == list(dynamic_outcomes["query_names"])
    ):
        raise ValueError("repair calibration query registries differ")
    if int(dynamic_outcomes["anchor_count"]) != int(anchor_count):
        raise ValueError("repair calibration map registry differs")
    cache = query_cache.get("queries", query_cache)
    occurrences, parents = _mode_occurrences(routed_audit, query_cache)
    if not occurrences:
        raise ValueError("repair calibration has no accepted modes")

    flat_descriptors = []
    flat_locations = []
    for key, values in sorted(occurrences.items()):
        for local, value in enumerate(values):
            flat_descriptors.append(value["descriptor"])
            flat_locations.append((key, local))
    transformed = _transform_descriptors(
        torch.stack(flat_descriptors),
        metric,
        device=device,
        batch_size=config.descriptor_batch_size,
    )
    for descriptor, (key, local) in zip(transformed, flat_locations):
        occurrences[key][local]["metric_descriptor"] = descriptor

    holdout_modes: dict[str, list[dict]] = defaultdict(list)
    mode_reports: dict[tuple[int, int], dict] = {}
    for key, values in sorted(occurrences.items()):
        trajectories = sorted({value["trajectory"] for value in values})
        report = {
            "route": int(key[0]),
            "representation": int(key[1]),
            "target_anchor": int(parents[key]),
            "support_trajectories": trajectories,
            "trajectory_outcomes": {},
        }
        mode_reports[key] = report
        for trajectory in trajectories:
            training = [
                value["metric_descriptor"]
                for value in values
                if value["trajectory"] != trajectory
            ]
            if not training:
                report["trajectory_outcomes"][trajectory] = {
                    "eligible": False,
                    "reason": "no_cross_trajectory_training_support",
                }
                continue
            holdout_modes[trajectory].append(
                {
                    "key": key,
                    "target_anchor": int(parents[key]),
                    "feature": _descriptor_medoid(torch.stack(training)),
                }
            )

    dynamic_by_name = {
        str(record["query_name"]): record
        for record in dynamic_outcomes["records"]
    }
    teacher_by_name = {
        str(record["query_name"]): record
        for record in positive_teacher["records"]
    }
    family_bias = torch.as_tensor(
        routed_family.get(
            "prototype_bias",
            torch.zeros(
                len(torch.as_tensor(routed_family["prototype_features"]))
            ),
        )
    ).float()
    counters: dict[tuple[int, int], Counter] = defaultdict(Counter)
    for trajectory, modes in sorted(holdout_modes.items()):
        mode_features = torch.stack([value["feature"] for value in modes]).to(
            device
        )
        mode_bias = torch.as_tensor(
            [
                (
                    float(family_bias[int(value["key"][1]) - anchor_count])
                    if int(value["key"][0]) == ROUTE_FAMILY
                    else 0.0
                )
                for value in modes
            ],
            dtype=torch.float32,
            device=device,
        )
        targets = {int(value["target_anchor"]) for value in modes}
        for name in names:
            if name.split("/", 1)[0] != trajectory:
                continue
            dynamic = dynamic_by_name[name]
            query_rows = torch.as_tensor(dynamic["query_rows"]).long()
            descriptors = torch.as_tensor(cache[name]["native_descriptors"])[
                query_rows
            ]
            query = _transform_descriptors(
                descriptors,
                metric,
                device=device,
                batch_size=config.descriptor_batch_size,
            ).to(device)
            baseline = torch.as_tensor(
                dynamic["top1_scores"]
            ).float().to(device)
            if len(query) != len(baseline):
                raise ValueError("calibration dynamic rows do not align")
            scores = query @ mode_features.T + mode_bias[None]
            activated = scores > (
                baseline[:, None] + float(config.activation_margin)
            )
            positives = _positive_rows_by_target(
                teacher_by_name[name], targets
            )
            for column, mode in enumerate(modes):
                active_rows = query_rows[
                    activated[:, column].cpu()
                ].tolist()
                positive_rows = positives.get(
                    int(mode["target_anchor"]), set()
                )
                counter = counters[mode["key"]]
                counter[(trajectory, "activation")] += len(active_rows)
                counter[(trajectory, "correct")] += sum(
                    int(int(row) in positive_rows) for row in active_rows
                )

    accepted: dict[tuple[int, int], bool] = {}
    for key, report in mode_reports.items():
        total_activation = 0
        total_correct = 0
        supported = 0
        trajectory_safe = True
        for trajectory in report["support_trajectories"]:
            outcome = report["trajectory_outcomes"].setdefault(
                trajectory, {"eligible": True}
            )
            if not outcome.get("eligible", True):
                trajectory_safe = False
                continue
            activation = int(counters[key][(trajectory, "activation")])
            correct = int(counters[key][(trajectory, "correct")])
            precision = correct / max(activation, 1)
            outcome.update(
                {
                    "activation_count": activation,
                    "correct_activation_count": correct,
                    "precision": precision,
                }
            )
            total_activation += activation
            total_correct += correct
            if correct >= int(
                config.minimum_true_activations_per_trajectory
            ):
                supported += 1
            if (
                correct
                < int(config.minimum_true_activations_per_trajectory)
                or precision < float(config.minimum_trajectory_precision)
            ):
                trajectory_safe = False
        global_precision = total_correct / max(total_activation, 1)
        keep = bool(
            trajectory_safe
            and supported >= int(config.minimum_supported_trajectories)
            and global_precision >= float(config.minimum_global_precision)
        )
        accepted[key] = keep
        report.update(
            {
                "activation_count": total_activation,
                "correct_activation_count": total_correct,
                "global_precision": global_precision,
                "supported_trajectory_count": supported,
                "accepted": keep,
            }
        )

    prototype_count = len(
        torch.as_tensor(routed_family["prototype_features"])
    )
    if not 0 <= int(base_family_count) <= prototype_count:
        raise ValueError("invalid base family count")
    kept_prototypes = list(range(int(base_family_count)))
    for prototype in range(int(base_family_count), prototype_count):
        key = (ROUTE_FAMILY, int(anchor_count) + prototype)
        if accepted.get(key, False):
            kept_prototypes.append(prototype)
    old_to_new_representation = {
        int(anchor_count) + old: int(anchor_count) + new
        for new, old in enumerate(kept_prototypes)
    }

    calibrated_records = []
    route_counts = Counter()
    for record in routed_audit["records"]:
        routes = torch.as_tensor(record["route"]).long().clone()
        representations = torch.as_tensor(
            record["target_representation"]
        ).long().clone()
        for local, (route, representation) in enumerate(
            zip(routes.tolist(), representations.tolist())
        ):
            if int(route) not in (ROUTE_PRIMARY, ROUTE_FAMILY):
                route_counts["reject"] += 1
                continue
            key = (int(route), int(representation))
            if not accepted.get(key, False):
                routes[local] = ROUTE_REJECT
                representations[local] = -1
                route_counts["reject"] += 1
            elif int(route) == ROUTE_FAMILY:
                representations[local] = old_to_new_representation[
                    int(representation)
                ]
                route_counts["family"] += 1
            else:
                route_counts["primary"] += 1
        calibrated_records.append(
            {
                **record,
                "route": routes.to(torch.int8),
                "target_representation": representations,
            }
        )

    teacher_records = []
    for audit_record, teacher_record in zip(
        calibrated_records, routed_teacher["records"]
    ):
        audit_rows = torch.as_tensor(audit_record["query_rows"]).long()
        audit_routes = torch.as_tensor(audit_record["route"]).long()
        rejected_rows = {
            int(row)
            for row, route in zip(
                audit_rows.tolist(), audit_routes.tolist()
            )
            if int(route) == ROUTE_REJECT
        }
        rows = torch.as_tensor(teacher_record["query_rows"]).long()
        offsets = torch.as_tensor(
            teacher_record["positive_offsets"]
        ).long()
        indices = torch.as_tensor(
            teacher_record["positive_indices"]
        ).long()
        blocks = []
        for local, row in enumerate(rows.tolist()):
            if int(row) in rejected_rows:
                blocks.append(torch.empty(0, dtype=torch.long))
            else:
                blocks.append(
                    indices[int(offsets[local]) : int(offsets[local + 1])]
                )
        counts = torch.as_tensor([len(value) for value in blocks]).long()
        new_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long), counts.cumsum(0))
        )
        new_indices = (
            torch.cat(blocks)
            if int(new_offsets[-1])
            else torch.empty(0, dtype=torch.long)
        )
        teacher_records.append(
            {
                **teacher_record,
                "positive_offsets": new_offsets,
                "positive_indices": new_indices,
            }
        )

    keep = torch.as_tensor(kept_prototypes).long()
    calibrated_family = {
        **routed_family,
        "prototype_features": torch.as_tensor(
            routed_family["prototype_features"]
        )[keep],
        "prototype_anchor_indices": torch.as_tensor(
            routed_family["prototype_anchor_indices"]
        )[keep],
        "prototype_bias": family_bias[keep],
        "prototype_temperature": torch.as_tensor(
            routed_family.get(
                "prototype_temperature",
                torch.ones(prototype_count),
            )
        )[keep],
    }
    report = {
        "schema": "lafgs_repair_activation_calibration",
        "version": 1,
        "config": asdict(config),
        "base_family_count": int(base_family_count),
        "input_family_count": prototype_count,
        "output_family_count": len(kept_prototypes),
        "candidate_mode_count": len(mode_reports),
        "accepted_mode_count": sum(accepted.values()),
        "rejected_mode_count": len(accepted) - sum(accepted.values()),
        "route_counts": dict(route_counts),
        "modes": [
            mode_reports[key] for key in sorted(mode_reports)
        ],
    }
    calibrated_audit = {
        **routed_audit,
        "version": int(routed_audit.get("version", 1)) + 1,
        "records": calibrated_records,
        "summary": {
            **dict(routed_audit.get("summary", {})),
            "activation_calibration": {
                key: value
                for key, value in report.items()
                if key != "modes"
            },
        },
        "repair_activation_calibration": report,
    }
    calibrated_teacher = {
        **routed_teacher,
        "version": int(routed_teacher.get("version", 1)) + 1,
        "records": teacher_records,
        "diagnostics": {
            **dict(routed_teacher.get("diagnostics", {})),
            "activation_calibration": calibrated_audit["summary"][
                "activation_calibration"
            ],
        },
    }
    calibrated_family["repair_activation_calibration"] = {
        key: value for key, value in report.items() if key != "modes"
    }
    return (
        calibrated_audit,
        calibrated_teacher,
        calibrated_family,
        report,
    )


def serialize_config(config: RepairActivationCalibrationConfig) -> dict:
    return asdict(config)
